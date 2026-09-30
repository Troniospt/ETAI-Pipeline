"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

This orchestrates the full pipeline:
    load config -> load data -> diagnose/clean (week 3) -> drop duplicate rows (training only, week 4)
    -> split features/target
    -> set the final test set aside, locked (week 4)
    -> compare all configured models with stratified cross-validation
    -> select by mean CV score and refit on the whole development set
    -> evaluate the selected model once on the locked test set -> save results
"""
import yaml
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data import load_data
from src.preprocessing import (
    clean_dataset, drop_duplicate_rows, split_features_target, build_preprocessor, split_dev_test,
)
from src.model import build_model
from src.evaluate import (
    cross_validate_pipeline, cv_report, oof_classification_report,
    test_classification_report, fairness_report,
)
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def configured_models(config: dict) -> list:
    """Return the configured model list, while accepting the old single-model format."""
    if "models" in config:
        return config["models"]
    return [config["model"]]


def main():
    config = load_config()

    # load + diagnose-and-clean (week 3): nothing here is learned from the data, so it's
    # safe to run on the whole dataset -- see src/preprocessing.py
    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw, config["diagnostics"])          # row-preserving: also safe for new data
    # training data only: the same person must not count twice, or sit in both dev and test
    df_clean = drop_duplicate_rows(df_clean, config["diagnostics"].get("id_column"))

    mnar_sources = config["preprocessing"].get("mnar_indicator_sources", [])
    X, y, extras = split_features_target(df_clean, config["data"], mnar_sources)

    # week 4: the final test set is set aside HERE and never used again in this script.
    # Every decision from now on (preprocessing, model, hyperparameters) is made on the
    # development set only. The test set is used only for the final assessment.
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    model_configs = configured_models(config)
    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)
    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=shuffle,
        random_state=cv_config.get("random_state") if shuffle else None,
    )
    cv_scoring = cv_config.get("scoring", "accuracy")
    cv_reports = []
    cv_rows = []

    print(f"\n{'#' * 60}\nCROSS-VALIDATION COMPARISON\n{'#' * 60}")
    for model_config in model_configs:
        model_name = model_config["type"]
        print(f"\n{'=' * 60}\nMODEL: {model_name}\n{'=' * 60}")
        pipeline = Pipeline([
            ("prep", build_preprocessor(config["preprocessing"])),
            ("model", build_model(model_config)),
        ])
        fold_scores, y_oof = cross_validate_pipeline(
            pipeline, X_dev, y_dev, cv, cv_scoring, n_jobs=cv_config.get("n_jobs", 1)
        )
        model_report = f"MODEL: {model_name}\n{'=' * 60}\n"
        model_report += cv_report(fold_scores, cv_scoring)
        model_report += "\n\n" + oof_classification_report(y_dev, y_oof)
        model_report += "\n" + fairness_report(
            y_dev, y_oof, extras_dev,
            sensitive_attr=config["data"]["sensitive_attr"],
        )
        cv_reports.append(model_report)
        cv_rows.append({
            "model": model_name,
            "train_mean": fold_scores["train"].mean(),
            "validation_mean": fold_scores["validation"].mean(),
            "validation_std": fold_scores["validation"].std(ddof=1),
            "gap_mean": fold_scores["gap"].mean(),
        })

    cv_comparison = pd.DataFrame(cv_rows).sort_values(
        "validation_mean", ascending=False
    ).reset_index(drop=True)
    cv_comparison_text = (
        f"\n{'=' * 60}\nCROSS-VALIDATION MODEL COMPARISON ({cv_scoring})\n{'=' * 60}\n"
        + cv_comparison.to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print(cv_comparison_text)

    # Cross-validation is the more stable estimate, so it determines the final model.
    best_name = cv_comparison.loc[0, "model"]
    best_config = next(model for model in configured_models(config) if model["type"] == best_name)
    final_model = Pipeline([
        ("prep", build_preprocessor(config["preprocessing"])),
        ("model", build_model(best_config)),
    ]).fit(X_dev, y_dev)
    refit = f"Final model: {best_name} refit on all {len(X_dev)} development rows."
    print(refit)

    y_test_pred = final_model.predict(X_test)
    final_test_report = test_classification_report(y_test, y_test_pred)
    final_test_report += "\n" + fairness_report(
        y_test, y_test_pred, extras_test,
        sensitive_attr=config["data"]["sensitive_attr"],
        evaluation_label="locked test set, final model only",
    )
    final_note = (
        f"Locked test set evaluated once: {len(X_test)} rows. "
        f"Development set: {len(X_dev)} rows."
    )
    print(final_note)
    report = (
        "CROSS-VALIDATION RESULTS\n" + "\n\n".join(cv_reports)
        + cv_comparison_text
        + "\n\n" + refit + "\n\n" + final_test_report + "\n" + final_note + "\n"
    )

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
