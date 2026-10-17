"""Run preprocessing, default CV, optional nested tuning, and final evaluation."""
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.evaluate import (
    cross_validate_pipeline, cv_report, fairness_report,
    oof_classification_report, test_classification_report,
)
from src.model import build_pipeline
from src.preprocessing import clean_dataset, drop_duplicate_rows, split_dev_test, split_features_target
from src.results import save_run
from src.tuning import nested_cross_validate, tune_pipeline, tuning_report


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def make_cv(config: dict) -> StratifiedKFold:
    values = config["cv"]
    shuffle = values.get("shuffle", True)
    return StratifiedKFold(
        n_splits=values["n_splits"], shuffle=shuffle,
        random_state=values.get("random_state") if shuffle else None,
    )


def main():
    config = load_config()
    df = clean_dataset(load_data(config["data"]["path"]), config["diagnostics"])
    df = drop_duplicate_rows(df, config["diagnostics"].get("id_column"))
    X, y, extras = split_features_target(
        df, config["data"], config["preprocessing"].get("mnar_indicator_sources", [])
    )
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras, test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    model_config = config["model"]
    model_name = model_config["type"]
    pipeline = build_pipeline(config["preprocessing"], model_config)
    outer_cv = make_cv(config)
    scoring = config["cv"].get("scoring", "accuracy")
    n_jobs = config["cv"].get("n_jobs", 1)

    default_scores, default_oof = cross_validate_pipeline(
        pipeline, X_dev, y_dev, outer_cv, scoring, n_jobs=n_jobs
    )
    report_parts = [
        f"MODEL: {model_name}\n" + "=" * 60,
        "DEFAULT CROSS-VALIDATION",
        cv_report(default_scores, scoring),
    ]
    evaluation_oof = default_oof

    tuning = config.get("tuning", {})
    search_space = tuning.get("search_spaces", {}).get(model_name)
    final_pipeline = pipeline
    if tuning.get("enabled", False):
        if not search_space:
            options = list((tuning.get("search_spaces") or {}).keys())
            raise ValueError(
                f"No tuning search space for model type {model_name!r}. "
                f"Available search spaces: {options}"
            )
        inner_cv = StratifiedKFold(
            n_splits=tuning["n_splits"], shuffle=True,
            random_state=tuning.get("random_state"),
        )
        nested_scores, nested_oof = nested_cross_validate(
            pipeline, X_dev, y_dev, outer_cv, inner_cv, scoring, search_space,
            tuning["n_trials"], tuning.get("random_state", 42), n_jobs,
        )
        final_pipeline, study = tune_pipeline(
            pipeline, X_dev, y_dev, inner_cv, scoring, search_space,
            tuning["n_trials"], tuning.get("random_state", 42), n_jobs,
        )
        evaluation_oof = nested_oof
        report_parts.extend([
            "TUNED NESTED CROSS-VALIDATION",
            cv_report(nested_scores, scoring),
            tuning_report(study, nested_scores, scoring),
        ])

    report_parts.extend([
        oof_classification_report(y_dev, evaluation_oof),
        fairness_report(
            y_dev, evaluation_oof, extras_dev, config["data"]["sensitive_attr"]
        ),
    ])

    final_pipeline.fit(X_dev, y_dev)
    y_test_pred = final_pipeline.predict(X_test)
    test_report = test_classification_report(y_test, y_test_pred)
    test_report += "\n" + fairness_report(
        y_test, y_test_pred, extras_test,
        sensitive_attr=config["data"]["sensitive_attr"],
        evaluation_label="locked test set, final model only",
    )
    report_parts.extend([
        f"Final model: {model_name}, refit on all {len(X_dev)} development rows.",
        test_report,
        f"Locked test set evaluated once: {len(X_test)} rows.",
    ])
    report = "\n\n".join(report_parts) + "\n"
    path = save_run(config.get("output", {}).get("results_dir", "results"), config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
