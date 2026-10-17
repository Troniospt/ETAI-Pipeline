"""Model and full preprocessing/model pipeline construction."""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.random_projection import SparseRandomProjection
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor


def oblique_random_forest(n_estimators: int = 100, n_components: int = 10, density="auto",
                          random_state=None, n_jobs=None, **tree_params):
    """Build a bagged forest whose trees split random linear projections."""
    tree_params.setdefault("max_features", "sqrt")
    oblique_tree = Pipeline([
        ("project", SparseRandomProjection(
            n_components=n_components, density=density, random_state=random_state
        )),
        ("tree", DecisionTreeClassifier(random_state=random_state, **tree_params)),
    ])
    return BaggingClassifier(
        estimator=oblique_tree,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def adaboost(max_depth: int = 1, random_state=None, **params):
    """Build AdaBoost with decision trees of configurable depth."""
    estimator = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    return AdaBoostClassifier(estimator=estimator, random_state=random_state, **params)

_MODELS = {
    # always predicts the most frequent class -- the floor every real model must beat
    "dummy": DummyClassifier,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "oblique_random_forest": oblique_random_forest,
    "adaboost": adaboost,
}


def build_model(model_config: dict):
    model_type = model_config["type"]
    params = model_config.get("params", {})

    if model_type not in _MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Options: {list(_MODELS)}")

    return _MODELS[model_type](**params)


def build_pipeline(preprocessing_config: dict, model_config: dict) -> Pipeline:
    """Keep every learned preprocessing operation inside the fitted estimator."""
    return Pipeline([
        ("prep", build_preprocessor(preprocessing_config)),
        ("model", build_model(model_config)),
    ])
