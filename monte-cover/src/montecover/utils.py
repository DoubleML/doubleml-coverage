from typing import Any, Callable, Dict, Tuple

from doubleml.utils import GlobalClassifier, GlobalRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingClassifier, StackingRegressor
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression, Ridge

LearnerInstantiator = Callable[[Dict[str, Any]], Any]
# Map learner abbreviations to their instantiation logic
LEARNER_REGISTRY: Dict[str, LearnerInstantiator] = {
    "LassoCV": lambda params: LassoCV(**params),
    "RF Regr.": lambda params: RandomForestRegressor(**params),
    "RF Clas.": lambda params: RandomForestClassifier(**params),
    "LGBM Regr.": lambda params: LGBMRegressor(**{**{"verbose": -1, "n_jobs": 1}, **params}),
    "LGBM Clas.": lambda params: LGBMClassifier(**{**{"verbose": -1, "n_jobs": 1}, **params}),
    "Linear": lambda params: LinearRegression(**params),
    "Logistic": lambda params: LogisticRegression(**params),
    "Global Linear": lambda params: GlobalRegressor(LinearRegression(**params)),
    "Global Logistic": lambda params: GlobalClassifier(LogisticRegression(**params)),
    "Stacked Regr.": lambda params: StackingRegressor(
        estimators=[
            ("lr", LinearRegression()),
            (
                "lgbm",
                LGBMRegressor(**{**{"verbose": -1, "n_jobs": 1}, **params}),
            ),
            ("glr", GlobalRegressor(LinearRegression())),
        ],
        final_estimator=Ridge(),
    ),
    "Stacked Clas.": lambda params: StackingClassifier(
        estimators=[
            ("lr", LogisticRegression()),
            (
                "lgbm",
                LGBMClassifier(**{**{"verbose": -1, "n_jobs": 1}, **params}),
            ),
            ("glr", GlobalClassifier(LogisticRegression())),
        ],
        final_estimator=LogisticRegression(),
    ),
}


def create_learner_from_config(learner_config: Dict[str, Any]) -> Tuple[str, Any]:
    """
    Instantiates a machine learning model based on a configuration dictionary.
    The 'name' in learner_config should use the defined abbreviations.

    Args:
        learner_config: A dictionary containing 'name' (str) for the learner
                        (e.g., "LassoCV", "RF Regr.", "RF Clas.", "LGBM Regr.",
                        "LGBM Clas.", "Linear", "Logostic")
                        and optionally 'params' (dict) for its hyperparameters.

    Returns:
        A tuple containing the learner's abbreviated name (str) and the instantiated learner object.

    Raises:
        ValueError: If the learner name in the config is unknown.
    """
    learner_name_abbr = learner_config["name"]
    params = learner_config.get("params", {})

    if learner_name_abbr not in LEARNER_REGISTRY:
        raise ValueError(
            f"Unknown learner name abbreviation in config: {learner_name_abbr}. "
            f"Available learners are: {', '.join(LEARNER_REGISTRY.keys())}"
        )

    instantiator = LEARNER_REGISTRY[learner_name_abbr]
    learner = instantiator(params)

    return (learner_name_abbr, learner)
