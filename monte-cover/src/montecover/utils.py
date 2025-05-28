from typing import Any, Callable, Dict, Tuple

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression

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
