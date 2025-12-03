def lgbm_reg_params(trial):
    """Parameter space for LightGBM regression tuning."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50, step=5),
        "max_depth": 3,
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }


def lgbm_cls_params(trial):
    """Parameter space for LightGBM classification tuning."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50, step=5),
        "max_depth": 3,
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }
