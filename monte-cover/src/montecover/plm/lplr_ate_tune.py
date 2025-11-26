from typing import Any, Dict, Optional

import doubleml as dml
import optuna
from doubleml.plm.datasets import make_lplr_LZZ2020

from montecover.base import BaseSimulation
from montecover.utils import create_learner_from_config


class LPLRATETuningCoverageSimulation(BaseSimulation):
    """Simulation class for coverage properties of DoubleMLPLR for ATE estimation."""

    def __init__(
        self,
        config_file: str,
        suppress_warnings: bool = True,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        use_failed_scores: bool = False,
    ):
        super().__init__(
            config_file=config_file,
            suppress_warnings=suppress_warnings,
            log_level=log_level,
            log_file=log_file,
        )

        # Calculate oracle values
        self._calculate_oracle_values()
        self._use_failed_scores = use_failed_scores

        # for simplicity, we use the same parameter space for all learners
        def ml_params(trial):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-3, 0.1, log=True
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", 20, 100, step=5
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 10, step=1),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            }

        self._param_space = {
            "ml_M": ml_params,
            "ml_t": ml_params,
            "ml_m": ml_params,
            "ml_a": ml_params,
        }

        self._optuna_settings = {
            "n_trials": 200,
            "show_progress_bar": False,
            "verbosity": optuna.logging.WARNING,  # Suppress Optuna logs
        }

    def _process_config_parameters(self):
        """Process simulation-specific parameters from config"""
        # Process ML models in parameter grid
        assert (
            "learners" in self.dml_parameters
        ), "No learners specified in the config file"

        required_learners = ["ml_m", "ml_M", "ml_t"]
        for learner in self.dml_parameters["learners"]:
            for ml in required_learners:
                assert ml in learner, f"No {ml} specified in the config file"

    def _calculate_oracle_values(self):
        """Calculate oracle values for the simulation."""
        self.logger.info("Calculating oracle values")

        self.oracle_values = dict()
        self.oracle_values["theta"] = self.dgp_parameters["alpha"]

    def run_single_rep(self, dml_data, dml_params) -> Dict[str, Any]:
        """Run a single repetition with the given parameters."""
        # Extract parameters
        learner_config = dml_params["learners"]
        learner_m_name, ml_m = create_learner_from_config(learner_config["ml_m"])
        learner_M_name, ml_M = create_learner_from_config(learner_config["ml_M"])
        learner_t_name, ml_t = create_learner_from_config(learner_config["ml_t"])
        score = dml_params["score"]

        model_inputs = {
            "obj_dml_data": dml_data,
            "ml_m": ml_m,
            "ml_M": ml_M,
            "ml_t": ml_t,
            "score": score,
            "error_on_convergence_failure": not self._use_failed_scores,
        }
        # Model
        dml_model = dml.DoubleMLLPLR(**model_inputs)
        dml_model_tuned = dml.DoubleMLLPLR(**model_inputs)
        dml_model_tuned.tune_ml_models(
            ml_param_space=self._param_space,
            optuna_settings=self._optuna_settings,
        )

        result = {
            "coverage": [],
        }

        for model in [dml_model, dml_model_tuned]:
            try:
                model.fit()
            except RuntimeError as e:
                self.logger.info(f"Exception during fit: {e}")
                return None

            for level in self.confidence_parameters["level"]:
                level_result = dict()
                level_result["coverage"] = self._compute_coverage(
                    thetas=model.coef,
                    oracle_thetas=self.oracle_values["theta"],
                    confint=model.confint(level=level),
                    joint_confint=None,
                )

                # add parameters to the result
                for res in level_result.values():
                    res.update(
                        {
                            "Learner m": learner_m_name,
                            "Learner M": learner_M_name,
                            "Learner t": learner_t_name,
                            "Score": score,
                            "level": level,
                            "Tuned": model is dml_model_tuned,
                        }
                    )
                for key, res in level_result.items():
                    result[key].append(res)

        return result

    def summarize_results(self):
        """Summarize the simulation results."""
        self.logger.info("Summarizing simulation results")

        # Group by parameter combinations
        groupby_cols = [
            "Learner m",
            "Learner M",
            "Learner t",
            "Score",
            "level",
            "Tuned",
        ]
        aggregation_dict = {
            "Coverage": "mean",
            "CI Length": "mean",
            "Bias": "mean",
            "repetition": "count",
        }

        # Aggregate results (possibly multiple result dfs)
        result_summary = dict()
        for result_name, result_df in self.results.items():
            result_summary[result_name] = (
                result_df.groupby(groupby_cols).agg(aggregation_dict).reset_index()
            )
            self.logger.debug(f"Summarized {result_name} results")

        return result_summary

    def _generate_dml_data(self, dgp_params) -> dml.DoubleMLData:
        """Generate data for the simulation."""
        return make_lplr_LZZ2020(**dgp_params)
