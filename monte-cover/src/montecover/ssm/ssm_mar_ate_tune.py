from typing import Any, Dict, Optional

import doubleml as dml
import optuna
from doubleml.irm.datasets import make_ssm_data

from montecover.base import BaseSimulation
from montecover.utils import create_learner_from_config
from montecover.utils_tuning import lgbm_reg_params, lgbm_cls_params


class SSMMarATETuningCoverageSimulation(BaseSimulation):
    """Simulation class for coverage properties of DoubleMLSSM with missing at random for ATE estimation with tuning."""

    def __init__(
        self,
        config_file: str,
        suppress_warnings: bool = True,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
    ):
        super().__init__(
            config_file=config_file,
            suppress_warnings=suppress_warnings,
            log_level=log_level,
            log_file=log_file,
        )

        # Calculate oracle values
        self._calculate_oracle_values()
        # tuning specific settings
        self._param_space = {"ml_g": lgbm_reg_params, "ml_m": lgbm_cls_params, "ml_pi": lgbm_cls_params}

        self._optuna_settings = {
            "n_trials": 50,
            "show_progress_bar": False,
            "verbosity": optuna.logging.WARNING,  # Suppress Optuna logs
        }

    def _process_config_parameters(self):
        """Process simulation-specific parameters from config"""
        # Process ML models in parameter grid
        assert (
            "learners" in self.dml_parameters
        ), "No learners specified in the config file"

        required_learners = ["ml_g", "ml_m", "ml_pi"]
        for learner in self.dml_parameters["learners"]:
            for ml in required_learners:
                assert ml in learner, f"No {ml} specified in the config file"

    def _calculate_oracle_values(self):
        """Calculate oracle values for the simulation."""
        self.logger.info("Calculating oracle values")

        self.oracle_values = dict()
        self.oracle_values["theta"] = self.dgp_parameters["theta"]

    def run_single_rep(
        self, dml_data: dml.DoubleMLData, dml_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single repetition with the given parameters."""
        # Extract parameters
        learner_config = dml_params["learners"]
        learner_g_name, ml_g = create_learner_from_config(learner_config["ml_g"])
        learner_m_name, ml_m = create_learner_from_config(learner_config["ml_m"])
        learner_pi_name, ml_pi = create_learner_from_config(learner_config["ml_pi"])

        # Model
        dml_model = dml.DoubleMLSSM(
            obj_dml_data=dml_data,
            ml_g=ml_g,
            ml_m=ml_m,
            ml_pi=ml_pi,
            score="missing-at-random",
        )

        dml_model_tuned = dml.DoubleMLSSM(
            obj_dml_data=dml_data,
            ml_g=ml_g,
            ml_m=ml_m,
            ml_pi=ml_pi,
            score="missing-at-random",
        )
        dml_model_tuned.tune_ml_models(
            ml_param_space=self._param_space,
            optuna_settings=self._optuna_settings,
        )

        result = {
            "coverage": [],
        }
        for model in [dml_model, dml_model_tuned]:
            model.fit()
            nuisance_loss = model.nuisance_loss
            for level in self.confidence_parameters["level"]:
                level_result = dict()
                level_result["coverage"] = self._compute_coverage(
                    thetas=model.coef,
                    oracle_thetas=self.oracle_values["theta"],
                    confint=model.confint(level=level),
                    joint_confint=None,
                )

                # add parameters to the result
                for res_metric in level_result.values():
                    res_metric.update(
                        {
                            "Learner g": learner_g_name,
                            "Learner m": learner_m_name,
                            "Learner pi": learner_pi_name,
                            "level": level,
                            "Tuned": model is dml_model_tuned,
                            "Loss g_d0": nuisance_loss["ml_g_d0"].mean(),
                            "Loss g_d1": nuisance_loss["ml_g_d1"].mean(),
                            "Loss m": nuisance_loss["ml_m"].mean(),
                            "Loss pi": nuisance_loss["ml_pi"].mean(),
                        }
                    )
                for key, res in level_result.items():
                    result[key].append(res)

        return result

    def summarize_results(self):
        """Summarize the simulation results."""
        self.logger.info("Summarizing simulation results")

        # Group by parameter combinations
        groupby_cols = ["Learner g", "Learner m", "Learner pi", "level", "Tuned"]
        aggregation_dict = {
            "Coverage": "mean",
            "CI Length": "mean",
            "Bias": "mean",
            "Loss g_d0": "mean",
            "Loss g_d1": "mean",
            "Loss m": "mean",
            "Loss pi": "mean",
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

    def _generate_dml_data(self, dgp_params: Dict[str, Any]) -> dml.DoubleMLData:
        """Generate data for the simulation."""
        data = make_ssm_data(
            theta=dgp_params["theta"],
            n_obs=dgp_params["n_obs"],
            dim_x=dgp_params["dim_x"],
            mar=True,
            return_type="DataFrame",
        )
        dml_data = dml.data.DoubleMLSSMData(data, "y", "d", s_col="s")
        return dml_data
