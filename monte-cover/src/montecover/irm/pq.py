from typing import Any, Dict, Optional

import doubleml as dml
import numpy as np
import pandas as pd

from montecover.base import BaseSimulation
from montecover.utils import create_learner_from_config


# define loc-scale model
def f_loc(D, X):
    loc = 0.5 * D + 2 * D * X[:, 4] + 2.0 * (X[:, 1] > 0.1) - 1.7 * (X[:, 0] * X[:, 2] > 0) - 3 * X[:, 3]
    return loc


def f_scale(D, X):
    scale = np.sqrt(0.5 * D + 0.3 * D * X[:, 1] + 2)
    return scale


def dgp(n=200, p=5):
    X = np.random.uniform(-1, 1, size=[n, p])
    D = ((X[:, 1] - X[:, 3] + 1.5 * (X[:, 0] > 0) + np.random.normal(size=n)) > 0) * 1.0
    epsilon = np.random.normal(size=n)

    Y = f_loc(D, X) + f_scale(D, X) * epsilon
    return Y, X, D, epsilon


class PQCoverageSimulation(BaseSimulation):
    """Simulation class for coverage properties of DoubleMLPQ for potential quantile estimation."""

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

    def _process_config_parameters(self):
        """Process simulation-specific parameters from config"""
        # Process ML models in parameter grid
        assert "learners" in self.dml_parameters, "No learners specified in the config file"

        required_learners = ["ml_g", "ml_m"]
        for learner in self.dml_parameters["learners"]:
            for ml in required_learners:
                assert ml in learner, f"No {ml} specified in the config file"

    def _calculate_oracle_values(self):
        """Calculate oracle values for the simulation."""
        self.logger.info("Calculating oracle values")

        # Parameters
        n_true = int(10e6)
        tau_vec = self.dml_parameters["tau_vec"][0]
        p = self.dgp_parameters["dim_x"][0]

        _, X_true, _, epsilon_true = dgp(n=n_true, p=p)
        D1 = np.ones(n_true)
        D0 = np.zeros(n_true)

        Y1 = f_loc(D1, X_true) + f_scale(D1, X_true) * epsilon_true
        Y0 = f_loc(D0, X_true) + f_scale(D0, X_true) * epsilon_true

        Y1_quant = np.quantile(Y1, q=tau_vec)
        Y0_quant = np.quantile(Y0, q=tau_vec)
        effect_quant = Y1_quant - Y0_quant

        self.oracle_values = dict()
        self.oracle_values["Y0_quant"] = Y0_quant
        self.oracle_values["Y1_quant"] = Y1_quant
        self.oracle_values["effect_quant"] = effect_quant

        self.logger.info(f"Oracle values: {self.oracle_values}")

    def run_single_rep(self, dml_data: dml.DoubleMLData, dml_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single repetition with the given parameters."""
        # Extract parameters
        learner_config = dml_params["learners"]
        learner_g_name, ml_g = create_learner_from_config(learner_config["ml_g"])
        learner_m_name, ml_m = create_learner_from_config(learner_config["ml_m"])
        tau_vec = dml_params["tau_vec"]
        trimming_threshold = dml_params["trimming_threshold"]
        Y0_quant = self.oracle_values["Y0_quant"]
        Y1_quant = self.oracle_values["Y1_quant"]
        effect_quant = self.oracle_values["effect_quant"]

        # Model
        dml_model = dml.DoubleMLQTE(
            obj_dml_data=dml_data,
            ml_g=ml_g,
            ml_m=ml_m,
            score="PQ",
            quantiles=tau_vec,
            trimming_threshold=trimming_threshold,
        )
        dml_model.fit()
        dml_model.bootstrap(n_rep_boot=2000)

        result = {
            "Y0_coverage": [],
            "Y1_coverage": [],
            "effect_coverage": [],
        }
        for level in self.confidence_parameters["level"]:
            level_result = dict()
            level_result["effect_coverage"] = self._compute_coverage(
                thetas=dml_model.coef,
                oracle_thetas=effect_quant,
                confint=dml_model.confint(level=level),
                joint_confint=dml_model.confint(level=level, joint=True),
            )

            Y0_estimates = np.full(len(tau_vec), np.nan)
            Y1_estimates = np.full(len(tau_vec), np.nan)

            Y0_confint = np.full((len(tau_vec), 2), np.nan)
            Y1_confint = np.full((len(tau_vec), 2), np.nan)

            for tau_idx in range(len(tau_vec)):
                model_Y0 = dml_model.modellist_0[tau_idx]
                model_Y1 = dml_model.modellist_1[tau_idx]

                Y0_estimates[tau_idx] = model_Y0.coef
                Y1_estimates[tau_idx] = model_Y1.coef

                Y0_confint[tau_idx, :] = model_Y0.confint(level=level)
                Y1_confint[tau_idx, :] = model_Y1.confint(level=level)

            Y0_confint_df = pd.DataFrame(Y0_confint, columns=["lower", "upper"])
            Y1_confint_df = pd.DataFrame(Y1_confint, columns=["lower", "upper"])

            level_result["Y0_coverage"] = self._compute_coverage(
                thetas=Y0_estimates,
                oracle_thetas=Y0_quant,
                confint=Y0_confint_df,
                joint_confint=None,
            )

            level_result["Y1_coverage"] = self._compute_coverage(
                thetas=Y1_estimates,
                oracle_thetas=Y1_quant,
                confint=Y1_confint_df,
                joint_confint=None,
            )

            # add parameters to the result
            for res_metric in level_result.values():
                res_metric.update(
                    {
                        "Learner g": learner_g_name,
                        "Learner m": learner_m_name,
                        "level": level,
                    }
                )
            for key, res in level_result.items():
                result[key].append(res)

        return result

    def summarize_results(self):
        """Summarize the simulation results."""
        self.logger.info("Summarizing simulation results")

        # Group by parameter combinations
        groupby_cols = ["Learner g", "Learner m", "level"]
        aggregation_dict = {
            "Coverage": "mean",
            "CI Length": "mean",
            "Bias": "mean",
            "repetition": "count",
        }

        result_summary = dict()
        # Aggregate results for Y0 and Y1
        for result_name in ["Y0_coverage", "Y1_coverage"]:
            df = self.results[result_name]
            result_summary[result_name] = df.groupby(groupby_cols).agg(aggregation_dict).reset_index()
            self.logger.debug(f"Summarized {result_name} results")

        uniform_aggregation_dict = {
            "Coverage": "mean",
            "CI Length": "mean",
            "Bias": "mean",
            "Uniform Coverage": "mean",
            "Uniform CI Length": "mean",
            "repetition": "count",
        }
        result_summary["effect_coverage"] = (
            self.results["effect_coverage"].groupby(groupby_cols).agg(uniform_aggregation_dict).reset_index()
        )
        self.logger.debug("Summarized effect_coverage results")

        return result_summary

    def _generate_dml_data(self, dgp_params: Dict[str, Any]) -> dml.DoubleMLData:
        """Generate data for the simulation."""
        Y, X, D, _ = dgp(n=dgp_params["n_obs"], p=dgp_params["dim_x"])
        dml_data = dml.DoubleMLData.from_arrays(X, Y, D)
        return dml_data
