from typing import Any, Dict, Optional

import doubleml as dml
import numpy as np
import pandas as pd
from doubleml.irm.datasets import make_confounded_irm_data

from montecover.base import BaseSimulation
from montecover.utils import create_learner_from_config


class IRMATESensitivityCoverageSimulation(BaseSimulation):
    """Simulation class for sensitivity properties of DoubleMLIRM for ATE estimation."""

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

        dgp_dict = make_confounded_irm_data(
            n_obs=int(1e6),
            theta=self.dgp_parameters["theta"][0],
            gamma_a=self.dgp_parameters["gamma_a"][0],
            beta_a=self.dgp_parameters["beta_a"][0],
            var_epsilon_y=self.dgp_parameters["var_epsilon_y"][0],
            trimming_threshold=self.dgp_parameters["trimming_threshold"][0],
            linear=self.dgp_parameters["linear"][0],
        )

        self.oracle_values = {
            "theta": self.dgp_parameters["theta"],
            "cf_y": dgp_dict["oracle_values"]["cf_y"],
            "cf_d": dgp_dict["oracle_values"]["cf_d_ate"],
            "rho": dgp_dict["oracle_values"]["rho_ate"],
        }
        self.logger.info(f"Oracle values: {self.oracle_values}")

    def run_single_rep(self, dml_data, dml_params) -> Dict[str, Any]:
        """Run a single repetition with the given parameters."""
        # Extract parameters
        learner_config = dml_params["learners"]
        learner_g_name, ml_g = create_learner_from_config(learner_config["ml_g"])
        learner_m_name, ml_m = create_learner_from_config(learner_config["ml_m"])
        trimming_threshold = dml_params["trimming_threshold"]
        theta = self.oracle_values["theta"][0]

        # Model
        dml_model = dml.DoubleMLIRM(
            obj_dml_data=dml_data,
            ml_g=ml_g,
            ml_m=ml_m,
            score="ATE",
            trimming_threshold=trimming_threshold,
        )
        dml_model.fit()

        result = {
            "coverage": [],
        }
        for level in self.confidence_parameters["level"]:
            level_result = dict()
            level_result["coverage"] = self._compute_coverage(
                thetas=dml_model.coef,
                oracle_thetas=theta,
                confint=dml_model.confint(level=level),
                joint_confint=None,
            )

            # sensitvity analysis
            dml_model.sensitivity_analysis(
                cf_y=self.oracle_values["cf_y"],
                cf_d=self.oracle_values["cf_d"],
                rho=self.oracle_values["rho"],
                level=level,
                null_hypothesis=theta,
            )
            sensitivity_results = {
                "Coverage (Lower)": theta >= dml_model.sensitivity_params["ci"]["lower"][0],
                "Coverage (Upper)": theta <= dml_model.sensitivity_params["ci"]["upper"][0],
                "RV": dml_model.sensitivity_params["rv"][0],
                "RVa": dml_model.sensitivity_params["rva"][0],
                "Bias (Lower)": abs(theta - dml_model.sensitivity_params["theta"]["lower"][0]),
                "Bias (Upper)": abs(theta - dml_model.sensitivity_params["theta"]["upper"][0]),
            }
            # add sensitivity results to the level result coverage
            level_result["coverage"].update(sensitivity_results)

            # add parameters to the result
            for res in level_result.values():
                res.update(
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
            "Coverage (Lower)": "mean",
            "Coverage (Upper)": "mean",
            "RV": "mean",
            "RVa": "mean",
            "Bias (Lower)": "mean",
            "Bias (Upper)": "mean",
            "repetition": "count",
        }

        # Aggregate results (possibly multiple result dfs)
        result_summary = dict()
        for result_name, result_df in self.results.items():
            result_summary[result_name] = result_df.groupby(groupby_cols).agg(aggregation_dict).reset_index()
            self.logger.debug(f"Summarized {result_name} results")

        return result_summary

    def _generate_dml_data(self, dgp_params) -> dml.DoubleMLData:
        """Generate data for the simulation."""
        dgp_dict = make_confounded_irm_data(
            n_obs=dgp_params["n_obs"],
            theta=dgp_params["theta"],
            gamma_a=dgp_params["gamma_a"],
            beta_a=dgp_params["beta_a"],
            var_epsilon_y=dgp_params["var_epsilon_y"],
            trimming_threshold=dgp_params["trimming_threshold"],
            linear=dgp_params["linear"],
        )
        x_cols = [f"X{i + 1}" for i in np.arange(dgp_dict["x"].shape[1])]
        df = pd.DataFrame(
            np.column_stack((dgp_dict["x"], dgp_dict["y"], dgp_dict["d"])),
            columns=x_cols + ["y", "d"],
        )
        dml_data = dml.DoubleMLData(df, "y", "d")
        return dml_data
