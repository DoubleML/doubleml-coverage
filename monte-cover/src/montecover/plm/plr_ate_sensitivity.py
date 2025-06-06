from typing import Any, Dict, Optional

import doubleml as dml
import numpy as np
import pandas as pd
from doubleml.datasets import make_confounded_plr_data

from montecover.base import BaseSimulation
from montecover.utils import create_learner_from_config


class PLRATESensitivityCoverageSimulation(BaseSimulation):
    """Simulation class for sensitivity properties of DoubleMLPLR for ATE estimation."""

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

        # hardcoded parameters for omitted confounders
        cf_y = 0.1
        cf_d = 0.1

        np.random.seed(42)
        dgp_dict = make_confounded_plr_data(n_obs=int(1e6), cf_y=cf_y, cf_d=cf_d, theta=self.dgp_parameters["theta"])
        oracle_dict = dgp_dict["oracle_values"]
        cf_y_test = np.mean(np.square(oracle_dict["g_long"] - oracle_dict["g_short"])) / np.mean(
            np.square(dgp_dict["y"] - oracle_dict["g_short"])
        )
        self.logger.info(f"Input cf_y:{cf_y} \nCalculated cf_y: {round(cf_y_test, 5)}")

        rr_long = (dgp_dict["d"] - oracle_dict["m_long"]) / np.mean(np.square(dgp_dict["d"] - oracle_dict["m_long"]))
        rr_short = (dgp_dict["d"] - oracle_dict["m_short"]) / np.mean(np.square(dgp_dict["d"] - oracle_dict["m_short"]))
        C2_D = (np.mean(np.square(rr_long)) - np.mean(np.square(rr_short))) / np.mean(np.square(rr_short))
        cf_d_test = C2_D / (1 + C2_D)
        self.logger.info(f"Input cf_d:{cf_d}\nCalculated cf_d: {round(cf_d_test, 5)}")

        # compute the value for rho
        rho = np.corrcoef((oracle_dict["g_long"] - oracle_dict["g_short"]), (rr_long - rr_short))[0, 1]
        self.logger.info(f"Correlation rho: {round(rho, 5)}")

        self.oracle_values = {
            "theta": self.dgp_parameters["theta"],
            "cf_y": cf_y,
            "cf_d": cf_d,
            "rho": rho,
        }

    def run_single_rep(self, dml_data, dml_params) -> Dict[str, Any]:
        """Run a single repetition with the given parameters."""
        # Extract parameters
        learner_config = dml_params["learners"]
        learner_g_name, ml_g = create_learner_from_config(learner_config["ml_g"])
        learner_m_name, ml_m = create_learner_from_config(learner_config["ml_m"])
        score = dml_params["score"]
        theta = self.oracle_values["theta"][0]

        # Model
        dml_model = dml.DoubleMLPLR(
            obj_dml_data=dml_data,
            ml_l=ml_g,
            ml_m=ml_m,
            ml_g=ml_g if score == "IV-type" else None,
            score=score,
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
                        "Score": score,
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
        groupby_cols = ["Learner g", "Learner m", "Score", "level"]
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
        dgp_dict = make_confounded_plr_data(
            n_obs=dgp_params["n_obs"],
            cf_y=self.oracle_values["cf_y"],
            cf_d=self.oracle_values["cf_d"],
            theta=dgp_params["theta"],
        )
        x_cols = [f"X{i + 1}" for i in np.arange(dgp_dict["x"].shape[1])]
        df = pd.DataFrame(
            np.column_stack((dgp_dict["x"], dgp_dict["y"], dgp_dict["d"])),
            columns=x_cols + ["y", "d"],
        )
        dml_data = dml.DoubleMLData(df, "y", "d")
        return dml_data
