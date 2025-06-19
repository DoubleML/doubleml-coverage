from typing import Any, Dict, Optional

import doubleml as dml
import numpy as np
from doubleml.datasets import make_irm_data
from scipy.linalg import toeplitz

from montecover.base import BaseSimulation
from montecover.utils import create_learner_from_config


class IRMATTECoverageSimulation(BaseSimulation):
    """Simulation class for coverage properties of DoubleMLIRM for ATTE estimation."""

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

        theta = self.dgp_parameters["theta"][0]
        dim_x = self.dgp_parameters["dim_x"][0]

        n_obs_atte = int(1e6)
        R2_d = 0.5
        R2_y = 0.5

        v = np.random.uniform(
            size=[
                n_obs_atte,
            ]
        )
        zeta = np.random.standard_normal(
            size=[
                n_obs_atte,
            ]
        )

        cov_mat = toeplitz([np.power(0.5, k) for k in range(dim_x)])
        x = np.random.multivariate_normal(
            np.zeros(dim_x),
            cov_mat,
            size=[
                n_obs_atte,
            ],
        )

        beta = [1 / (k**2) for k in range(1, dim_x + 1)]
        b_sigma_b = np.dot(np.dot(cov_mat, beta), beta)
        c_y = np.sqrt(R2_y / ((1 - R2_y) * b_sigma_b))
        c_d = np.sqrt(np.pi**2 / 3.0 * R2_d / ((1 - R2_d) * b_sigma_b))

        xx = np.exp(np.dot(x, np.multiply(beta, c_d)))
        d = 1.0 * ((xx / (1 + xx)) > v)

        # y = d * theta + d * np.dot(x, np.multiply(beta, c_y)) + zeta
        y0 = zeta
        y1 = theta + np.dot(x, np.multiply(beta, c_y)) + zeta

        self.oracle_values = dict()
        self.oracle_values["theta"] = np.mean(y1[d == 1] - y0[d == 1])
        self.logger.info(f"Oracle ATTE value: {self.oracle_values['theta']}")

    def run_single_rep(self, dml_data: dml.DoubleMLData, dml_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single repetition with the given parameters."""
        # Extract parameters
        learner_config = dml_params["learners"]
        learner_g_name, ml_g = create_learner_from_config(learner_config["ml_g"])
        learner_m_name, ml_m = create_learner_from_config(learner_config["ml_m"])

        # Model
        dml_model = dml.DoubleMLIRM(
            obj_dml_data=dml_data,
            ml_g=ml_g,
            ml_m=ml_m,
            score="ATTE",
        )
        dml_model.fit()

        result = {
            "coverage": [],
        }
        for level in self.confidence_parameters["level"]:
            level_result = dict()
            level_result["coverage"] = self._compute_coverage(
                thetas=dml_model.coef,
                oracle_thetas=self.oracle_values["theta"],
                confint=dml_model.confint(level=level),
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

        # Aggregate results (possibly multiple result dfs)
        result_summary = dict()
        for result_name, result_df in self.results.items():
            result_summary[result_name] = result_df.groupby(groupby_cols).agg(aggregation_dict).reset_index()
            self.logger.debug(f"Summarized {result_name} results")

        return result_summary

    def _generate_dml_data(self, dgp_params: Dict[str, Any]) -> dml.DoubleMLData:
        """Generate data for the simulation."""
        data = make_irm_data(
            theta=dgp_params["theta"],
            n_obs=dgp_params["n_obs"],
            dim_x=dgp_params["dim_x"],
            return_type="DataFrame",
        )
        dml_data = dml.DoubleMLData(data, "y", "d")
        return dml_data
