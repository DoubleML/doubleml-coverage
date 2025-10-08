from typing import Any, Dict, Optional

import doubleml as dml
from doubleml.plm.datasets import make_pliv_multiway_cluster_CKMS2021

from montecover.base import BaseSimulation
from montecover.utils import create_learner_from_config


class PLIVLATEClusterCoverageSimulation(BaseSimulation):
    """Simulation class for PLIV LATE coverage with cluster data."""

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
        assert (
            "learners" in self.dml_parameters
        ), "No learners specified in the config file"

        required_learners = ["ml_g", "ml_m", "ml_r"]
        for learner in self.dml_parameters["learners"]:
            for ml in required_learners:
                assert ml in learner, f"No {ml} specified in the config file"

    def _calculate_oracle_values(self):
        """Calculate oracle values for the simulation."""
        self.logger.info("Calculating oracle values")

        self.oracle_values = dict()
        self.oracle_values["theta"] = self.dgp_parameters["theta"]

    def run_single_rep(self, dml_data, dml_params) -> Dict[str, Any]:
        """Run a single repetition with the given parameters."""

        # Extract parameters
        learner_config = dml_params["learners"]
        learner_g_name, ml_g = create_learner_from_config(learner_config["ml_g"])
        learner_m_name, ml_m = create_learner_from_config(learner_config["ml_m"])
        learner_r_name, ml_r = create_learner_from_config(learner_config["ml_r"])
        score = dml_params["score"]

        # Model
        dml_model = dml.DoubleMLPLIV(
            obj_dml_data=dml_data,
            ml_l=ml_g,
            ml_m=ml_m,
            ml_g=ml_g if score == "IV-type" else None,
            ml_r=ml_r,
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
                oracle_thetas=self.oracle_values["theta"],
                confint=dml_model.confint(level=level),
                joint_confint=None,
            )

            # add parameters to the result
            for res in level_result.values():
                res.update(
                    {
                        "Learner g": learner_g_name,
                        "Learner m": learner_m_name,
                        "Learner r": learner_r_name,
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
        groupby_cols = ["Learner g", "Learner m", "Learner r", "Score", "level"]
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
        dml_data = make_pliv_multiway_cluster_CKMS2021(
            alpha=dgp_params["theta"],
            n_obs=dgp_params["n"],
            dim_x=dgp_params["m"],
            dim_z=dgp_params["dim_X"],
            return_type="DoubleMLData",
        )
        return dml_data
