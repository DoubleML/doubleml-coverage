from typing import Any, Dict, Optional

import doubleml as dml
from doubleml.plm.datasets import make_plr_cluster_data

from montecover.base import BaseSimulation
from montecover.utils import create_learner_from_config


class PLRATEClusterCoverageSimulation(BaseSimulation):
    """Simulation class for coverage properties of DoubleMLPLR for ATE estimation."""

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

        self.oracle_values = dict()
        self.oracle_values["theta"] = self.dgp_parameters["alpha"]

    def run_single_rep(self, dml_data, dml_params) -> Dict[str, Any]:
        """Run a single repetition with the given parameters."""
        # Extract parameters
        learner_config = dml_params["learners"]
        learner_g_name, ml_g = create_learner_from_config(learner_config["ml_g"])
        learner_m_name, ml_m = create_learner_from_config(learner_config["ml_m"])
        score = dml_params["score"]

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
        data_dict = make_plr_cluster_data(
            n_clusters1=dgp_params["n_clusters1"],
            n_clusters2=dgp_params["n_clusters2"],
            dim_x=dgp_params["dim_x"],
            alpha=dgp_params["alpha"],
            obs_per_cluster=dgp_params["obs_per_cluster"],
            linear=dgp_params["linear"],
            cluster_correlation=dgp_params["cluster_correlation"],
            error_correlation=dgp_params["error_correlation"],
            cluster_size_variation=dgp_params["cluster_size_variation"],
        )
        x_cols = [f"X{i}" for i in range(1, dgp_params["dim_x"] + 1)]
        if dgp_params["n_clusters2"] is None:
            cluster_cols = ["cluster1"]
        else:
            cluster_cols = ["cluster1", "cluster2"]
        dml_data = dml.DoubleMLData(data_dict["data"], "y", "d", x_cols=x_cols, cluster_cols=cluster_cols)
        return dml_data
