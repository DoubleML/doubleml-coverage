from typing import Any, Dict, Optional

import doubleml as dml
import numpy as np
import pandas as pd
from doubleml.datasets import make_heterogeneous_data

from montecover.base import BaseSimulation
from montecover.utils import create_learner_from_config


class IRMGATECoverageSimulation(BaseSimulation):
    """Simulation class for coverage properties of DoubleMLIRM for GATE estimation."""

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
        self.logger.info(f"Oracle values: {self.oracle_values}")

    def _process_config_parameters(self):
        """Process simulation-specific parameters from config"""
        # Process ML models in parameter grid
        assert "learners" in self.dml_parameters, "No learners specified in the config file"

        required_learners = ["ml_g", "ml_m"]
        for learner in self.dml_parameters["learners"]:
            for ml in required_learners:
                assert ml in learner, f"No {ml} specified in the config file"

    def _generate_groups(self, data):
        """Generate groups for the simulation."""
        groups = pd.DataFrame(
            np.column_stack(
                (
                    data["X_0"] <= 0.3,
                    (data["X_0"] > 0.3) & (data["X_0"] <= 0.7),
                    data["X_0"] > 0.7,
                )
            ),
            columns=["Group 1", "Group 2", "Group 3"],
        )
        return groups

    def _calculate_oracle_values(self):
        """Calculate oracle values for the simulation."""
        # Oracle values
        data_oracle = make_heterogeneous_data(
            n_obs=int(1e6),
            p=self.dgp_parameters["p"][0],
            support_size=self.dgp_parameters["support_size"][0],
            n_x=self.dgp_parameters["n_x"][0],
            binary_treatment=True,
        )

        self.logger.info("Calculating oracle values")
        groups = self._generate_groups(data_oracle["data"])
        oracle_gates = [data_oracle["effects"][groups[group]].mean() for group in groups.columns]

        self.oracle_values = dict()
        self.oracle_values["gates"] = oracle_gates

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
        )
        dml_model.fit()

        # gate
        groups = self._generate_groups(dml_data.data)
        gate_model = dml_model.gate(groups=groups)

        result = {
            "coverage": [],
        }
        for level in self.confidence_parameters["level"]:
            level_result = dict()
            confint = gate_model.confint(level=level)
            effects = confint["effect"]
            uniform_confint = gate_model.confint(level=0.95, joint=True, n_rep_boot=2000)
            level_result["coverage"] = self._compute_coverage(
                thetas=effects,
                oracle_thetas=self.oracle_values["gates"],
                confint=confint.iloc[:, [0, 2]],
                joint_confint=uniform_confint.iloc[:, [0, 2]],
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
            "Uniform Coverage": "mean",
            "Uniform CI Length": "mean",
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
        data = make_heterogeneous_data(
            n_obs=dgp_params["n_obs"],
            p=dgp_params["p"],
            support_size=dgp_params["support_size"],
            n_x=dgp_params["n_x"],
            binary_treatment=True,
        )
        dml_data = dml.DoubleMLData(data["data"], "y", "d")
        return dml_data
