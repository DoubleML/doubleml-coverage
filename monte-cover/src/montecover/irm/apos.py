from typing import Any, Dict, Optional

import doubleml as dml
import numpy as np
import pandas as pd
from doubleml.irm.datasets import make_irm_data_discrete_treatments

from montecover.base import BaseSimulation
from montecover.utils import create_learner_from_config


class APOSCoverageSimulation(BaseSimulation):
    """Simulation class for coverage properties of DoubleMLAPOs for APO estimation."""

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

        required_learners = ["ml_g", "ml_m"]
        for learner in self.dml_parameters["learners"]:
            for ml in required_learners:
                assert ml in learner, f"No {ml} specified in the config file"

    def _calculate_oracle_values(self):
        """Calculate oracle values for the simulation."""
        self.logger.info("Calculating oracle values")

        n_levels = self.dgp_parameters["n_levels"][0]
        data_apo_oracle = make_irm_data_discrete_treatments(
            n_obs=int(1e6), n_levels=n_levels, linear=self.dgp_parameters["linear"][0]
        )

        y0 = data_apo_oracle["oracle_values"]["y0"]
        ite = data_apo_oracle["oracle_values"]["ite"]
        d = data_apo_oracle["d"]

        average_ites = np.full(n_levels + 1, np.nan)
        apos = np.full(n_levels + 1, np.nan)
        for i in range(n_levels + 1):
            average_ites[i] = np.mean(ite[d == i]) * (i > 0)
            apos[i] = np.mean(y0) + average_ites[i]

        ates = np.full(n_levels, np.nan)
        for i in range(n_levels):
            ates[i] = apos[i + 1] - apos[0]

        self.logger.info(
            f"Levels and their counts:\n{np.unique(d, return_counts=True)}"
        )
        self.logger.info(f"True APOs: {apos}")
        self.logger.info(f"True ATEs: {ates}")

        self.oracle_values = dict()
        self.oracle_values["apos"] = apos
        self.oracle_values["ates"] = ates

    def run_single_rep(
        self, dml_data: dml.DoubleMLData, dml_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single repetition with the given parameters."""
        # Extract parameters
        learner_config = dml_params["learners"]
        learner_g_name, ml_g = create_learner_from_config(learner_config["ml_g"])
        learner_m_name, ml_m = create_learner_from_config(learner_config["ml_m"])
        treatment_levels = dml_params["treatment_levels"]
        trimming_threshold = dml_params["trimming_threshold"]

        # Model
        dml_model = dml.DoubleMLAPOS(
            obj_dml_data=dml_data,
            ml_g=ml_g,
            ml_m=ml_m,
            treatment_levels=treatment_levels,
            trimming_threshold=trimming_threshold,
        )
        dml_model.fit()
        dml_model.bootstrap(n_rep_boot=2000)

        causal_contrast_model = dml_model.causal_contrast(reference_levels=0)
        causal_contrast_model.bootstrap(n_rep_boot=2000)

        result = {
            "coverage": [],
            "causal_contrast": [],
        }
        for level in self.confidence_parameters["level"]:
            level_result = dict()
            level_result["coverage"] = self._compute_coverage(
                thetas=dml_model.coef,
                oracle_thetas=self.oracle_values["apos"],
                confint=dml_model.confint(level=level),
                joint_confint=dml_model.confint(level=level, joint=True),
            )
            level_result["causal_contrast"] = self._compute_coverage(
                thetas=causal_contrast_model.thetas,
                oracle_thetas=self.oracle_values["ates"],
                confint=causal_contrast_model.confint(level=level),
                joint_confint=causal_contrast_model.confint(level=level, joint=True),
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
            result_summary[result_name] = (
                result_df.groupby(groupby_cols).agg(aggregation_dict).reset_index()
            )
            self.logger.debug(f"Summarized {result_name} results")

        return result_summary

    def _generate_dml_data(self, dgp_params: Dict[str, Any]) -> dml.DoubleMLData:
        """Generate data for the simulation."""
        data = make_irm_data_discrete_treatments(
            n_obs=dgp_params["n_obs"],
            n_levels=dgp_params["n_levels"],
            linear=dgp_params["linear"],
        )
        df_apo = pd.DataFrame(
            np.column_stack((data["y"], data["d"], data["x"])),
            columns=["y", "d"] + ["x" + str(i) for i in range(data["x"].shape[1])],
        )
        dml_data = dml.DoubleMLData(df_apo, "y", "d")
        return dml_data
