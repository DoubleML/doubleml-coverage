from typing import Any, Dict, Optional

import doubleml as dml
from doubleml.datasets import make_plr_CCDDHNR2018
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV

from montecover.base import BaseSimulation


class PLRATECoverageSimulation(BaseSimulation):
    """Simulation study for coverage properties of DoubleMLPLR for ATE estimation."""

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

        # Additional results storage for aggregated results
        self.results_aggregated = []

        # Calculate oracle values
        self._calculate_oracle_values()

    def _process_config_parameters(self):
        """Process simulation-specific parameters from config"""
        # Process ML models in parameter grid
        assert "learners" in self.dml_parameters, "No learners specified in the config file"
        for learner in self.dml_parameters["learners"]:
            assert "ml_g" in learner, "No ml_g specified in the config file"
            assert "ml_m" in learner, "No ml_m specified in the config file"

            # Convert ml_g strings to actual objects
            learner["ml_g"] = self._convert_ml_string_to_object(learner["ml_g"][0])
            learner["ml_m"] = self._convert_ml_string_to_object(learner["ml_m"][0])

    def _convert_ml_string_to_object(self, ml_string):
        """Convert a string to a machine learning object."""
        if ml_string == "Lasso":
            learner = LassoCV()
        elif ml_string == "Random Forest":
            learner = RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
        elif ml_string == "LGBM":
            learner = LGBMRegressor(n_estimators=100, learning_rate=0.05, verbose=-1, n_jobs=1)
        else:
            raise ValueError(f"Unknown learner type: {ml_string}")

        return (ml_string, learner)

    def _calculate_oracle_values(self):
        """Calculate oracle values for the simulation."""
        self.logger.info("Calculating oracle values")

        self.oracle_values = dict()
        self.oracle_values["theta"] = self.dgp_parameters["theta"]

    def run_single_rep(self, dml_data, dml_params) -> Dict[str, Any]:
        """Run a single repetition with the given parameters."""
        # Extract parameters
        learner_g_name, ml_g = dml_params["learners"]["ml_g"]
        learner_m_name, ml_m = dml_params["learners"]["ml_m"]
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
        }

        # Aggregate results (possibly multiple result dfs)
        result_summary = dict()
        for result_name, result_df in self.results.items():
            result_summary[result_name] = result_df.groupby(groupby_cols).agg(aggregation_dict).reset_index()
            self.logger.debug(f"Summarized {result_name} results")

        return result_summary

    def _generate_dml_data(self, dgp_params) -> dml.DoubleMLData:
        """Generate data for the simulation."""
        data = make_plr_CCDDHNR2018(
            alpha=dgp_params["theta"], n_obs=dgp_params["n_obs"], dim_x=dgp_params["dim_x"], return_type="DataFrame"
        )
        dml_data = dml.DoubleMLData(data, "y", "d")
        return dml_data
