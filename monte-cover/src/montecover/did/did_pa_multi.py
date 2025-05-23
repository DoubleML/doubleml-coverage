from typing import Any, Dict, Optional

import doubleml as dml
import numpy as np
import pandas as pd
from doubleml.did.datasets import make_did_CS2021
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from montecover.base import BaseSimulation


class DIDMultiCoverageSimulation(BaseSimulation):
    """Simulation study for coverage properties of DoubleMLDIDMulti."""

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

        assert (
            "learners" in self.dml_parameters
        ), "No learners specified in the config file"
        for learner in self.dml_parameters["learners"]:
            assert "ml_g" in learner, "No ml_g specified in the config file"
            assert "ml_m" in learner, "No ml_m specified in the config file"

            # Convert ml_g strings to actual objects
            if learner["ml_g"][0] == "Linear":
                learner["ml_g"] = ("Linear", LinearRegression())
            elif learner["ml_g"][0] == "LGBM":
                learner["ml_g"] = (
                    "LGBM",
                    LGBMRegressor(
                        n_estimators=500, learning_rate=0.02, verbose=-1, n_jobs=1
                    ),
                )
            else:
                raise ValueError(f"Unknown learner type: {learner['ml_g']}")

            # Convert ml_m strings to actual objects
            if learner["ml_m"][0] == "Linear":
                learner["ml_m"] = ("Linear", LogisticRegression())
            elif learner["ml_m"][0] == "LGBM":
                learner["ml_m"] = (
                    "LGBM",
                    LGBMClassifier(
                        n_estimators=500, learning_rate=0.02, verbose=-1, n_jobs=1
                    ),
                )
            else:
                raise ValueError(f"Unknown learner type: {learner['ml_m']}")

    def _calculate_oracle_values(self):
        """Calculate oracle values for the simulation."""
        self.logger.info("Calculating oracle values")

        self.oracle_values = dict()
        # Oracle values
        df_oracle = make_did_CS2021(
            n_obs=int(1e6), dgp_type=1
        )  # does not depend on the DGP type
        df_oracle["ite"] = df_oracle["y1"] - df_oracle["y0"]
        self.oracle_values["detailed"] = (
            df_oracle.groupby(["d", "t"])["ite"].mean().reset_index()
        )

        # Oracle group aggregation
        df_oracle_post_treatment = df_oracle[df_oracle["t"] >= df_oracle["d"]]
        self.oracle_values["group"] = df_oracle_post_treatment.groupby("d")[
            "ite"
        ].mean()

        # Oracle time aggregation
        self.oracle_values["time"] = df_oracle_post_treatment.groupby("t")["ite"].mean()

        # Oracle eventstudy aggregation
        df_oracle["e"] = pd.to_datetime(df_oracle["t"]).values.astype(
            "datetime64[M]"
        ) - pd.to_datetime(df_oracle["d"]).values.astype("datetime64[M]")
        self.oracle_values["eventstudy"] = df_oracle.groupby("e")["ite"].mean()[1:]

    def run_single_rep(self, dml_data, dml_params) -> Dict[str, Any]:
        """Run a single repetition with the given parameters."""
        # Extract parameters
        learner_g_name, ml_g = dml_params["learners"]["ml_g"]
        learner_m_name, ml_m = dml_params["learners"]["ml_m"]
        score = dml_params["score"]
        in_sample_normalization = dml_params["in_sample_normalization"]

        # Model
        dml_model = dml.did.DoubleMLDIDMulti(
            obj_dml_data=dml_data,
            ml_g=ml_g,
            ml_m=None if score == "experimental" else ml_m,
            gt_combinations="standard",
            score=score,
            in_sample_normalization=in_sample_normalization,
        )
        dml_model.fit()
        dml_model.bootstrap(n_rep_boot=2000)

        # Oracle values for this model
        oracle_thetas = np.full_like(dml_model.coef, np.nan)
        for i, (g, _, t) in enumerate(dml_model.gt_combinations):
            group_index = self.oracle_values["detailed"]["d"] == g
            time_index = self.oracle_values["detailed"]["t"] == t
            oracle_thetas[i] = self.oracle_values["detailed"][group_index & time_index][
                "ite"
            ].iloc[0]

        result = {
            "detailed": [],
            "group": [],
            "time": [],
            "eventstudy": [],
        }
        for level in self.confidence_parameters["level"]:
            level_result = dict()
            level_result["detailed"] = self._compute_coverage(
                thetas=dml_model.coef,
                oracle_thetas=oracle_thetas,
                confint=dml_model.confint(level=level),
                joint_confint=dml_model.confint(level=level, joint=True),
            )

            for aggregation_method in ["group", "time", "eventstudy"]:
                agg_obj = dml_model.aggregate(aggregation=aggregation_method)
                agg_obj.aggregated_frameworks.bootstrap(n_rep_boot=2000)

                level_result[aggregation_method] = self._compute_coverage(
                    thetas=agg_obj.aggregated_frameworks.thetas,
                    oracle_thetas=self.oracle_values[aggregation_method].values,
                    confint=agg_obj.aggregated_frameworks.confint(level=level),
                    joint_confint=agg_obj.aggregated_frameworks.confint(
                        level=level, joint=True
                    ),
                )

            # add parameters to the result
            for res in level_result.values():
                res.update(
                    {
                        "Learner g": learner_g_name,
                        "Learner m": learner_m_name,
                        "Score": score,
                        "In-sample-norm.": in_sample_normalization,
                        "level": level,
                    }
                )
            for key, res in level_result.items():
                result[key].append(res)

        return result

    def summarize_results(self):
        """Summarize the simulation results."""
        self.logger.info("Summarizing simulation results")

        groupby_cols = [
            "Learner g",
            "Learner m",
            "Score",
            "In-sample-norm.",
            "DGP",
            "level",
        ]
        aggregation_dict = {
            "Coverage": "mean",
            "CI Length": "mean",
            "Bias": "mean",
            "Uniform Coverage": "mean",
            "Uniform CI Length": "mean",
            "repetition": "count",
        }

        result_summary = dict()
        for result_name, result_df in self.results.items():
            result_summary[result_name] = (
                result_df.groupby(groupby_cols).agg(aggregation_dict).reset_index()
            )
            self.logger.debug(f"Summarized {result_name} results")

        return result_summary

    def _generate_dml_data(self, dgp_params) -> dml.data.DoubleMLPanelData:
        """Generate data for the simulation."""
        data = make_did_CS2021(n_obs=dgp_params["n_obs"], dgp_type=dgp_params["DGP"])
        dml_data = dml.data.DoubleMLPanelData(
            data,
            y_col="y",
            d_cols="d",
            id_col="id",
            t_col="t",
            x_cols=["Z1", "Z2", "Z3", "Z4"],
        )
        return dml_data
