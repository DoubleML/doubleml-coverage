from typing import Any, Dict, List

import doubleml as dml
import numpy as np
import pandas as pd
from doubleml.did.datasets import make_did_CS2021
from sklearn.linear_model import LinearRegression, LogisticRegression

from montecover.base import BaseSimulation


class DIDMultiCoverageSimulation(BaseSimulation):
    """Simulation study for coverage properties of DoubleMLDIDMulti."""

    def __init__(
        self,
        repetitions: int = 20,
        n_obs: int = 500,
        max_runtime: float = 5.5 * 3600,
        random_seed: int = 42,
        suppress_warnings: bool = True,
    ):
        super().__init__(
            repetitions=repetitions,
            max_runtime=max_runtime,
            random_seed=random_seed,
            suppress_warnings=suppress_warnings,
        )
        self.n_obs = n_obs

        # Calculate oracle values
        self._calculate_oracle_values()

        # Additional results storage for aggregated results
        self.results_aggregated = []

    def _calculate_oracle_values(self):
        """Calculate oracle values for the simulation."""

        self.oracle_values = dict()
        # Oracle values
        df_oracle = make_did_CS2021(n_obs=int(1e6), dgp_type=1)  # does not depend on the DGP type
        df_oracle["ite"] = df_oracle["y1"] - df_oracle["y0"]
        self.oracle_values["detailed"] = df_oracle.groupby(["d", "t"])["ite"].mean().reset_index()

        # Oracle group aggregation
        df_oracle_post_treatment = df_oracle[df_oracle["t"] >= df_oracle["d"]]
        self.oracle_values["group"] = df_oracle_post_treatment.groupby("d")["ite"].mean()

        # Oracle time aggregation
        self.oracle_values["time"] = df_oracle_post_treatment.groupby("t")["ite"].mean()

        # Oracle eventstudy aggregation
        df_oracle["e"] = pd.to_datetime(df_oracle["t"]).values.astype("datetime64[M]") - pd.to_datetime(
            df_oracle["d"]
        ).values.astype("datetime64[M]")
        self.oracle_values["eventstudy"] = df_oracle.groupby("e")["ite"].mean()[1:]

    def setup_parameters(self) -> Dict[str, List[Any]]:
        """Define simulation parameters."""
        return {
            "DGP": [1, 4, 6],
            "score": ["observational", "experimental"],
            "in_sample_normalization": [True, False],
            "learner_g": [("Linear", LinearRegression())],
            "learner_m": [("Linear", LogisticRegression())],
            "level": [0.95, 0.90],
        }

    def run_single_rep(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single repetition with the given parameters."""
        # Extract parameters
        dgp_type = params["DGP"]
        score = params["score"]
        in_sample_normalization = params["in_sample_normalization"]
        learner_g_name, ml_g = params["learner_g"]
        learner_m_name, ml_m = params["learner_m"]
        level = params["level"]

        dml_data = self._generate_data(dgp_type=dgp_type)

        # Model
        dml_model = dml.did.DoubleMLDIDMulti(
            obj_dml_data=dml_data,
            ml_g=ml_g,
            ml_m=None if score == "experimental" else ml_m,
            gt_combinations="standard",
            score=score,
            in_sample_normalization=in_sample_normalization,
        )
        dml_model.fit(n_jobs_cv=5)
        dml_model.bootstrap(n_rep_boot=2000)

        # Oracle values for this model
        oracle_thetas = np.full_like(dml_model.coef, np.nan)
        for i, (g, _, t) in enumerate(dml_model.gt_combinations):
            group_index = self.oracle_values["detailed"]["d"] == g
            time_index = self.oracle_values["detailed"]["t"] == t
            oracle_thetas[i] = self.oracle_values["detailed"][group_index & time_index]["ite"].iloc[0]

        result = dict()
        result["detailed"] = self._compute_coverage(
            thetas=dml_model.coef,
            oracle_thetas=oracle_thetas,
            confint=dml_model.confint(level=level),
            joint_confint=dml_model.confint(level=level, joint=True),
            )

        for aggregation_method in ["group", "time", "eventstudy"]:
            agg_obj = dml_model.aggregate(aggregation=aggregation_method)
            agg_obj.aggregated_frameworks.bootstrap(n_rep_boot=2000)

            result[aggregation_method] = self._compute_coverage(
                thetas=agg_obj.aggregated_frameworks.thetas,
                oracle_thetas=self.oracle_values[aggregation_method].values,
                confint=agg_obj.aggregated_frameworks.confint(level=level),
                joint_confint=agg_obj.aggregated_frameworks.confint(level=level, joint=True),
            )

        # add parameters to the result
        for result_dict in result.values():
            result_dict.update({
                "Learner g": learner_g_name,
                "Learner m": learner_m_name,
                "Score": score,
                "In-sample-norm.": in_sample_normalization,
                "DGP": dgp_type,
                "level": level,
            })

        return result

    def summarize_results(self):
        """Summarize the simulation results."""

        groupby_cols = ["Learner g", "Learner m", "Score", "In-sample-norm.", "DGP", "level"]
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
                result_df.groupby(groupby_cols)
                .agg(aggregation_dict)
                .reset_index()
            )

        return result_summary

    def _generate_data(self, dgp_type: int) -> dml.data.DoubleMLPanelData:
        """Generate data for the simulation."""
        data = make_did_CS2021(n_obs=self.n_obs, dgp_type=dgp_type)
        dml_data = dml.data.DoubleMLPanelData(
            data,
            y_col="y",
            d_cols="d",
            id_col="id",
            t_col="t",
            x_cols=["Z1", "Z2", "Z3", "Z4"],
        )
        return dml_data
