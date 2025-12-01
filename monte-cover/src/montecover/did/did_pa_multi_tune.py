from typing import Any, Dict, Optional

import doubleml as dml
import numpy as np
import optuna
import pandas as pd
from doubleml.did.datasets import make_did_CS2021

from montecover.base import BaseSimulation
from montecover.utils import create_learner_from_config
from montecover.utils_tuning import lgbm_reg_params, lgbm_cls_params


class DIDMultiTuningCoverageSimulation(BaseSimulation):
    """Simulation study for coverage properties of DoubleMLDIDMulti with hyperparameter tuning."""

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

        # tuning specific settings
        self._param_space = {"ml_g": lgbm_reg_params, "ml_m": lgbm_cls_params}

        self._optuna_settings = {
            "n_trials": 50,
            "show_progress_bar": False,
            "verbosity": optuna.logging.WARNING,  # Suppress Optuna logs
        }

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
        learner_config = dml_params["learners"]
        learner_g_name, ml_g = create_learner_from_config(learner_config["ml_g"])
        learner_m_name, ml_m = create_learner_from_config(learner_config["ml_m"])
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
        # Tuning
        dml_model_tuned = dml.did.DoubleMLDIDMulti(
            obj_dml_data=dml_data,
            ml_g=ml_g,
            ml_m=None if score == "experimental" else ml_m,
            gt_combinations="standard",
            score=score,
            in_sample_normalization=in_sample_normalization,
        )
        dml_model_tuned.tune_ml_models(
            ml_param_space=self._param_space,
            optuna_settings=self._optuna_settings,
        )

        # sort out oracle thetas
        oracle_thetas = np.full(len(dml_model.gt_combinations), np.nan)
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
        for model in [dml_model, dml_model_tuned]:
            model.fit()
            model.bootstrap(n_rep_boot=2000)
            nuisance_loss = model.nuisance_loss
            for level in self.confidence_parameters["level"]:
                level_result = dict()
                level_result["detailed"] = self._compute_coverage(
                    thetas=model.coef,
                    oracle_thetas=oracle_thetas,
                    confint=model.confint(level=level),
                    joint_confint=model.confint(level=level, joint=True),
                )

                for aggregation_method in ["group", "time", "eventstudy"]:
                    agg_obj = model.aggregate(aggregation=aggregation_method)
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
                            "Tuned": model is dml_model_tuned,
                            "Loss g_control": nuisance_loss["ml_g0"].mean(),
                            "Loss g_treated": nuisance_loss["ml_g1"].mean(),
                            "Loss m": nuisance_loss["ml_m"].mean(),
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
            "Tuned",
        ]
        aggregation_dict = {
            "Coverage": "mean",
            "CI Length": "mean",
            "Bias": "mean",
            "Uniform Coverage": "mean",
            "Uniform CI Length": "mean",
            "Loss g_control": "mean",
            "Loss g_treated": "mean",
            "Loss m": "mean",
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
