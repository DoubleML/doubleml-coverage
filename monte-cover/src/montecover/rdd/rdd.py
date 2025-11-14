import time
import warnings
from itertools import product
from typing import Any, Dict, Optional

import doubleml as dml
import numpy as np
import pandas as pd
from doubleml.rdd.datasets import make_simple_rdd_data
from rdrobust import rdrobust
from statsmodels.nonparametric.kernel_regression import KernelReg

from montecover.base import BaseSimulation
from montecover.utils import create_learner_from_config


class RDDCoverageSimulation(BaseSimulation):
    """Simulation class for coverage properties of DoubleML RDFlex for RDD."""

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

        self.fuzzy = self.dgp_parameters.get("fuzzy", [False])[0]
        self.cutoff = self.dgp_parameters.get("cutoff", [0.0])[0]
        # Calculate oracle values
        self._calculate_oracle_values()

    def _process_config_parameters(self):
        """Process simulation-specific parameters from config."""

        # Process ML models in parameter grid
        assert "learners" in self.dml_parameters, "No learners specified in the config file"

        required_learners = ["ml_g"]
        for learner in self.dml_parameters["learners"]:
            for ml in required_learners:
                assert ml in learner, f"No {ml} specified in the config file"

    def _calculate_oracle_values(self):
        """Calculate oracle values for the simulation."""
        self.logger.info("Calculating oracle values")

        data_oracle = make_simple_rdd_data(n_obs=int(1e6), fuzzy=self.fuzzy, cutoff=self.cutoff)
        # get oracle value
        score = data_oracle["score"]
        ite = data_oracle["oracle_values"]["Y1"] - data_oracle["oracle_values"]["Y0"]

        # subset score and ite for faster computation
        score_subset = (score >= (self.cutoff - 0.02)) & (score <= (self.cutoff + 0.02))
        self.logger.info(f"Oracle score subset size: {np.sum(score_subset)}")
        kernel_reg = KernelReg(endog=ite[score_subset], exog=score[score_subset], var_type="c", reg_type="ll")
        effect_at_cutoff, _ = kernel_reg.fit(np.array([self.cutoff]))
        oracle_effect = effect_at_cutoff[0]

        self.logger.info(f"Oracle effect at cutoff: {oracle_effect}")
        self.oracle_values = dict()
        self.oracle_values["theta"] = oracle_effect

    def _process_repetition(self, i_rep):
        """Process a single repetition with all parameter combinations."""
        if self.suppress_warnings:
            warnings.simplefilter(action="ignore", category=UserWarning)

        i_param_comb = 0
        rep_results = {
            "coverage": [],
        }

        # loop through all parameter combinations
        for dgp_param_values in product(*self.dgp_parameters.values()):
            dgp_params = dict(zip(self.dgp_parameters.keys(), dgp_param_values))
            dml_data = self._generate_dml_data(dgp_params)

            # --- Run rdrobust benchmark ---
            self.logger.debug(f"Rep {i_rep+1}: Running rdrobust benchmark for DGP {dgp_params}")
            param_start_time_rd_benchmark = time.time()

            # Call the dedicated benchmark function
            # Pass dml_data, current dgp_params, and repetition index
            benchmark_result_list = self._rdrobust_benchmark(dml_data, dgp_params, i_rep)
            if benchmark_result_list:
                rep_results["coverage"].extend(benchmark_result_list)

            param_duration_rd_benchmark = time.time() - param_start_time_rd_benchmark
            self.logger.debug(f"rdrobust benchmark for DGP {dgp_params} completed in {param_duration_rd_benchmark:.2f}s")

            for dml_param_values in product(*self.dml_parameters.values()):
                dml_params = dict(zip(self.dml_parameters.keys(), dml_param_values))
                i_param_comb += 1

                comb_results = self._process_parameter_combination(i_rep, i_param_comb, dgp_params, dml_params, dml_data)
                rep_results["coverage"].extend(comb_results["coverage"])

        return rep_results

    def _rdrobust_benchmark(self, dml_data, dml_params, i_rep):
        """Run a benchmark using rdrobust for RDD."""

        # Extract parameters
        score = dml_data.data[dml_data.score_col]
        Y = dml_data.data[dml_data.y_col]
        Z = dml_data.data[dml_data.x_cols]

        benchmark_results_list = []
        for level in self.confidence_parameters["level"]:
            if self.fuzzy:
                D = dml_data.data[dml_data.d_cols]
                rd_model = rdrobust(y=Y, x=score, fuzzy=D, covs=Z, c=self.cutoff, level=level * 100)
            else:
                rd_model = rdrobust(y=Y, x=score, covs=Z, c=self.cutoff, level=level * 100)
            coef_rd = rd_model.coef.loc["Robust", "Coeff"]
            ci_lower_rd = rd_model.ci.loc["Robust", "CI Lower"]
            ci_upper_rd = rd_model.ci.loc["Robust", "CI Upper"]

            confint_for_compute = pd.DataFrame({"lower": [ci_lower_rd], "upper": [ci_upper_rd]})
            theta_for_compute = np.array([coef_rd])

            coverage_metrics = self._compute_coverage(
                thetas=theta_for_compute,
                oracle_thetas=self.oracle_values["theta"],
                confint=confint_for_compute,
                joint_confint=None,
            )

            # Add metadata
            coverage_metrics.update(
                {
                    "repetition": i_rep,
                    "Learner g": "Linear",
                    "Learner m": "Logistic",
                    "Method": "rdrobust",
                    "fs_specification": "cutoff",
                    "level": level,
                }
            )
            benchmark_results_list.append(coverage_metrics)

        return benchmark_results_list

    def run_single_rep(self, dml_data, dml_params) -> Dict[str, Any]:
        """Run a single repetition with the given parameters."""

        # Extract parameters
        learner_config = dml_params["learners"]
        learner_g_name, ml_g = create_learner_from_config(learner_config["ml_g"])
        if self.fuzzy:
            learner_m_name, ml_m = create_learner_from_config(learner_config["ml_m"])
        else:
            learner_m_name, ml_m = "N/A", None
        fs_specification = dml_params["fs_specification"]

        # Model
        dml_model = dml.rdd.RDFlex(
            obj_dml_data=dml_data,
            ml_g=ml_g,
            ml_m=ml_m,
            n_folds=5,
            n_rep=1,
            fuzzy=self.fuzzy,
            cutoff=self.cutoff,
            fs_specification=fs_specification,
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
                        "Method": "RDFlex",
                        "fs_specification": fs_specification,
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
        groupby_cols = ["Method", "fs_specification", "Learner g", "Learner m", "level"]
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
        data = make_simple_rdd_data(
            n_obs=dgp_params["n_obs"],
            fuzzy=dgp_params["fuzzy"],
            cutoff=dgp_params["cutoff"],
        )

        x_cols = ["x" + str(i) for i in range(data["X"].shape[1])]
        columns = ["y", "d", "score"] + x_cols
        df = pd.DataFrame(np.column_stack((data["Y"], data["D"], data["score"], data["X"])), columns=columns)

        dml_data = dml.data.DoubleMLRDDData(
            data=df,
            y_col="y",
            d_cols=["d"],
            x_cols=x_cols,
            score_col="score",
        )
        return dml_data
