import numpy as np
import pandas as pd
from datetime import datetime
import sys
from typing import Dict, List, Any, Tuple

from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml
from doubleml.did.datasets import make_did_CS2021

from montecover.base import BaseSimulation

class DIDMultiCoverageSimulation(BaseSimulation):
    """Simulation study for coverage properties of DoubleMLDIDMulti."""
    
    def __init__(
        self,
        n_rep: int = 20,
        n_obs: int = 500,
        max_runtime: float = 5.5 * 3600,
        random_seed: int = 42,
        output_path: str = "results/did/did_pa_multi_coverage",
        suppress_warnings: bool = True,
    ):
        super().__init__(
            n_rep=n_rep,
            max_runtime=max_runtime,
            random_seed=random_seed,
            output_path=output_path,
            suppress_warnings=suppress_warnings,
        )
        self.n_obs = n_obs
        
        # Calculate oracle values
        self._calculate_oracle_values()
        
        # Additional results storage for aggregated results
        self.results_aggregated = []
    
    def _calculate_oracle_values(self):
        """Calculate oracle values for the simulation."""
        # Oracle values
        df_oracle = make_did_CS2021(n_obs=int(1e+6), dgp_type=1)  # does not depend on the DGP type
        df_oracle["ite"] = df_oracle["y1"] - df_oracle["y0"]
        self.oracle_thetas = df_oracle.groupby(["d", "t"])["ite"].mean().reset_index()
        print("ATTs:")
        print(self.oracle_thetas)
        
        # Oracle group aggregation
        df_oracle_post_treatment = df_oracle[df_oracle["t"] >= df_oracle["d"]]
        self.oracle_agg_group = df_oracle_post_treatment.groupby("d")["ite"].mean()
        print("Group aggregated ATTs:")
        print(self.oracle_agg_group)
        
        # Oracle time aggregation
        self.oracle_agg_time = df_oracle_post_treatment.groupby("t")["ite"].mean()
        print("Time aggregated ATTs:")
        print(self.oracle_agg_time)
        
        # Oracle eventstudy aggregation
        df_oracle["e"] = pd.to_datetime(df_oracle["t"]).values.astype("datetime64[M]") - \
            pd.to_datetime(df_oracle["d"]).values.astype("datetime64[M]")
        self.oracle_agg_eventstudy = df_oracle.groupby("e")["ite"].mean()
        print("Event Study aggregated ATTs:")
        print(self.oracle_agg_eventstudy)
    
    def setup_parameters(self) -> Dict[str, List[Any]]:
        """Define simulation parameters."""
        return {
            "DGP": [1, 4, 6],
            "score": ["observational", "experimental"],
            "in_sample_normalization": [True, False],
            "learner_g": [("Linear", LinearRegression())],
            "learner_m": [("Linear", LogisticRegression())],
            "level": [0.95, 0.90]
        }
    
    def run_single_rep(self, rep_idx: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single repetition with the given parameters."""
        # Extract parameters
        dgp_type = params["DGP"]
        score = params["score"]
        in_sample_normalization = params["in_sample_normalization"]
        learner_g_name, ml_g = params["learner_g"]
        learner_m_name, ml_m = params["learner_m"]
        level = params["level"]
        
        # Generate data
        data = make_did_CS2021(n_obs=self.n_obs, dgp_type=dgp_type)
        obj_dml_data = dml.data.DoubleMLPanelData(
            data,
            y_col="y",
            d_cols="d",
            id_col="id",
            t_col="t",
            x_cols=["Z1", "Z2", "Z3", "Z4"],
        )
        
        # Fit model
        if score == "experimental":
            dml_DiD = dml.did.DoubleMLDIDMulti(
                obj_dml_data=obj_dml_data,
                ml_g=ml_g,
                ml_m=None,
                gt_combinations="standard",
                score=score,
                in_sample_normalization=in_sample_normalization)
        else:
            dml_DiD = dml.did.DoubleMLDIDMulti(
                obj_dml_data=obj_dml_data,
                ml_g=ml_g,
                ml_m=ml_m,
                gt_combinations="standard",
                score=score,
                in_sample_normalization=in_sample_normalization)
        
        # Fit the model
        dml_DiD.fit(n_jobs_cv=5)
        
        # Oracle values for this model
        theta = np.full_like(dml_DiD.coef, np.nan)
        for i, (g, _, t) in enumerate(dml_DiD.gt_combinations):
            group_index = self.oracle_thetas["d"] == g
            time_index = self.oracle_thetas["t"] == t
            theta[i] = self.oracle_thetas[group_index & time_index]["ite"].iloc[0]
        
        # Confidence intervals and metrics
        confint = dml_DiD.confint(level=level)
        coverage = np.mean((confint.iloc[:, 0] < theta) & (theta < confint.iloc[:, 1]))
        ci_length = np.mean(confint.iloc[:, 1] - confint.iloc[:, 0])
        bias = np.mean(abs(dml_DiD.coef - theta))
        
        # Bootstrap for uniform confidence intervals
        dml_DiD.bootstrap(n_rep_boot=2000)
        confint_uniform = dml_DiD.confint(level=level, joint=True)
        
        coverage_uniform = all((confint_uniform.iloc[:, 0] < theta) & (theta < confint_uniform.iloc[:, 1]))
        ci_length_uniform = np.mean(confint_uniform.iloc[:, 1] - confint_uniform.iloc[:, 0])
        
        # Detailed results
        result_detailed = {
            "Coverage": coverage,
            "CI Length": ci_length,
            "Bias": bias,
            "Uniform Coverage": coverage_uniform,
            "Uniform CI Length": ci_length_uniform,
            "Learner g": learner_g_name,
            "Learner m": learner_m_name,
            "Score": score,
            "In-sample-norm.": in_sample_normalization,
            "DGP": dgp_type,
            "level": level,
        }
        
        # Group aggregation
        group_agg = dml_DiD.aggregate(aggregation="group")
        group_confint = group_agg.aggregated_frameworks.confint(level=level)
        group_coverage = np.mean((group_confint.iloc[:, 0] < self.oracle_agg_group.values) & 
                                (self.oracle_agg_group.values < group_confint.iloc[:, 1]))
        group_ci_length = np.mean(group_confint.iloc[:, 1] - group_confint.iloc[:, 0])
        group_bias = np.mean(abs(group_agg.aggregated_frameworks.thetas - self.oracle_agg_group.values))
        
        group_agg.aggregated_frameworks.bootstrap(n_rep_boot=2000)
        group_confint_uniform = group_agg.aggregated_frameworks.confint(level=level, joint=True)
        group_coverage_uniform = all((group_confint_uniform.iloc[:, 0] < self.oracle_agg_group.values) & 
                                   (self.oracle_agg_group.values < group_confint_uniform.iloc[:, 1]))
        group_ci_length_uniform = np.mean(group_confint_uniform.iloc[:, 1] - group_confint_uniform.iloc[:, 0])
        
        # Aggregated results
        result_aggregated = {
            "Coverage": group_coverage,
            "CI Length": group_ci_length,
            "Bias": group_bias,
            "Uniform Coverage": group_coverage_uniform,
            "Uniform CI Length": group_ci_length_uniform,
            "Learner g": learner_g_name,
            "Learner m": learner_m_name,
            "Score": score,
            "In-sample-norm.": in_sample_normalization,
            "DGP": dgp_type,
            "level": level,
        }
        
        # Store aggregated results
        self.results_aggregated.append(result_aggregated)
        
        return result_detailed
    
    def summarize_results(self):
        """Summarize results and save to CSV."""
        df_results_detailed = pd.DataFrame(self.results)
        df_results_aggregated = pd.DataFrame(self.results_aggregated)
        
        # Aggregate results by parameters
        df_summary_detailed = df_results_detailed.groupby(
            ["Learner g", "Learner m", "Score", "In-sample-norm.", "DGP", "level"]
        ).agg({
            "Coverage": "mean",
            "CI Length": "mean",
            "Bias": "mean",
            "Uniform Coverage": "mean",
            "Uniform CI Length": "mean",
            "repetition": "count"
        }).reset_index()
        
        df_summary_aggregated = df_results_aggregated.groupby(
            ["Learner g", "Learner m", "Score", "In-sample-norm.", "DGP", "level"]
        ).agg({
            "Coverage": "mean",
            "CI Length": "mean",
            "Bias": "mean",
            "Uniform Coverage": "mean",
            "Uniform CI Length": "mean",
            "repetition": "count"
        }).reset_index()
        
        # Save results
        metadata = pd.DataFrame({
            'DoubleML Version': [dml.__version__],
            'Script': ['DIDMultiCoverageSimulation'],
            'Date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            'Total Runtime (seconds)': [self.total_runtime],
            'Python Version': [f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"],
        })
        
        df_summary_detailed.to_csv(f"{self.output_path}.csv", index=False)
        df_summary_aggregated.to_csv(f"{self.output_path}_aggregated.csv", index=False)
        metadata.to_csv(f"{self.output_path}_metadata.csv", index=False)
        
        return df_summary_detailed, df_summary_aggregated, metadata