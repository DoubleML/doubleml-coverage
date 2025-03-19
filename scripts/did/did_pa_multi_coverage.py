import numpy as np
import pandas as pd
from datetime import datetime
import time
import sys
import warnings

from sklearn.linear_model import LinearRegression, LogisticRegression
from lightgbm import LGBMRegressor, LGBMClassifier

import doubleml as dml
from doubleml.did.datasets import make_did_CS2021

# Suppress warnings
warnings.simplefilter(action='ignore', category=UserWarning)

# Number of repetitions
n_rep = 20
max_runtime = 5.5 * 3600  # 5.5 hours in seconds

# Oracle values
df_oracle = make_did_CS2021(n_obs=int(1e+6), dgp_type=1)  # does not depend on the DGP type
df_oracle["ite"] = df_oracle["y1"] - df_oracle["y0"]
df_oracle_thetas = df_oracle.groupby(["d", "t"])["ite"].mean().reset_index()
print("ATTs:")
print(df_oracle_thetas)

df_oracle_post_treatment = df_oracle[df_oracle["t"] >= df_oracle["d"]]
# Group aggregation
df_oracle_agg_group = df_oracle_post_treatment.groupby("d")["ite"].mean()
print("Group aggregated ATTs:")
print(df_oracle_agg_group)

# Time aggregation
df_oracle_agg_time = df_oracle_post_treatment.groupby("t")["ite"].mean()
print("Time aggregated ATTs:")
print(df_oracle_agg_time)

# Eventstudy aggregation
df_oracle["e"] = pd.to_datetime(df_oracle["t"]).values.astype("datetime64[M]") - \
    pd.to_datetime(df_oracle["d"]).values.astype("datetime64[M]")
df_oracle_agg_eventstudy = df_oracle.groupby("e")["ite"].mean()
print("Event Study aggregated ATTs:")
print(df_oracle_agg_eventstudy)

# DGP pars
n_obs = 500
dgp_types = [1, 4, 6]

# set up hyperparameters
hyperparam_dict = {
    "DGP": dgp_types,
    "score": ["observational", "experimental"],
    "in sample normalization": [True, False],
    "learner_g": [("Linear", LinearRegression()),],
    "learner_m": [("Linear", LogisticRegression()),],
    "level": [0.95, 0.90]
}

# set up the results dataframe
df_results_detailed = pd.DataFrame()
df_results_agg = pd.DataFrame()

# start simulation
np.random.seed(42)
start_time = time.time()

for i_rep in range(n_rep):
    print(f"Repetition: {i_rep + 1}/{n_rep}", end="\r")

    # Check the elapsed time
    elapsed_time = time.time() - start_time
    if elapsed_time > max_runtime:
        print("Maximum runtime exceeded. Stopping the simulation.")
        break

    for i_dgp, dgp_type in enumerate(dgp_types):
        # define the DoubleML data object
        data = make_did_CS2021(n_obs=n_obs, dgp_type=dgp_type)
        obj_dml_data = dml.data.DoubleMLPanelData(
            data,
            y_col="y",
            d_cols="d",
            id_col="id",
            t_col="t",
            x_cols=["Z1", "Z2", "Z3", "Z4"],
        )

        for learner_g_idx, (learner_g_name, ml_g) in enumerate(hyperparam_dict["learner_g"]):
            for learner_m_idx, (learner_m_name, ml_m) in enumerate(hyperparam_dict["learner_m"]):
                for score in hyperparam_dict["score"]:
                    for in_sample_normalization in hyperparam_dict["in sample normalization"]:
                        if score == "experimental":
                            dml_DiD = dml.did.DoubleMLDIDMulti(
                                obj_dml_data=obj_dml_data,
                                ml_g=ml_g,
                                ml_m=None,
                                gt_combinations="standard",
                                score=score,
                                in_sample_normalization=in_sample_normalization)
                        else:
                            assert score == "observational"
                            dml_DiD = dml.did.DoubleMLDIDMulti(
                                obj_dml_data=obj_dml_data,
                                ml_g=ml_g,
                                ml_m=ml_m,
                                gt_combinations="standard",
                                score=score,
                                in_sample_normalization=in_sample_normalization)
                        dml_DiD.fit(n_jobs_cv=5)

                        # oracle values
                        theta = np.full_like(dml_DiD.coef, np.nan)
                        for i, (g, _, t) in enumerate(dml_DiD.gt_combinations):
                            group_index = df_oracle_thetas["d"] == g
                            time_index = df_oracle_thetas["t"] == t
                            theta[i] = df_oracle_thetas[group_index & time_index]["ite"].iloc[0]

                        for level_idx, level in enumerate(hyperparam_dict["level"]):
                            confint = dml_DiD.confint(level=level)
                            coverage = np.mean((confint.iloc[:, 0] < theta) & (theta < confint.iloc[:, 1]))
                            ci_length = np.mean(confint.iloc[:, 1] - confint.iloc[:, 0])
                            bias = np.mean(abs(dml_DiD.coef - theta))

                            dml_DiD.bootstrap(n_rep_boot=2000)
                            confint_uniform = dml_DiD.confint(level=level, joint=True)

                            coverage_uniform = all((confint_uniform.iloc[:, 0] < theta) & (theta < confint_uniform.iloc[:, 1]))
                            ci_length_uniform = np.mean(confint_uniform.iloc[:, 1] - confint_uniform.iloc[:, 0])

                            df_results_detailed = pd.concat(
                                (df_results_detailed,
                                 pd.DataFrame({
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
                                    "repetition": i_rep}, index=[0])),
                                ignore_index=True)
                            
                            # group aggregation
                            group_agg = dml_DiD.aggregate(aggregation="group")
                            group_confint = group_agg.aggregated_frameworks.confint(level=level)
                            group_coverage = np.mean((group_confint.iloc[:, 0] < df_oracle_agg_group.values) & (df_oracle_agg_group.values < group_confint.iloc[:, 1]))
                            group_ci_length = np.mean(group_confint.iloc[:, 1] - group_confint.iloc[:, 0])
                            group_bias = np.mean(abs(group_agg.aggregated_frameworks.thetas - df_oracle_agg_group.values))

                            group_agg.aggregated_frameworks.bootstrap(n_rep_boot=2000)
                            group_confint_uniform = group_agg.aggregated_frameworks.confint(level=level, joint=True)
                            group_coverage_uniform = all((group_confint_uniform.iloc[:, 0] < df_oracle_agg_group.values) & (df_oracle_agg_group.values < group_confint_uniform.iloc[:, 1]))
                            group_ci_length_uniform = np.mean(group_confint_uniform.iloc[:, 1] - group_confint_uniform.iloc[:, 0])

                            df_results_agg = pd.concat(
                                (df_results_agg,
                                 pd.DataFrame({
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
                                    "repetition": i_rep}, index=[0]),
                                ),
                                ignore_index=True)


df_results = df_results_detailed.groupby(
    ["Learner g", "Learner m", "Score", "In-sample-norm.", "DGP", "level"]).agg(
        {"Coverage": "mean",
         "CI Length": "mean",
         "Bias": "mean",
         "Uniform Coverage": "mean",
         "Uniform CI Length": "mean",
         "repetition": "count"}
    ).reset_index()
print(df_results)

df_results_agg = df_results_agg.groupby(
    ["Learner g", "Learner m", "Score", "In-sample-norm.", "DGP", "level"]).agg(
        {"Coverage": "mean",
         "CI Length": "mean",
         "Bias": "mean",
         "Uniform Coverage": "mean",
         "Uniform CI Length": "mean",
         "repetition": "count"}
    ).reset_index()
print(df_results_agg)

end_time = time.time()
total_runtime = end_time - start_time

# save results
script_name = "did_pa_multi_coverage.py"
path = "results/did/did_pa_multi_coverage"

metadata = pd.DataFrame({
    'DoubleML Version': [dml.__version__],
    'Script': [script_name],
    'Date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    'Total Runtime (seconds)': [total_runtime],
    'Python Version': [f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"],
})
print(metadata)

df_results.to_csv(f"{path}.csv", index=False)
metadata.to_csv(f"{path}_metadata.csv", index=False)
