import numpy as np
import pandas as pd
from datetime import datetime
import time
import sys

from sklearn.linear_model import LinearRegression, LogisticRegression
from lightgbm import LGBMRegressor, LGBMClassifier

import doubleml as dml
from doubleml.did.datasets import make_did_CS2021

# Number of repetitions
n_rep = 1000
max_runtime = 5.5 * 3600  # 5.5 hours in seconds

# DGP pars
dgp_dict = {}

df_oracle = make_did_CS2021(n_obs=int(1e+6), dgp_type=1)  # does not depend on the DGP type
df_oracle["ite"] = df_oracle["y1"] - df_oracle["y0"]
df_oracle_thetas = df_oracle.groupby(["d", "t"])["ite"].mean().reset_index()
# drop
print(df_oracle_thetas)

n_obs = 2000

# to get the best possible comparison between different learners (and settings) we first simulate all datasets
np.random.seed(42)

dgp_types = [1, 2, 3, 4, 5, 6]
n_dgps = len(dgp_types)
datasets = []
for dgp_type in dgp_types:
    datasets_dgp = []
    for i in range(n_rep):
        data = make_did_CS2021(n_obs=n_obs, dgp_type=dgp_type)
        datasets_dgp.append(data)
    datasets.append(datasets_dgp)


# set up hyperparameters
hyperparam_dict = {
    "DGP": dgp_types,
    "score": ["observational", "experimental"],
    "in sample normalization": [True, False],
    "learner_g": [("Linear", LinearRegression()),
                  ("LGBM", LGBMRegressor(n_estimators=300, learning_rate=0.05, verbose=-1)),],
    "learner_m": [("Linear", LogisticRegression()),
                  ("LGBM", LGBMClassifier(n_estimators=300, learning_rate=0.05, verbose=-1))],
    "level": [0.95, 0.90]
}

# set up the results dataframe
df_results_detailed = pd.DataFrame()

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
        obj_dml_data = dml.data.DoubleMLPanelData(
            datasets[i_dgp][i_rep],
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
