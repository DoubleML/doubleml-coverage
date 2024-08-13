import numpy as np
import pandas as pd
from datetime import datetime
import time
import sys

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV

import doubleml as dml
from doubleml.datasets import make_pliv_CHS2015


# Number of repetitions
n_rep = 1000

# DGP pars
theta = 0.5
n_obs = 500
dim_x = 20
dim_z = 1

# to get the best possible comparison between different learners (and settings) we first simulate all datasets
np.random.seed(42)

datasets = []
for i in range(n_rep):
    data = make_pliv_CHS2015(alpha=theta, n_obs=n_obs, dim_x=dim_x, dim_z=dim_z, return_type='DataFrame')
    datasets.append(data)

# set up hyperparameters
hyperparam_dict = {
    "score": ["partialling out", "IV-type"],
    "learner_g": [("Lasso", LassoCV()),
                  ("Random Forest",
                   RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2))],
    "learner_m": [("Lasso", LassoCV()),
                  ("Random Forest",
                   RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2))],
    "learner_r": [("Lasso", LassoCV()),
                  ("Random Forest",
                   RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2))],
    "level": [0.95, 0.90]
}

# set up the results dataframe
df_results_detailed = pd.DataFrame()

# start simulation
np.random.seed(42)
start_time = time.time()

for i_rep in range(n_rep):
    print(f"Repetition: {i_rep}/{n_rep}", end="\r")

    # define the DoubleML data object
    obj_dml_data = dml.DoubleMLData(datasets[i_rep], 'y', 'd', z_cols='Z1')

    for score_idx, score in enumerate(hyperparam_dict["score"]):
        for learner_g_idx, (learner_g_name, ml_g) in enumerate(hyperparam_dict["learner_g"]):
            for learner_m_idx, (learner_m_name, ml_m) in enumerate(hyperparam_dict["learner_m"]):
                for learner_r_idx, (learner_r_name, ml_r) in enumerate(hyperparam_dict["learner_r"]):
                    if score == "IV-type":
                        # Set machine learning methods for g & m
                        dml_pliv = dml.DoubleMLPLIV(
                            obj_dml_data=obj_dml_data,
                            ml_l=ml_g,
                            ml_m=ml_m,
                            ml_g=ml_g,
                            ml_r=ml_r,
                            score="IV-type",
                        )
                    else:
                        # Set machine learning methods for g & m
                        dml_pliv = dml.DoubleMLPLIV(
                            obj_dml_data=obj_dml_data,
                            ml_l=ml_g,
                            ml_m=ml_m,
                            ml_r=ml_r,
                            score=score,
                        )
                    dml_pliv.fit(n_jobs_cv=5)

                    for level_idx, level in enumerate(hyperparam_dict["level"]):
                        confint = dml_pliv.confint(level=level)
                        coverage = (confint.iloc[0, 0] < theta) & (theta < confint.iloc[0, 1])
                        ci_length = confint.iloc[0, 1] - confint.iloc[0, 0]

                        df_results_detailed = pd.concat(
                            (df_results_detailed,
                             pd.DataFrame({
                                "Coverage": coverage.astype(int),
                                "CI Length": confint.iloc[0, 1] - confint.iloc[0, 0],
                                "Bias": abs(dml_pliv.coef[0] - theta),
                                "score": score,
                                "Learner g": learner_g_name,
                                "Learner m": learner_m_name,
                                "Learner r": learner_r_name,
                                "level": level,
                                "repetition": i_rep}, index=[0])),
                            ignore_index=True)

df_results = df_results_detailed.groupby(
    ["Learner g", "Learner m", "Learner r", "score", "level"]).agg(
        {"Coverage": "mean",
         "CI Length": "mean",
         "Bias": "mean",
         "repetition": "count"}
    ).reset_index()
print(df_results)

end_time = time.time()
total_runtime = end_time - start_time

# save results
script_name = "pliv_late_coverage.py"
path = "results/plm/pliv_late_coverage"

metadata = pd.DataFrame({
    'DoubleML Version': [dml.__version__],
    'Script': [script_name],
    'Date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    'Total Runtime (seconds)': [total_runtime],
    'Python Version': [f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"],
})
print(metadata)

df_results.to_csv(f"../../{path}.csv", index=False)
metadata.to_csv(f"../../{path}_metadata.csv", index=False)