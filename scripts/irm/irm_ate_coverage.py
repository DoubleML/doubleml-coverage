import numpy as np
import pandas as pd
from datetime import datetime
import time
import sys

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, LogisticRegressionCV

import doubleml as dml
from doubleml.datasets import make_irm_data

# Number of repetitions
n_rep = 1000
max_runtime = 5.5 * 3600  # 5.5 hours in seconds

# DGP pars
theta = 0.5
n_obs = 500
dim_x = 20

# to get the best possible comparison between different learners (and settings) we first simulate all datasets
np.random.seed(42)
datasets = []
for i in range(n_rep):
    data = make_irm_data(theta=theta, n_obs=n_obs, dim_x=dim_x, return_type='DataFrame')
    datasets.append(data)

# set up hyperparameters
hyperparam_dict = {
    "learner_g": [("Lasso", LassoCV()),
                  ("Random Forest",
                   RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2))],
    "learner_m": [("Logistic Regression", LogisticRegressionCV()),
                  ("Random Forest",
                   RandomForestClassifier(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2))],
    "level": [0.95, 0.90]
}

# set up the results dataframe
df_results_detailed = pd.DataFrame()

# start simulation
np.random.seed(42)
start_time = time.time()

for i_rep in range(n_rep):
    print(f"Repetition: {i_rep}/{n_rep}", end="\r")

    # Check the elapsed time
    elapsed_time = time.time() - start_time
    if elapsed_time > max_runtime:
        print("Maximum runtime exceeded. Stopping the simulation.")
        break

    # define the DoubleML data object
    obj_dml_data = dml.DoubleMLData(datasets[i_rep], 'y', 'd')

    for learner_g_idx, (learner_g_name, ml_g) in enumerate(hyperparam_dict["learner_g"]):
        for learner_m_idx, (learner_m_name, ml_m) in enumerate(hyperparam_dict["learner_m"]):
            # Set machine learning methods for g & m
            dml_irm = dml.DoubleMLIRM(
                obj_dml_data=obj_dml_data,
                ml_g=ml_g,
                ml_m=ml_m,
            )
            dml_irm.fit(n_jobs_cv=5)

            for level_idx, level in enumerate(hyperparam_dict["level"]):
                confint = dml_irm.confint(level=level)
                coverage = (confint.iloc[0, 0] < theta) & (theta < confint.iloc[0, 1])
                ci_length = confint.iloc[0, 1] - confint.iloc[0, 0]

                df_results_detailed = pd.concat(
                    (df_results_detailed,
                     pd.DataFrame({
                        "Coverage": coverage.astype(int),
                        "CI Length": confint.iloc[0, 1] - confint.iloc[0, 0],
                        "Bias": abs(dml_irm.coef[0] - theta),
                        "Learner g": learner_g_name,
                        "Learner m": learner_m_name,
                        "level": level,
                        "repetition": i_rep}, index=[0])),
                    ignore_index=True)

df_results = df_results_detailed.groupby(
    ["Learner g", "Learner m", "level"]).agg(
        {"Coverage": "mean",
         "CI Length": "mean",
         "Bias": "mean",
         "repetition": "count"}
    ).reset_index()
print(df_results)

end_time = time.time()
total_runtime = end_time - start_time

# save results
script_name = "irm_ate_coverage.py"
path = "results/irm/irm_ate_coverage"

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
