import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, LogisticRegressionCV

import doubleml as dml
from doubleml.datasets import make_irm_data_discrete_treatments

# Number of repetitions
n_rep = 1000

# DGP pars (APO for D=0 is 210)
theta = 210
n_obs = 500
n_levels = 2
treatment_level = 0

# to get the best possible comparison between different learners (and settings) we first simulate all datasets
np.random.seed(42)
datasets = []
for i in range(n_rep):
    data_apo = make_irm_data_discrete_treatments(n_obs=n_obs, n_levels=n_levels)
    df_apo = pd.DataFrame(
        np.column_stack((data_apo['y'], data_apo['d'], data_apo['x'])),
        columns=['y', 'd'] + ['x' + str(i) for i in range(data_apo['x'].shape[1])]
    )
    datasets.append(df_apo)

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
df_results_detailed = pd.DataFrame(
    columns=["Coverage", "CI Length",
             "Bias",
             "Learner g", "Learner m",
             "level", "repetition"])

# start simulation
np.random.seed(42)

for i_rep in range(n_rep):
    print(f"Repetition: {i_rep}/{n_rep}", end="\r")

    # define the DoubleML data object
    obj_dml_data = dml.DoubleMLData(datasets[i_rep], 'y', 'd')

    for learner_g_idx, (learner_g_name, ml_g) in enumerate(hyperparam_dict["learner_g"]):
        for learner_m_idx, (learner_m_name, ml_m) in enumerate(hyperparam_dict["learner_m"]):
            # Set machine learning methods for g & m
            dml_obj = dml.DoubleMLAPO(
                obj_dml_data=obj_dml_data,
                ml_g=ml_g,
                ml_m=ml_m,
                treatment_level=treatment_level,
            )
            dml_obj.fit(n_jobs_cv=5)

            for level_idx, level in enumerate(hyperparam_dict["level"]):
                confint = dml_obj.confint(level=level)
                coverage = (confint.iloc[0, 0] < theta) & (theta < confint.iloc[0, 1])
                ci_length = confint.iloc[0, 1] - confint.iloc[0, 0]

                df_results_detailed = pd.concat(
                    (df_results_detailed,
                     pd.DataFrame({
                        "Coverage": coverage.astype(int),
                        "CI Length": confint.iloc[0, 1] - confint.iloc[0, 0],
                        "Bias": abs(dml_obj.coef[0] - theta),
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

# save results
df_results.to_csv("results/irm_apo_coverage.csv", index=False)
