import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, LogisticRegressionCV

import doubleml as dml
from doubleml.datasets import make_heterogeneous_data

# Number of repetitions
n_rep = 1000

# DGP pars
n_obs = 500
p = 10
support_size = 5
n_x = 1

# to get the best possible comparison between different learners (and settings) we first simulate all datasets
np.random.seed(42)
datasets = []
for i in range(n_rep):
    data = make_heterogeneous_data(n_obs=n_obs, p=p, support_size=support_size, n_x=n_x, binary_treatment=True)
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
df_results_detailed = pd.DataFrame(
    columns=["Coverage", "CI Length",
             "Bias", "Uniform Coverage", "Uniform CI Length",
             "Learner g", "Learner m",
             "level", "repetition"])
df_results_detailed["Uniform Coverage"] = df_results_detailed["Uniform Coverage"].astype(bool)

# start simulation
np.random.seed(42)

for i_rep in range(n_rep):
    print(f"Repetition: {i_rep}/{n_rep}", end="\r")

    # define the DoubleML data object
    data = datasets[i_rep]['data']
    ite = datasets[i_rep]['effects']

    groups = pd.DataFrame(
        np.column_stack((data['X_0'] <= 0.3,
                         (data['X_0'] > 0.3) & (data['X_0'] <= 0.7),
                         data['X_0'] > 0.7)),
        columns=['Group 1', 'Group 2', 'Group 3'])
    true_effects = [ite[groups[group]].mean() for group in groups.columns]

    obj_dml_data = dml.DoubleMLData(data, 'y', 'd')

    for learner_g_idx, (learner_g_name, ml_g) in enumerate(hyperparam_dict["learner_g"]):
        for learner_m_idx, (learner_m_name, ml_m) in enumerate(hyperparam_dict["learner_m"]):
            # Set machine learning methods for g & m
            dml_irm = dml.DoubleMLIRM(
                obj_dml_data=obj_dml_data,
                ml_g=ml_g,
                ml_m=ml_m,
            )
            dml_irm.fit(n_jobs_cv=5)
            gate = dml_irm.gate(groups=groups)

            for level_idx, level in enumerate(hyperparam_dict["level"]):
                confint = gate.confint(level=level)
                coverage = (confint.iloc[:, 0] < true_effects) & (true_effects < confint.iloc[:, 2])
                ci_length = confint.iloc[:, 2] - confint.iloc[:, 0]
                confint_uniform = gate.confint(level=0.95, joint=True, n_rep_boot=2000)
                coverage_uniform = all((confint_uniform.iloc[:, 0] < true_effects) &
                                       (true_effects < confint_uniform.iloc[:, 2]))
                ci_length_uniform = confint_uniform.iloc[:, 2] - confint_uniform.iloc[:, 0]
                df_results_detailed = pd.concat(
                    (df_results_detailed,
                     pd.DataFrame({
                        "Coverage": coverage.mean(),
                        "CI Length": ci_length.mean(),
                        "Bias": abs(dml_irm.coef - true_effects).mean(),
                        "Uniform Coverage": coverage_uniform,
                        "Uniform CI Length": ci_length_uniform.mean(),
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
         "Uniform Coverage": "mean",
         "Uniform CI Length": "mean",
         "repetition": "count"}
    ).reset_index()
print(df_results)

# save results
df_results.to_csv("results/irm_gate_coverage.csv", index=False)
