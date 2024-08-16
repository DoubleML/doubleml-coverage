import numpy as np
import pandas as pd
from datetime import datetime
import time
import sys

from lightgbm import LGBMRegressor
from sklearn.linear_model import LassoCV

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
    data = make_heterogeneous_data(n_obs=n_obs, p=p, support_size=support_size, n_x=n_x, binary_treatment=False)
    datasets.append(data)

# set up hyperparameters
hyperparam_dict = {
    "learner_g": [("Lasso", LassoCV()),
                  ("LGBM", LGBMRegressor(n_estimators=200, learning_rate=0.05, verbose=-1))],
    "learner_m": [("Lasso", LassoCV()),
                  ("LGBM", LGBMRegressor(n_estimators=200, learning_rate=0.05, verbose=-1))],
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
            dml_plr = dml.DoubleMLPLR(
                obj_dml_data=obj_dml_data,
                ml_l=ml_g,
                ml_m=ml_m,
            )
            dml_plr.fit(n_jobs_cv=5)
            gate = dml_plr.gate(groups=groups)

            for level_idx, level in enumerate(hyperparam_dict["level"]):
                confint = gate.confint(level=level)
                effects = confint["effect"]
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
                        "Bias": abs(effects - true_effects).mean(),
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

end_time = time.time()
total_runtime = end_time - start_time

# save results
script_name = "plr_gate_coverage.py"
path = "results/plm/plr_gate_coverage"

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
