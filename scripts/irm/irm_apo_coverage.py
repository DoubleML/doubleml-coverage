import numpy as np
import pandas as pd
from datetime import datetime
import time
import sys

from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml
from doubleml.datasets import make_irm_data_discrete_treatments

# Number of repetitions
n_rep = 1000

# DGP pars
n_obs = 500
n_levels = 2

# generate the APOs true values
data_apo_large = make_irm_data_discrete_treatments(n_obs=int(1e+6), n_levels=n_levels, linear=True)
y0 = data_apo_large['oracle_values']['y0']
ite = data_apo_large['oracle_values']['ite']
d = data_apo_large['d']

average_ites = np.full(n_levels + 1, np.nan)
apos = np.full(n_levels + 1, np.nan)
for i in range(n_levels + 1):
    average_ites[i] = np.mean(ite[d == i])
    apos[i] = np.mean(y0) + average_ites[i]

print(f"Levels and their counts:\n{np.unique(d, return_counts=True)}")
print(f"True APOs: {apos}")

# to get the best possible comparison between different learners (and settings) we first simulate all datasets
np.random.seed(42)
datasets = []
for i in range(n_rep):
    data_apo = make_irm_data_discrete_treatments(n_obs=n_obs, n_levels=n_levels, linear=True)
    df_apo = pd.DataFrame(
        np.column_stack((data_apo['y'], data_apo['d'], data_apo['x'])),
        columns=['y', 'd'] + ['x' + str(i) for i in range(data_apo['x'].shape[1])]
    )
    datasets.append(df_apo)

# set up hyperparameters
hyperparam_dict = {
    "learner_g":
        [("Linear", LinearRegression()),
         ("LGBM", LGBMRegressor(verbose=-1))],
    "learner_m":
        [("Logistic", LogisticRegression()),
         ("LGBM", LGBMClassifier(verbose=-1))],
    "treatment_levels": [0, 1, 2],
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
    obj_dml_data = dml.DoubleMLData(datasets[i_rep], 'y', 'd')

    for learner_g_idx, (learner_g_name, ml_g) in enumerate(hyperparam_dict["learner_g"]):
        for learner_m_idx, (learner_m_name, ml_m) in enumerate(hyperparam_dict["learner_m"]):
            for treatment_idx, treatment_level in enumerate(hyperparam_dict["treatment_levels"]):
                # Set machine learning methods for g & m
                dml_obj = dml.DoubleMLAPO(
                    obj_dml_data=obj_dml_data,
                    ml_g=ml_g,
                    ml_m=ml_m,
                    treatment_level=treatment_level,
                    score='APO',
                )
                dml_obj.fit(n_jobs_cv=5)

                for level_idx, level in enumerate(hyperparam_dict["level"]):
                    confint = dml_obj.confint(level=level)
                    coverage = (confint.iloc[0, 0] < apos[treatment_idx]) & (apos[treatment_idx] < confint.iloc[0, 1])
                    ci_length = confint.iloc[0, 1] - confint.iloc[0, 0]

                    df_results_detailed = pd.concat(
                        (df_results_detailed,
                            pd.DataFrame({
                                "Coverage": coverage.astype(int),
                                "CI Length": confint.iloc[0, 1] - confint.iloc[0, 0],
                                "Bias": abs(dml_obj.coef[0] - apos[treatment_idx]),
                                "Treatment Level": treatment_level,
                                "Learner g": learner_g_name,
                                "Learner m": learner_m_name,
                                "level": level,
                                "repetition": i_rep}, index=[0])),
                        ignore_index=True)

df_results = df_results_detailed.groupby(
    ["Learner g", "Learner m", "Treatment Level", "level"]).agg(
        {"Coverage": "mean",
         "CI Length": "mean",
         "Bias": "mean",
         "repetition": "count"}
    ).reset_index()
print(df_results)

end_time = time.time()
total_runtime = end_time - start_time

# save results
script_name = "irm_apo_coverage.py"
path = "results/irm/irm_apo_coverage"

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
