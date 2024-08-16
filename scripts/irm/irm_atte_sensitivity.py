import numpy as np
import pandas as pd
from datetime import datetime
import time
import sys

from sklearn.linear_model import LinearRegression, LogisticRegression
from lightgbm import LGBMRegressor, LGBMClassifier

import doubleml as dml
from doubleml.datasets import make_confounded_irm_data

# Number of repetitions
n_rep = 500

# DGP pars
n_obs = 5000
theta = 5.0
trimming_threshold = 0.05

dgp_pars = {
    "gamma_a": 0.151,
    "beta_a": 0.580,
    "theta": theta,
    "var_epsilon_y": 1.0,
    "trimming_threshold": trimming_threshold,
    "linear": False,
}

# test inputs
np.random.seed(42)
dgp_dict = make_confounded_irm_data(n_obs=int(1e+6), **dgp_pars)

oracle_dict = dgp_dict['oracle_values']
rho = oracle_dict['rho_atte']
cf_y = oracle_dict['cf_y']
cf_d = oracle_dict['cf_d_atte']

print(f"Confounding factor for Y: {cf_y}")
print(f"Confounding factor for D: {cf_d}")
print(f"Rho: {rho}")

# to get the best possible comparison between different learners (and settings) we first simulate all datasets
np.random.seed(42)
datasets = []
for i in range(n_rep):
    dgp_dict = make_confounded_irm_data(n_obs=n_obs, **dgp_pars)
    datasets.append(dgp_dict)

# set up hyperparameters
hyperparam_dict = {
    "learner_g": [("Linear Reg.", LinearRegression()),
                  ("LGBM", LGBMRegressor(n_estimators=500, learning_rate=0.01, min_child_samples=10, verbose=-1))],
    "learner_m": [("Logistic Regr.", LogisticRegression()),
                  ("LGBM", LGBMClassifier(n_estimators=500, learning_rate=0.01, min_child_samples=10, verbose=-1)),],
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
    dgp_dict = datasets[i_rep]

    x_cols = [f'X{i + 1}' for i in np.arange(dgp_dict['x'].shape[1])]
    df = pd.DataFrame(np.column_stack((dgp_dict['x'], dgp_dict['y'], dgp_dict['d'])), columns=x_cols + ['y', 'd'])
    obj_dml_data = dml.DoubleMLData(df, 'y', 'd')

    for learner_g_idx, (learner_g_name, ml_g) in enumerate(hyperparam_dict["learner_g"]):
        for learner_m_idx, (learner_m_name, ml_m) in enumerate(hyperparam_dict["learner_m"]):
            # Set machine learning methods for g & m
            dml_irm = dml.DoubleMLIRM(
                obj_dml_data=obj_dml_data,
                score='ATTE',
                ml_g=ml_g,
                ml_m=ml_m,
                trimming_threshold=trimming_threshold
            )
            dml_irm.fit(n_jobs_cv=5)

            for level_idx, level in enumerate(hyperparam_dict["level"]):
                estimate = dml_irm.coef[0]
                confint = dml_irm.confint(level=level)
                coverage = (confint.iloc[0, 0] < theta) & (theta < confint.iloc[0, 1])
                ci_length = confint.iloc[0, 1] - confint.iloc[0, 0]

                # test sensitivity parameters
                dml_irm.sensitivity_analysis(cf_y=cf_y, cf_d=cf_d, rho=rho, level=level, null_hypothesis=theta)
                cover_lower = theta >= dml_irm.sensitivity_params['ci']['lower']
                cover_upper = theta <= dml_irm.sensitivity_params['ci']['upper']
                rv = dml_irm.sensitivity_params['rv']
                rva = dml_irm.sensitivity_params['rva']
                bias_lower = abs(theta - dml_irm.sensitivity_params['theta']['lower'])
                bias_upper = abs(theta - dml_irm.sensitivity_params['theta']['upper'])

                df_results_detailed = pd.concat(
                    (df_results_detailed,
                     pd.DataFrame({
                        "Coverage": coverage.astype(int),
                        "CI Length": confint.iloc[0, 1] - confint.iloc[0, 0],
                        "Bias": abs(estimate - theta),
                        "Coverage (Lower)": cover_lower.astype(int),
                        "Coverage (Upper)": cover_upper.astype(int),
                        "RV": rv,
                        "RVa": rva,
                        "Bias (Lower)": bias_lower,
                        "Bias (Upper)": bias_upper,
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
         "Coverage (Lower)": "mean",
         "Coverage (Upper)": "mean",
         "RV": "mean",
         "RVa": "mean",
         "Bias (Lower)": "mean",
         "Bias (Upper)": "mean",
         "repetition": "count"}
    ).reset_index()
print(df_results)
end_time = time.time()
total_runtime = end_time - start_time

# save results
script_name = "irm_atte_sensitivity.py"
path = "results/irm/irm_atte_sensitivity"

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
