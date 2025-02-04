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
n_rep = 250
max_runtime = 5.5 * 3600  # 5.5 hours in seconds

# DGP pars
n_obs = 5000
theta = 5.0
trimming_threshold = 0.05

dgp_pars = {
    "gamma_a": 0.198,
    "beta_a": 0.582,
    "theta": theta,
    "var_epsilon_y": 1.0,
    "trimming_threshold": trimming_threshold,
    "linear": False,
}

# test inputs
np.random.seed(42)
dgp_dict = make_confounded_irm_data(n_obs=int(1e+6), **dgp_pars)

oracle_dict = dgp_dict['oracle_values']
rho = oracle_dict['rho_ate']
cf_y = oracle_dict['cf_y']
cf_d = oracle_dict['cf_d_ate']

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
    "level": [0.95, 0.90],
    "treatment_levels": [0.0, 1.0],
}

# set up the results dataframe
df_results_detailed = pd.DataFrame()
sensitivity_err_count = 0

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
    dgp_dict = datasets[i_rep]

    x_cols = [f'X{i + 1}' for i in np.arange(dgp_dict['x'].shape[1])]
    df = pd.DataFrame(np.column_stack((dgp_dict['x'], dgp_dict['y'], dgp_dict['d'])), columns=x_cols + ['y', 'd'])
    obj_dml_data = dml.DoubleMLData(df, 'y', 'd')

    for learner_g_idx, (learner_g_name, ml_g) in enumerate(hyperparam_dict["learner_g"]):
        for learner_m_idx, (learner_m_name, ml_m) in enumerate(hyperparam_dict["learner_m"]):
            # Set machine learning methods for g & m
            # calculate the APOs
            dml_apos = dml.DoubleMLAPOS(
                obj_dml_data=obj_dml_data,
                ml_g=ml_g,
                ml_m=ml_m,
                treatment_levels=hyperparam_dict["treatment_levels"],
            )

            for level_idx, level in enumerate(hyperparam_dict["level"]):
                dml_apos.fit(n_jobs_cv=5)
            effects = dml_apos.coef
            dml_apos.fit(n_jobs_cv=5)
            effects = dml_apos.coef

            causal_contrast_model = dml_apos.causal_contrast(reference_levels=0)
            estimate = causal_contrast_model.thetas

            for level_idx, level in enumerate(hyperparam_dict["level"]):
                # estimate = causal_contrast_model.coef[0]
                confint = causal_contrast_model.confint(level=level)
                coverage = (confint.iloc[0, 0] < theta) & (theta < confint.iloc[0, 1])
                ci_length = confint.iloc[0, 1] - confint.iloc[0, 0]

                # test sensitivity parameters
                # try to run sensitivity analysis
                try:
                    causal_contrast_model.sensitivity_analysis(cf_y=cf_y, cf_d=cf_d, rho=rho, level=level ,null_hypothesis=theta)
                    cover_lower = theta >= causal_contrast_model.sensitivity_params['ci']['lower']
                    cover_upper = theta <= causal_contrast_model.sensitivity_params['ci']['upper']
                    rv = causal_contrast_model.sensitivity_params['rv']
                    rva = causal_contrast_model.sensitivity_params['rva']
                    bias_lower = abs(theta - causal_contrast_model.sensitivity_params['theta']['lower'])
                    bias_upper = abs(theta - causal_contrast_model.sensitivity_params['theta']['upper'])
                    success_eval = 1
                except Exception as e:
                    sensitivity_err_count += 1
                    continue

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
                        "repetition": i_rep,
                        "success_eval": success_eval}, index=[0])),
                    ignore_index=True)

# aggregate results only if success_eval == 1
df_results_detailed = df_results_detailed[df_results_detailed["success_eval"] == 1]

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
script_name = "irm_apo_sensitivity.py"
path = "results/irm/irm_apo_sensitivity"

metadata = pd.DataFrame({
    'DoubleML Version': [dml.__version__],
    'Script': [script_name],
    'Date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    'Total Runtime (seconds)': [total_runtime],
    'Python Version': [f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"],
    'Sensitivity Errors': [sensitivity_err_count],
})
print(metadata)

df_results.to_csv(f"{path}.csv", index=False)
metadata.to_csv(f"{path}_metadata.csv", index=False)
