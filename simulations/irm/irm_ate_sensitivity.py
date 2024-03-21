import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

import doubleml as dml
from doubleml.datasets import make_confounded_irm_SZ2020

# Number of repetitions
n_rep = 20

# DGP pars
n_obs = 10000
gamma_a = 0.11
beta_a = 0.6
theta = 0.0
dgp_type = 4

# test inputs
dgp_dict = make_confounded_irm_SZ2020(
    n_obs=int(1e+6),
    theta=theta,
    gamma_a=gamma_a,
    beta_a=beta_a,
    dgp_type=dgp_type,
    var_epsilon_y=1.0)

oracle_dict = dgp_dict['oracle_values']
rho = oracle_dict['rho']
cf_y = oracle_dict['cf_y']
cf_d = oracle_dict['cf_d']

print(f"Confounding factor for Y: {cf_y}")
print(f"Confounding factor for D: {cf_d}")
print(f"Rho: {rho}")

# to get the best possible comparison between different learners (and settings) we first simulate all datasets
np.random.seed(42)
datasets = []
for i in range(n_rep):
    data = dgp_dict = make_confounded_irm_SZ2020(
        n_obs=n_obs,
        theta=theta,
        gamma_a=gamma_a,
        beta_a=beta_a,
        dgp_type=dgp_type,
        var_epsilon_y=1.0)
    datasets.append(data)

# set up hyperparameters
hyperparam_dict = {
    "learner_g": [("LGBM", LGBMRegressor(n_estimators=1000, learning_rate=0.05, min_child_samples=5)),
                  ("Random Forest",
                   RandomForestRegressor(n_estimators=200, max_features=20, max_depth=5, min_samples_leaf=2))],
    "learner_m": [("LGBM", LGBMClassifier(n_estimators=100, learning_rate=0.05, min_child_samples=20)),
                  ("Random Forest",
                   RandomForestClassifier(n_estimators=200, max_features=20, max_depth=5, min_samples_leaf=20))],
    "level": [0.95, 0.90]
}

# set up the results dataframe
df_results_detailed = pd.DataFrame(
    columns=["Coverage", "CI Length",
             "Bias", "Coverage (Lower)", "Coverage (Upper)",
             "RV", "RVa", "Bias (Lower)", "Bias (Upper)", "score",
             "Learner g", "Learner m",
             "level", "repetition"])

# start simulation
np.random.seed(42)

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
                ml_g=ml_g,
                ml_m=ml_m,
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

# save results
df_results.to_csv("results/irm_ate_sensitivity.csv", index=False)
