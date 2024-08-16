import numpy as np
import pandas as pd
from datetime import datetime
import time
import sys

from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

import doubleml as dml
from doubleml.datasets import make_confounded_plr_data


# Number of repetitions
n_rep = 500

# DGP pars
n_obs = 1000
cf_y = 0.1
cf_d = 0.1
theta = 5.0

# to get the best possible comparison between different learners (and settings) we first simulate all datasets
np.random.seed(42)

# test inputs
dgp_dict = make_confounded_plr_data(n_obs=int(1e+6), cf_y=cf_y, cf_d=cf_d)
oracle_dict = dgp_dict['oracle_values']

cf_y_test = np.mean(np.square(oracle_dict['g_long'] - oracle_dict['g_short'])) / \
    np.mean(np.square(dgp_dict['y'] - oracle_dict['g_short']))
print(f'Input cf_y:{cf_y} \nCalculated cf_y: {round(cf_y_test, 5)}')

rr_long = (dgp_dict['d'] - oracle_dict['m_long']) / np.mean(np.square(dgp_dict['d'] - oracle_dict['m_long']))
rr_short = (dgp_dict['d'] - oracle_dict['m_short']) / np.mean(np.square(dgp_dict['d'] - oracle_dict['m_short']))
C2_D = (np.mean(np.square(rr_long)) - np.mean(np.square(rr_short))) / np.mean(np.square(rr_short))
cf_d_test = C2_D / (1 + C2_D)
print(f'Input cf_d:{cf_d}\nCalculated cf_d: {round(cf_d_test, 5)}')

# compute the value for rho
rho = np.corrcoef((oracle_dict['g_long'] - oracle_dict['g_short']), (rr_long - rr_short))[0, 1]
print(f'Correlation rho: {round(rho, 5)}')

datasets = []
for i in range(n_rep):
    data = make_confounded_plr_data(n_obs=n_obs, cf_y=cf_y, cf_d=cf_d, theta=theta)
    datasets.append(data)

# set up hyperparameters
hyperparam_dict = {
    "score": ["partialling out", "IV-type"],
    "learner_g": [("LGBM", LGBMRegressor(n_estimators=500, learning_rate=0.05, min_child_samples=5, verbose=-1)),
                  ("Random Forest",
                   RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2))],
    "learner_m": [("LGBM", LGBMRegressor(n_estimators=500, learning_rate=0.05, min_child_samples=2, verbose=-1)),
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
    print(f"Repetition: {i_rep + 1}/{n_rep}", end="\r")

    # define the DoubleML data object
    dgp_dict = datasets[i_rep]
    x_cols = [f'X{i + 1}' for i in np.arange(dgp_dict['x'].shape[1])]
    df = pd.DataFrame(np.column_stack((dgp_dict['x'], dgp_dict['y'], dgp_dict['d'])), columns=x_cols + ['y', 'd'])
    obj_dml_data = dml.DoubleMLData(df, 'y', 'd')

    for score_idx, score in enumerate(hyperparam_dict["score"]):
        for learner_g_idx, (learner_g_name, ml_g) in enumerate(hyperparam_dict["learner_g"]):
            for learner_m_idx, (learner_m_name, ml_m) in enumerate(hyperparam_dict["learner_m"]):
                if score == "IV-type":
                    # Set machine learning methods for g & m
                    dml_plr = dml.DoubleMLPLR(
                        obj_dml_data=obj_dml_data,
                        ml_l=ml_g,
                        ml_m=ml_m,
                        ml_g=ml_g,
                        score="IV-type",
                    )
                else:
                    # Set machine learning methods for g & m
                    dml_plr = dml.DoubleMLPLR(
                        obj_dml_data=obj_dml_data,
                        ml_l=ml_g,
                        ml_m=ml_m,
                        score=score,
                    )
                dml_plr.fit(n_jobs_cv=5)

                for level_idx, level in enumerate(hyperparam_dict["level"]):

                    estimate = dml_plr.coef[0]
                    confint = dml_plr.confint(level=level)
                    coverage = (confint.iloc[0, 0] < theta) & (theta < confint.iloc[0, 1])
                    ci_length = confint.iloc[0, 1] - confint.iloc[0, 0]

                    # test sensitivity parameters
                    dml_plr.sensitivity_analysis(cf_y=cf_y, cf_d=cf_d, rho=rho, level=level, null_hypothesis=theta)
                    cover_lower = theta >= dml_plr.sensitivity_params['ci']['lower']
                    cover_upper = theta <= dml_plr.sensitivity_params['ci']['upper']
                    rv = dml_plr.sensitivity_params['rv']
                    rva = dml_plr.sensitivity_params['rva']
                    bias_lower = abs(theta - dml_plr.sensitivity_params['theta']['lower'])
                    bias_upper = abs(theta - dml_plr.sensitivity_params['theta']['upper'])

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
                            "score": score,
                            "Learner g": learner_g_name,
                            "Learner m": learner_m_name,
                            "level": level,
                            "repetition": i_rep}, index=[0])),
                        ignore_index=True)

df_results = df_results_detailed.groupby(
    ["Learner g", "Learner m", "score", "level"]).agg(
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
script_name = "plr_ate_sensitivity.py"
path = "results/plm/plr_ate_sensitivity"

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
