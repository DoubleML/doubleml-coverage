import numpy as np
import pandas as pd
from datetime import datetime
import time
import sys
import patsy

from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.linear_model import LassoCV, LogisticRegressionCV

import doubleml as dml
from doubleml.datasets import make_heterogeneous_data

# Number of repetitions
n_rep = 1000
max_runtime = 5.5 * 3600  # 5.5 hours in seconds

# DGP pars
n_obs = 2000
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
                  ("LGBM", LGBMRegressor(n_estimators=200, learning_rate=0.05, verbose=-1))],
    "learner_m": [("Logistic Regression", LogisticRegressionCV()),
                  ("LGBM", LGBMClassifier(n_estimators=200, learning_rate=0.05, verbose=-1))],
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
    data = datasets[i_rep]['data']
    design_matrix = patsy.dmatrix("bs(x, df=5, degree=2)", {"x": data["X_0"]})
    spline_basis = pd.DataFrame(design_matrix)

    true_effects = datasets[i_rep]['effects']

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
            cate = dml_irm.cate(spline_basis)

            for level_idx, level in enumerate(hyperparam_dict["level"]):
                confint = cate.confint(basis=spline_basis, level=level)
                effects = confint["effect"]
                coverage = (confint.iloc[:, 0] < true_effects) & (true_effects < confint.iloc[:, 2])
                ci_length = confint.iloc[:, 2] - confint.iloc[:, 0]
                confint_uniform = cate.confint(basis=spline_basis, level=0.95, joint=True, n_rep_boot=2000)
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
script_name = "irm_cate_coverage.py"
path = "results/irm/irm_cate_coverage"

metadata = pd.DataFrame({
    'DoubleML Version': [dml.__version__],
    'Script': [script_name],
    'Date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    'Total Runtime (seconds)': [total_runtime],
    'Python Version': [f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"],
    'Number of observations': [n_obs],
    'Number of repetitions': [n_rep],
})
print(metadata)

df_results.to_csv(f"{path}.csv", index=False)
metadata.to_csv(f"{path}_metadata.csv", index=False)
