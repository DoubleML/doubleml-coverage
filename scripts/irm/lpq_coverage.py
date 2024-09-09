import numpy as np
import pandas as pd
import multiprocessing
from datetime import datetime
import time
import sys

from sklearn.linear_model import LogisticRegressionCV
from lightgbm import LGBMClassifier

import doubleml as dml

# set up parallelization
n_cores = multiprocessing.cpu_count()
print(f"Number of Cores: {n_cores}")
cores_used = n_cores-1

# Number of repetitions
n_rep = 100
max_runtime = 5.5 * 3600  # 5.5 hours in seconds

# DGP pars
n_obs = 5000
tau_vec = np.arange(0.3, 0.75, 0.05)
p = 5


# define loc-scale model
def f_loc(D, X, X_conf):
    loc = 0.5*D + 2*D*X[:, 4] + 2.0*(X[:, 1] > 0.1) - 1.7*(X[:, 0] * X[:, 2] > 0) - 3*X[:, 3] - 2*X_conf[:, 0]
    return loc


def f_scale(D, X, X_conf):
    scale = np.sqrt(0.5*D + 3*D*X[:, 0] + 0.4*X_conf[:, 0] + 2)
    return scale


def generate_treatment(Z, X, X_conf):
    eta = np.random.normal(size=len(Z))
    d = ((0.5*Z - 0.3*X[:, 0] + 0.7*X_conf[:, 0] + eta) > 0)*1.0
    return d


def dgp(n=200, p=5):
    X = np.random.uniform(0, 1, size=[n, p])
    X_conf = np.random.uniform(-1, 1, size=[n, 1])
    Z = np.random.binomial(1, p=0.5, size=n)
    D = generate_treatment(Z, X, X_conf)
    epsilon = np.random.normal(size=n)

    Y = f_loc(D, X, X_conf) + f_scale(D, X, X_conf)*epsilon

    return Y, X, D, Z


# Estimate true LPQ and LQTE with counterfactuals on large sample

n_true = int(10e+6)

X_true = np.random.uniform(0, 1, size=[n_true, p])
X_conf_true = np.random.uniform(-1, 1, size=[n_true, 1])
Z_true = np.random.binomial(1, p=0.5, size=n_true)
eta_true = np.random.normal(size=n_true)
D1_true = generate_treatment(np.ones_like(Z_true), X_true, X_conf_true)
D0_true = generate_treatment(np.zeros_like(Z_true), X_true, X_conf_true)
epsilon_true = np.random.normal(size=n_true)

compliers = (D1_true == 1) * (D0_true == 0)
print(f'Compliance probability: {str(compliers.mean())}')
n_compliers = compliers.sum()
Y1 = f_loc(np.ones(n_compliers), X_true[compliers, :], X_conf_true[compliers, :]) +\
    f_scale(np.ones(n_compliers), X_true[compliers, :], X_conf_true[compliers, :])*epsilon_true[compliers]
Y0 = f_loc(np.zeros(n_compliers), X_true[compliers, :], X_conf_true[compliers, :]) +\
    f_scale(np.zeros(n_compliers), X_true[compliers, :], X_conf_true[compliers, :])*epsilon_true[compliers]

Y0_quant = np.quantile(Y0, q=tau_vec)
Y1_quant = np.quantile(Y1, q=tau_vec)
print(f'Local Potential Quantile Y(0): {Y0_quant}')
print(f'Local Potential Quantile Y(1): {Y1_quant}')
LQTE = Y1_quant - Y0_quant
print(f'Local Quantile Treatment Effect: {LQTE}')


# to get the best possible comparison between different learners (and settings) we first simulate all datasets
np.random.seed(42)
datasets = []
for i in range(n_rep):
    Y, X, D, Z = dgp(n=n_obs, p=p)
    data = dml.DoubleMLData.from_arrays(X, Y, D, Z)
    datasets.append(data)

# set up hyperparameters
hyperparam_dict = {
    "learner_g": [("Logistic Regression", LogisticRegressionCV()),
                  ("LGBM", LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=10, verbose=-1))],
    "learner_m": [("Logistic Regression", LogisticRegressionCV()),
                  ("LGBM", LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=10, verbose=-1))],
    "level": [0.95, 0.90]
}

# set up the results dataframe
df_results_detailed_qte = pd.DataFrame()
df_results_detailed_pq0 = pd.DataFrame()
df_results_detailed_pq1 = pd.DataFrame()

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
    obj_dml_data = datasets[i_rep]

    for learner_g_idx, (learner_g_name, ml_g) in enumerate(hyperparam_dict["learner_g"]):
        for learner_m_idx, (learner_m_name, ml_m) in enumerate(hyperparam_dict["learner_m"]):
            # Set machine learning methods for g & m
            dml_qte = dml.DoubleMLQTE(
                obj_dml_data=obj_dml_data,
                ml_g=ml_g,
                ml_m=ml_m,
                score='LPQ',
                quantiles=tau_vec
            )
            dml_qte.fit(n_jobs_models=cores_used)
            effects = dml_qte.coef

            for level_idx, level in enumerate(hyperparam_dict["level"]):
                confint = dml_qte.confint(level=level)
                coverage = np.mean((confint.iloc[:, 0] < LQTE) & (LQTE < confint.iloc[:, 1]))
                ci_length = np.mean(confint.iloc[:, 1] - confint.iloc[:, 0])

                dml_qte.bootstrap(n_rep_boot=2000)
                confint_uniform = dml_qte.confint(level=level, joint=True)
                coverage_uniform = all((confint_uniform.iloc[:, 0] < LQTE) &
                                       (LQTE < confint_uniform.iloc[:, 1]))
                ci_length_uniform = np.mean(confint_uniform.iloc[:, 1] - confint_uniform.iloc[:, 0])
                df_results_detailed_qte = pd.concat(
                    (df_results_detailed_qte,
                     pd.DataFrame({
                        "Coverage": coverage,
                        "CI Length": ci_length,
                        "Bias": np.mean(abs(effects - LQTE)),
                        "Uniform Coverage": coverage_uniform,
                        "Uniform CI Length": ci_length_uniform,
                        "Learner g": learner_g_name,
                        "Learner m": learner_m_name,
                        "level": level,
                        "repetition": i_rep}, index=[0])),
                    ignore_index=True)

                # evaluate each model
                coverage_0 = np.zeros(len(tau_vec))
                coverage_1 = np.zeros(len(tau_vec))

                ci_length_0 = np.zeros(len(tau_vec))
                ci_length_1 = np.zeros(len(tau_vec))

                bias_0 = np.zeros(len(tau_vec))
                bias_1 = np.zeros(len(tau_vec))
                for tau_idx, tau in enumerate(tau_vec):
                    model_0 = dml_qte.modellist_0[tau_idx]
                    model_1 = dml_qte.modellist_1[tau_idx]

                    confint_0 = model_0.confint(level=level)
                    confint_1 = model_1.confint(level=level)

                    coverage_0[tau_idx] = (confint_0.iloc[0, 0] < Y0_quant[tau_idx]) & \
                        (Y0_quant[tau_idx] < confint_0.iloc[0, 1])
                    coverage_1[tau_idx] = (confint_1.iloc[0, 0] < Y1_quant[tau_idx]) & \
                        (Y1_quant[tau_idx] < confint_1.iloc[0, 1])

                    ci_length_0[tau_idx] = confint_0.iloc[0, 1] - confint_0.iloc[0, 0]
                    ci_length_1[tau_idx] = confint_1.iloc[0, 1] - confint_1.iloc[0, 0]

                    bias_0[tau_idx] = abs(model_0.coef[0] - Y0_quant[tau_idx])
                    bias_1[tau_idx] = abs(model_1.coef[0] - Y1_quant[tau_idx])

                df_results_detailed_pq0 = pd.concat(
                    (df_results_detailed_pq0,
                     pd.DataFrame({
                        "Coverage": np.mean(coverage_0),
                        "CI Length": np.mean(ci_length_0),
                        "Bias": np.mean(bias_0),
                        "Learner g": learner_g_name,
                        "Learner m": learner_m_name,
                        "level": level,
                        "repetition": i_rep}, index=[0])),
                    ignore_index=True)

                df_results_detailed_pq1 = pd.concat(
                    (df_results_detailed_pq1,
                     pd.DataFrame({
                        "Coverage": np.mean(coverage_1),
                        "CI Length": np.mean(ci_length_1),
                        "Bias": np.mean(bias_1),
                        "Learner g": learner_g_name,
                        "Learner m": learner_m_name,
                        "level": level,
                        "repetition": i_rep}, index=[0])),
                    ignore_index=True)

df_results_qte = df_results_detailed_qte.groupby(
    ["Learner g", "Learner m", "level"]).agg(
        {"Coverage": "mean",
         "CI Length": "mean",
         "Bias": "mean",
         "Uniform Coverage": "mean",
         "Uniform CI Length": "mean",
         "repetition": "count"}
    ).reset_index()
print(df_results_qte)

df_results_pq0 = df_results_detailed_pq0.groupby(
    ["Learner g", "Learner m", "level"]).agg(
        {"Coverage": "mean",
         "CI Length": "mean",
         "Bias": "mean",
         "repetition": "count"}
    ).reset_index()
print(df_results_pq0)

df_results_pq1 = df_results_detailed_pq1.groupby(
    ["Learner g", "Learner m", "level"]).agg(
        {"Coverage": "mean",
         "CI Length": "mean",
         "Bias": "mean",
         "repetition": "count"}
    ).reset_index()
print(df_results_pq1)

end_time = time.time()
total_runtime = end_time - start_time

# save results
script_name = "lpq_coverage.py"
path = "results/irm/lpq_coverage"

metadata = pd.DataFrame({
    'DoubleML Version': [dml.__version__],
    'Script': [script_name],
    'Date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    'Total Runtime (seconds)': [total_runtime],
    'Python Version': [f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"],
})
print(metadata)

df_results_qte.to_csv(f"{path}_lqte.csv", index=False)
df_results_pq0.to_csv(f"{path}_lpq0.csv", index=False)
df_results_pq1.to_csv(f"{path}_lpq1.csv", index=False)
metadata.to_csv(f"{path}_metadata.csv", index=False)
