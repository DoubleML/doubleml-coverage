import numpy as np
import pandas as pd
import multiprocessing
from datetime import datetime
import time
import sys

from sklearn.linear_model import LogisticRegressionCV, LinearRegression
from lightgbm import LGBMClassifier, LGBMRegressor

import doubleml as dml

# set up parallelization
n_cores = multiprocessing.cpu_count()
print(f"Number of Cores: {n_cores}")
cores_used = n_cores - 1

# Number of repetitions
n_rep = 100
max_runtime = 5.5 * 3600  # 5.5 hours in seconds

# DGP pars
n_obs = 5000
tau_vec = np.arange(0.2, 0.85, 0.05)
p = 5


# define loc-scale model
def f_loc(D, X):
    loc = (
        0.5 * D
        + 2 * D * X[:, 4]
        + 2.0 * (X[:, 1] > 0.1)
        - 1.7 * (X[:, 0] * X[:, 2] > 0)
        - 3 * X[:, 3]
    )
    return loc


def f_scale(D, X):
    scale = np.sqrt(0.5 * D + 0.3 * D * X[:, 1] + 2)
    return scale


def dgp(n=200, p=5):
    X = np.random.uniform(-1, 1, size=[n, p])
    D = ((X[:, 1] - X[:, 3] + 1.5 * (X[:, 0] > 0) + np.random.normal(size=n)) > 0) * 1.0
    epsilon = np.random.normal(size=n)

    Y = f_loc(D, X) + f_scale(D, X) * epsilon
    return Y, X, D, epsilon


# Estimate true  and QTE with counterfactuals on large sample
n_true = int(10e6)

_, X_true, _, epsilon_true = dgp(n=n_true, p=p)
D1 = np.ones(n_true)
D0 = np.zeros(n_true)

Y1 = f_loc(D1, X_true) + f_scale(D1, X_true) * epsilon_true
Y0 = f_loc(D0, X_true) + f_scale(D0, X_true) * epsilon_true

Y1_quant = np.quantile(Y1, q=tau_vec)
Y0_quant = np.quantile(Y0, q=tau_vec)
Y1_cvar = [Y1[Y1 >= quant].mean() for quant in Y1_quant]
Y0_cvar = [Y0[Y0 >= quant].mean() for quant in Y0_quant]
CVAR = np.array(Y1_cvar) - np.array(Y0_cvar)

print(f"Conditional Value at Risk Y(0): {Y0_cvar}")
print(f"Conditional Value at Risk Y(1): {Y1_cvar}")
print(f"Conditional Value at Risk Effect: {CVAR}")

# to get the best possible comparison between different learners (and settings) we first simulate all datasets
np.random.seed(42)
datasets = []
for i in range(n_rep):
    Y, X, D, _ = dgp(n=n_obs, p=p)
    data = dml.DoubleMLData.from_arrays(X, Y, D)
    datasets.append(data)

# set up hyperparameters
hyperparam_dict = {
    "learner_g": [
        ("Linear", LinearRegression()),
        (
            "LGBM",
            LGBMRegressor(
                n_estimators=300, learning_rate=0.05, num_leaves=10, verbose=-1
            ),
        ),
    ],
    "learner_m": [
        ("Logistic Regression", LogisticRegressionCV()),
        (
            "LGBM",
            LGBMClassifier(
                n_estimators=300, learning_rate=0.05, num_leaves=10, verbose=-1
            ),
        ),
    ],
    "level": [0.95, 0.90],
}

# set up the results dataframe
df_results_detailed_qte = pd.DataFrame()
df_results_detailed_pq0 = pd.DataFrame()
df_results_detailed_pq1 = pd.DataFrame()

# start simulation
np.random.seed(42)
start_time = time.time()

for i_rep in range(n_rep):
    print(f"Repetition: {i_rep + 1}/{n_rep}", end="\r")

    # Check the elapsed time
    elapsed_time = time.time() - start_time
    if elapsed_time > max_runtime:
        print("Maximum runtime exceeded. Stopping the simulation.")
        break

    # define the DoubleML data object
    obj_dml_data = datasets[i_rep]

    for learner_g_idx, (learner_g_name, ml_g) in enumerate(
        hyperparam_dict["learner_g"]
    ):
        for learner_m_idx, (learner_m_name, ml_m) in enumerate(
            hyperparam_dict["learner_m"]
        ):
            # Set machine learning methods for g & m
            dml_qte = dml.DoubleMLQTE(
                obj_dml_data=obj_dml_data,
                ml_g=ml_g,
                ml_m=ml_m,
                score="CVaR",
                quantiles=tau_vec,
            )
            dml_qte.fit(n_jobs_models=cores_used)
            effects = dml_qte.coef

            for level_idx, level in enumerate(hyperparam_dict["level"]):
                confint = dml_qte.confint(level=level)
                coverage = np.mean(
                    (confint.iloc[:, 0] < CVAR) & (CVAR < confint.iloc[:, 1])
                )
                ci_length = np.mean(confint.iloc[:, 1] - confint.iloc[:, 0])

                dml_qte.bootstrap(n_rep_boot=2000)
                confint_uniform = dml_qte.confint(level=level, joint=True)
                coverage_uniform = all(
                    (confint_uniform.iloc[:, 0] < CVAR)
                    & (CVAR < confint_uniform.iloc[:, 1])
                )
                ci_length_uniform = np.mean(
                    confint_uniform.iloc[:, 1] - confint_uniform.iloc[:, 0]
                )
                df_results_detailed_qte = pd.concat(
                    (
                        df_results_detailed_qte,
                        pd.DataFrame(
                            {
                                "Coverage": coverage,
                                "CI Length": ci_length,
                                "Bias": np.mean(abs(effects - CVAR)),
                                "Uniform Coverage": coverage_uniform,
                                "Uniform CI Length": ci_length_uniform,
                                "Learner g": learner_g_name,
                                "Learner m": learner_m_name,
                                "level": level,
                                "repetition": i_rep,
                            },
                            index=[0],
                        ),
                    ),
                    ignore_index=True,
                )

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

                    coverage_0[tau_idx] = (confint_0.iloc[0, 0] < Y0_cvar[tau_idx]) & (
                        Y0_cvar[tau_idx] < confint_0.iloc[0, 1]
                    )
                    coverage_1[tau_idx] = (confint_1.iloc[0, 0] < Y1_cvar[tau_idx]) & (
                        Y1_cvar[tau_idx] < confint_1.iloc[0, 1]
                    )

                    ci_length_0[tau_idx] = confint_0.iloc[0, 1] - confint_0.iloc[0, 0]
                    ci_length_1[tau_idx] = confint_1.iloc[0, 1] - confint_1.iloc[0, 0]

                    bias_0[tau_idx] = abs(model_0.coef[0] - Y0_cvar[tau_idx])
                    bias_1[tau_idx] = abs(model_1.coef[0] - Y1_cvar[tau_idx])

                df_results_detailed_pq0 = pd.concat(
                    (
                        df_results_detailed_pq0,
                        pd.DataFrame(
                            {
                                "Coverage": np.mean(coverage_0),
                                "CI Length": np.mean(ci_length_0),
                                "Bias": np.mean(bias_0),
                                "Learner g": learner_g_name,
                                "Learner m": learner_m_name,
                                "level": level,
                                "repetition": i_rep,
                            },
                            index=[0],
                        ),
                    ),
                    ignore_index=True,
                )

                df_results_detailed_pq1 = pd.concat(
                    (
                        df_results_detailed_pq1,
                        pd.DataFrame(
                            {
                                "Coverage": np.mean(coverage_1),
                                "CI Length": np.mean(ci_length_1),
                                "Bias": np.mean(bias_1),
                                "Learner g": learner_g_name,
                                "Learner m": learner_m_name,
                                "level": level,
                                "repetition": i_rep,
                            },
                            index=[0],
                        ),
                    ),
                    ignore_index=True,
                )

df_results_qte = (
    df_results_detailed_qte.groupby(["Learner g", "Learner m", "level"])
    .agg(
        {
            "Coverage": "mean",
            "CI Length": "mean",
            "Bias": "mean",
            "Uniform Coverage": "mean",
            "Uniform CI Length": "mean",
            "repetition": "count",
        }
    )
    .reset_index()
)
print(df_results_qte)

df_results_pq0 = (
    df_results_detailed_pq0.groupby(["Learner g", "Learner m", "level"])
    .agg(
        {"Coverage": "mean", "CI Length": "mean", "Bias": "mean", "repetition": "count"}
    )
    .reset_index()
)
print(df_results_pq0)

df_results_pq1 = (
    df_results_detailed_pq1.groupby(["Learner g", "Learner m", "level"])
    .agg(
        {"Coverage": "mean", "CI Length": "mean", "Bias": "mean", "repetition": "count"}
    )
    .reset_index()
)
print(df_results_pq1)

end_time = time.time()
total_runtime = end_time - start_time

# save results
script_name = "cvar_coverage.py"
path = "results/irm/cvar_coverage"

metadata = pd.DataFrame(
    {
        "DoubleML Version": [dml.__version__],
        "Script": [script_name],
        "Date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Total Runtime (seconds)": [total_runtime],
        "Python Version": [
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ],
    }
)
print(metadata)

df_results_qte.to_csv(f"{path}_qte.csv", index=False)
df_results_pq0.to_csv(f"{path}_pq0.csv", index=False)
df_results_pq1.to_csv(f"{path}_pq1.csv", index=False)
metadata.to_csv(f"{path}_metadata.csv", index=False)
