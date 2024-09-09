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
max_runtime = 30  # 5.5 * 3600  # 5.5 hours in seconds

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
    average_ites[i] = np.mean(ite[d == i]) * (i > 0)
    apos[i] = np.mean(y0) + average_ites[i]

ates = np.full(n_levels, np.nan)
for i in range(n_levels):
    ates[i] = apos[i + 1] - apos[0]

print(f"Levels and their counts:\n{np.unique(d, return_counts=True)}")
print(f"True APOs: {apos}")
print(f"True ATEs: {ates}")

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
    "treatment_levels": [0.0, 1.0, 2.0],
    "level": [0.95, 0.90],
    "trimming_threshold": 0.01
}

# set up the results dataframe
df_results_detailed_apo = pd.DataFrame()
df_results_detailed_apos = pd.DataFrame()
df_results_detailed_apos_constrast = pd.DataFrame()

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
    obj_dml_data = dml.DoubleMLData(datasets[i_rep], 'y', 'd')

    for learner_g_idx, (learner_g_name, ml_g) in enumerate(hyperparam_dict["learner_g"]):
        for learner_m_idx, (learner_m_name, ml_m) in enumerate(hyperparam_dict["learner_m"]):
            for treatment_idx, treatment_level in enumerate(hyperparam_dict["treatment_levels"]):
                dml_apo = dml.DoubleMLAPO(
                    obj_dml_data=obj_dml_data,
                    ml_g=ml_g,
                    ml_m=ml_m,
                    treatment_level=treatment_level,
                    trimming_threshold=hyperparam_dict["trimming_threshold"]
                )
                dml_apo.fit(n_jobs_cv=5)

                for level_idx, level in enumerate(hyperparam_dict["level"]):
                    confint = dml_apo.confint(level=level)
                    coverage = (confint.iloc[0, 0] < apos[treatment_idx]) & (apos[treatment_idx] < confint.iloc[0, 1])
                    ci_length = confint.iloc[0, 1] - confint.iloc[0, 0]

                    df_results_detailed_apo = pd.concat(
                        (df_results_detailed_apo,
                            pd.DataFrame({
                                "Coverage": coverage.astype(int),
                                "CI Length": confint.iloc[0, 1] - confint.iloc[0, 0],
                                "Bias": abs(dml_apo.coef[0] - apos[treatment_idx]),
                                "Treatment Level": treatment_level,
                                "Learner g": learner_g_name,
                                "Learner m": learner_m_name,
                                "level": level,
                                "repetition": i_rep}, index=[0])),
                        ignore_index=True)

            # calculate the APOs
            dml_apos = dml.DoubleMLAPOS(
                obj_dml_data=obj_dml_data,
                ml_g=ml_g,
                ml_m=ml_m,
                treatment_levels=hyperparam_dict["treatment_levels"],
                trimming_threshold=hyperparam_dict["trimming_threshold"]
            )
            dml_apos.fit(n_jobs_cv=5)
            effects = dml_apos.coef

            causal_contrast_model = dml_apos.causal_contrast(reference_levels=0)
            est_ates = causal_contrast_model.thetas

            for level_idx, level in enumerate(hyperparam_dict["level"]):
                confint = dml_apos.confint(level=level)
                coverage = np.mean((confint.iloc[:, 0] < apos) & (apos < confint.iloc[:, 1]))
                ci_length = np.mean(confint.iloc[:, 1] - confint.iloc[:, 0])

                dml_apos.bootstrap(n_rep_boot=2000)
                confint_uniform = dml_apos.confint(level=level, joint=True)
                coverage_uniform = all((confint_uniform.iloc[:, 0] < apos) & (apos < confint_uniform.iloc[:, 1]))
                ci_length_uniform = np.mean(confint_uniform.iloc[:, 1] - confint_uniform.iloc[:, 0])
                df_results_detailed_apos = pd.concat(
                    (df_results_detailed_apos,
                        pd.DataFrame({
                            "Coverage": coverage,
                            "CI Length": ci_length,
                            "Bias": np.mean(abs(effects - apos)),
                            "Uniform Coverage": coverage_uniform,
                            "Uniform CI Length": ci_length_uniform,
                            "Learner g": learner_g_name,
                            "Learner m": learner_m_name,
                            "level": level,
                            "repetition": i_rep}, index=[0])),
                    ignore_index=True)

                # calculate the ATEs
                confint_contrast = causal_contrast_model.confint(level=level)
                coverage_contrast = np.mean((confint_contrast.iloc[:, 0] < ates) & (ates < confint_contrast.iloc[:, 1]))
                ci_length_contrast = np.mean(confint_contrast.iloc[:, 1] - confint_contrast.iloc[:, 0])

                causal_contrast_model.bootstrap(n_rep_boot=2000)
                confint_contrast_uniform = causal_contrast_model.confint(level=level, joint=True)
                coverage_contrast_uniform = all(
                    (confint_contrast_uniform.iloc[:, 0] < ates) & (ates < confint_contrast_uniform.iloc[:, 1]))
                ci_length_contrast_uniform = np.mean(confint_contrast_uniform.iloc[:, 1] - confint_contrast_uniform.iloc[:, 0])
                df_results_detailed_apos_constrast = pd.concat(
                    (df_results_detailed_apos_constrast,
                        pd.DataFrame({
                            "Coverage": coverage_contrast,
                            "CI Length": ci_length_contrast,
                            "Bias": np.mean(abs(est_ates - ates)),
                            "Uniform Coverage": coverage_contrast_uniform,
                            "Uniform CI Length": ci_length_contrast_uniform,
                            "Learner g": learner_g_name,
                            "Learner m": learner_m_name,
                            "level": level,
                            "repetition": i_rep}, index=[0])),
                    ignore_index=True)

df_results_apo = df_results_detailed_apo.groupby(
    ["Learner g", "Learner m", "Treatment Level", "level"]).agg(
        {"Coverage": "mean",
         "CI Length": "mean",
         "Bias": "mean",
         "repetition": "count"}
    ).reset_index()
print(df_results_apo)

df_results_apos = df_results_detailed_apos.groupby(
    ["Learner g", "Learner m", "level"]).agg(
        {"Coverage": "mean",
         "CI Length": "mean",
         "Bias": "mean",
         "Uniform Coverage": "mean",
         "Uniform CI Length": "mean",
         "repetition": "count"}
    ).reset_index()
print(df_results_apos)

df_results_apos_contrast = df_results_detailed_apos_constrast.groupby(
    ["Learner g", "Learner m", "level"]).agg(
        {"Coverage": "mean",
         "CI Length": "mean",
         "Bias": "mean",
         "Uniform Coverage": "mean",
         "Uniform CI Length": "mean",
         "repetition": "count"}
    ).reset_index()
print(df_results_apos_contrast)

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

df_results_apo.to_csv(f"{path}_apo.csv", index=False)
df_results_apos.to_csv(f"/{path}_apos.csv", index=False)
df_results_apos_contrast.to_csv(f"{path}_apos_contrast.csv", index=False)
metadata.to_csv(f"{path}_metadata.csv", index=False)
