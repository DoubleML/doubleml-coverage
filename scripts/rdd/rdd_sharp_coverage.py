import numpy as np
import pandas as pd
from datetime import datetime
import time
import sys

from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from rdrobust import rdrobust

import doubleml as dml
from doubleml.rdd import RDFlex
from doubleml.rdd.datasets import make_simple_rdd_data
from doubleml.utils import GlobalRegressor

from statsmodels.nonparametric.kernel_regression import KernelReg


# Number of repetitions
n_rep = 500
max_runtime = 5.5 * 3600  # 5.5 hours in seconds

# DGP pars
n_obs = 1000
cutoff = 0

# to get the best possible comparison between different learners (and settings) we first simulate all datasets
np.random.seed(42)

datasets = []
for i in range(n_rep):
    data = make_simple_rdd_data(n_obs=n_obs, fuzzy=False, cutoff=cutoff)
    datasets.append(data)

# set up hyperparameters
hyperparam_dict = {
    "fs_specification": ["cutoff", "cutoff and score", "interacted cutoff and score"],
    "learner_g": [
        ("Linear", LinearRegression()),
        ("LGBM", LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, verbose=-1)),
        ("Global linear", GlobalRegressor(LinearRegression())),
        ("Stacked", StackingRegressor(
            estimators=[
                ('lr', LinearRegression()),
                ('lgbm', LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, verbose=-1)),
                ('glr', GlobalRegressor(LinearRegression()))],
            final_estimator=Ridge()))],
    "level": [0.95, 0.90]}

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

    data = datasets[i_rep]
    # get oracle value
    score = data["score"]
    ite = data["oracle_values"]['Y1'] - data["oracle_values"]['Y0']

    kernel_reg = KernelReg(endog=ite, exog=score, var_type='c', reg_type='ll')
    effect_at_cutoff, _ = kernel_reg.fit(np.array([cutoff]))
    oracle_effect = effect_at_cutoff[0]

    Y = data["Y"]
    Z = data["X"].reshape(n_obs, -1)
    D = data["D"]

    # baseline
    for level_idx, level in enumerate(hyperparam_dict["level"]):
        res = rdrobust(y=Y, x=score, covs=Z, c=cutoff, level=level*100)
        coef = res.coef.loc["Robust", "Coeff"]
        ci_lower = res.ci.loc["Robust", "CI Lower"]
        ci_upper = res.ci.loc["Robust", "CI Upper"]

        coverage = (ci_lower < oracle_effect) & (oracle_effect < ci_upper)
        ci_length = ci_upper - ci_lower

        df_results_detailed = pd.concat(
            (df_results_detailed,
                pd.DataFrame({
                    "Coverage": coverage.astype(int),
                    "CI Length": ci_length,
                    "Bias": abs(coef - oracle_effect),
                    "Learner g": "linear",
                    "Method": "rdrobust",
                    "fs specification": "cutoff",
                    "level": level,
                    "repetition": i_rep}, index=[0])),
            ignore_index=True)

    # define the DoubleML data object
    obj_dml_data = dml.DoubleMLData.from_arrays(y=Y, d=D, x=Z, s=score)

    for learner_g_idx, (learner_g_name, ml_g) in enumerate(hyperparam_dict["learner_g"]):
        for fs_specification_idx, fs_specification in enumerate(hyperparam_dict["fs_specification"]):
            rdflex_model = RDFlex(
                obj_dml_data,
                ml_g=ml_g,
                n_folds=5,
                n_rep=1,
                cutoff=cutoff,
                fuzzy=False,
                fs_specification=fs_specification)
            rdflex_model.fit(n_iterations=2)

            for level_idx, level in enumerate(hyperparam_dict["level"]):
                confint = rdflex_model.confint(level=level)
                coverage = (confint.iloc[2, 0] < oracle_effect) & (oracle_effect < confint.iloc[2, 1])
                ci_length = confint.iloc[2, 1] - confint.iloc[2, 0]

                df_results_detailed = pd.concat(
                    (df_results_detailed,
                        pd.DataFrame({
                            "Coverage": coverage.astype(int),
                            "CI Length": ci_length,
                            "Bias": abs(rdflex_model.coef[2] - oracle_effect),
                            "Learner g": learner_g_name,
                            "Method": "rdflex",
                            "fs specification": fs_specification,
                            "level": level,
                            "repetition": i_rep}, index=[0])),
                    ignore_index=True)

df_results = df_results_detailed.groupby(
    ["Method", "fs specification", "Learner g", "level"]).agg(
        {"Coverage": "mean",
         "CI Length": "mean",
         "Bias": "mean",
         "repetition": "count"}
    ).reset_index()
print(df_results)

end_time = time.time()
total_runtime = end_time - start_time

# save results
script_name = "rdd_sharp_coverage.py"
path = "results/rdd/rdd_sharp_coverage"

metadata = pd.DataFrame({
    'DoubleML Version': [dml.__version__],
    'Script': [script_name],
    'Date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    'Total Runtime (seconds)': [total_runtime],
    'Python Version': [f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"],
})
print(metadata)

df_results.to_csv(f"{path}.csv", index=False)
metadata.to_csv(f"{path}_metadata.csv", index=False)
