import time
import warnings
import pandas as pd
from abc import ABC, abstractmethod
from itertools import product
from typing import Any, Dict, List

import sys
from datetime import datetime
import numpy as np
import doubleml as dml


class BaseSimulation(ABC):
    """Abstract base class for simulation studies."""

    def __init__(
        self,
        repetitions: int = 20,
        max_runtime: float = 5.5 * 3600,
        random_seed: int = 42,
        output_path: str = "results",
        suppress_warnings: bool = True,
    ):
        self.repetitions = repetitions
        self.max_runtime = max_runtime
        self.random_seed = random_seed
        self.output_path = output_path
        self.suppress_warnings = suppress_warnings

        # Results storage
        self.results = dict()
        self.result_summary = None

        # Timing
        self.start_time = None
        self.end_time = None
        self.total_runtime = None

        # Set up environment
        if suppress_warnings:
            warnings.simplefilter(action="ignore", category=UserWarning)

        np.random.seed(self.random_seed)

    @abstractmethod
    def setup_parameters(self) -> Dict[str, List[Any]]:
        """Define simulation parameters."""
        pass

    @abstractmethod
    def run_single_rep(self, rep_idx: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single repetition with the given parameters."""
        pass

    @abstractmethod
    def summarize_results(self) -> Dict[str, Any]:
        """Summarize the simulation results."""
        pass

    def run_simulation(self):
        """Run the full simulation."""
        self.start_time = time.time()
        param_grid = self.setup_parameters()

        # Loop through repetitions
        for i_rep in range(self.repetitions):
            print(f"Repetition: {i_rep + 1}/{self.repetitions}", end="\r")

            # Check elapsed time
            elapsed_time = time.time() - self.start_time
            if elapsed_time > self.max_runtime:
                print("Maximum runtime exceeded. Stopping the simulation.")
                break

            keys = param_grid.keys()
            for values in product(*param_grid.values()):
                params = dict(zip(keys, values))
                repetition_results = self.run_single_rep(params)
                if repetition_results is not None:
                    assert isinstance(repetition_results, dict), "The result must be a dictionary."
                    # Process each dataframe in the result dictionary
                    for result_name, repetition_result in repetition_results.items():
                        assert isinstance(repetition_result, dict), "Each result must be a dictionary."
                        repetition_result["repetition"] = i_rep

                        # Initialize key in results dict if not exists
                        if result_name not in self.results:
                            self.results[result_name] = []
                        self.results[result_name].append(repetition_result)

        # convert results to dataframes
        for key, value in self.results.items():
            self.results[key] = pd.DataFrame(value)

        self.end_time = time.time()
        self.total_runtime = self.end_time - self.start_time

        # Summarize & save results
        self.result_summary = self.summarize_results()
        print("Simulation finished.")

    def save_results(self):
        """Save the simulation results."""
        metadata = pd.DataFrame(
            {
                "DoubleML Version": [dml.__version__],
                "Script": ["DIDMultiCoverageSimulation"],
                "Date": [datetime.now().strftime("%Y-%m-%d %H")],
                "Total Runtime (seconds)": [self.total_runtime],
                "Python Version": [f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"],
            }
        )

        for df_name, df in self.result_summary.items():
            df.to_csv(f"{self.output_path}_{df_name}.csv", index=False)
        metadata.to_csv(f"{self.output_path}_metadata.csv", index=False)

    @staticmethod
    def _compute_coverage(thetas, oracle_thetas, confint, joint_confint=None):
        """Compute coverage, CI length, and bias."""

        coverage = np.mean((confint.iloc[:, 0] < oracle_thetas) & (oracle_thetas < confint.iloc[:, 1]))
        ci_length = np.mean(confint.iloc[:, 1] - confint.iloc[:, 0])
        bias = np.mean(abs(thetas - oracle_thetas))
        result_dict = {
            "Coverage": coverage,
            "CI Length": ci_length,
            "Bias": bias,
        }

        if joint_confint is not None:
            coverage_uniform = all((joint_confint.iloc[:, 0] < oracle_thetas) & (oracle_thetas < joint_confint.iloc[:, 1]))
            ci_length_uniform = np.mean(joint_confint.iloc[:, 1] - joint_confint.iloc[:, 0])

            result_dict["Uniform Coverage"] = coverage_uniform
            result_dict["Uniform CI Length"] = ci_length_uniform

        return result_dict
