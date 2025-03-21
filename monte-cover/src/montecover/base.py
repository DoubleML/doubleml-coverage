import logging
import os
import sys
import time
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from itertools import product
from typing import Any, Dict, Optional

import doubleml as dml
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed, parallel_backend


class BaseSimulation(ABC):
    """Abstract base class for simulation studies."""

    def __init__(
        self,
        config_file: str,
        suppress_warnings: bool = True,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
    ):
        # Load config first to get default values if needed
        self.config_file = config_file
        self.config = self._load_config(config_file)

        # Apply parameters from config
        self.simulation_parameters = self.config.get("simulation_parameters", {})
        self.repetitions = self.simulation_parameters.get("repetitions", 20)
        self.max_runtime = self.simulation_parameters.get("max_runtime", 5.5 * 3600)
        self.random_seed = self.simulation_parameters.get("random_seed", 42)
        self.default_n_jobs = self.simulation_parameters.get("n_jobs", 1)
        self.suppress_warnings = suppress_warnings

        # Set up logging
        self._setup_logging(log_level, log_file)
        self.logger.info(f"Loaded configuration from {config_file}")

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
        self.logger.info(f"Initialized simulation with random seed {self.random_seed}")

        # Let child classes process any specific parameters
        self.dgp_parameters = self.config.get("dgp_parameters", {})
        self.dml_parameters = self.config.get("dml_parameters", {})
        self.confidence_parameters = self.config.get("confidence_parameters", {})
        self._process_config_parameters()

    @abstractmethod
    def _generate_dml_data(self, dgp_params: Dict[str, Any]) -> dml.DoubleMLData:
        """Generate data for the DoubleML simulation."""
        pass

    @abstractmethod
    def run_single_rep(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single repetition with the given parameters."""
        pass

    @abstractmethod
    def summarize_results(self) -> Dict[str, Any]:
        """Summarize the simulation results."""
        pass

    def run_simulation(self, n_jobs=None):
        """
        Run the full simulation.

        Parameters:
        -----------
        n_jobs : int, optional (default=None)
            Number of jobs to run in parallel. If None, uses the value from the configuration file.
            If 1, runs sequentially. If > 1, runs in parallel with the specified number of workers.
            If -1, uses all available CPU cores .
            If -2, uses all available CPU cores except one.
        """
        # Use n_jobs from parameter, or fall back to config value
        n_jobs = n_jobs if n_jobs is not None else self.default_n_jobs
        self._log_parameters(n_jobs=n_jobs)

        if n_jobs == 1:
            for i_rep in range(self.repetitions):
                rep_start_time = time.time()
                self.logger.info(f"Starting repetition {i_rep + 1}/{self.repetitions}")

                if self._stop_simulation():
                    break

                # Running actual simulation
                self._process_repetition(i_rep)

                rep_end_time = time.time()
                rep_duration = rep_end_time - rep_start_time
                self.logger.info(f"Repetition {i_rep+1} completed in {rep_duration:.2f}s")

        else:
            self.logger.info(f"Starting parallel execution with n_jobs={n_jobs}")
            with parallel_backend("loky", inner_max_num_threads=1):
                results = Parallel(n_jobs=n_jobs, verbose=10)(
                    delayed(self._process_repetition)(i_rep) for i_rep in range(self.repetitions) if not self._stop_simulation()
                )

            # Process results from parallel execution
            for worker_results in results:
                if worker_results:  # Check if we have results
                    for result_name, result_list in worker_results.items():
                        if result_name not in self.results:
                            self.results[result_name] = []
                        self.results[result_name].extend(result_list)

        self._process_results()

    def save_results(self, output_path: str = "results", file_prefix: str = ""):
        """Save the simulation results."""
        os.makedirs(output_path, exist_ok=True)

        metadata = pd.DataFrame(
            {
                "DoubleML Version": [dml.__version__],
                "Script": [self.__class__.__name__],
                "Date": [datetime.now().strftime("%Y-%m-%d %H:%M")],
                "Total Runtime (minutes)": [self.total_runtime / 60],
                "Python Version": [f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"],
                "Config File": [self.config_file],
            }
        )

        for df_name, df in self.result_summary.items():
            output_file = os.path.join(output_path, f"{file_prefix}_{df_name}.csv")
            df.to_csv(output_file, index=False)
            self.logger.info(f"Results saved to {output_file}")

        metadata_file = os.path.join(output_path, f"{file_prefix}_metadata.csv")
        metadata.to_csv(metadata_file, index=False)
        self.logger.info(f"Metadata saved to {metadata_file}")

    def save_config(self, output_path: str):
        """Save the current configuration to a YAML file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if not (output_path.endswith(".yaml") or output_path.endswith(".yml")):
            output_path += ".yaml"
            self.logger.warning(f"Adding .yaml extension to output path: {output_path}")

        with open(output_path, "w") as file:
            yaml.dump(self.config, file)

        self.logger.info(f"Configuration saved to {output_path}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from a YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
        else:
            raise ValueError(f"Unsupported config file format: {config_path}. Use .yaml or .yml")

        return config

    def _process_config_parameters(self):
        """
        Process any special configuration parameters.
        Child classes should override this method to handle specific parameters.
        """
        pass

    def _setup_logging(self, log_level: str, log_file: Optional[str]):
        """Set up logging configuration."""
        level = getattr(logging, log_level.upper())
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(level)

        # Clear any existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # File handler if specified
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def _log_parameters(self, n_jobs):
        """Initialize timing and calculate parameter combination metrics."""
        self.start_time = time.time()
        self.logger.info("Starting simulation")
        self.logger.info(f"DGP Parameters: {self.dgp_parameters}")
        self.logger.info(f"DML Parameters: {self.dml_parameters}")
        self.logger.info(f"Confidence Parameters: {self.confidence_parameters}")

        # Calculate expected iterations
        dgp_combinations = [len(v) for v in self.dgp_parameters.values()]
        dml_combinations = [len(v) for v in self.dml_parameters.values()]
        self.total_combinations = np.prod(dgp_combinations + dml_combinations)
        self.total_iterations = self.total_combinations * self.repetitions

        self.logger.info(f"Total parameter combinations: {self.total_combinations}")
        self.logger.info(f"Expected total iterations: {self.total_iterations}")
        if n_jobs <= 1:
            self.logger.info("Running simulation sequentially")
        else:
            self.logger.info(f"Running simulation in parallel with {n_jobs} workers")

    def _stop_simulation(self) -> bool:
        """Check if simulation should be stopped based on criteria like runtime."""
        # Check if maximum runtime is exceeded
        if self.max_runtime and time.time() - self.start_time > self.max_runtime:
            self.logger.warning("Maximum runtime exceeded. Stopping the simulation.")
            return True
        return False

    def _process_repetition(self, i_rep):
        """Process a single repetition with all parameter combinations."""
        if self.suppress_warnings:
            warnings.simplefilter(action="ignore", category=UserWarning)

        i_param_comb = 0
        rep_results = {}

        # loop through all parameter combinations
        for dgp_param_values in product(*self.dgp_parameters.values()):
            dgp_params = dict(zip(self.dgp_parameters.keys(), dgp_param_values))
            dml_data = self._generate_dml_data(dgp_params)

            for dml_param_values in product(*self.dml_parameters.values()):
                dml_params = dict(zip(self.dml_parameters.keys(), dml_param_values))
                i_param_comb += 1

                comb_results = self._process_parameter_combination(i_rep, i_param_comb, dgp_params, dml_params, dml_data)

                # Merge results
                for result_name, result_list in comb_results.items():
                    if result_name not in rep_results:
                        rep_results[result_name] = []
                    rep_results[result_name].extend(result_list)

        return rep_results

    def _process_parameter_combination(self, i_rep, i_param_comb, dgp_params, dml_params, dml_data):
        """Process a single parameter combination."""
        # Log parameter combination
        self.logger.debug(
            f"Rep {i_rep+1}, Combo {i_param_comb}/{self.total_combinations}: " f"DGPs {dgp_params}, DML {dml_params}"
        )
        param_start_time = time.time()

        try:
            repetition_results = self.run_single_rep(dml_data, dml_params)

            # Log timing
            param_duration = time.time() - param_start_time
            self.logger.debug(f"Parameter combination completed in {param_duration:.2f}s")

            # Process results
            if repetition_results is None:
                return {}

            # Add metadata to results
            processed_results = {}
            for result_name, repetition_result in repetition_results.items():
                processed_results[result_name] = []
                for res in repetition_result:
                    res["repetition"] = i_rep
                    res.update(dgp_params)
                    processed_results[result_name].append(res)

            return processed_results

        except Exception as e:
            self.logger.error(
                f"Error: repetition {i_rep+1}, DGP parameters {dgp_params}, " f"DML parameters {dml_params}: {str(e)}"
            )
            self.logger.exception("Exception details:")
            return {}

    def _process_results(self):
        """Process collected results and log completion metrics."""
        # Convert results to dataframes incrementally
        for key, value in self.results.items():
            self.results[key] = pd.DataFrame(value)

        self.end_time = time.time()
        self.total_runtime = self.end_time - self.start_time
        self.logger.info(f"Simulation completed in {self.total_runtime:.2f}s")

        # Summarize results
        self.logger.info("Summarizing results")
        self.result_summary = self.summarize_results()

    @staticmethod
    def _compute_coverage(thetas, oracle_thetas, confint, joint_confint=None):
        """Compute coverage, CI length, and bias."""
        lower_bound = confint.iloc[:, 0]
        upper_bound = confint.iloc[:, 1]
        coverage_mask = (lower_bound < oracle_thetas) & (oracle_thetas < upper_bound)

        result_dict = {
            "Coverage": np.mean(coverage_mask),
            "CI Length": np.mean(upper_bound - lower_bound),
            "Bias": np.mean(np.abs(thetas - oracle_thetas)),
        }

        if joint_confint is not None:
            joint_lower_bound = joint_confint.iloc[:, 0]
            joint_upper_bound = joint_confint.iloc[:, 1]
            joint_coverage_mark = (joint_lower_bound < oracle_thetas) & (oracle_thetas < joint_upper_bound)

            result_dict["Uniform Coverage"] = np.all(joint_coverage_mark)
            result_dict["Uniform CI Length"] = np.mean(joint_upper_bound - joint_lower_bound)

        return result_dict
