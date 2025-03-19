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

    def run_simulation(self):
        """Run the full simulation."""
        self.start_time = time.time()
        self.logger.info("Starting simulation")

        self.logger.info(f"DGP Parameters: {self.dgp_parameters}")
        self.logger.info(f"DML Parameters: {self.dml_parameters}")
        self.logger.info(f"Confidence Parameters: {self.confidence_parameters}")

        # Calculate total number of parameter combinations
        dgp_combinations = [len(v) for v in self.dgp_parameters.values()]
        dml_combinations = [len(v) for v in self.dml_parameters.values()]
        total_combinations = np.prod(dgp_combinations + dml_combinations)
        self.logger.info(f"Total parameter combinations: {total_combinations}")

        # Calculate expected total iterations
        total_iterations = total_combinations * self.repetitions
        self.logger.info(f"Expected total iterations: {total_iterations}")

        # Loop through repetitions
        for i_rep in range(self.repetitions):
            rep_start_time = time.time()
            self.logger.info(f"Starting repetition {i_rep + 1}/{self.repetitions}")

            # Check elapsed time
            elapsed_time = time.time() - self.start_time
            if elapsed_time > self.max_runtime:
                self.logger.warning("Maximum runtime exceeded. Stopping the simulation.")
                break

            param_combo = 0
            # loop through all
            for dgp_param_values in product(*self.dgp_parameters.values()):
                dgp_params = dict(zip(self.dgp_parameters.keys(), dgp_param_values))
                dml_data = self._generate_dml_data(dgp_params)

                for dml_param_values in product(*self.dml_parameters.values()):
                    dml_params = dict(zip(self.dml_parameters.keys(), dml_param_values))

                    param_combo += 1
                    # Log parameter combination
                    self.logger.debug(
                        f"Rep {i_rep+1}, Combo {param_combo}/{total_combinations}: DGPs {dgp_params}, DML {dml_params}"
                    )
                    param_start_time = time.time()

                    try:
                        repetition_results = self.run_single_rep(dml_data, dml_params)
                        param_end_time = time.time()
                        param_duration = param_end_time - param_start_time

                        if repetition_results is not None:
                            assert isinstance(repetition_results, dict), "The result must be a dictionary."
                            # Process each dataframe in the result dictionary
                            for result_name, repetition_result in repetition_results.items():
                                assert isinstance(repetition_result, list), "Each repetition_result must be a list."
                                for res in repetition_result:
                                    assert isinstance(res, dict), "Each res must be a dictionary."
                                    res["repetition"] = i_rep
                                    # add dgp parameters to the result
                                    res.update(dgp_params)

                                # Initialize key in results dict if not exists
                                if result_name not in self.results:
                                    self.results[result_name] = []
                                self.results[result_name].extend(repetition_result)

                            self.logger.debug(f"Parameter combination completed in {param_duration:.2f}s")
                    except Exception as e:
                        self.logger.error(
                            f"Error: repetition {i_rep+1}, DGP parameters {dgp_params}, DML parameters {dml_params}: {str(e)}"
                        )
                        self.logger.exception("Exception details:")

            rep_end_time = time.time()
            rep_duration = rep_end_time - rep_start_time
            self.logger.info(f"Repetition {i_rep+1} completed in {rep_duration:.2f}s")

        # convert results to dataframes
        for key, value in self.results.items():
            self.results[key] = pd.DataFrame(value)

        self.end_time = time.time()
        self.total_runtime = self.end_time - self.start_time
        self.logger.info(f"Simulation completed in {self.total_runtime:.2f}s")

        # Summarize & save results
        self.logger.info("Summarizing results")
        self.result_summary = self.summarize_results()

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
