import time
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from itertools import product
import numpy as np


class BaseSimulation(ABC):
    """Abstract base class for simulation studies."""

    def __init__(
        self,
        n_rep: int = 20,
        max_runtime: float = 5.5 * 3600,
        random_seed: int = 42,
        output_path: str = "results",
        suppress_warnings: bool = True,
    ):
        self.n_rep = n_rep
        self.max_runtime = max_runtime
        self.random_seed = random_seed
        self.output_path = output_path
        self.suppress_warnings = suppress_warnings

        # Results storage
        self.results = []
        self.aggregated_results = None

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

    def run_simulation(self):
        """Run the full simulation."""
        self.start_time = time.time()
        param_grid = self.setup_parameters()

        # Loop through repetitions
        for i_rep in range(self.n_rep):
            print(f"Repetition: {i_rep + 1}/{self.n_rep}", end="\r")

            # Check elapsed time
            elapsed_time = time.time() - self.start_time
            if elapsed_time > self.max_runtime:
                print("Maximum runtime exceeded. Stopping the simulation.")
                break

            keys = param_grid.keys()
            for values in product(*param_grid.values()):
                params = dict(zip(keys, values))
                result = self.run_single_rep(i_rep, params)
                if result is not None:
                    result["repetition"] = i_rep
                    self.results.append(result)

        self.end_time = time.time()
        self.total_runtime = self.end_time - self.start_time

    # Add other methods here...
