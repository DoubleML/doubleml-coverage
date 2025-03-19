"""Monte Carlo coverage simulation package for statistical methods."""

from importlib.metadata import version

__version__ = version("montecover")

from .base import BaseSimulation

__all__ = ["BaseSimulation"]


def main() -> None:
    print("Hello from monte-cover!")
