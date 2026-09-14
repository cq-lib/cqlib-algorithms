"""Composable quantum algorithm components for Cqlib."""

from importlib.metadata import version

from .protocols import (
    Ansatz,
    Estimator,
    EstimatorRequest,
    EstimatorResult,
    Gradient,
    GradientRequest,
    GradientResult,
    Sampler,
    SamplerRequest,
    StatePreparation,
)
from .transforms import QFT

__version__ = version("cqlib-algorithms")

__all__ = [
    "__version__",
    "Ansatz",
    "Estimator",
    "EstimatorRequest",
    "EstimatorResult",
    "Gradient",
    "GradientRequest",
    "GradientResult",
    "Sampler",
    "SamplerRequest",
    "StatePreparation",
    "QFT",
]
