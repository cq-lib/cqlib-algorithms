# This code is part of Cqlib.
#
# (C) Copyright China Telecom Quantum Group 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Composable quantum algorithm components for Cqlib."""

from importlib.metadata import version

from .algorithms import (
    BB84,
    CHSH,
    QAOA,
    VQE,
    BB84Basis,
    BB84Request,
    BB84Result,
    BernsteinVazirani,
    BernsteinVaziraniRequest,
    BernsteinVaziraniResult,
    CHSHRequest,
    CHSHResult,
    DeutschJozsa,
    DeutschJozsaRequest,
    DeutschJozsaResult,
    QAOAResult,
    Simon,
    SimonRequest,
    SimonResult,
    VQAResult,
    VQEResult,
)
from .protocols import (
    Ansatz,
    Estimator,
    EstimatorRequest,
    EstimatorResult,
    Gradient,
    GradientRequest,
    GradientResult,
    Optimizer,
    OptimizerRequest,
    OptimizerResult,
    OptimizerStatus,
    Sampler,
    SamplerRequest,
    StatePreparation,
)
from .transforms import QFT

__version__ = version("cqlib-algorithms")

__all__ = [
    "BB84",
    "CHSH",
    "QAOA",
    "QFT",
    "VQE",
    "Ansatz",
    "BB84Basis",
    "BB84Request",
    "BB84Result",
    "BernsteinVazirani",
    "BernsteinVaziraniRequest",
    "BernsteinVaziraniResult",
    "CHSHRequest",
    "CHSHResult",
    "DeutschJozsa",
    "DeutschJozsaRequest",
    "DeutschJozsaResult",
    "Estimator",
    "EstimatorRequest",
    "EstimatorResult",
    "Gradient",
    "GradientRequest",
    "GradientResult",
    "Optimizer",
    "OptimizerRequest",
    "OptimizerResult",
    "OptimizerStatus",
    "QAOAResult",
    "Sampler",
    "SamplerRequest",
    "Simon",
    "SimonRequest",
    "SimonResult",
    "StatePreparation",
    "VQAResult",
    "VQEResult",
    "__version__",
]
