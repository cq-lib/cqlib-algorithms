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

"""Public component abstract base classes and supporting value types."""

from .ansatz import Ansatz
from .estimator import Estimator, EstimatorRequest, EstimatorResult
from .gradient import Gradient, GradientRequest, GradientResult
from .optimizer import Optimizer, OptimizerRequest, OptimizerResult, OptimizerStatus
from .quantum_kernel import KernelLoss, QuantumKernel, TrainableQuantumKernel
from .sampler import (
    Sampler,
    SamplerRequest,
    spawn_seeds,
    validate_sampler_result,
    validate_sampler_results,
)
from .scalar_minimizer import ScalarMinimizer, ScalarMinimizerResult
from .state_preparation import StatePreparation

__all__ = [
    "Ansatz",
    "Estimator",
    "EstimatorRequest",
    "EstimatorResult",
    "Gradient",
    "GradientRequest",
    "GradientResult",
    "KernelLoss",
    "Optimizer",
    "OptimizerRequest",
    "OptimizerResult",
    "OptimizerStatus",
    "QuantumKernel",
    "Sampler",
    "SamplerRequest",
    "ScalarMinimizer",
    "ScalarMinimizerResult",
    "StatePreparation",
    "TrainableQuantumKernel",
    "spawn_seeds",
    "validate_sampler_result",
    "validate_sampler_results",
]
