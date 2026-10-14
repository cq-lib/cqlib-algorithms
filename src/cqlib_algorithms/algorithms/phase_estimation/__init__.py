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

"""Quantum and Hamiltonian phase-estimation algorithms."""

from .hamiltonian import (
    HamiltonianPhaseEstimation,
    HamiltonianPhaseEstimationRequest,
    HamiltonianPhaseEstimationResult,
)
from .iterative import (
    IterativePhaseEstimation,
    IterativePhaseEstimationRequest,
    IterativePhaseEstimationResult,
)
from .problem import PhaseEstimationProblem
from .standard import PhaseEstimation, PhaseEstimationRequest, PhaseEstimationResult

__all__ = [
    "HamiltonianPhaseEstimation",
    "HamiltonianPhaseEstimationRequest",
    "HamiltonianPhaseEstimationResult",
    "IterativePhaseEstimation",
    "IterativePhaseEstimationRequest",
    "IterativePhaseEstimationResult",
    "PhaseEstimation",
    "PhaseEstimationProblem",
    "PhaseEstimationRequest",
    "PhaseEstimationResult",
]
