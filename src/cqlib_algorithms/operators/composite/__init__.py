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

"""Composite operator components built from primitive operators."""

from .grover_operator import GroverOperator
from .hadamard_phase_query import HadamardPhaseQuery
from .hidden_linear_circuit import HiddenLinearFunctionCircuit
from .phase_estimation_circuit import PhaseEstimationCircuit
from .quantum_singular_value_transformation import (
    ProjectorControlledPhase,
    QuantumSignalProcessing,
    QuantumSingularValueTransformation,
    RealValuedQSVT,
)

__all__ = [
    "GroverOperator",
    "HadamardPhaseQuery",
    "HiddenLinearFunctionCircuit",
    "PhaseEstimationCircuit",
    "ProjectorControlledPhase",
    "QuantumSignalProcessing",
    "QuantumSingularValueTransformation",
    "RealValuedQSVT",
]
