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

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .grover_operator import GroverOperator
    from .hadamard_phase_query import HadamardPhaseQuery
    from .hidden_linear_circuit import HiddenLinearFunctionCircuit
    from .phase_estimation_circuit import PhaseEstimationCircuit
    from .qft import QFT
    from .quantum_singular_value_transformation import (
        ProjectorControlledPhase,
        QuantumSignalProcessing,
        QuantumSingularValueTransformation,
        RealValuedQSVT,
    )

_EXPORTS: dict[str, tuple[str, str]] = {
    "GroverOperator": (".grover_operator", "GroverOperator"),
    "HadamardPhaseQuery": (".hadamard_phase_query", "HadamardPhaseQuery"),
    "HiddenLinearFunctionCircuit": (
        ".hidden_linear_circuit",
        "HiddenLinearFunctionCircuit",
    ),
    "PhaseEstimationCircuit": (
        ".phase_estimation_circuit",
        "PhaseEstimationCircuit",
    ),
    "ProjectorControlledPhase": (
        ".quantum_singular_value_transformation",
        "ProjectorControlledPhase",
    ),
    "QFT": (".qft", "QFT"),
    "QuantumSignalProcessing": (
        ".quantum_singular_value_transformation",
        "QuantumSignalProcessing",
    ),
    "QuantumSingularValueTransformation": (
        ".quantum_singular_value_transformation",
        "QuantumSingularValueTransformation",
    ),
    "RealValuedQSVT": (
        ".quantum_singular_value_transformation",
        "RealValuedQSVT",
    ),
}

__all__ = [
    "QFT",
    "GroverOperator",
    "HadamardPhaseQuery",
    "HiddenLinearFunctionCircuit",
    "PhaseEstimationCircuit",
    "ProjectorControlledPhase",
    "QuantumSignalProcessing",
    "QuantumSingularValueTransformation",
    "RealValuedQSVT",
]


def __getattr__(name: str) -> object:
    """Resolve and cache a public composite component on first access."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module globals together with lazily exported names."""
    return sorted(set(globals()) | set(__all__))
