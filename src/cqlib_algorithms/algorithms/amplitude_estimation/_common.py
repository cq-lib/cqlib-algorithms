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

"""Shared construction and validation for amplitude-estimation algorithms."""

from __future__ import annotations

import copy

from cqlib import Circuit
from cqlib.device import ExecutionResult

from .problem import AmplitudeEstimationProblem, _validate_problem


def _validate_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _validate_nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _build_grover_power_circuit(
    problem: AmplitudeEstimationProblem,
    power: int,
) -> Circuit:
    """Build ``Q**power A`` without modifying either problem circuit."""
    _validate_problem(problem)
    _validate_nonnegative_integer(power, "power")
    circuit = Circuit(problem.num_qubits)
    circuit.compose(copy.deepcopy(problem.state_preparation), circuit.qubits)
    for _ in range(power):
        circuit.compose(copy.deepcopy(problem.grover_operator), circuit.qubits)
    return circuit


def _count_good_outcomes(
    result: ExecutionResult,
    problem: AmplitudeEstimationProblem,
) -> int:
    """Return the number of sampled objective outcomes classified as good."""
    good = 0
    for bitstring, count in result.counts.items():
        classified = problem.is_good_state(bitstring)
        if not isinstance(classified, bool):
            raise TypeError("is_good_state must return a bool")
        if classified:
            good += count
    return good
