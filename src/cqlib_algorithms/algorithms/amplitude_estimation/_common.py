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
from collections.abc import Sequence

import numpy as np
from cqlib import Circuit, Qubit
from cqlib.circuit.gates import UnitaryGate
from cqlib.device import ExecutionResult

from .problem import _UNITARY_ATOL, AmplitudeEstimationProblem, _validate_problem

_MAX_DENSE_WORK_BYTES = 256 * 1024 * 1024


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


def _dense_grover_matrix(
    problem: AmplitudeEstimationProblem,
    num_controlled_powers: int = 1,
) -> np.ndarray:
    """Return the full Grover matrix after enforcing the allocation limit."""
    _validate_problem(problem)
    _validate_positive_integer(num_controlled_powers, "num_controlled_powers")
    dimension = 1 << problem.num_qubits
    # The source and matrix-power temporaries coexist with every controlled
    # matrix already retained by the circuit. Reject before allocating U.
    matrix_units = 8 + 4 * num_controlled_powers
    estimated_bytes = (
        matrix_units * dimension * dimension * np.dtype(np.complex128).itemsize
    )
    if estimated_bytes > _MAX_DENSE_WORK_BYTES:
        raise ValueError("request exceeds the dense controlled-matrix limit")
    matrix = np.asarray(
        problem.grover_operator.to_matrix(list(range(problem.num_qubits))),
        dtype=np.complex128,
    )
    if matrix.shape != (dimension, dimension) or not np.all(np.isfinite(matrix)):
        raise ValueError("grover_operator must produce a finite square matrix")
    identity = np.eye(dimension, dtype=np.complex128)
    if not np.allclose(
        matrix.conj().T @ matrix,
        identity,
        rtol=0.0,
        atol=_UNITARY_ATOL,
    ):
        raise ValueError("grover_operator matrix must be unitary")
    return matrix


def _append_dense_controlled_power(
    circuit: Circuit,
    control: Qubit,
    system_targets: Sequence[Qubit],
    grover_matrix: np.ndarray,
    power: int,
) -> None:
    """Append ``diag(I, Q**power)`` with Core's gate-target convention.

    ``Circuit.to_matrix([0, ..., n-1])`` interprets that order from least to
    most significant. ``append_unitary_gate`` interprets its first target as
    the local matrix's most-significant tensor factor. Reversing the system
    target sequence is therefore required after placing the control first.
    """
    _validate_positive_integer(power, "power")
    dimension = grover_matrix.shape[0]
    powered = np.linalg.matrix_power(grover_matrix, power)
    if not np.all(np.isfinite(powered)):
        raise ValueError("powered grover_operator matrix must be finite")
    identity = np.eye(dimension, dtype=np.complex128)
    if not np.allclose(
        powered.conj().T @ powered,
        identity,
        rtol=0.0,
        atol=_UNITARY_ATOL,
    ):
        raise ValueError("powered grover_operator matrix must be unitary")

    controlled = np.zeros((2 * dimension, 2 * dimension), dtype=np.complex128)
    controlled[:dimension, :dimension] = identity
    controlled[dimension:, dimension:] = powered
    gate = UnitaryGate(
        f"controlled-grover-power-{power}", len(system_targets) + 1
    ).with_matrix(controlled)
    circuit.append_unitary_gate(gate, [control, *reversed(system_targets)])


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
