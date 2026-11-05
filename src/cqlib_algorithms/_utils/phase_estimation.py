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

"""Shared numerical and circuit support for phase estimation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from cqlib import Circuit, Qubit
from cqlib.circuit.gates import UnitaryGate

from .circuit import (
    snapshot_bound_static_unitary_circuit,
    validate_bound_static_unitary_circuit,
)

_MAX_DENSE_WORK_BYTES = 256 * 1024 * 1024
_MAX_EVALUATION_QUBITS = 52
_MAX_EXACT_BINARY64_INTEGER = (1 << 53) - 1
_UNITARY_ATOL = 1e-10


def normalize_evaluation_qubits(value: object) -> int:
    """Return a valid binary64 phase-register width."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("num_evaluation_qubits must be an integer")
    if value <= 0:
        raise ValueError("num_evaluation_qubits must be greater than zero")
    if value > _MAX_EVALUATION_QUBITS:
        raise ValueError(
            "num_evaluation_qubits must not exceed 52 for binary64 phase output"
        )
    return value


def normalize_shots(value: object, name: str = "shots") -> int:
    """Return a positive shot count exactly representable in binary64."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    if value > _MAX_EXACT_BINARY64_INTEGER:
        raise ValueError(f"{name} must not exceed 2**53 - 1")
    return value


def validate_static_unitary_circuit(circuit: Circuit, name: str) -> None:
    """Validate a fully bound static circuit accepted by phase estimation."""
    validate_bound_static_unitary_circuit(
        circuit,
        name,
        check_numeric_unitaries=False,
    )


def snapshot_static_unitary_circuit(circuit: Circuit, name: str) -> Circuit:
    """Return a defensive phase-estimation circuit snapshot."""
    return snapshot_bound_static_unitary_circuit(
        circuit,
        name,
        check_numeric_unitaries=False,
    )


def unitary_matrix(
    unitary: Circuit,
    num_evaluation_qubits: int,
) -> np.ndarray:
    """Return the checked dense matrix used by controlled powers."""
    dimension = 1 << unitary.num_qubits
    matrix_units = 3 + 4 * num_evaluation_qubits
    estimated_bytes = (
        matrix_units * dimension * dimension * np.dtype(np.complex128).itemsize
    )
    if estimated_bytes > _MAX_DENSE_WORK_BYTES:
        raise ValueError("request exceeds the dense controlled-matrix limit")

    matrix = np.asarray(
        unitary.to_matrix(list(range(unitary.num_qubits))),
        dtype=np.complex128,
    )
    identity = np.eye(dimension, dtype=np.complex128)
    if matrix.shape != (dimension, dimension) or not np.all(np.isfinite(matrix)):
        raise ValueError("unitary must produce a finite square matrix")
    if not np.allclose(
        matrix.conj().T @ matrix,
        identity,
        rtol=0.0,
        atol=_UNITARY_ATOL,
    ):
        raise ValueError("unitary matrix must be unitary")
    return matrix


def append_controlled_matrix(
    circuit: Circuit,
    control: Qubit,
    system_targets: Sequence[Qubit],
    matrix: np.ndarray,
    *,
    label: str,
) -> None:
    """Append one checked dense controlled-unitary matrix."""
    dimension = matrix.shape[0]
    identity = np.eye(dimension, dtype=np.complex128)
    if not np.all(np.isfinite(matrix)) or not np.allclose(
        matrix.conj().T @ matrix,
        identity,
        rtol=0.0,
        atol=_UNITARY_ATOL,
    ):
        raise ValueError("controlled unitary power must remain unitary")

    controlled = np.zeros((2 * dimension, 2 * dimension), dtype=np.complex128)
    controlled[:dimension, :dimension] = identity
    controlled[dimension:, dimension:] = matrix
    gate = UnitaryGate(label, len(system_targets) + 1).with_matrix(controlled)
    # Core's numerical gate treats the first target as its most-significant
    # tensor factor, while Circuit.to_matrix orders system qubits LSB first.
    circuit.append_unitary_gate(gate, [control, *reversed(system_targets)])


__all__: list[str] = []
