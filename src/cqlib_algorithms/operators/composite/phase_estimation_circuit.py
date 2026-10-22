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

"""Reusable quantum phase-estimation circuit construction.

References
----------
[1] R. Cleve, A. Ekert, C. Macchiavello, and M. Mosca, "Quantum algorithms
    revisited," Proceedings of the Royal Society A, 454, 339-354, 1998.
    DOI: 10.1098/rspa.1998.0164.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import final

import numpy as np
from cqlib import Circuit, Qubit
from cqlib.circuit import CircuitError
from cqlib.circuit.gates import UnitaryGate

from ...transforms import QFT

_MAX_DENSE_WORK_BYTES = 256 * 1024 * 1024
_MAX_EVALUATION_QUBITS = 52
_UNITARY_ATOL = 1e-10


def _validate_evaluation_qubits(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("num_evaluation_qubits must be an integer")
    if value <= 0:
        raise ValueError("num_evaluation_qubits must be greater than zero")
    if value > _MAX_EVALUATION_QUBITS:
        raise ValueError(
            "num_evaluation_qubits must not exceed 52 for binary64 phase output"
        )
    return value


def _validate_static_unitary_circuit(circuit: Circuit, name: str) -> None:
    try:
        circuit.validate()
    except CircuitError as error:
        raise ValueError(f"{name} must be a valid circuit") from error
    if circuit.num_qubits <= 0:
        raise ValueError(f"{name} must contain at least one qubit")
    if circuit.used_symbols:
        raise ValueError(f"{name} must be fully bound")
    if circuit.classical_vars or circuit.classical_values:
        raise ValueError(f"{name} must not contain classical data")

    try:
        operations = circuit.decompose().operations
    except CircuitError as error:
        raise ValueError(f"{name} must be statically decomposable") from error
    for operation in operations:
        if operation.is_directive and operation.name == "barrier":
            continue
        if (
            operation.is_classical_data
            or operation.is_classical_control
            or operation.is_delay
            or operation.is_directive
            or operation.instruction_type not in {"standard", "mcgate", "unitary"}
        ):
            raise ValueError(
                f"{name} must contain only static unitary operations and barriers"
            )


def _snapshot_static_unitary_circuit(circuit: Circuit, name: str) -> Circuit:
    if not isinstance(circuit, Circuit):
        raise TypeError(f"{name} must be a Circuit")
    source = copy.deepcopy(circuit)
    _validate_static_unitary_circuit(source, name)
    normalized = Circuit(source.num_qubits)
    normalized.compose(source, normalized.qubits)
    _validate_static_unitary_circuit(normalized, name)
    return normalized


def _unitary_matrix(
    unitary: Circuit,
    num_evaluation_qubits: int,
) -> np.ndarray:
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


def _append_controlled_matrix(
    circuit: Circuit,
    control: Qubit,
    system_targets: Sequence[Qubit],
    matrix: np.ndarray,
    *,
    label: str,
) -> None:
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


@final
class PhaseEstimationCircuit:
    """Build the standard multi-qubit phase-estimation circuit.

    For an eigenstate ``|psi>`` satisfying
    ``U|psi> = exp(2*pi*i*phi)|psi>``, evaluation qubit ``j`` controls
    ``U**(2**j)``. An inverse QFT then maps ``phi`` to the evaluation
    register. The returned circuit contains no state preparation or
    measurement operations.

    Arbitrary Core circuits cannot currently be promoted directly to a
    quantum-controlled circuit. This component therefore constructs dense
    controlled matrices, preserving the unitary's global phase. The path has
    exponential memory cost in the system width and enforces a pre-allocation
    memory limit.

    Args:
        unitary: Positive-width, fully bound static unitary circuit.
        num_evaluation_qubits: Positive phase-register width.
        approximation_degree: Small-angle inverse-QFT layers to omit.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the circuit, width, approximation, or memory bound is invalid.
    """

    __slots__ = (
        "_approximation_degree",
        "_num_evaluation_qubits",
        "_unitary",
    )

    def __init__(
        self,
        unitary: Circuit,
        num_evaluation_qubits: int,
        *,
        approximation_degree: int = 0,
    ) -> None:
        self._num_evaluation_qubits = _validate_evaluation_qubits(num_evaluation_qubits)
        if isinstance(approximation_degree, bool) or not isinstance(
            approximation_degree, int
        ):
            raise TypeError("approximation_degree must be an integer")
        if not 0 <= approximation_degree < self._num_evaluation_qubits:
            raise ValueError(
                "approximation_degree must satisfy "
                "0 <= approximation_degree < num_evaluation_qubits"
            )
        self._approximation_degree = approximation_degree
        self._unitary = _snapshot_static_unitary_circuit(unitary, "unitary")

    @property
    def num_evaluation_qubits(self) -> int:
        """Return the evaluation-register width."""
        return self._num_evaluation_qubits

    @property
    def num_system_qubits(self) -> int:
        """Return the unitary's system-register width."""
        return int(self._unitary.num_qubits)

    @property
    def num_qubits(self) -> int:
        """Return the total circuit width."""
        return self.num_evaluation_qubits + self.num_system_qubits

    @property
    def approximation_degree(self) -> int:
        """Return the inverse-QFT approximation degree."""
        return self._approximation_degree

    def build_circuit(self) -> Circuit:
        """Return a fresh, unmeasured phase-estimation circuit."""
        _validate_static_unitary_circuit(self._unitary, "unitary")
        unitary_matrix = _unitary_matrix(
            self._unitary,
            self._num_evaluation_qubits,
        )
        circuit = Circuit(self.num_qubits)
        evaluation_targets = tuple(circuit.qubits[: self._num_evaluation_qubits])
        system_targets = tuple(circuit.qubits[self._num_evaluation_qubits :])

        for target in evaluation_targets:
            circuit.h(target)

        powered = unitary_matrix
        for index, control in enumerate(evaluation_targets):
            _append_controlled_matrix(
                circuit,
                control,
                system_targets,
                powered,
                label=f"controlled-unitary-power-{1 << index}",
            )
            if index + 1 < self._num_evaluation_qubits:
                powered = powered @ powered

        circuit.compose(
            QFT(
                self._num_evaluation_qubits,
                approximation_degree=self._approximation_degree,
                inverse=True,
            ).build_circuit(),
            list(evaluation_targets),
        )
        return circuit


__all__ = ["PhaseEstimationCircuit"]
