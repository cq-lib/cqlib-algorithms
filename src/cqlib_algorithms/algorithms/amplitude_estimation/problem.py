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

"""Problem definition for amplitude-estimation algorithms."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from cqlib import Circuit
from cqlib.circuit import CircuitError

_UNITARY_ATOL = 1e-10


def _validate_static_unitary_circuit(circuit: Circuit, name: str) -> None:
    """Validate a fully bound circuit containing only static unitary work."""
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
        expanded = circuit.decompose()
    except CircuitError as error:
        raise ValueError(f"{name} must be statically decomposable") from error

    for operation in expanded.operations:
        if operation.is_directive and operation.name == "barrier":
            continue
        if (
            operation.is_classical_data
            or operation.is_classical_control
            or operation.is_delay
            or operation.is_directive
        ):
            raise ValueError(
                f"{name} must contain only static unitary operations and barriers"
            )
        if operation.instruction_type not in {"standard", "mcgate", "unitary"}:
            raise ValueError(
                f"{name} must contain only static unitary operations and barriers"
            )
        if operation.instruction_type != "unitary":
            continue
        try:
            matrix = np.asarray(operation.matrix(), dtype=np.complex128)
        except (TypeError, ValueError, CircuitError) as error:
            raise ValueError(f"{name} contains an invalid numeric unitary") from error
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} contains a non-finite numeric unitary")
        identity = np.eye(matrix.shape[0], dtype=np.complex128)
        if not np.allclose(
            matrix.conj().T @ matrix,
            identity,
            rtol=0.0,
            atol=_UNITARY_ATOL,
        ):
            raise ValueError(f"{name} contains a non-unitary numeric gate")


def _snapshot_circuit(circuit: Circuit, name: str) -> Circuit:
    if not isinstance(circuit, Circuit):
        raise TypeError(f"{name} must be a Circuit")
    source = copy.deepcopy(circuit)
    _validate_static_unitary_circuit(source, name)
    normalized = Circuit(source.num_qubits)
    normalized.compose(source, normalized.qubits)
    _validate_static_unitary_circuit(normalized, name)
    return normalized


@dataclass(frozen=True, slots=True)
class AmplitudeEstimationProblem:
    """A state preparation and Grover iterate for amplitude estimation.

    Source circuits are defensively copied and normalized by qubit insertion
    position onto contiguous qubits ``0..n-1``. ``objective_qubits`` contains
    positions in that normalized system register. A Sampler reports a measured
    objective bitstring with its rightmost bit corresponding to the first
    objective qubit; ``is_good_state`` must classify that string.

    The caller is responsible for ensuring that ``grover_operator`` has the
    phase-rotation semantics associated with ``state_preparation`` and the
    good-state predicate. This object validates circuit mechanics, not that
    semantic relationship.
    """

    state_preparation: Circuit
    grover_operator: Circuit
    objective_qubits: tuple[int, ...]
    is_good_state: Callable[[str], bool]

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        state_preparation = _snapshot_circuit(
            self.state_preparation, "state_preparation"
        )
        grover_operator = _snapshot_circuit(self.grover_operator, "grover_operator")
        if state_preparation.num_qubits != grover_operator.num_qubits:
            raise ValueError(
                "state_preparation and grover_operator must have the same qubit count"
            )
        if not isinstance(self.objective_qubits, tuple):
            raise TypeError("objective_qubits must be a tuple")
        for qubit in self.objective_qubits:
            if isinstance(qubit, bool) or not isinstance(qubit, int):
                raise TypeError("objective_qubits must contain only integers")
            if qubit < 0 or qubit >= state_preparation.num_qubits:
                raise ValueError(
                    "objective_qubits entries must be valid system positions"
                )
        if not self.objective_qubits:
            raise ValueError("objective_qubits must not be empty")
        if len(set(self.objective_qubits)) != len(self.objective_qubits):
            raise ValueError("objective_qubits must not contain duplicates")
        if not callable(self.is_good_state):
            raise TypeError("is_good_state must be callable")

        object.__setattr__(self, "state_preparation", state_preparation)
        object.__setattr__(self, "grover_operator", grover_operator)

    @property
    def num_qubits(self) -> int:
        """Return the system-register width."""
        return int(self.state_preparation.num_qubits)

    def validate(self) -> None:
        """Revalidate the shallowly frozen problem before execution."""
        _validate_static_unitary_circuit(self.state_preparation, "state_preparation")
        _validate_static_unitary_circuit(self.grover_operator, "grover_operator")
        if self.state_preparation.num_qubits != self.grover_operator.num_qubits:
            raise ValueError(
                "state_preparation and grover_operator must have the same qubit count"
            )
        if any(
            qubit < 0 or qubit >= self.state_preparation.num_qubits
            for qubit in self.objective_qubits
        ):
            raise ValueError("objective_qubits entries must be valid system positions")


def _validate_problem(problem: object) -> AmplitudeEstimationProblem:
    if not isinstance(problem, AmplitudeEstimationProblem):
        raise TypeError("problem must be an AmplitudeEstimationProblem")
    problem.validate()
    return problem


__all__ = ["AmplitudeEstimationProblem"]
