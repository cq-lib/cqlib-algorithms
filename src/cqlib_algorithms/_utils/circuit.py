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

"""Private validation and snapshot helpers for static unitary circuits."""

from __future__ import annotations

import copy

import numpy as np
from cqlib import Circuit
from cqlib.circuit import CircuitError

_UNITARY_ATOL = 1e-10


def _validate_decomposable_static_unitary(
    circuit: Circuit,
    *,
    classical_error: str,
    decomposition_error: str,
    operation_error: str,
) -> None:
    """Apply the shared structural checks with caller-specific errors."""
    if circuit.classical_vars or circuit.classical_values:
        raise ValueError(classical_error)

    try:
        expanded = circuit.decompose()
    except CircuitError as error:
        raise ValueError(decomposition_error) from error

    for operation in expanded.operations:
        if (
            operation.is_classical_data
            or operation.is_classical_control
            or operation.is_delay
            or (operation.is_directive and operation.name != "barrier")
        ):
            raise ValueError(operation_error)


def validate_static_unitary_structure(circuit: Circuit, *, label: str) -> None:
    """Validate a decomposable static unitary circuit structure."""
    _validate_decomposable_static_unitary(
        circuit,
        classical_error=f"{label} must be a static unitary circuit",
        decomposition_error=(f"{label} must be a decomposable static unitary circuit"),
        operation_error=(
            f"{label} must contain only static unitary operations "
            "without measurement, reset, delay, or classical control"
        ),
    )


def validate_unitary_state_preparation(circuit: Circuit) -> None:
    """Validate a decomposable unitary state-preparation circuit."""
    _validate_decomposable_static_unitary(
        circuit,
        classical_error="estimation requires a unitary circuit",
        decomposition_error="estimation requires a decomposable unitary circuit",
        operation_error=(
            "estimation requires a unitary circuit without measurement, "
            "reset, delay, or classical control"
        ),
    )


def validate_bound_static_unitary_circuit(
    circuit: Circuit,
    name: str,
    *,
    check_numeric_unitaries: bool = True,
) -> None:
    """Validate a fully bound static unitary circuit."""
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
        if not check_numeric_unitaries or operation.instruction_type != "unitary":
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


def snapshot_bound_static_unitary_circuit(
    circuit: Circuit,
    name: str,
    *,
    check_numeric_unitaries: bool = True,
) -> Circuit:
    """Return a defensive fully bound static unitary circuit snapshot."""
    if not isinstance(circuit, Circuit):
        raise TypeError(f"{name} must be a Circuit")
    source = copy.deepcopy(circuit)
    validate_bound_static_unitary_circuit(
        source,
        name,
        check_numeric_unitaries=check_numeric_unitaries,
    )
    normalized = Circuit(source.num_qubits)
    normalized.compose(source, normalized.qubits)
    validate_bound_static_unitary_circuit(
        normalized,
        name,
        check_numeric_unitaries=check_numeric_unitaries,
    )
    return normalized


__all__: list[str] = []
