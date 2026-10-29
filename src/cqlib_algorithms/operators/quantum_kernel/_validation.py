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

# Modified to integrate the shared fidelity kernel into the operator layer.

"""Private numeric and fully bound circuit validation for quantum kernels."""

from __future__ import annotations

import numpy as np
from cqlib import Circuit
from cqlib.circuit import CircuitError
from numpy.typing import ArrayLike, NDArray


def _real_array(values: ArrayLike, name: str, ndim: int) -> NDArray[np.float64]:
    """Copy finite arrays of real numbers."""
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular real numeric array") from error
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values, excluding bool")
    if raw.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(raw, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


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
