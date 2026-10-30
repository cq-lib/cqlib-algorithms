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

"""Numeric data and circuit validation for machine-learning algorithms."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from types import MappingProxyType

import numpy as np
from cqlib import Circuit, Parameter
from cqlib.circuit import CircuitError
from numpy.typing import ArrayLike, NDArray

from ..operators.state_preparation import UserCircuitPreparation

_UNITARY_ATOL = 1e-10


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


def _real_hyperparameter(value: float, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (result < 0.0 if allow_zero else result <= 0.0):
        raise ValueError(
            f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}"
        )
    return result


def _targets(values: ArrayLike, size: int, name: str) -> NDArray[np.float64]:
    result = _real_array(values, name, 1)
    if len(result) != size:
        raise ValueError(f"{name} length must match the number of samples")
    return result


def names(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence of names")
    result = tuple(values)
    if any(not isinstance(value, str) for value in result):
        raise TypeError(f"{label} must contain strings")
    if (
        not result
        or any(not value for value in result)
        or len(set(result)) != len(result)
    ):
        raise ValueError(f"{label} must contain nonempty unique names")
    return result


def bindings(values: Mapping[str, float] | None, label: str) -> dict[str, float]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} must be a mapping")
    result: dict[str, float] = {}
    for name, value in values.items():
        if not isinstance(name, str):
            raise TypeError(f"{label} names must be strings")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{label} values must be real numbers")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{label} values must be finite")
        result[name] = numeric
    return result


def data(
    values: ArrayLike, width: int, *, nonempty: bool = False
) -> NDArray[np.float64]:
    result = _real_array(values, "X", 2)
    if result.shape[1] != width:
        raise ValueError("X feature width must match input_parameters")
    if nonempty and not len(result):
        raise ValueError("fit requires at least one sample")
    return result


def binary_labels(
    values: ArrayLike, size: int
) -> tuple[NDArray[np.float64], tuple[float, float]]:
    labels = _targets(values, size, "y")
    unique = np.unique(labels)
    if len(unique) != 2:
        raise ValueError("y must contain exactly two distinct classes")
    classes = (float(unique[0]), float(unique[1]))
    return np.where(labels == classes[0], -1.0, 1.0), classes


def initial(
    values: Mapping[str, float], parameters: tuple[str, ...]
) -> NDArray[np.float64]:
    result = bindings(values, "initial_point")
    if set(result) != set(parameters):
        raise ValueError("initial_point must cover exactly the trainable parameters")
    return np.array([result[name] for name in parameters], dtype=np.float64)


def immutable_parameters(
    parameters: tuple[str, ...], point: Sequence[float] | NDArray[np.float64]
) -> Mapping[str, float]:
    return MappingProxyType(dict(zip(parameters, (float(value) for value in point))))


def _validate_static_unitary(circuit: Circuit, *, label: str) -> None:
    """Validate static unitary circuit structure."""
    if circuit.classical_vars or circuit.classical_values:
        raise ValueError(f"{label} must be a static unitary circuit")

    try:
        expanded = circuit.decompose()
    except CircuitError as exc:
        raise ValueError(
            f"{label} must be a decomposable static unitary circuit"
        ) from exc

    for operation in expanded.operations:
        if (
            operation.is_classical_data
            or operation.is_classical_control
            or operation.is_delay
            or (operation.is_directive and operation.name != "barrier")
        ):
            raise ValueError(
                f"{label} must contain only static unitary operations "
                "without measurement, reset, delay, or classical control"
            )


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


def template(
    circuit: Circuit,
    inputs: tuple[str, ...],
    weights: tuple[str, ...],
    fixed: Mapping[str, float],
) -> Circuit:
    if set(inputs) & set(weights) or (set(inputs) | set(weights)) & set(fixed):
        raise ValueError("input, trainable and fixed parameter names must be disjoint")
    normalized = UserCircuitPreparation(circuit).build_circuit("")
    if set(inputs) | set(weights) | set(fixed) != set(normalized.used_symbols):
        raise ValueError(
            "parameter names must cover exactly the active circuit symbols"
        )
    partial = normalized.assign_parameters(dict(fixed))
    _validate_static_unitary(partial, label="circuit")
    partial.inverse()
    constants = Circuit(partial.num_qubits)
    if not partial.global_phase.symbols:
        constants.set_global_phase(partial.global_phase)
    for operation in partial.decompose().operations:
        if operation.is_directive and operation.name == "barrier":
            constants.append(operation)
            continue
        if operation.instruction_type not in {"standard", "mcgate", "unitary"}:
            raise ValueError(
                "circuit must contain only static unitary operations and barriers"
            )
        if not any(
            isinstance(value, Parameter) and value.symbols for value in operation.params
        ):
            constants.append(operation)
    _validate_static_unitary_circuit(constants, "circuit constant operations")
    return normalized
