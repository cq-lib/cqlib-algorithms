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

# Modified to use the shared fidelity quantum-kernel operator.

"""Private validation and scalar optimization shared by learning algorithms."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Protocol, cast

import numpy as np
from cqlib import Circuit, Parameter
from numpy.typing import ArrayLike, NDArray

from ..operators.quantum_kernel._validation import _positive_integer, _real_array
from ..operators.state_fidelity.compute_uncompute import _validate_static_unitary
from ..operators.state_preparation import UserCircuitPreparation
from .amplitude_estimation.problem import _validate_static_unitary_circuit
from .qsvr import _real_hyperparameter, _targets


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
    parameters: tuple[str, ...], point: NDArray[np.float64]
) -> Mapping[str, float]:
    return MappingProxyType(dict(zip(parameters, (float(value) for value in point))))


def configuration(
    max_evaluations: int, dimension: int, tolerance: float, initial_step: float
) -> tuple[float, float]:
    _positive_integer(max_evaluations, "max_evaluations")
    if max_evaluations < dimension + 2:
        raise ValueError(
            "max_evaluations must be at least the trainable dimension plus two"
        )
    return (
        _real_hyperparameter(tolerance, "tolerance"),
        _real_hyperparameter(initial_step, "initial_step"),
    )


class _Minimize(Protocol):
    def __call__(
        self,
        fun: Callable[[NDArray[np.float64]], float],
        x0: NDArray[np.float64],
        *,
        method: str,
        options: Mapping[str, float | int],
    ) -> object: ...


class _Result(Protocol):
    x: ArrayLike
    fun: object
    success: object
    status: object
    message: object
    nfev: object


@dataclass(frozen=True, slots=True)
class Optimization:
    point: NDArray[np.float64]
    success: bool
    status: int
    message: str
    evaluations: int


def minimize(
    objective: Callable[[NDArray[np.float64]], float],
    point: NDArray[np.float64],
    budget: int,
    tolerance: float,
    initial_step: float,
) -> Optimization:
    try:
        module = importlib.import_module("scipy.optimize")
    except ModuleNotFoundError as error:
        if error.name not in {"scipy", "scipy.optimize"}:
            raise
        raise ImportError(
            "fit requires SciPy; install cqlib-algorithms[machine-learning]"
        ) from error
    calls = 0

    def counted(candidate: NDArray[np.float64]) -> float:
        nonlocal calls
        if calls >= budget:
            raise ValueError("optimizer exceeded max_evaluations")
        numeric = _real_array(candidate, "optimizer candidate", 1)
        if numeric.shape != point.shape:
            raise ValueError("optimizer candidate has the wrong dimension")
        calls += 1
        value = objective(numeric)
        if not math.isfinite(value):
            raise ValueError("objective must be finite")
        return value

    solver = cast(_Minimize, module.minimize)
    result = cast(
        _Result,
        solver(
            counted,
            point.copy(),
            method="COBYLA",
            options={"maxiter": budget, "tol": tolerance, "rhobeg": initial_step},
        ),
    )
    optimum = _real_array(result.x, "optimizer result", 1)
    if optimum.shape != point.shape:
        raise ValueError("optimizer result has the wrong dimension")
    if not isinstance(result.success, (bool, np.bool_)):
        raise TypeError("optimizer success must be bool")
    if isinstance(result.status, bool) or not isinstance(
        result.status, (int, np.integer)
    ):
        raise TypeError("optimizer status must be an integer")
    if not isinstance(result.message, str):
        raise TypeError("optimizer message must be a string")
    if isinstance(result.nfev, bool) or not isinstance(result.nfev, (int, np.integer)):
        raise TypeError("optimizer nfev must be an integer")
    if int(result.nfev) != calls:
        raise ValueError("optimizer nfev must match actual objective evaluations")
    if (
        isinstance(result.fun, bool)
        or not isinstance(result.fun, Real)
        or not math.isfinite(float(result.fun))
    ):
        raise ValueError("optimizer fun must be a finite real number")
    if calls == 0:
        raise ValueError("optimizer must evaluate the objective")
    return Optimization(
        optimum, bool(result.success), int(result.status), result.message, calls
    )


def gram(values: ArrayLike, size: int) -> NDArray[np.float64]:
    matrix = _real_array(values, "kernel matrix", 2)
    if matrix.shape != (size, size) or not np.array_equal(matrix, matrix.T):
        raise ValueError("kernel must return a symmetric square training matrix")
    if np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise ValueError("kernel fidelities must lie within [0, 1]")
    return matrix


def mse(prediction: NDArray[np.float64], targets: NDArray[np.float64]) -> float:
    with np.errstate(over="raise", invalid="raise"):
        try:
            residual = prediction - targets
            scale = float(np.max(np.abs(residual)))
            value = (
                0.0
                if scale == 0.0
                else float(np.mean((residual / scale) ** 2)) * scale * scale
            )
        except FloatingPointError as error:
            raise ValueError("squared error overflowed") from error
    if not math.isfinite(value):
        raise ValueError("squared error must be finite")
    return value
