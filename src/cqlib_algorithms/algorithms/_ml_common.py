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

"""Scalar optimization and numeric objectives shared by learning algorithms."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Mapping
from numbers import Real
from typing import Protocol, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._utils.numeric import copy_finite_real_array, validate_positive_integer
from ..protocols.scalar_minimizer import ScalarMinimizer, ScalarMinimizerResult
from ._ml_validation import _real_hyperparameter


def configuration(
    max_evaluations: int, dimension: int, tolerance: float, initial_step: float
) -> tuple[float, float]:
    validate_positive_integer(max_evaluations, "max_evaluations")
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


def minimize(
    objective: Callable[[NDArray[np.float64]], float],
    point: NDArray[np.float64],
    budget: int,
    tolerance: float,
    initial_step: float,
    optimizer: ScalarMinimizer | None = None,
) -> ScalarMinimizerResult:
    calls = 0

    def counted(candidate: NDArray[np.float64]) -> float:
        nonlocal calls
        if calls >= budget:
            raise ValueError("optimizer exceeded max_evaluations")
        numeric = copy_finite_real_array(candidate, "optimizer candidate", 1)
        if numeric.shape != point.shape:
            raise ValueError("optimizer candidate has the wrong dimension")
        calls += 1
        value = objective(numeric)
        if not math.isfinite(value):
            raise ValueError("objective must be finite")
        return value

    if optimizer is not None:
        custom_result = optimizer(counted, point.copy())
        if not isinstance(custom_result, ScalarMinimizerResult):
            raise TypeError("optimizer must return ScalarMinimizerResult")
        if len(custom_result.point) != len(point):
            raise ValueError("optimizer result has the wrong dimension")
        if custom_result.evaluations != calls:
            raise ValueError(
                "optimizer evaluations must match actual objective evaluations"
            )
        return custom_result
    try:
        module = importlib.import_module("scipy.optimize")
    except ModuleNotFoundError as error:
        if error.name not in {"scipy", "scipy.optimize"}:
            raise
        raise ImportError(
            "fit requires SciPy; install cqlib-algorithms[machine-learning]"
        ) from error
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
    optimum = copy_finite_real_array(result.x, "optimizer result", 1)
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
    return ScalarMinimizerResult(
        tuple(float(x) for x in optimum),
        float(result.fun),
        bool(result.success),
        int(result.status),
        result.message,
        calls,
    )


def gram(values: ArrayLike, size: int) -> NDArray[np.float64]:
    matrix = copy_finite_real_array(values, "kernel matrix", 2)
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
