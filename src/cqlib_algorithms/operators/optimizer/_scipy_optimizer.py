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

"""Shared adaptation of SciPy minimizers to estimator objectives."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from numbers import Real
from typing import Any

import numpy as np

from ...protocols.estimator import Estimator, EstimatorRequest, EstimatorResult
from ...protocols.optimizer import (
    OptimizerRequest,
    OptimizerResult,
    OptimizerStatus,
)

_Solver = Callable[
    [Estimator, OptimizerRequest, np.ndarray],
    tuple[np.ndarray, int, OptimizerStatus],
]


def _run_scipy_optimizer(
    estimator: Estimator,
    requests: Sequence[OptimizerRequest],
    *,
    maxiter: int,
    zero_budget_status: OptimizerStatus,
    solve: _Solver,
) -> list[OptimizerResult]:
    """Apply shared request and result handling around one SciPy solver."""
    if not isinstance(estimator, Estimator):
        raise TypeError("estimator must be an Estimator")
    request_list = list(requests)
    if any(not isinstance(request, OptimizerRequest) for request in request_list):
        raise TypeError("requests must contain only OptimizerRequest values")
    for request in request_list:
        request.validate()
    if not request_list:
        return []

    bindings = [
        dict(request.estimator_request.parameter_bindings) for request in request_list
    ]
    iterations: list[int] = []
    statuses: list[OptimizerStatus] = []

    for position, request in enumerate(request_list):
        names = request.parameter_names
        if not names:
            iterations.append(0)
            statuses.append(OptimizerStatus.NO_PARAMETERS)
            continue
        if maxiter == 0:
            iterations.append(0)
            statuses.append(zero_budget_status)
            continue

        initial_point = np.asarray(
            [bindings[position][name] for name in names], dtype=np.float64
        )
        point, iteration_count, status = solve(estimator, request, initial_point)
        point = _validate_point(point, len(names))
        for name, value in zip(names, point, strict=True):
            bindings[position][name] = float(value)
        iterations.append(iteration_count)
        statuses.append(status)

    final_requests = [
        _estimator_request(request, bindings[position])
        for position, request in enumerate(request_list)
    ]
    final_results = list(estimator.run(final_requests))
    if len(final_results) != len(request_list):
        raise ValueError(
            "estimator must return exactly one result per final request, "
            f"expected {len(request_list)}, got {len(final_results)}"
        )

    results: list[OptimizerResult] = []
    for position, (request, final_result) in enumerate(
        zip(request_list, final_results, strict=True)
    ):
        if not isinstance(final_result, EstimatorResult):
            raise TypeError("estimator must return EstimatorResult values")
        parameters = tuple(bindings[position][name] for name in request.parameter_names)
        results.append(
            OptimizerResult(
                request.parameter_names,
                parameters,
                final_result.expectation,
                iterations[position],
                statuses[position],
            )
        )
    return results


def _estimator_request(
    request: OptimizerRequest,
    bindings: dict[str, float],
) -> EstimatorRequest:
    base = request.estimator_request
    return EstimatorRequest(
        base.circuit,
        base.observable,
        bindings,
        base.shots,
        base.seed,
    )


def _evaluate(
    estimator: Estimator,
    request: OptimizerRequest,
    point: np.ndarray,
) -> float:
    """Evaluate one finite point through the public estimator contract."""
    point = _validate_point(point, len(request.parameter_names))
    bindings = dict(request.estimator_request.parameter_bindings)
    for name, value in zip(request.parameter_names, point, strict=True):
        bindings[name] = float(value)
    results = list(estimator.run([_estimator_request(request, bindings)]))
    if len(results) != 1:
        raise ValueError(
            "estimator must return exactly one result per objective evaluation, "
            f"got {len(results)}"
        )
    result = results[0]
    if not isinstance(result, EstimatorResult):
        raise TypeError("estimator must return EstimatorResult values")
    return result.expectation


def _validate_point(point: np.ndarray, expected_size: int) -> np.ndarray:
    numeric = np.asarray(point, dtype=np.float64)
    if numeric.shape != (expected_size,):
        raise ValueError(
            "SciPy optimizer returned a point with an invalid shape, "
            f"expected {(expected_size,)}, got {numeric.shape}"
        )
    if not np.all(np.isfinite(numeric)):
        raise ValueError("SciPy optimizer returned a non-finite point")
    return numeric


def _result_integer(result: Any, name: str, fallback: int = 0) -> int:
    value = getattr(result, name, fallback)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return fallback
    return max(0, int(value))


def _result_message(result: Any) -> str:
    return str(getattr(result, "message", "unknown SciPy failure"))


def _nonnegative_real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return numeric


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
