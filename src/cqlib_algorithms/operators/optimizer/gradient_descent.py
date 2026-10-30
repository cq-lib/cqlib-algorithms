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

"""First-order gradient descent for estimator expectation objectives."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import final

import numpy as np

from ...protocols.estimator import Estimator, EstimatorRequest
from ...protocols.gradient import Gradient, GradientRequest, GradientResult
from ...protocols.optimizer import (
    Optimizer,
    OptimizerRequest,
    OptimizerResult,
    OptimizerStatus,
)


def _positive_real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return numeric_value


def _tolerance(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("tolerance must be a real number or None")
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    return numeric_value


def _iteration_limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("maxiter must be an integer")
    if value < 0:
        raise ValueError("maxiter must be non-negative")
    return value


def _step_norm(step: np.ndarray) -> float:
    """Return a scaling-stable Euclidean norm for a non-empty step."""
    return math.hypot(*(float(value) for value in step))


@final
class GradientDescent(Optimizer):
    """Minimize estimator objectives with fixed-step gradient descent.

    Every iteration batches all still-active requests into one call to the
    injected :class:`Gradient`.  A request converges when the Euclidean norm
    of its applied update is less than or equal to ``tolerance``.  With
    ``tolerance=None``, only the iteration limit terminates non-empty requests.

    Args:
        gradient: Gradient component used for objective derivatives.
        learning_rate: Positive fixed update scale.
        maxiter: Nonnegative maximum update count.
        tolerance: Nonnegative step-norm threshold, or ``None``.

    Raises:
        TypeError: If an option has the wrong type.
        ValueError: If a numeric option is outside its valid range.
    """

    __slots__ = ("_gradient", "_learning_rate", "_maxiter", "_tolerance")

    def __init__(
        self,
        gradient: Gradient,
        learning_rate: float = 0.1,
        *,
        maxiter: int = 100,
        tolerance: float | None = None,
    ) -> None:
        if not isinstance(gradient, Gradient):
            raise TypeError("gradient must be a Gradient")
        self._gradient = gradient
        self._learning_rate = _positive_real(learning_rate, "learning_rate")
        self._maxiter = _iteration_limit(maxiter)
        self._tolerance = _tolerance(tolerance)

    @property
    def gradient(self) -> Gradient:
        """The injected gradient implementation."""
        return self._gradient

    @property
    def learning_rate(self) -> float:
        """The fixed learning rate."""
        return self._learning_rate

    @property
    def maxiter(self) -> int:
        """The maximum number of update steps per request."""
        return self._maxiter

    @property
    def tolerance(self) -> float | None:
        """The optional Euclidean step-norm convergence tolerance."""
        return self._tolerance

    def run(
        self,
        estimator: Estimator,
        requests: Sequence[OptimizerRequest],
        /,
    ) -> list[OptimizerResult]:
        """Optimize a batch and return final points in request order."""
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
            dict(request.estimator_request.parameter_bindings)
            for request in request_list
        ]
        iterations = [0 for _ in request_list]
        statuses: list[OptimizerStatus | None] = [
            OptimizerStatus.NO_PARAMETERS if not request.parameter_names else None
            for request in request_list
        ]
        active = [
            position
            for position, request in enumerate(request_list)
            if request.parameter_names and self._maxiter > 0
        ]

        for _ in range(self._maxiter):
            if not active:
                break
            gradient_requests = []
            for position in active:
                request = request_list[position]
                base = request.estimator_request
                objective = EstimatorRequest(
                    base.circuit,
                    base.observable,
                    bindings[position],
                    base.shots,
                )
                gradient_requests.append(
                    GradientRequest(objective, request.parameter_names)
                )
            gradient_results = list(self._gradient.run(estimator, gradient_requests))
            if len(gradient_results) != len(active):
                raise ValueError(
                    "gradient must return exactly one result per active request, "
                    f"expected {len(active)}, got {len(gradient_results)}"
                )

            next_active: list[int] = []
            for position, gradient_result in zip(active, gradient_results, strict=True):
                if not isinstance(gradient_result, GradientResult):
                    raise TypeError("gradient must return GradientResult values")
                names = request_list[position].parameter_names
                if gradient_result.parameter_names != names:
                    raise ValueError(
                        "gradient result parameter names must match the request"
                    )
                step = self._learning_rate * np.asarray(
                    gradient_result.values, dtype=np.float64
                )
                if not np.all(np.isfinite(step)):
                    raise ValueError("gradient descent produced a non-finite step")
                for name, value in zip(names, step, strict=True):
                    bindings[position][name] -= float(value)
                iterations[position] += 1
                if self._tolerance is not None and _step_norm(step) <= self._tolerance:
                    statuses[position] = OptimizerStatus.STEP_TOLERANCE
                else:
                    next_active.append(position)
            active = next_active

        for position, status in enumerate(statuses):
            if status is None:
                statuses[position] = OptimizerStatus.MAX_ITERATIONS

        final_requests = []
        for position, request in enumerate(request_list):
            base = request.estimator_request
            final_requests.append(
                EstimatorRequest(
                    base.circuit,
                    base.observable,
                    bindings[position],
                    base.shots,
                )
            )
        final_results = list(estimator.run(final_requests))
        if len(final_results) != len(request_list):
            raise ValueError(
                "estimator must return exactly one result per final request, "
                f"expected {len(request_list)}, got {len(final_results)}"
            )

        response: list[OptimizerResult] = []
        for position, (request, final_result) in enumerate(
            zip(request_list, final_results, strict=True)
        ):
            status = statuses[position]
            if status is None:
                raise RuntimeError("optimizer terminated without a status")
            parameters = tuple(
                bindings[position][name] for name in request.parameter_names
            )
            response.append(
                OptimizerResult(
                    request.parameter_names,
                    parameters,
                    final_result.expectation,
                    iterations[position],
                    status,
                )
            )
        return response
