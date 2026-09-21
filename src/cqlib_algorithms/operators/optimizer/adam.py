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

"""Adam optimization for estimator expectation objectives."""

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
from .gradient_descent import (
    _iteration_limit,
    _positive_real,
    _step_norm,
    _tolerance,
)


def _decay(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or not 0.0 <= numeric_value < 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1)")
    return numeric_value


@final
class Adam(Optimizer):
    """Minimize estimator objectives with bias-corrected Adam updates.

    First- and second-moment vectors are independent for every request and
    reset at the beginning of each :meth:`run`.  Active requests share one
    gradient batch per iteration but stop independently by step norm.
    """

    __slots__ = (
        "_beta1",
        "_beta2",
        "_epsilon",
        "_gradient",
        "_learning_rate",
        "_maxiter",
        "_tolerance",
    )

    def __init__(
        self,
        gradient: Gradient,
        learning_rate: float = 0.001,
        *,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        maxiter: int = 100,
        tolerance: float | None = None,
    ) -> None:
        if not isinstance(gradient, Gradient):
            raise TypeError("gradient must be a Gradient")
        self._gradient = gradient
        self._learning_rate = _positive_real(learning_rate, "learning_rate")
        self._beta1 = _decay(beta1, "beta1")
        self._beta2 = _decay(beta2, "beta2")
        self._epsilon = _positive_real(epsilon, "epsilon")
        self._maxiter = _iteration_limit(maxiter)
        self._tolerance = _tolerance(tolerance)

    @property
    def gradient(self) -> Gradient:
        """The injected gradient implementation."""
        return self._gradient

    @property
    def learning_rate(self) -> float:
        """The Adam learning rate."""
        return self._learning_rate

    @property
    def beta1(self) -> float:
        """The first-moment decay rate."""
        return self._beta1

    @property
    def beta2(self) -> float:
        """The second-moment decay rate."""
        return self._beta2

    @property
    def epsilon(self) -> float:
        """The positive denominator stabilizer."""
        return self._epsilon

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
        moments = [np.zeros(len(request.parameter_names)) for request in request_list]
        variances = [np.zeros(len(request.parameter_names)) for request in request_list]
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
                gradient_values = np.asarray(gradient_result.values, dtype=np.float64)
                if not np.all(np.isfinite(gradient_values)):
                    raise ValueError("Adam received a non-finite gradient")
                iteration = iterations[position] + 1
                with np.errstate(
                    divide="ignore", invalid="ignore", over="ignore", under="ignore"
                ):
                    next_moment = (
                        self._beta1 * moments[position]
                        + (1.0 - self._beta1) * gradient_values
                    )
                    next_variance = self._beta2 * variances[position] + (
                        1.0 - self._beta2
                    ) * np.square(gradient_values)
                    moment_correction = 1.0 - self._beta1**iteration
                    variance_correction = 1.0 - self._beta2**iteration
                    moment_hat = next_moment / moment_correction
                    variance_hat = next_variance / variance_correction
                    denominator = np.sqrt(variance_hat) + self._epsilon
                    step = self._learning_rate * moment_hat / denominator
                if not np.all(np.isfinite(next_moment)):
                    raise ValueError("Adam first moment became non-finite")
                if not np.all(np.isfinite(next_variance)):
                    raise ValueError("Adam second moment became non-finite")
                if not math.isfinite(moment_correction) or moment_correction <= 0.0:
                    raise ValueError("Adam first-moment correction became invalid")
                if not math.isfinite(variance_correction) or variance_correction <= 0.0:
                    raise ValueError("Adam second-moment correction became invalid")
                if not np.all(np.isfinite(moment_hat)):
                    raise ValueError("Adam corrected first moment became non-finite")
                if not np.all(np.isfinite(variance_hat)) or np.any(variance_hat < 0.0):
                    raise ValueError("Adam corrected second moment became invalid")
                if not np.all(np.isfinite(denominator)) or np.any(denominator <= 0.0):
                    raise ValueError("Adam denominator became invalid")
                if not np.all(np.isfinite(step)):
                    raise ValueError("Adam produced a non-finite step")
                moments[position] = next_moment
                variances[position] = next_variance
                for name, value in zip(names, step, strict=True):
                    bindings[position][name] -= float(value)
                iterations[position] = iteration
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
