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

"""Simultaneous-perturbation stochastic approximation optimizer.

References
----------
[1] J. C. Spall, "Multivariate stochastic approximation using a simultaneous
    perturbation gradient approximation," IEEE Transactions on Automatic
    Control, 37(3), 332-341, 1992.
    DOI: 10.1109/9.119632.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import final

import numpy as np

from ...protocols.estimator import Estimator, EstimatorRequest
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


def _non_negative_real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return numeric_value


def _schedule_value(
    numerator: float,
    denominator: float,
    exponent: float,
    name: str,
) -> float:
    """Return one positive finite power-law schedule value."""
    try:
        log_value = math.log(numerator) - exponent * math.log(denominator)
        value = math.exp(log_value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} schedule must remain finite and positive") from error
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} schedule must remain finite and positive")
    return value


@final
class SPSAOptimizer(Optimizer):
    """Minimize estimator objectives with two evaluations per active request.

    At request-local iteration ``k`` (zero based), the learning and
    perturbation scales are ``a/(A+k+1)^alpha`` and ``c/(k+1)^gamma``.
    One Rademacher vector supplies the simultaneous two-sided gradient
    estimate.  The random stream advances across :meth:`run` calls.
    """

    __slots__ = (
        "_alpha",
        "_gamma",
        "_learning_rate",
        "_maxiter",
        "_perturbation",
        "_rng",
        "_stability_constant",
        "_tolerance",
    )

    def __init__(
        self,
        learning_rate: float,
        perturbation: float,
        *,
        alpha: float = 0.602,
        gamma: float = 0.101,
        stability_constant: float = 0.0,
        maxiter: int = 100,
        tolerance: float | None = None,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        self._learning_rate = _positive_real(learning_rate, "learning_rate")
        self._perturbation = _positive_real(perturbation, "perturbation")
        self._alpha = _positive_real(alpha, "alpha")
        self._gamma = _positive_real(gamma, "gamma")
        self._stability_constant = _non_negative_real(
            stability_constant, "stability_constant"
        )
        self._maxiter = _iteration_limit(maxiter)
        self._tolerance = _tolerance(tolerance)
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, (int, np.random.Generator))
        ):
            raise TypeError("seed must be an int, a Generator or None")
        self._rng = (
            seed
            if isinstance(seed, np.random.Generator)
            else np.random.default_rng(seed)
        )

    @property
    def learning_rate(self) -> float:
        """The numerator ``a`` of the learning-rate schedule."""
        return self._learning_rate

    @property
    def perturbation(self) -> float:
        """The numerator ``c`` of the perturbation schedule."""
        return self._perturbation

    @property
    def alpha(self) -> float:
        """The positive learning-rate decay exponent."""
        return self._alpha

    @property
    def gamma(self) -> float:
        """The positive perturbation decay exponent."""
        return self._gamma

    @property
    def stability_constant(self) -> float:
        """The non-negative learning-rate stability constant ``A``."""
        return self._stability_constant

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
            perturbations: dict[int, np.ndarray] = {}
            scales: dict[int, tuple[float, float]] = {}
            evaluations: list[EstimatorRequest] = []
            for position in active:
                request = request_list[position]
                base = request.estimator_request
                iteration = iterations[position]
                learning_rate = _schedule_value(
                    self._learning_rate,
                    self._stability_constant + iteration + 1,
                    self._alpha,
                    "learning_rate",
                )
                perturbation = _schedule_value(
                    self._perturbation,
                    float(iteration + 1),
                    self._gamma,
                    "perturbation",
                )
                delta = (
                    self._rng.integers(
                        0,
                        2,
                        size=len(request.parameter_names),
                        dtype=np.int64,
                    )
                    * 2
                    - 1
                )
                perturbations[position] = delta
                scales[position] = (learning_rate, perturbation)
                for sign in (1.0, -1.0):
                    shifted = dict(bindings[position])
                    for name_index, name in enumerate(request.parameter_names):
                        shifted[name] += sign * perturbation * float(delta[name_index])
                    evaluations.append(
                        EstimatorRequest(
                            base.circuit,
                            base.observable,
                            shifted,
                            base.shots,
                        )
                    )

            estimates = list(estimator.run(evaluations))
            if len(estimates) != len(evaluations):
                raise ValueError(
                    "estimator must return exactly one result per perturbation, "
                    f"expected {len(evaluations)}, got {len(estimates)}"
                )

            next_active: list[int] = []
            for active_index, position in enumerate(active):
                plus = estimates[2 * active_index].expectation
                minus = estimates[2 * active_index + 1].expectation
                learning_rate, perturbation = scales[position]
                direction = (plus - minus) / (2.0 * perturbation)
                gradient = direction * perturbations[position]
                step = learning_rate * gradient
                if not np.all(np.isfinite(step)):
                    raise ValueError("SPSA produced a non-finite step")
                names = request_list[position].parameter_names
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
