# This code is part of Cqlib.
#
# (C) Copyright China Telecom Quantum Group 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# use this file except in compliance with the License, obtained from the
# LICENSE.txt file in the root directory of this source tree or
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Finite-difference gradients of estimator expectation values."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import final

from ...protocols.estimator import Estimator, EstimatorRequest
from ...protocols.gradient import (
    Gradient,
    GradientRequest,
    GradientResult,
)

_FORWARD = "forward"
_CENTRAL = "central"
_METHODS = (_FORWARD, _CENTRAL)


@final
class FiniteDifferenceGradient(Gradient):
    """Numerical gradients of expectation values by finite differences.

    Unlike the analytic components this class performs no circuit
    analysis: every circuit the injected estimator accepts can be
    differentiated, including gates whose generators have no exact
    shift rule.  The derivative of parameter ``t`` is approximated as

    ``central``: ``(E(t + eps) - E(t - eps)) / (2 eps)`` with truncation
    error ``O(eps^2)``; ``forward``: ``(E(t + eps) - E(t)) / eps`` with
    truncation error ``O(eps)``.

    All difference evaluations of all requests are flattened into a
    single :meth:`Estimator.run` batch.  Requests with empty
    ``parameter_names`` return an empty gradient without estimator
    evaluations.  Shot-based estimator requests are passed through
    unchanged; the statistical quality of a shot-based gradient is that
    of the underlying estimator evaluations.

    Raises:
        TypeError: If the estimator or a request has the wrong type.
        ValueError: If ``epsilon`` is not a finite positive number,
            ``method`` is unknown, a request is invalid, or the estimator
            returns a mismatched result count.
    """

    __slots__ = ("_epsilon", "_method")

    def __init__(self, epsilon: float = 1e-6, method: str = _CENTRAL) -> None:
        """Configure the difference step and scheme.

        Raises:
            TypeError: If ``epsilon`` or ``method`` has the wrong type.
            ValueError: If ``epsilon`` is not a finite positive number or
                ``method`` is neither ``"forward"`` nor ``"central"``.
        """
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
            raise TypeError("epsilon must be a real number")
        numeric_epsilon = float(epsilon)
        if not math.isfinite(numeric_epsilon) or numeric_epsilon <= 0.0:
            raise ValueError("epsilon must be a finite positive number")
        if not isinstance(method, str):
            raise TypeError("method must be a string")
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS!r}, got {method!r}")
        self._epsilon = numeric_epsilon
        self._method = method

    @property
    def epsilon(self) -> float:
        """The finite difference step."""
        return self._epsilon

    @property
    def method(self) -> str:
        """The difference scheme, ``"forward"`` or ``"central"``."""
        return self._method

    def run(
        self,
        estimator: Estimator,
        requests: Sequence[GradientRequest],
        /,
    ) -> list[GradientResult]:
        """Differentiate a batch of estimator expectation objectives."""
        if not isinstance(estimator, Estimator):
            raise TypeError("estimator must be an Estimator")
        request_list = list(requests)
        if any(not isinstance(request, GradientRequest) for request in request_list):
            raise TypeError("requests must contain only GradientRequest values")
        for request in request_list:
            request.validate()

        central = self._method == _CENTRAL
        # (weight, offset) pairs: central uses t±eps, forward compares the
        # shifted point against the unshifted base point.
        points = (
            ((1.0, self._epsilon), (-1.0, -self._epsilon))
            if central
            else ((1.0, self._epsilon), (-1.0, 0.0))
        )

        evaluations: list[EstimatorRequest] = []
        # One entry per evaluation: (owner request index, name index, weight).
        owners: list[tuple[int, int, float]] = []
        for position, request in enumerate(request_list):
            base = request.estimator_request
            bindings = dict(base.parameter_bindings)
            for name_index, name in enumerate(request.parameter_names):
                for weight, offset in points:
                    shifted = dict(bindings)
                    if offset != 0.0:
                        shifted[name] = bindings[name] + offset
                    evaluations.append(
                        type(base)(
                            base.circuit,
                            base.observable,
                            shifted,
                            base.shots,
                        )
                    )
                    owners.append((position, name_index, weight))

        results = list(estimator.run(evaluations)) if evaluations else []
        if len(results) != len(evaluations):
            raise ValueError(
                "estimator must return exactly one result per evaluation, "
                f"expected {len(evaluations)}, got {len(results)}"
            )

        differences: dict[tuple[int, int], float] = {}
        for (position, name_index, weight), result in zip(owners, results):
            key = (position, name_index)
            differences[key] = differences.get(key, 0.0) + weight * result.expectation

        scale = 2.0 * self._epsilon if central else self._epsilon
        response: list[GradientResult] = []
        for position, request in enumerate(request_list):
            names = request.parameter_names
            if not names:
                response.append(GradientResult((), ()))
                continue
            values = tuple(
                differences[(position, name_index)] / scale
                for name_index in range(len(names))
            )
            response.append(GradientResult(names, values))
        return response
