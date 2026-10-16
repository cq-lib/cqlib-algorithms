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

"""Simultaneous-perturbation stochastic approximation gradients.

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
from typing import final

import numpy as np

from ...protocols.estimator import Estimator, EstimatorRequest
from ...protocols.gradient import (
    Gradient,
    GradientRequest,
    GradientResult,
)


@final
class SPSAGradient(Gradient):
    """Stochastic gradient estimates via simultaneous perturbations.

    Each :meth:`run` call draws, for every request, one Rademacher
    perturbation vector ``delta`` over the request's parameters and
    evaluates the objective at ``theta + c * delta`` and
    ``theta - c * delta``, giving the estimate

    ``g_hat[i] = (E(+i) - E(-i)) / (2 c) * delta[i]``

    with two estimator evaluations per request regardless of the
    parameter count (Spall, IEEE Trans. Autom. Control 37, 332 (1992)).
    The estimate is unbiased for smooth objectives up to ``O(c)`` bias.

    No circuit analysis is performed, so every circuit the injected
    estimator accepts can be differentiated.  Randomness comes from a
    NumPy generator: passing an integer seed makes runs reproducible,
    passing a ``numpy.random.Generator`` shares its stream, and the
    default ``None`` seeds a fresh generator from OS entropy.  The
    generator state advances across :meth:`run` calls, so repeated calls
    on one instance use different perturbations while rebuilding with
    the same seed replays the same sequence.  Multi-perturbation
    averaging with a configurable batch size is a possible extension and
    is intentionally not implemented here.

    All evaluations of all requests are flattened into a single
    :meth:`Estimator.run` batch.  Requests with empty
    ``parameter_names`` draw no perturbations and return an empty
    gradient without estimator evaluations.  Shot-based estimator
    requests are passed through unchanged.

    Raises:
        TypeError: If the estimator, a request, or ``seed`` has the
            wrong type.
        ValueError: If ``perturbation`` is not a finite positive
            number, a request is invalid, or the estimator returns a
            mismatched result count.
    """

    __slots__ = ("_perturbation", "_rng")

    def __init__(
        self,
        perturbation: float,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        """Configure the perturbation scale and randomness.

        Raises:
            TypeError: If ``perturbation`` or ``seed`` has the wrong
                type.
            ValueError: If ``perturbation`` is not a finite positive
                number.
        """
        if isinstance(perturbation, bool) or not isinstance(perturbation, (int, float)):
            raise TypeError("perturbation must be a real number")
        numeric_perturbation = float(perturbation)
        if not math.isfinite(numeric_perturbation) or numeric_perturbation <= 0.0:
            raise ValueError("perturbation must be a finite positive number")
        if seed is not None and not isinstance(seed, (int, np.random.Generator)):
            raise TypeError("seed must be an int, a Generator or None")
        self._perturbation = numeric_perturbation
        if isinstance(seed, np.random.Generator):
            self._rng = seed
        else:
            self._rng = np.random.default_rng(seed)

    @property
    def perturbation(self) -> float:
        """The simultaneous perturbation scale ``c``."""
        return self._perturbation

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

        perturbations: list[np.ndarray] = []
        for request in request_list:
            count = len(request.parameter_names)
            perturbations.append(
                self._rng.integers(0, 2, size=count, dtype=np.int64) * 2 - 1
                if count
                else np.zeros(0, dtype=np.int64)
            )

        evaluations: list[EstimatorRequest] = []
        # (request index, sign): sign selects theta + c*delta or theta - c*delta.
        owners: list[tuple[int, int]] = []
        for position, request in enumerate(request_list):
            base = request.estimator_request
            bindings = dict(base.parameter_bindings)
            delta = perturbations[position]
            trainable = bool(request.parameter_names)
            for sign in (1.0, -1.0):
                if not trainable:
                    break
                shifted = dict(bindings)
                for name_index, name in enumerate(request.parameter_names):
                    shifted[name] = bindings[name] + sign * self._perturbation * float(
                        delta[name_index]
                    )
                evaluations.append(
                    type(base)(
                        base.circuit,
                        base.observable,
                        shifted,
                        base.shots,
                    )
                )
                owners.append((position, int(sign)))

        results = list(estimator.run(evaluations)) if evaluations else []
        if len(results) != len(evaluations):
            raise ValueError(
                "estimator must return exactly one result per evaluation, "
                f"expected {len(evaluations)}, got {len(results)}"
            )

        plus: dict[int, float] = {}
        minus: dict[int, float] = {}
        for (position, sign), result in zip(owners, results):
            if sign > 0:
                plus[position] = result.expectation
            else:
                minus[position] = result.expectation

        scale = 2.0 * self._perturbation
        response: list[GradientResult] = []
        for position, request in enumerate(request_list):
            names = request.parameter_names
            if not names:
                response.append(GradientResult((), ()))
                continue
            delta = perturbations[position]
            direction = (plus[position] - minus[position]) / scale
            values = tuple(
                direction * float(delta[name_index]) for name_index in range(len(names))
            )
            response.append(GradientResult(names, values))
        return response
