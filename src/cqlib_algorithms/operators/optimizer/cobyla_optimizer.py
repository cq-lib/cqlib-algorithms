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

"""COBYLA minimization of estimator expectation objectives."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]

from ...protocols.estimator import Estimator
from ...protocols.optimizer import (
    Optimizer,
    OptimizerRequest,
    OptimizerResult,
    OptimizerStatus,
)
from ._scipy_optimizer import (
    _evaluate,
    _result_integer,
    _result_message,
    _run_scipy_optimizer,
)
from .gradient_descent import _iteration_limit, _positive_real, _tolerance


class COBYLAOptimizer(Optimizer):
    """Minimize estimator objectives with SciPy's derivative-free COBYLA.

    COBYLA constructs linear models inside a changing trust region and does
    not require a :class:`~cqlib_algorithms.protocols.Gradient`.  The current
    optimizer request contract contains no general constraint objects, so this
    component performs unconstrained parameter optimization.  ``maxiter`` is
    passed to SciPy as the maximum objective-evaluation budget.

    Args:
        maxiter: Nonnegative maximum number of objective evaluations.
        rhobeg: Positive initial trust-region radius.
        tolerance: Optional nonnegative final trust-region tolerance.

    Raises:
        TypeError: If an option has the wrong type.
        ValueError: If a numeric option is outside its valid range.
    """

    __slots__ = ("_maxiter", "_rhobeg", "_tolerance")

    def __init__(
        self,
        *,
        maxiter: int = 1000,
        rhobeg: float = 1.0,
        tolerance: float | None = None,
    ) -> None:
        self._maxiter = _iteration_limit(maxiter)
        self._rhobeg = _positive_real(rhobeg, "rhobeg")
        self._tolerance = _tolerance(tolerance)

    @property
    def maxiter(self) -> int:
        """The maximum number of objective evaluations."""
        return self._maxiter

    @property
    def rhobeg(self) -> float:
        """The initial trust-region radius."""
        return self._rhobeg

    @property
    def tolerance(self) -> float | None:
        """The optional final trust-region tolerance."""
        return self._tolerance

    def run(
        self,
        estimator: Estimator,
        requests: Sequence[OptimizerRequest],
        /,
    ) -> list[OptimizerResult]:
        """Optimize requests sequentially and preserve their input order."""
        return _run_scipy_optimizer(
            estimator,
            requests,
            maxiter=self._maxiter,
            zero_budget_status=OptimizerStatus.MAX_EVALUATIONS,
            solve=self._minimize,
        )

    def _minimize(
        self,
        estimator: Estimator,
        request: OptimizerRequest,
        initial_point: np.ndarray,
    ) -> tuple[np.ndarray, int, OptimizerStatus]:
        minimum_evaluations = initial_point.size + 2
        if self._maxiter < minimum_evaluations:
            raise ValueError(
                "COBYLA maxiter must be at least the number of trainable "
                f"parameters plus two ({minimum_evaluations})"
            )
        iterations = 0

        def objective(point: np.ndarray) -> float:
            return _evaluate(estimator, request, point)

        def callback(point: np.ndarray) -> None:
            nonlocal iterations
            iterations += 1

        options: dict[str, float | int] = {
            "maxiter": self._maxiter,
            "rhobeg": self._rhobeg,
        }
        if self._tolerance is not None:
            options["tol"] = self._tolerance
        result: Any = minimize(
            objective,
            initial_point,
            method="COBYLA",
            callback=callback,
            options=options,
        )
        if bool(result.success):
            return np.asarray(result.x), iterations, OptimizerStatus.CONVERGED
        evaluations = _result_integer(result, "nfev")
        if evaluations >= self._maxiter:
            return (
                np.asarray(result.x),
                iterations,
                OptimizerStatus.MAX_EVALUATIONS,
            )
        raise RuntimeError(f"COBYLA failed: {_result_message(result)}")
