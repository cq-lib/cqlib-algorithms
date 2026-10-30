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

"""Bound-constrained L-BFGS-B minimization of estimator objectives."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]

from ...protocols.estimator import Estimator
from ...protocols.gradient import Gradient, GradientRequest, GradientResult
from ...protocols.optimizer import (
    Optimizer,
    OptimizerRequest,
    OptimizerResult,
    OptimizerStatus,
)
from ._scipy_optimizer import (
    _estimator_request,
    _evaluate,
    _nonnegative_real,
    _positive_integer,
    _result_integer,
    _result_message,
    _run_scipy_optimizer,
)
from .gradient_descent import _iteration_limit, _positive_real

ParameterBound = tuple[float | None, float | None]


class LBFGSBOptimizer(Optimizer):
    """Minimize estimator objectives with limited-memory BFGS and bounds.

    Bounds are keyed by trainable parameter name; omitted names are unbounded.
    A supplied :class:`~cqlib_algorithms.protocols.Gradient` provides analytic
    or quantum-derived derivatives.  Without one, SciPy estimates derivatives
    by forward finite differences using ``finite_difference_step``.  Requests
    are solved sequentially because each trajectory chooses its next point
    from the preceding objective and gradient results.

    Args:
        gradient: Optional gradient component, or ``None`` for finite differences.
        bounds: Optional immutable snapshot of per-parameter lower and upper bounds.
        maxiter: Nonnegative maximum number of solver iterations.
        maxfun: Positive approximate maximum number of objective evaluations.
        maxcor: Positive number of stored correction vectors.
        ftol: Nonnegative relative objective-decrease tolerance.
        gtol: Nonnegative projected-gradient tolerance.
        finite_difference_step: Positive absolute finite-difference step.
        maxls: Positive maximum line-search steps per iteration.

    Raises:
        TypeError: If an option or bound has the wrong type.
        ValueError: If a numeric option or bound is invalid.
    """

    __slots__ = (
        "_bounds",
        "_finite_difference_step",
        "_ftol",
        "_gradient",
        "_gtol",
        "_maxcor",
        "_maxfun",
        "_maxiter",
        "_maxls",
    )

    def __init__(
        self,
        gradient: Gradient | None = None,
        *,
        bounds: Mapping[str, ParameterBound] | None = None,
        maxiter: int = 1000,
        maxfun: int = 15000,
        maxcor: int = 10,
        ftol: float = 2.220446049250313e-9,
        gtol: float = 1e-5,
        finite_difference_step: float = 1e-8,
        maxls: int = 20,
    ) -> None:
        if gradient is not None and not isinstance(gradient, Gradient):
            raise TypeError("gradient must be a Gradient or None")
        self._maxiter = _iteration_limit(maxiter)
        self._gradient = gradient
        self._bounds = _parameter_bounds(bounds)
        self._maxfun = _positive_integer(maxfun, "maxfun")
        self._maxcor = _positive_integer(maxcor, "maxcor")
        self._ftol = _nonnegative_real(ftol, "ftol")
        self._gtol = _nonnegative_real(gtol, "gtol")
        self._finite_difference_step = _positive_real(
            finite_difference_step, "finite_difference_step"
        )
        self._maxls = _positive_integer(maxls, "maxls")

    @property
    def maxiter(self) -> int:
        """The maximum number of solver iterations."""
        return self._maxiter

    @property
    def gradient(self) -> Gradient | None:
        """The injected gradient or ``None`` for numerical differentiation."""
        return self._gradient

    @property
    def bounds(self) -> Mapping[str, ParameterBound]:
        """The immutable per-parameter bound mapping."""
        return self._bounds

    @property
    def maxfun(self) -> int:
        """The approximate maximum number of objective evaluations."""
        return self._maxfun

    @property
    def maxcor(self) -> int:
        """The number of correction vectors retained by L-BFGS-B."""
        return self._maxcor

    @property
    def ftol(self) -> float:
        """The relative objective-decrease tolerance."""
        return self._ftol

    @property
    def gtol(self) -> float:
        """The projected-gradient tolerance."""
        return self._gtol

    @property
    def finite_difference_step(self) -> float:
        """The finite-difference step used when no gradient is supplied."""
        return self._finite_difference_step

    @property
    def maxls(self) -> int:
        """The maximum line-search steps per iteration."""
        return self._maxls

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
            zero_budget_status=OptimizerStatus.MAX_ITERATIONS,
            solve=self._minimize,
        )

    def _minimize(
        self,
        estimator: Estimator,
        request: OptimizerRequest,
        initial_point: np.ndarray,
    ) -> tuple[np.ndarray, int, OptimizerStatus]:
        def objective(point: np.ndarray) -> float:
            return _evaluate(estimator, request, point)

        jacobian = None
        if self._gradient is not None:

            def jacobian(point: np.ndarray) -> np.ndarray:
                return self._gradient_at(estimator, request, point)

        request_bounds = [
            self._bounds.get(name, (None, None)) for name in request.parameter_names
        ]
        options = {
            "maxiter": self._maxiter,
            "maxfun": self._maxfun,
            "maxcor": self._maxcor,
            "ftol": self._ftol,
            "gtol": self._gtol,
            "eps": self._finite_difference_step,
            "maxls": self._maxls,
        }
        result: Any = minimize(
            objective,
            initial_point,
            method="L-BFGS-B",
            jac=jacobian,
            bounds=request_bounds,
            options=options,
        )
        iterations = _result_integer(result, "nit")
        if bool(result.success):
            return np.asarray(result.x), iterations, OptimizerStatus.CONVERGED
        evaluations = _result_integer(result, "nfev")
        if evaluations >= self._maxfun:
            return (
                np.asarray(result.x),
                iterations,
                OptimizerStatus.MAX_EVALUATIONS,
            )
        if iterations >= self._maxiter:
            return np.asarray(result.x), iterations, OptimizerStatus.MAX_ITERATIONS
        raise RuntimeError(f"L-BFGS-B failed: {_result_message(result)}")

    def _gradient_at(
        self,
        estimator: Estimator,
        request: OptimizerRequest,
        point: np.ndarray,
    ) -> np.ndarray:
        if self._gradient is None:
            raise RuntimeError("L-BFGS-B gradient is unavailable")
        bindings = dict(request.estimator_request.parameter_bindings)
        for name, value in zip(request.parameter_names, point, strict=True):
            bindings[name] = float(value)
        objective = _estimator_request(request, bindings)
        results = list(
            self._gradient.run(
                estimator,
                [GradientRequest(objective, request.parameter_names)],
            )
        )
        if len(results) != 1:
            raise ValueError(
                "gradient must return exactly one result per derivative evaluation, "
                f"got {len(results)}"
            )
        result = results[0]
        if not isinstance(result, GradientResult):
            raise TypeError("gradient must return GradientResult values")
        if result.parameter_names != request.parameter_names:
            raise ValueError("gradient result parameter names must match the request")
        values = np.asarray(result.values, dtype=np.float64)
        if values.shape != point.shape or not np.all(np.isfinite(values)):
            raise ValueError("gradient returned invalid values")
        return values


def _parameter_bounds(
    bounds: Mapping[str, ParameterBound] | None,
) -> Mapping[str, ParameterBound]:
    if bounds is None:
        return MappingProxyType({})
    if not isinstance(bounds, Mapping):
        raise TypeError("bounds must be a mapping or None")
    normalized: dict[str, ParameterBound] = {}
    for name, pair in bounds.items():
        if not isinstance(name, str):
            raise TypeError("bound names must be strings")
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError("each bound must be a two-item tuple")
        lower = _bound_endpoint(pair[0], "lower")
        upper = _bound_endpoint(pair[1], "upper")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("a lower bound must not exceed its upper bound")
        normalized[name] = (lower, upper)
    return MappingProxyType(normalized)


def _bound_endpoint(value: float | None, side: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{side} bound must be a real number or None")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{side} bound must be finite or None")
    return numeric
