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

"""Abstract optimizer interface and immutable request and result types."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Real

from .estimator import Estimator, EstimatorRequest


class OptimizerStatus(Enum):
    """Termination reason for one optimizer request."""

    CONVERGED = "converged"
    STEP_TOLERANCE = "step_tolerance"
    MAX_EVALUATIONS = "max_evaluations"
    MAX_ITERATIONS = "max_iterations"
    NO_PARAMETERS = "no_parameters"


@dataclass(frozen=True, slots=True)
class OptimizerRequest:
    """One estimator objective and its ordered trainable parameter subset.

    Parameters not listed in ``parameter_names`` remain fixed at their values
    in ``estimator_request.parameter_bindings``.  The requested names must be
    unique and already bound by the nested estimator request.  An empty tuple
    requests objective evaluation without optimization.

    The dataclass is shallowly frozen.  The nested Core circuit and observable
    remain mutable, so optimizers call :meth:`validate` before beginning a run.

    Args:
        estimator_request: Fully specified expectation-value objective.
        parameter_names: Ordered subset of bound parameters to optimize.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If names are duplicated or not bound by the objective.
    """

    estimator_request: EstimatorRequest
    parameter_names: tuple[str, ...]

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Revalidate this request and its nested estimator request."""
        if not isinstance(self.estimator_request, EstimatorRequest):
            raise TypeError("estimator_request must be an EstimatorRequest")
        if not isinstance(self.parameter_names, tuple):
            raise TypeError("parameter_names must be a tuple")
        if any(not isinstance(name, str) for name in self.parameter_names):
            raise TypeError("parameter names must be strings")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("parameter names must be unique")

        self.estimator_request.validate()
        bound_names = set(self.estimator_request.parameter_bindings)
        unknown = [name for name in self.parameter_names if name not in bound_names]
        if unknown:
            raise ValueError(f"optimizer parameters must be bound: {unknown!r}")


@dataclass(frozen=True, slots=True)
class OptimizerResult:
    """Final optimization point, objective value and termination metadata.

    ``optimal_parameters`` is the final iterate in ``parameter_names`` order,
    not the best point seen during the run.  ``optimal_value`` is obtained by
    an independent estimator evaluation at that final iterate.

    Attributes:
        parameter_names: Optimized names in result order.
        optimal_parameters: Final parameter values in the same order.
        optimal_value: Objective evaluated at the final point.
        iterations: Number of applied updates.
        status: Termination reason.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If names are duplicated, lengths differ, a numeric value
            is non-finite, or ``iterations`` is negative.
    """

    parameter_names: tuple[str, ...]
    optimal_parameters: tuple[float, ...]
    optimal_value: float
    iterations: int
    status: OptimizerStatus

    def __post_init__(self) -> None:
        if not isinstance(self.parameter_names, tuple):
            raise TypeError("parameter_names must be a tuple")
        if not isinstance(self.optimal_parameters, tuple):
            raise TypeError("optimal_parameters must be a tuple")
        if any(not isinstance(name, str) for name in self.parameter_names):
            raise TypeError("parameter names must be strings")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("parameter names must be unique")
        if len(self.parameter_names) != len(self.optimal_parameters):
            raise ValueError(
                "parameter names and optimal parameters must have equal length"
            )

        parameters: list[float] = []
        for value in self.optimal_parameters:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("optimal parameters must be real numbers")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError("optimal parameters must be finite")
            parameters.append(numeric_value)
        object.__setattr__(self, "optimal_parameters", tuple(parameters))

        if isinstance(self.optimal_value, bool) or not isinstance(
            self.optimal_value, Real
        ):
            raise TypeError("optimal_value must be a real number")
        optimal_value = float(self.optimal_value)
        if not math.isfinite(optimal_value):
            raise ValueError("optimal_value must be finite")
        object.__setattr__(self, "optimal_value", optimal_value)

        if isinstance(self.iterations, bool) or not isinstance(self.iterations, int):
            raise TypeError("iterations must be an integer")
        if self.iterations < 0:
            raise ValueError("iterations must be non-negative")
        if not isinstance(self.status, OptimizerStatus):
            raise TypeError("status must be an OptimizerStatus")

    @property
    def converged(self) -> bool:
        """Whether termination represents convergence rather than exhaustion."""
        return self.status in (
            OptimizerStatus.CONVERGED,
            OptimizerStatus.STEP_TOLERANCE,
            OptimizerStatus.NO_PARAMETERS,
        )


class Optimizer(ABC):
    """Define synchronous batched minimization of Estimator objectives."""

    __slots__ = ()

    @abstractmethod
    def run(
        self,
        estimator: Estimator,
        requests: Sequence[OptimizerRequest],
        /,
    ) -> list[OptimizerResult]:
        """Optimize a batch and return results in request order.

        Args:
            estimator: Expectation estimator for objective evaluations.
            requests: Optimization requests in output order.

        Returns:
            One optimization result per request.
        """
        raise NotImplementedError
