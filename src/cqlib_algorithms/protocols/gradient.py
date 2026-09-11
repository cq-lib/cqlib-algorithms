"""Structural gradient protocol and value types for estimator objectives."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Protocol

from .estimator import Estimator, EstimatorRequest


@dataclass(frozen=True, slots=True)
class GradientRequest:
    """A gradient request for one estimator expectation objective.

    The scalar objective is the ``expectation`` returned for
    ``estimator_request``. ``parameter_names`` is an explicit ordered,
    duplicate-free subset of that request's bindings. An empty tuple requests
    an empty gradient.

    The dataclass is only shallowly frozen because its estimator request holds
    mutable Core objects. Callers must not modify those objects during the
    request's lifetime. A gradient implementation must nevertheless call
    :meth:`validate` immediately before constructing perturbed requests.
    Revalidation detects invalid mutations but cannot detect every semantically
    different mutation that remains valid.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If parameter names are duplicated or unbound, or if the
            nested estimator request is no longer valid.
        cqlib.circuit.CircuitError: If nested ``Circuit.validate`` fails.
    """

    estimator_request: EstimatorRequest
    parameter_names: tuple[str, ...]

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Revalidate this request and its nested estimator request.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If names are duplicated or unbound, or if the nested
                estimator request violates an invariant.
            cqlib.circuit.CircuitError: If nested ``Circuit.validate`` fails.
        """
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
            raise ValueError(f"gradient parameters must be bound: {unknown!r}")


@dataclass(frozen=True, slots=True)
class GradientResult:
    """Finite gradient values paired with their parameter names.

    Raises:
        TypeError: If fields are not tuples or contain values of the wrong type.
        ValueError: If names are duplicated, lengths differ, or a value is not finite.
    """

    parameter_names: tuple[str, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.parameter_names, tuple):
            raise TypeError("parameter_names must be a tuple")
        if not isinstance(self.values, tuple):
            raise TypeError("values must be a tuple")
        if any(not isinstance(name, str) for name in self.parameter_names):
            raise TypeError("parameter names must be strings")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("parameter names must be unique")
        if len(self.parameter_names) != len(self.values):
            raise ValueError(
                "parameter names and gradient values must have equal length"
            )

        normalized: list[float] = []
        for value in self.values:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("gradient values must be real numbers")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError("gradient values must be finite")
            normalized.append(numeric_value)
        object.__setattr__(self, "values", tuple(normalized))


class Gradient(Protocol):
    """Structural protocol for synchronous batched estimator gradients.

    Implementations differentiate only ``EstimatorResult.expectation``. They
    call :meth:`GradientRequest.validate` before evaluation, should flatten all
    perturbed points into as few estimator batches as possible, and return one
    result per request in input order. The estimator is explicit so the
    protocol does not impose a constructor on implementations.
    """

    def run(
        self,
        estimator: Estimator,
        requests: Sequence[GradientRequest],
        /,
    ) -> list[GradientResult]:
        """Differentiate a batch of estimator expectation objectives."""
        ...


__all__ = ["GradientRequest", "GradientResult", "Gradient"]
