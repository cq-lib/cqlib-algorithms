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

"""Shared results for variational quantum algorithms."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Literal

from cqlib import Circuit

from ...protocols.optimizer import OptimizerResult


def _immutable_parameters(parameters: Mapping[str, float]) -> Mapping[str, float]:
    if not isinstance(parameters, Mapping):
        raise TypeError("optimal_parameters must be a mapping")

    normalized: dict[str, float] = {}
    for name, value in parameters.items():
        if not isinstance(name, str):
            raise TypeError("optimal parameter names must be strings")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("optimal parameter values must be real numbers")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError("optimal parameter values must be finite")
        normalized[name] = numeric_value
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class VQAResult:
    """An objective value, final point, bound circuit and optimizer metadata.

    ``optimal_parameters`` preserves the optimizer's parameter order and is
    defensively copied into an immutable mapping. Algorithms supply an
    independent, fully bound circuit; this dataclass does not copy a manually
    supplied circuit. Core circuit objects remain mutable despite the frozen
    result fields.

    The value refers to the original objective at the final optimizer iterate,
    not the best historical iterate. For "minimize" it equals the optimizer's
    value; for "maximize" it is the negative of that minimized value.
    No checkpoint, variance or global optimality guarantee is implied.

    Raises:
        TypeError: If values, metadata or the circuit have invalid types.
        ValueError: If values are non-finite, metadata is inconsistent, the
            direction is unsupported or the circuit is not fully bound.
    """

    optimal_value: float
    optimal_parameters: Mapping[str, float]
    optimal_circuit: Circuit
    optimizer_result: OptimizerResult = field(repr=False)
    objective_sense: Literal["minimize", "maximize"] = field(
        default="minimize", kw_only=True
    )

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if isinstance(self.optimal_value, bool) or not isinstance(
            self.optimal_value, Real
        ):
            raise TypeError("optimal_value must be a real number")
        value = float(self.optimal_value)
        if not math.isfinite(value):
            raise ValueError("optimal_value must be finite")
        object.__setattr__(self, "optimal_value", value)

        if not isinstance(self.optimizer_result, OptimizerResult):
            raise TypeError("optimizer_result must be an OptimizerResult")
        parameters = _immutable_parameters(self.optimal_parameters)
        object.__setattr__(self, "optimal_parameters", parameters)

        names = tuple(parameters)
        if names != self.optimizer_result.parameter_names:
            raise ValueError("optimal parameter names must match the optimizer result")
        if tuple(parameters.values()) != self.optimizer_result.optimal_parameters:
            raise ValueError("optimal parameter values must match the optimizer result")
        if self.objective_sense not in ("minimize", "maximize"):
            raise ValueError("objective_sense must be 'minimize' or 'maximize'")
        expected = self.optimizer_result.optimal_value
        if self.objective_sense == "maximize":
            expected = -expected
        if value != expected:
            raise ValueError("optimal_value must match the signed optimizer result")

        if not isinstance(self.optimal_circuit, Circuit):
            raise TypeError("optimal_circuit must be a cqlib.Circuit")
        self.optimal_circuit.validate()
        if self.optimal_circuit.used_symbols:
            raise ValueError("optimal_circuit must be fully bound")

    @property
    def eigenvalue(self) -> float:
        """Return the objective expectation (not a guaranteed exact eigenvalue)."""
        return self.optimal_value

    def __repr__(self) -> str:
        """Summarize the final iterate without printing the circuit or parameters."""
        return (
            f"VQAResult(optimal_value={self.optimal_value!r}, "
            f"num_parameters={len(self.optimal_parameters)}, "
            f"objective_sense={self.objective_sense!r}, "
            f"iterations={self.optimizer_result.iterations}, "
            f"status={self.optimizer_result.status.name!r})"
        )
