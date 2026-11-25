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

"""Private helpers for symbolic parameter names and finite bindings."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from numbers import Real
from types import MappingProxyType

from cqlib import Circuit


def validate_parameter_prefix(prefix: str) -> None:
    """Validate a symbolic parameter-name prefix."""
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a string")


def snapshot_parameter_bindings(
    bindings: Mapping[str, float],
    *,
    name: str,
) -> Mapping[str, float]:
    """Return an immutable finite copy of symbolic parameter bindings."""
    if not isinstance(bindings, Mapping):
        raise TypeError(f"{name} must be a mapping")

    normalized: dict[str, float] = {}
    for parameter_name, value in bindings.items():
        if not isinstance(parameter_name, str):
            raise TypeError("parameter binding names must be strings")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("parameter binding values must be real numbers")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError("parameter binding values must be finite")
        normalized[parameter_name] = numeric_value
    return MappingProxyType(normalized)


def bind_circuit_parameters(
    circuit: Circuit,
    parameter_bindings: Mapping[str, float],
) -> Circuit:
    """Return an independent circuit with exactly its active symbols bound.

    Finite real values are substituted into every parameter expression.
    Missing or unexpected symbol names raise ValueError. The input circuit
    and its global phase are preserved.
    """
    bindings = snapshot_parameter_bindings(
        parameter_bindings, name="parameter_bindings"
    )
    required = set(circuit.used_symbols)
    provided = set(bindings)
    if required != provided:
        raise ValueError(
            "parameter bindings must match used circuit symbols exactly: "
            f"missing={sorted(required - provided)!r}, "
            f"unexpected={sorted(provided - required)!r}"
        )
    return deepcopy(circuit).assign_parameters(dict(bindings))


__all__: list[str] = []
