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

"""Shared validation for lattice-model parameters."""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from typing import TypeAlias, cast

import numpy as np

from .topology import Lattice

RealParameter: TypeAlias = complex | float | int
RealVectorParameter: TypeAlias = RealParameter | Sequence[RealParameter]
RealTriple: TypeAlias = tuple[float, float, float]
FieldParameter: TypeAlias = Sequence[RealParameter] | Sequence[Sequence[RealParameter]]


def require_lattice(lattice: Lattice) -> None:
    """Require a lattice-model input."""
    if not isinstance(lattice, Lattice):
        raise TypeError("lattice must be a Lattice")


def finite_real(value: RealParameter, *, name: str) -> float:
    """Return a finite real scalar without silently dropping an imaginary part."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Complex):
        raise TypeError(f"{name} must be a real number")
    result = complex(value)
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise ValueError(f"{name} must be finite")
    if result.imag != 0.0:
        raise ValueError(f"{name} must be real")
    return result.real


def real_vector(
    value: RealVectorParameter,
    *,
    length: int,
    name: str,
) -> tuple[float, ...]:
    """Normalize a real scalar or fixed-length real vector."""
    if isinstance(value, numbers.Complex) and not isinstance(value, (bool, np.bool_)):
        scalar = finite_real(value, name=name)
        return (scalar,) * length
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a real number or a sequence of real numbers")
    copied = tuple(value)
    if len(copied) != length:
        raise ValueError(f"{name} length must equal the number of lattice nodes")
    return tuple(
        finite_real(item, name=f"{name}[{position}]")
        for position, item in enumerate(copied)
    )


def real_triple(value: Sequence[RealParameter], *, name: str) -> RealTriple:
    """Normalize a three-component real vector."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of three real numbers")
    copied = tuple(value)
    if len(copied) != 3:
        raise ValueError(f"{name} must contain exactly three components")
    return (
        finite_real(copied[0], name=f"{name}[0]"),
        finite_real(copied[1], name=f"{name}[1]"),
        finite_real(copied[2], name=f"{name}[2]"),
    )


def external_fields(
    value: FieldParameter,
    *,
    num_nodes: int,
) -> tuple[RealTriple, ...]:
    """Normalize one uniform field or one three-vector per lattice node."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(
            "external_field must be one three-vector or one three-vector per node"
        )
    copied = tuple(value)
    if all(
        isinstance(item, numbers.Complex) and not isinstance(item, (bool, np.bool_))
        for item in copied
    ):
        uniform = real_triple(
            cast(Sequence[RealParameter], copied),
            name="external_field",
        )
        return (uniform,) * num_nodes
    if len(copied) != num_nodes:
        raise ValueError(
            "external_field must contain one three-vector per lattice node"
        )
    return tuple(
        real_triple(
            cast(Sequence[RealParameter], item),
            name=f"external_field[{position}]",
        )
        for position, item in enumerate(copied)
    )


def real_interaction_edges(lattice: Lattice) -> tuple[tuple[int, int, float], ...]:
    """Return real non-self edges and reject ambiguous lattice self-loops."""
    result: list[tuple[int, int, float]] = []
    for source, target, weight in lattice.weighted_edges:
        if source == target:
            if weight != 0.0:
                raise ValueError(
                    "spin-model lattices must not contain nonzero self-loop weights; "
                    "use the explicit field parameter"
                )
            continue
        result.append(
            (
                source,
                target,
                finite_real(weight, name=f"edge weight ({source}, {target})"),
            )
        )
    return tuple(result)
