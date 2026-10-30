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

"""Validated linear combinations of fermionic excitation generators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from cqlib.qis import FermionicOp

from .excitation import FermionicExcitation


@dataclass(frozen=True, slots=True)
class GroupedFermionicGenerator:
    """A fixed real linear combination sharing one evolution parameter.

    Args:
        terms: Non-empty excitation/coefficient pairs with a common width.

    Raises:
        TypeError: If a term or coefficient has the wrong type.
        ValueError: If coefficients are non-finite or zero, excitations are
            duplicated or reversed, or widths disagree.
    """

    terms: tuple[tuple[FermionicExcitation, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.terms, tuple):
            raise TypeError("terms must be a tuple")
        if not self.terms:
            raise ValueError("terms must not be empty")

        snapshot: list[tuple[FermionicExcitation, float]] = []
        seen: set[FermionicExcitation] = set()
        seen_undirected: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        width: int | None = None
        for item in self.terms:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(
                    "terms must contain (FermionicExcitation, coefficient) tuples"
                )
            excitation, coefficient = item
            if not isinstance(excitation, FermionicExcitation):
                raise TypeError("term excitations must be FermionicExcitation values")
            if isinstance(coefficient, bool) or not isinstance(coefficient, Real):
                raise TypeError("term coefficients must be real numbers")
            numeric_coefficient = float(coefficient)
            if not math.isfinite(numeric_coefficient):
                raise ValueError("term coefficients must be finite")
            if numeric_coefficient == 0.0:
                raise ValueError("term coefficients must be nonzero")
            if excitation in seen:
                raise ValueError("term excitations must be unique")
            seen.add(excitation)
            direction = (excitation.sources, excitation.targets)
            undirected = (
                direction
                if excitation.sources < excitation.targets
                else (excitation.targets, excitation.sources)
            )
            if undirected in seen_undirected:
                raise ValueError(
                    "term excitations must not contain both forward and reverse directions"
                )
            seen_undirected.add(undirected)
            if width is None:
                width = excitation.num_spin_orbitals
            elif excitation.num_spin_orbitals != width:
                raise ValueError("all term excitations must use the same width")
            snapshot.append((excitation, numeric_coefficient))

        object.__setattr__(self, "terms", tuple(snapshot))

    @property
    def num_spin_orbitals(self) -> int:
        """Return the common excitation width."""
        return self.terms[0][0].num_spin_orbitals

    def to_fermionic_generator(self) -> FermionicOp:
        """Return an independent Core sum of Hermitian excitation generators."""
        first_excitation, first_coefficient = self.terms[0]
        result = first_excitation.to_fermionic_generator().scaled(first_coefficient)
        for excitation, coefficient in self.terms[1:]:
            result = result + excitation.to_fermionic_generator().scaled(coefficient)
        return result


__all__ = ["GroupedFermionicGenerator"]
