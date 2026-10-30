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

"""Immutable molecular input data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..validation.numerical import validate_finite_real, validate_strict_int

LengthUnit = Literal["angstrom", "bohr"]


@dataclass(frozen=True, slots=True)
class Atom:
    """An element symbol and Cartesian coordinates.

    Args:
        symbol: Non-empty alphabetic chemical symbol.
        coordinates: Three finite Cartesian coordinates in the molecule unit.

    Raises:
        TypeError: If a field has the wrong container or scalar type.
        ValueError: If the symbol or coordinates are invalid.
    """

    symbol: str
    coordinates: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str):
            raise TypeError("symbol must be a str")
        if not self.symbol.strip() or not self.symbol.isalpha():
            raise ValueError("symbol must contain only letters")
        if not isinstance(self.coordinates, tuple):
            raise TypeError("coordinates must be a tuple")
        if len(self.coordinates) != 3:
            raise ValueError("coordinates must contain exactly three values")
        for coordinate in self.coordinates:
            validate_finite_real(coordinate, name="coordinate")


@dataclass(frozen=True, slots=True)
class MolecularSpecification:
    """Molecular geometry and electronic-structure calculation settings.

    Args:
        atoms: Non-empty immutable atom sequence.
        basis: Non-empty orbital basis identifier.
        charge: Integer molecular charge.
        multiplicity: Positive spin multiplicity ``2S + 1``.
        unit: Coordinate unit, either ``"angstrom"`` or ``"bohr"``.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If geometry or calculation settings are invalid.
    """

    atoms: tuple[Atom, ...]
    basis: str
    charge: int = 0
    multiplicity: int = 1
    unit: LengthUnit = "angstrom"

    def __post_init__(self) -> None:
        if not isinstance(self.atoms, tuple):
            raise TypeError("atoms must be a tuple")
        if not self.atoms:
            raise ValueError("atoms must not be empty")
        if any(not isinstance(atom, Atom) for atom in self.atoms):
            raise TypeError("atoms must contain only Atom values")
        if not isinstance(self.basis, str):
            raise TypeError("basis must be a str")
        if not self.basis.strip():
            raise ValueError("basis must not be empty")
        validate_strict_int(self.charge, name="charge", minimum=None)
        validate_strict_int(self.multiplicity, name="multiplicity", minimum=1)
        if self.unit not in ("angstrom", "bohr"):
            raise ValueError("unit must be 'angstrom' or 'bohr'")


__all__ = ["Atom", "LengthUnit", "MolecularSpecification"]
