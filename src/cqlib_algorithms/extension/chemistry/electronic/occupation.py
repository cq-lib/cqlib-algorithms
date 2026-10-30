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

"""Spin-resolved electronic occupation data."""

from __future__ import annotations

from dataclasses import dataclass

from ..validation.numerical import validate_strict_int


@dataclass(frozen=True, slots=True)
class ElectronicOccupation:
    """Index-ordered occupation in interleaved alpha/beta spin-orbital order.

    Occupied modes are selected from the lowest spatial-orbital indices. This
    is an Aufbau occupation only when those indices are ordered by increasing
    orbital energy.

    Args:
        num_spatial_orbitals: Positive number of spatial orbitals.
        num_alpha: Number of alpha electrons.
        num_beta: Number of beta electrons.

    Raises:
        TypeError: If a count is not an integer.
        ValueError: If a count is negative or exceeds orbital capacity.
    """

    num_spatial_orbitals: int
    num_alpha: int
    num_beta: int

    def __post_init__(self) -> None:
        validate_strict_int(
            self.num_spatial_orbitals,
            name="num_spatial_orbitals",
            minimum=1,
        )
        validate_strict_int(self.num_alpha, name="num_alpha")
        validate_strict_int(self.num_beta, name="num_beta")
        if self.num_alpha > self.num_spatial_orbitals:
            raise ValueError("num_alpha exceeds the spatial-orbital capacity")
        if self.num_beta > self.num_spatial_orbitals:
            raise ValueError("num_beta exceeds the spatial-orbital capacity")

    @classmethod
    def closed_shell(
        cls,
        num_spatial_orbitals: int,
        num_electrons: int,
    ) -> ElectronicOccupation:
        """Construct an equal-spin reference occupation ordered by index.

        Args:
            num_spatial_orbitals: Positive number of spatial orbitals.
            num_electrons: Non-negative even electron count.

        Returns:
            Closed-shell interleaved alpha/beta occupation.

        Raises:
            TypeError: If a count is not an integer.
            ValueError: If the electron count is odd or exceeds capacity.
        """
        validate_strict_int(
            num_spatial_orbitals,
            name="num_spatial_orbitals",
            minimum=1,
        )
        validate_strict_int(num_electrons, name="num_electrons")
        if num_electrons % 2 != 0:
            raise ValueError("num_electrons must be even for a closed shell")
        if num_electrons > 2 * num_spatial_orbitals:
            raise ValueError("num_electrons exceeds the spin-orbital capacity")
        return cls(
            num_spatial_orbitals=num_spatial_orbitals,
            num_alpha=num_electrons // 2,
            num_beta=num_electrons // 2,
        )

    @property
    def num_spin_orbitals(self) -> int:
        """Return twice the number of spatial orbitals."""
        return 2 * self.num_spatial_orbitals

    @property
    def num_electrons(self) -> int:
        """Return the total electron count."""
        return self.num_alpha + self.num_beta

    @property
    def occupied_spin_orbitals(self) -> tuple[int, ...]:
        """Return modes occupied from the lowest spatial-orbital indices."""
        alpha = range(0, 2 * self.num_alpha, 2)
        beta = range(1, 2 * self.num_beta, 2)
        return tuple(sorted((*alpha, *beta)))

    @property
    def virtual_spin_orbitals(self) -> tuple[int, ...]:
        """Return unoccupied spin-orbital indices in ascending order."""
        occupied = set(self.occupied_spin_orbitals)
        return tuple(
            index for index in range(self.num_spin_orbitals) if index not in occupied
        )

    @property
    def spin_orbital_occupation(self) -> tuple[bool, ...]:
        """Return the little-endian interleaved occupation bit sequence."""
        occupied = set(self.occupied_spin_orbitals)
        return tuple(index in occupied for index in range(self.num_spin_orbitals))


__all__ = ["ElectronicOccupation"]
