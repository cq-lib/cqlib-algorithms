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

"""Computational-basis configurations for finite lattice systems."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .sectors import FermionicSector, SpinMagnetizationSector


def _snapshot_sites(
    sites: Sequence[int],
    *,
    num_sites: int,
    name: str,
) -> tuple[int, ...]:
    if isinstance(sites, (str, bytes)) or not isinstance(sites, Sequence):
        raise TypeError(f"{name} must be a sequence of site indices")
    snapshot = tuple(sites)
    seen: set[int] = set()
    for site in snapshot:
        if type(site) is not int:
            raise TypeError(f"{name} must contain only int values")
        if not 0 <= site < num_sites:
            raise ValueError(f"{name} contains a site index out of range")
        if site in seen:
            raise ValueError(f"{name} must not contain duplicate site indices")
        seen.add(site)
    return tuple(sorted(snapshot))


@dataclass(frozen=True, slots=True)
class FermionicLatticeConfiguration:
    """A spin-resolved occupation configuration on interleaved modes.

    Site ``i`` uses mode ``2*i`` for spin up and mode ``2*i + 1`` for spin
    down. An occupied mode is represented by computational basis value one.

    Args:
        num_sites: Positive number of lattice sites.
        occupied_up_sites: Sites occupied by spin-up fermions.
        occupied_down_sites: Sites occupied by spin-down fermions.

    Raises:
        TypeError: If a count or site sequence has the wrong type.
        ValueError: If a site is duplicated or out of range.
    """

    num_sites: int
    occupied_up_sites: Sequence[int]
    occupied_down_sites: Sequence[int]

    def __post_init__(self) -> None:
        FermionicSector(self.num_sites, 0, 0)
        object.__setattr__(
            self,
            "occupied_up_sites",
            _snapshot_sites(
                self.occupied_up_sites,
                num_sites=self.num_sites,
                name="occupied_up_sites",
            ),
        )
        object.__setattr__(
            self,
            "occupied_down_sites",
            _snapshot_sites(
                self.occupied_down_sites,
                num_sites=self.num_sites,
                name="occupied_down_sites",
            ),
        )

    @classmethod
    def neel(
        cls,
        num_sites: int,
        *,
        start_with_up: bool = True,
    ) -> FermionicLatticeConfiguration:
        """Return a half-filled alternating single-occupation configuration."""
        if not isinstance(start_with_up, bool):
            raise TypeError("start_with_up must be a bool")
        up_parity = 0 if start_with_up else 1
        return cls(
            num_sites,
            tuple(range(up_parity, num_sites, 2)),
            tuple(range(1 - up_parity, num_sites, 2)),
        )

    @property
    def occupied_up(self) -> tuple[int, ...]:
        """Return occupied spin-up sites in ascending order."""
        return tuple(self.occupied_up_sites)

    @property
    def occupied_down(self) -> tuple[int, ...]:
        """Return occupied spin-down sites in ascending order."""
        return tuple(self.occupied_down_sites)

    @property
    def sector(self) -> FermionicSector:
        """Return the spin-resolved particle-number sector."""
        return FermionicSector(
            self.num_sites,
            len(self.occupied_up_sites),
            len(self.occupied_down_sites),
        )

    @property
    def occupation(self) -> tuple[int, ...]:
        """Return occupations in interleaved mode order."""
        up = set(self.occupied_up_sites)
        down = set(self.occupied_down_sites)
        return tuple(
            occupied
            for site in range(self.num_sites)
            for occupied in (int(site in up), int(site in down))
        )

    @property
    def basis_index(self) -> int:
        """Return the little-endian computational-basis index."""
        return sum(bit << mode for mode, bit in enumerate(self.occupation))


@dataclass(frozen=True, slots=True)
class SpinLatticeConfiguration:
    r"""A spin-one-half computational-basis configuration.

    ``|0>`` denotes :math:`S_z=+1/2` and ``|1>`` denotes
    :math:`S_z=-1/2`. The stored sites are therefore the zero-valued qubits.

    Args:
        num_sites: Positive number of spin-one-half sites.
        up_sites: Sites prepared in the spin-up state.

    Raises:
        TypeError: If a count or site sequence has the wrong type.
        ValueError: If a site is duplicated or out of range.
    """

    num_sites: int
    up_sites: Sequence[int]

    def __post_init__(self) -> None:
        SpinMagnetizationSector(self.num_sites, 0)
        object.__setattr__(
            self,
            "up_sites",
            _snapshot_sites(
                self.up_sites,
                num_sites=self.num_sites,
                name="up_sites",
            ),
        )

    @classmethod
    def all_up(cls, num_sites: int) -> SpinLatticeConfiguration:
        """Return the all-up product state."""
        return cls(num_sites, tuple(range(num_sites)))

    @classmethod
    def all_down(cls, num_sites: int) -> SpinLatticeConfiguration:
        """Return the all-down product state."""
        return cls(num_sites, ())

    @classmethod
    def neel(
        cls,
        num_sites: int,
        *,
        start_with_up: bool = True,
    ) -> SpinLatticeConfiguration:
        """Return an alternating up/down spin configuration."""
        if not isinstance(start_with_up, bool):
            raise TypeError("start_with_up must be a bool")
        up_parity = 0 if start_with_up else 1
        return cls(num_sites, tuple(range(up_parity, num_sites, 2)))

    @property
    def occupied_up(self) -> tuple[int, ...]:
        """Return spin-up sites in ascending order."""
        return tuple(self.up_sites)

    @property
    def sector(self) -> SpinMagnetizationSector:
        """Return the total-magnetization sector."""
        return SpinMagnetizationSector(self.num_sites, len(self.up_sites))

    @property
    def basis_index(self) -> int:
        """Return the little-endian index with one on every down-spin site."""
        up = set(self.up_sites)
        return sum(1 << site for site in range(self.num_sites) if site not in up)


LatticeConfiguration = FermionicLatticeConfiguration | SpinLatticeConfiguration

__all__ = [
    "FermionicLatticeConfiguration",
    "LatticeConfiguration",
    "SpinLatticeConfiguration",
]
