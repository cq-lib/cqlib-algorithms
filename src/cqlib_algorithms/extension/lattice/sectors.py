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

"""Conserved sectors for finite lattice systems."""

from __future__ import annotations

from dataclasses import dataclass


def _validate_int(value: int, *, name: str, minimum: int = 0) -> None:
    """Validate an integer lower bound."""
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


@dataclass(frozen=True, slots=True)
class FermionicSector:
    """Fixed spin-resolved particle numbers on a spinful lattice.

    ``num_up`` and ``num_down`` are independent conserved populations. Each
    site can hold at most one fermion of each spin species.

    Args:
        num_sites: Positive number of lattice sites.
        num_up: Number of spin-up fermions.
        num_down: Number of spin-down fermions.

    Raises:
        TypeError: If a count is not an integer.
        ValueError: If a count is negative or exceeds site capacity.
    """

    num_sites: int
    num_up: int
    num_down: int

    def __post_init__(self) -> None:
        _validate_int(self.num_sites, name="num_sites", minimum=1)
        _validate_int(self.num_up, name="num_up")
        _validate_int(self.num_down, name="num_down")
        if self.num_up > self.num_sites:
            raise ValueError("num_up exceeds the number of lattice sites")
        if self.num_down > self.num_sites:
            raise ValueError("num_down exceeds the number of lattice sites")

    @property
    def num_particles(self) -> int:
        """Return the total fermion count."""
        return self.num_up + self.num_down

    @property
    def num_modes(self) -> int:
        """Return the number of interleaved spin-orbital modes."""
        return 2 * self.num_sites

    @property
    def spin_projection(self) -> float:
        r"""Return :math:`(N_\uparrow-N_\downarrow)/2`."""
        return (self.num_up - self.num_down) / 2.0


@dataclass(frozen=True, slots=True)
class SpinMagnetizationSector:
    r"""Fixed total-:math:`S_z` sector for spin-one-half lattice sites.

    ``num_up`` counts sites in the :math:`S_z=+1/2` state, represented by
    computational basis state ``|0>``. Consequently,
    :math:`M_z = N_\uparrow - N/2`.

    Args:
        num_sites: Positive number of spin-one-half sites.
        num_up: Number of sites in the spin-up state.

    Raises:
        TypeError: If a count is not an integer.
        ValueError: If a count is negative or exceeds site capacity.
    """

    num_sites: int
    num_up: int

    def __post_init__(self) -> None:
        _validate_int(self.num_sites, name="num_sites", minimum=1)
        _validate_int(self.num_up, name="num_up")
        if self.num_up > self.num_sites:
            raise ValueError("num_up exceeds the number of lattice sites")

    @property
    def num_down(self) -> int:
        """Return the number of spin-down sites."""
        return self.num_sites - self.num_up

    @property
    def magnetization(self) -> float:
        r"""Return the total :math:`S_z` eigenvalue."""
        return self.num_up - self.num_sites / 2.0


__all__ = ["FermionicSector", "SpinMagnetizationSector"]
