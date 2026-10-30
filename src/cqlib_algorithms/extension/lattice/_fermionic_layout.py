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

"""Shared interleaved spin-orbital layout for lattice components."""

from typing import Literal, TypeAlias

FermionSpin: TypeAlias = Literal["up", "down"]
FERMION_SPINS: tuple[FermionSpin, FermionSpin] = ("up", "down")


def fermionic_mode(site: int, spin: FermionSpin) -> int:
    """Return the interleaved mode for a non-negative site and spin label."""
    if type(site) is not int:
        raise TypeError("site must be an int")
    if site < 0:
        raise ValueError("site must be non-negative")
    if not isinstance(spin, str):
        raise TypeError("spin must be a str")
    if spin not in FERMION_SPINS:
        raise ValueError("spin must be 'up' or 'down'")
    return 2 * site + (spin == "down")


def fermionic_mode_layout(num_sites: int) -> tuple[tuple[int, FermionSpin], ...]:
    """Return ``(site, spin)`` entries in interleaved mode order."""
    if type(num_sites) is not int:
        raise TypeError("num_sites must be an int")
    if num_sites < 1:
        raise ValueError("num_sites must be at least 1")
    return tuple((site, spin) for site in range(num_sites) for spin in FERMION_SPINS)
