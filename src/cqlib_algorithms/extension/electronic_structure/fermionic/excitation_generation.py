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

"""Deterministic alpha/beta-number-conserving excitation generation."""

from __future__ import annotations

from itertools import combinations

from ..electronic.occupation import ElectronicOccupation
from .excitation import FermionicExcitation
from .generators import GroupedFermionicGenerator


def _validate_occupation(occupation: ElectronicOccupation) -> None:
    if not isinstance(occupation, ElectronicOccupation):
        raise TypeError("occupation must be an ElectronicOccupation")


def _validate_closed_shell(occupation: ElectronicOccupation) -> None:
    _validate_occupation(occupation)
    if occupation.num_alpha != occupation.num_beta:
        raise ValueError("occupation must be closed shell")


def _spin_counts(indices: tuple[int, ...]) -> tuple[int, int]:
    alpha = sum(index % 2 == 0 for index in indices)
    return alpha, len(indices) - alpha


def generate_single_excitations(
    occupation: ElectronicOccupation,
) -> tuple[FermionicExcitation, ...]:
    """Generate all spin-preserving occupied-to-virtual singles.

    Args:
        occupation: Reference occupation defining occupied and virtual modes.

    Returns:
        Deterministically ordered single excitations.

    Raises:
        TypeError: If ``occupation`` has the wrong type.
    """
    _validate_occupation(occupation)
    excitations = [
        FermionicExcitation(
            occupation.num_spin_orbitals,
            (source,),
            (target,),
        )
        for source in occupation.occupied_spin_orbitals
        for target in occupation.virtual_spin_orbitals
        if source % 2 == target % 2
    ]
    return tuple(excitations)


def generate_double_excitations(
    occupation: ElectronicOccupation,
) -> tuple[FermionicExcitation, ...]:
    """Generate spin-composition-preserving occupied-to-virtual doubles.

    Args:
        occupation: Reference occupation defining occupied and virtual modes.

    Returns:
        Deterministically ordered double excitations.

    Raises:
        TypeError: If ``occupation`` has the wrong type.
    """
    _validate_occupation(occupation)
    sources = tuple(combinations(occupation.occupied_spin_orbitals, 2))
    targets = tuple(combinations(occupation.virtual_spin_orbitals, 2))
    excitations = [
        FermionicExcitation(
            occupation.num_spin_orbitals,
            source,
            target,
        )
        for source in sources
        for target in targets
        if _spin_counts(source) == _spin_counts(target)
    ]
    return tuple(excitations)


def generate_uccsd_excitations(
    occupation: ElectronicOccupation,
) -> tuple[FermionicExcitation, ...]:
    """Return alpha/beta-number-conserving singles followed by doubles.

    Args:
        occupation: Reference occupation defining the excitation manifold.

    Returns:
        Deterministically ordered UCCSD excitation pool.

    Raises:
        TypeError: If ``occupation`` has the wrong type.
    """
    _validate_occupation(occupation)
    return generate_single_excitations(occupation) + generate_double_excitations(
        occupation
    )


def generate_paired_double_excitations(
    occupation: ElectronicOccupation,
) -> tuple[FermionicExcitation, ...]:
    """Generate occupied-to-virtual closed-shell pair excitations.

    Args:
        occupation: Closed-shell reference occupation.

    Returns:
        Deterministically ordered paired-double excitation pool.

    Raises:
        TypeError: If ``occupation`` has the wrong type.
        ValueError: If the occupation is not closed shell.
    """
    _validate_closed_shell(occupation)
    num_occupied = occupation.num_alpha
    return tuple(
        FermionicExcitation(
            occupation.num_spin_orbitals,
            (2 * source, 2 * source + 1),
            (2 * target, 2 * target + 1),
        )
        for source in range(num_occupied)
        for target in range(num_occupied, occupation.num_spatial_orbitals)
    )


def generate_puccsd_generators(
    occupation: ElectronicOccupation,
) -> tuple[FermionicExcitation | GroupedFermionicGenerator, ...]:
    """Generate the generalized spin-adapted paired-UCCSD pool.

    Args:
        occupation: Closed-shell reference occupation.

    Returns:
        Grouped spin-adapted singles followed by paired doubles.

    Raises:
        TypeError: If ``occupation`` has the wrong type.
        ValueError: If the occupation is not closed shell.
    """
    _validate_closed_shell(occupation)
    orbital_pairs = tuple(combinations(range(occupation.num_spatial_orbitals), 2))
    grouped_singles = tuple(
        GroupedFermionicGenerator(
            (
                (
                    FermionicExcitation(
                        occupation.num_spin_orbitals,
                        (2 * source,),
                        (2 * target,),
                    ),
                    1.0,
                ),
                (
                    FermionicExcitation(
                        occupation.num_spin_orbitals,
                        (2 * source + 1,),
                        (2 * target + 1,),
                    ),
                    1.0,
                ),
            )
        )
        for source, target in orbital_pairs
    )
    paired_doubles = tuple(
        FermionicExcitation(
            occupation.num_spin_orbitals,
            (2 * source, 2 * source + 1),
            (2 * target, 2 * target + 1),
        )
        for source, target in orbital_pairs
    )
    return grouped_singles + paired_doubles


__all__ = [
    "generate_double_excitations",
    "generate_paired_double_excitations",
    "generate_puccsd_generators",
    "generate_single_excitations",
    "generate_uccsd_excitations",
]
