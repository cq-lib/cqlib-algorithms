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

"""Validated fermionic excitation descriptors."""

from __future__ import annotations

from dataclasses import dataclass

from cqlib.qis import FermionicLadder, FermionicOp

from .._numeric import validate_strict_int


def _validate_indices(
    indices: tuple[int, ...],
    *,
    name: str,
    num_spin_orbitals: int,
) -> None:
    if not isinstance(indices, tuple):
        raise TypeError(f"{name} must be a tuple")
    if not indices:
        raise ValueError(f"{name} must not be empty")
    previous = -1
    for index in indices:
        validate_strict_int(index, name=f"{name} index")
        if index >= num_spin_orbitals:
            raise ValueError(f"{name} index is out of range")
        if index <= previous:
            raise ValueError(f"{name} must be strictly increasing")
        previous = index


@dataclass(frozen=True, slots=True)
class FermionicExcitation:
    """Particle-number-conserving excitation from source to target modes.

    Args:
        num_spin_orbitals: Positive total number of fermionic modes.
        sources: Strictly increasing occupied source-mode indices.
        targets: Strictly increasing unoccupied target-mode indices.

    Raises:
        TypeError: If counts or index containers have the wrong type.
        ValueError: If indices are invalid, overlap, or have unequal rank.
    """

    num_spin_orbitals: int
    sources: tuple[int, ...]
    targets: tuple[int, ...]

    def __post_init__(self) -> None:
        validate_strict_int(
            self.num_spin_orbitals,
            name="num_spin_orbitals",
            minimum=1,
        )
        _validate_indices(
            self.sources,
            name="sources",
            num_spin_orbitals=self.num_spin_orbitals,
        )
        _validate_indices(
            self.targets,
            name="targets",
            num_spin_orbitals=self.num_spin_orbitals,
        )
        if len(self.sources) != len(self.targets):
            raise ValueError("sources and targets must have equal rank")
        if set(self.sources) & set(self.targets):
            raise ValueError("sources and targets must be disjoint")

    @property
    def rank(self) -> int:
        """Return the number of promoted electrons."""
        return len(self.sources)

    def to_fermionic_generator(self) -> FermionicOp:
        """Return Core's Hermitian generator ``i(T - T†)``."""
        excitation = self.to_fermionic_op()
        adjoint = excitation.adjoint()
        return FermionicOp.from_terms(
            self.num_spin_orbitals,
            [
                *((term, 1j * coefficient) for term, coefficient in excitation.terms),
                *((term, -1j * coefficient) for term, coefficient in adjoint.terms),
            ],
        )

    def to_fermionic_op(self) -> FermionicOp:
        """Return ``T`` with target creators then reversed source annihilators."""
        term = tuple(FermionicLadder.create(target) for target in self.targets) + tuple(
            FermionicLadder.annihilate(source) for source in reversed(self.sources)
        )
        return FermionicOp.from_term(self.num_spin_orbitals, term)


__all__ = ["FermionicExcitation"]
