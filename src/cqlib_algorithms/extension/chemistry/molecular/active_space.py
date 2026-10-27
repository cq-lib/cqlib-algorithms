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

"""Explicit restricted and unrestricted active-space transformations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..electronic.integrals import ElectronicIntegrals
from ..electronic.occupation import ElectronicOccupation
from ..electronic.spin_resolved_integrals import SpinResolvedElectronicIntegrals
from ..validation.numerical import validate_strict_int


def _validate_indices(indices: tuple[int, ...], *, name: str) -> None:
    if not isinstance(indices, tuple):
        raise TypeError(f"{name} must be a tuple")
    previous = -1
    for index in indices:
        validate_strict_int(index, name=f"{name} index")
        if index <= previous:
            raise ValueError(f"{name} must be strictly increasing")
        previous = index


@dataclass(frozen=True, slots=True)
class ActiveSpace:
    """Explicit frozen occupied and retained active spatial orbitals.

    Args:
        frozen_orbitals: Strictly increasing occupied orbitals to integrate out.
        active_orbitals: Non-empty strictly increasing retained orbitals.

    Raises:
        TypeError: If an index collection or entry has the wrong type.
        ValueError: If indices are unordered, duplicated, empty, or overlap.
    """

    frozen_orbitals: tuple[int, ...]
    active_orbitals: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_indices(self.frozen_orbitals, name="frozen_orbitals")
        _validate_indices(self.active_orbitals, name="active_orbitals")
        if not self.active_orbitals:
            raise ValueError("active_orbitals must not be empty")
        if set(self.frozen_orbitals) & set(self.active_orbitals):
            raise ValueError("frozen_orbitals and active_orbitals must be disjoint")

    @classmethod
    def full(cls, num_spatial_orbitals: int) -> ActiveSpace:
        """Select every spatial orbital and freeze none."""
        validate_strict_int(
            num_spatial_orbitals, name="num_spatial_orbitals", minimum=1
        )
        return cls((), tuple(range(num_spatial_orbitals)))

    @classmethod
    def manual(
        cls,
        *,
        frozen_orbitals: tuple[int, ...] = (),
        active_orbitals: tuple[int, ...],
    ) -> ActiveSpace:
        """Create an explicit active-space selection."""
        return cls(frozen_orbitals, active_orbitals)


@dataclass(frozen=True, slots=True)
class ActiveSpaceResult:
    """Active electronic integrals and electron/orbital bookkeeping.

    Args:
        integrals: Transformed restricted active-space integrals.
        occupation: Active alpha/beta occupation.
        frozen_orbitals: Frozen full-space orbital indices.
        active_orbitals: Retained full-space orbital indices.

    Raises:
        TypeError: If a component has the wrong type.
        ValueError: If dimensions or orbital selections disagree.
    """

    integrals: ElectronicIntegrals
    occupation: ElectronicOccupation
    frozen_orbitals: tuple[int, ...]
    active_orbitals: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.integrals, ElectronicIntegrals):
            raise TypeError("integrals must be ElectronicIntegrals")
        if not isinstance(self.occupation, ElectronicOccupation):
            raise TypeError("occupation must be an ElectronicOccupation")
        if self.occupation.num_spatial_orbitals != self.integrals.num_spatial_orbitals:
            raise ValueError("occupation must match the active integral dimensions")
        _validate_indices(self.frozen_orbitals, name="frozen_orbitals")
        _validate_indices(self.active_orbitals, name="active_orbitals")
        if set(self.frozen_orbitals) & set(self.active_orbitals):
            raise ValueError("frozen_orbitals and active_orbitals must be disjoint")
        if len(self.active_orbitals) != self.integrals.num_spatial_orbitals:
            raise ValueError("active_orbitals must match the integral dimensions")

    @property
    def num_alpha(self) -> int:
        """Return the number of active alpha electrons."""
        return self.occupation.num_alpha

    @property
    def num_beta(self) -> int:
        """Return the number of active beta electrons."""
        return self.occupation.num_beta

    @property
    def num_electrons(self) -> int:
        """Return the total active electron count."""
        return self.occupation.num_electrons


@dataclass(frozen=True, slots=True)
class SpinResolvedActiveSpace:
    """Explicit frozen and active orbitals for separate alpha/beta bases.

    Alpha and beta active selections may contain different orbital indices but
    must have equal lengths because :class:`SpinResolvedElectronicIntegrals`
    represents the same number of modes in both spin sectors.

    Args:
        frozen_alpha_orbitals: Occupied alpha orbitals to integrate out.
        frozen_beta_orbitals: Occupied beta orbitals to integrate out.
        active_alpha_orbitals: Retained alpha orbitals.
        active_beta_orbitals: Retained beta orbitals.

    Raises:
        TypeError: If an index collection or entry has the wrong type.
        ValueError: If selections are invalid or active widths differ.
    """

    frozen_alpha_orbitals: tuple[int, ...]
    frozen_beta_orbitals: tuple[int, ...]
    active_alpha_orbitals: tuple[int, ...]
    active_beta_orbitals: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in (
            "frozen_alpha_orbitals",
            "frozen_beta_orbitals",
            "active_alpha_orbitals",
            "active_beta_orbitals",
        ):
            _validate_indices(getattr(self, name), name=name)
        if not self.active_alpha_orbitals or not self.active_beta_orbitals:
            raise ValueError("alpha and beta active orbitals must not be empty")
        if len(self.active_alpha_orbitals) != len(self.active_beta_orbitals):
            raise ValueError("alpha and beta active selections must have equal lengths")
        if set(self.frozen_alpha_orbitals) & set(self.active_alpha_orbitals):
            raise ValueError("frozen and active alpha orbitals must be disjoint")
        if set(self.frozen_beta_orbitals) & set(self.active_beta_orbitals):
            raise ValueError("frozen and active beta orbitals must be disjoint")

    @classmethod
    def full(cls, num_spatial_orbitals: int) -> SpinResolvedActiveSpace:
        """Select every alpha and beta orbital and freeze none."""
        validate_strict_int(
            num_spatial_orbitals, name="num_spatial_orbitals", minimum=1
        )
        active = tuple(range(num_spatial_orbitals))
        return cls((), (), active, active)

    @classmethod
    def paired(
        cls,
        *,
        frozen_orbitals: tuple[int, ...] = (),
        active_orbitals: tuple[int, ...],
    ) -> SpinResolvedActiveSpace:
        """Use the same orbital positions in the alpha and beta bases."""
        return cls(
            frozen_orbitals,
            frozen_orbitals,
            active_orbitals,
            active_orbitals,
        )

    @classmethod
    def manual(
        cls,
        *,
        frozen_alpha_orbitals: tuple[int, ...] = (),
        frozen_beta_orbitals: tuple[int, ...] = (),
        active_alpha_orbitals: tuple[int, ...],
        active_beta_orbitals: tuple[int, ...],
    ) -> SpinResolvedActiveSpace:
        """Create independent explicit alpha and beta selections."""
        return cls(
            frozen_alpha_orbitals,
            frozen_beta_orbitals,
            active_alpha_orbitals,
            active_beta_orbitals,
        )


@dataclass(frozen=True, slots=True)
class SpinResolvedActiveSpaceResult:
    """Spin-resolved active integrals and orbital bookkeeping.

    Args:
        integrals: Transformed spin-resolved active-space integrals.
        occupation: Active alpha/beta occupation.
        frozen_alpha_orbitals: Frozen alpha indices in the full space.
        frozen_beta_orbitals: Frozen beta indices in the full space.
        active_alpha_orbitals: Retained alpha indices in the full space.
        active_beta_orbitals: Retained beta indices in the full space.

    Raises:
        TypeError: If a component has the wrong type.
        ValueError: If dimensions or orbital selections disagree.
    """

    integrals: SpinResolvedElectronicIntegrals
    occupation: ElectronicOccupation
    frozen_alpha_orbitals: tuple[int, ...]
    frozen_beta_orbitals: tuple[int, ...]
    active_alpha_orbitals: tuple[int, ...]
    active_beta_orbitals: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.integrals, SpinResolvedElectronicIntegrals):
            raise TypeError("integrals must be SpinResolvedElectronicIntegrals")
        if not isinstance(self.occupation, ElectronicOccupation):
            raise TypeError("occupation must be an ElectronicOccupation")
        if self.occupation.num_spatial_orbitals != self.integrals.num_spatial_orbitals:
            raise ValueError("occupation must match the active integral dimensions")
        selection = SpinResolvedActiveSpace(
            self.frozen_alpha_orbitals,
            self.frozen_beta_orbitals,
            self.active_alpha_orbitals,
            self.active_beta_orbitals,
        )
        if len(selection.active_alpha_orbitals) != self.integrals.num_spatial_orbitals:
            raise ValueError("active orbitals must match the integral dimensions")

    @property
    def num_alpha(self) -> int:
        """Return the number of active alpha electrons."""
        return self.occupation.num_alpha

    @property
    def num_beta(self) -> int:
        """Return the number of active beta electrons."""
        return self.occupation.num_beta

    @property
    def num_electrons(self) -> int:
        """Return the total active electron count."""
        return self.occupation.num_electrons


def transform_active_space(
    integrals: ElectronicIntegrals,
    occupation: ElectronicOccupation,
    active_space: ActiveSpace,
) -> ActiveSpaceResult:
    """Freeze paired core orbitals and restrict to explicit active orbitals.

    The input two-electron tensor follows ``V[p,q,r,s] = (pq|rs)``. Every
    initially occupied spatial orbital must be either frozen or active.
    Frozen orbitals must be doubly occupied because integrating out a singly
    occupied orbital would require a spin-dependent effective Hamiltonian,
    which cannot be represented by :class:`ElectronicIntegrals`.

    Args:
        integrals: Restricted full-space electronic integrals.
        occupation: Full-space reference occupation.
        active_space: Explicit frozen and retained orbital selection.

    Returns:
        Restricted active-space integrals and updated occupation.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If dimensions, indices, or frozen occupations are invalid.
    """
    if not isinstance(integrals, ElectronicIntegrals):
        raise TypeError("integrals must be ElectronicIntegrals")
    if not isinstance(occupation, ElectronicOccupation):
        raise TypeError("occupation must be an ElectronicOccupation")
    if not isinstance(active_space, ActiveSpace):
        raise TypeError("active_space must be an ActiveSpace")

    num_orbitals = integrals.num_spatial_orbitals
    if occupation.num_spatial_orbitals != num_orbitals:
        raise ValueError("occupation and integrals must use the same orbital count")
    selected = active_space.frozen_orbitals + active_space.active_orbitals
    if any(index >= num_orbitals for index in selected):
        raise ValueError("active-space orbital index is out of range")

    doubly_occupied = set(range(min(occupation.num_alpha, occupation.num_beta)))
    occupied = set(range(max(occupation.num_alpha, occupation.num_beta)))
    frozen = set(active_space.frozen_orbitals)
    active = set(active_space.active_orbitals)
    if not frozen <= doubly_occupied:
        raise ValueError("only initially doubly occupied orbitals may be frozen")
    if not occupied <= frozen | active:
        raise ValueError("every initially occupied orbital must be frozen or active")

    one_body = np.array(integrals.one_body, copy=True)
    two_body = integrals.two_body
    constant = integrals.constant
    for i in active_space.frozen_orbitals:
        constant += 2.0 * integrals.one_body[i, i]
        for j in active_space.frozen_orbitals:
            constant += 2.0 * two_body[i, i, j, j] - two_body[i, j, j, i]
        for p in active_space.active_orbitals:
            for q in active_space.active_orbitals:
                one_body[p, q] += 2.0 * two_body[p, q, i, i] - two_body[p, i, i, q]

    active_indices = np.asarray(active_space.active_orbitals, dtype=np.intp)
    active_one_body = one_body[np.ix_(active_indices, active_indices)]
    active_two_body = two_body[
        np.ix_(active_indices, active_indices, active_indices, active_indices)
    ]
    active_integrals = ElectronicIntegrals(
        constant,
        active_one_body,
        active_two_body,
    )
    num_frozen = len(active_space.frozen_orbitals)
    active_occupation = ElectronicOccupation(
        num_spatial_orbitals=active_integrals.num_spatial_orbitals,
        num_alpha=occupation.num_alpha - num_frozen,
        num_beta=occupation.num_beta - num_frozen,
    )
    return ActiveSpaceResult(
        integrals=active_integrals,
        occupation=active_occupation,
        frozen_orbitals=active_space.frozen_orbitals,
        active_orbitals=active_space.active_orbitals,
    )


def transform_spin_resolved_active_space(
    integrals: SpinResolvedElectronicIntegrals,
    occupation: ElectronicOccupation,
    active_space: SpinResolvedActiveSpace,
) -> SpinResolvedActiveSpaceResult:
    """Freeze occupied UHF spin orbitals and retain explicit active orbitals.

    The mixed tensor stores ``(p_alpha q_alpha | r_beta s_beta)``. Frozen
    alpha and beta orbitals contribute their determinant expectation to the
    scalar term and their Coulomb/exchange fields to the corresponding active
    one-body blocks. Two-body blocks are then restricted without changing
    their coefficients.

    Args:
        integrals: Spin-resolved full-space electronic integrals.
        occupation: Full-space reference occupation.
        active_space: Independent alpha and beta orbital selection.

    Returns:
        Spin-resolved active-space integrals and updated occupation.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If dimensions, indices, or frozen occupations are invalid.
    """
    if not isinstance(integrals, SpinResolvedElectronicIntegrals):
        raise TypeError("integrals must be SpinResolvedElectronicIntegrals")
    if not isinstance(occupation, ElectronicOccupation):
        raise TypeError("occupation must be an ElectronicOccupation")
    if not isinstance(active_space, SpinResolvedActiveSpace):
        raise TypeError("active_space must be a SpinResolvedActiveSpace")

    num_orbitals = integrals.num_spatial_orbitals
    if occupation.num_spatial_orbitals != num_orbitals:
        raise ValueError("occupation and integrals must use the same orbital count")

    alpha_selected = (
        active_space.frozen_alpha_orbitals + active_space.active_alpha_orbitals
    )
    beta_selected = (
        active_space.frozen_beta_orbitals + active_space.active_beta_orbitals
    )
    if any(index >= num_orbitals for index in alpha_selected + beta_selected):
        raise ValueError("active-space orbital index is out of range")

    occupied_alpha = set(range(occupation.num_alpha))
    occupied_beta = set(range(occupation.num_beta))
    frozen_alpha = set(active_space.frozen_alpha_orbitals)
    frozen_beta = set(active_space.frozen_beta_orbitals)
    active_alpha = set(active_space.active_alpha_orbitals)
    active_beta = set(active_space.active_beta_orbitals)
    if not frozen_alpha <= occupied_alpha:
        raise ValueError("only initially occupied alpha orbitals may be frozen")
    if not frozen_beta <= occupied_beta:
        raise ValueError("only initially occupied beta orbitals may be frozen")
    if not occupied_alpha <= frozen_alpha | active_alpha:
        raise ValueError(
            "every initially occupied alpha orbital must be frozen or active"
        )
    if not occupied_beta <= frozen_beta | active_beta:
        raise ValueError(
            "every initially occupied beta orbital must be frozen or active"
        )

    one_alpha = np.array(integrals.one_body_alpha, copy=True)
    one_beta = np.array(integrals.one_body_beta, copy=True)
    two_alpha = integrals.two_body_alpha_alpha
    two_beta = integrals.two_body_beta_beta
    two_mixed = integrals.two_body_alpha_beta

    constant = integrals.constant
    for i in active_space.frozen_alpha_orbitals:
        constant += integrals.one_body_alpha[i, i]
        for j in active_space.frozen_alpha_orbitals:
            constant += 0.5 * (two_alpha[i, i, j, j] - two_alpha[i, j, j, i])
        for j in active_space.frozen_beta_orbitals:
            constant += two_mixed[i, i, j, j]
    for i in active_space.frozen_beta_orbitals:
        constant += integrals.one_body_beta[i, i]
        for j in active_space.frozen_beta_orbitals:
            constant += 0.5 * (two_beta[i, i, j, j] - two_beta[i, j, j, i])

    for p in active_space.active_alpha_orbitals:
        for q in active_space.active_alpha_orbitals:
            for i in active_space.frozen_alpha_orbitals:
                one_alpha[p, q] += two_alpha[p, q, i, i] - two_alpha[p, i, i, q]
            for i in active_space.frozen_beta_orbitals:
                one_alpha[p, q] += two_mixed[p, q, i, i]

    for p in active_space.active_beta_orbitals:
        for q in active_space.active_beta_orbitals:
            for i in active_space.frozen_beta_orbitals:
                one_beta[p, q] += two_beta[p, q, i, i] - two_beta[p, i, i, q]
            for i in active_space.frozen_alpha_orbitals:
                one_beta[p, q] += two_mixed[i, i, p, q]

    alpha_indices = np.asarray(active_space.active_alpha_orbitals, dtype=np.intp)
    beta_indices = np.asarray(active_space.active_beta_orbitals, dtype=np.intp)
    active_integrals = SpinResolvedElectronicIntegrals(
        constant=constant,
        one_body_alpha=one_alpha[np.ix_(alpha_indices, alpha_indices)],
        one_body_beta=one_beta[np.ix_(beta_indices, beta_indices)],
        two_body_alpha_alpha=two_alpha[
            np.ix_(alpha_indices, alpha_indices, alpha_indices, alpha_indices)
        ],
        two_body_beta_beta=two_beta[
            np.ix_(beta_indices, beta_indices, beta_indices, beta_indices)
        ],
        two_body_alpha_beta=two_mixed[
            np.ix_(alpha_indices, alpha_indices, beta_indices, beta_indices)
        ],
    )
    active_occupation = ElectronicOccupation(
        num_spatial_orbitals=active_integrals.num_spatial_orbitals,
        num_alpha=occupation.num_alpha - len(active_space.frozen_alpha_orbitals),
        num_beta=occupation.num_beta - len(active_space.frozen_beta_orbitals),
    )
    return SpinResolvedActiveSpaceResult(
        integrals=active_integrals,
        occupation=active_occupation,
        frozen_alpha_orbitals=active_space.frozen_alpha_orbitals,
        frozen_beta_orbitals=active_space.frozen_beta_orbitals,
        active_alpha_orbitals=active_space.active_alpha_orbitals,
        active_beta_orbitals=active_space.active_beta_orbitals,
    )


__all__ = [
    "ActiveSpace",
    "ActiveSpaceResult",
    "SpinResolvedActiveSpace",
    "SpinResolvedActiveSpaceResult",
    "transform_active_space",
    "transform_spin_resolved_active_space",
]
