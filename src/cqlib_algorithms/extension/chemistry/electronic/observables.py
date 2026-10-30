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

"""Electronic particle-number and spin observable factories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cqlib.qis import FermionicLadder, FermionicOp, FermionToQubitMapper, Hamiltonian
from numpy.typing import NDArray

from ...many_body._fermionic_mapping import map_fermionic_operator
from ..validation.numerical import immutable_real_array, validate_strict_int


def _spin_z(num_spatial_orbitals: int) -> FermionicOp:
    terms: list[tuple[tuple[FermionicLadder, ...], float]] = []
    for spatial in range(num_spatial_orbitals):
        terms.extend(
            (
                (
                    (
                        FermionicLadder.create(2 * spatial),
                        FermionicLadder.annihilate(2 * spatial),
                    ),
                    0.5,
                ),
                (
                    (
                        FermionicLadder.create(2 * spatial + 1),
                        FermionicLadder.annihilate(2 * spatial + 1),
                    ),
                    -0.5,
                ),
            )
        )
    return FermionicOp.from_terms(2 * num_spatial_orbitals, terms)


def _spin_plus(
    num_spatial_orbitals: int,
    alpha_beta_overlap: NDArray[np.float64],
) -> FermionicOp:
    return FermionicOp.from_terms(
        2 * num_spatial_orbitals,
        [
            (
                (
                    FermionicLadder.create(2 * alpha),
                    FermionicLadder.annihilate(2 * beta + 1),
                ),
                float(alpha_beta_overlap[alpha, beta]),
            )
            for alpha in range(num_spatial_orbitals)
            for beta in range(num_spatial_orbitals)
            if alpha_beta_overlap[alpha, beta] != 0.0
        ],
    )


@dataclass(frozen=True, slots=True)
class ParticleNumber:
    """Factory for ``N = sum_p a†_p a_p``.

    Args:
        num_spatial_orbitals: Positive number of spatial orbitals.

    Raises:
        TypeError: If the orbital count is not an integer.
        ValueError: If the orbital count is not positive.
    """

    num_spatial_orbitals: int

    def __post_init__(self) -> None:
        validate_strict_int(
            self.num_spatial_orbitals,
            name="num_spatial_orbitals",
            minimum=1,
        )

    def to_fermionic_op(self) -> FermionicOp:
        """Return the total particle-number operator."""
        return FermionicOp.total_number(2 * self.num_spatial_orbitals)

    def to_qubit_hamiltonian(
        self,
        mapper: FermionToQubitMapper | None = None,
    ) -> Hamiltonian:
        """Map the particle-number operator to a Hermitian qubit observable."""
        return map_fermionic_operator(
            self.to_fermionic_op(),
            mapper,
            name="particle-number observable",
        )


@dataclass(frozen=True, slots=True)
class Magnetization:
    """Factory for ``S_z = (N_alpha - N_beta) / 2``.

    Args:
        num_spatial_orbitals: Positive number of spatial orbitals.

    Raises:
        TypeError: If the orbital count is not an integer.
        ValueError: If the orbital count is not positive.
    """

    num_spatial_orbitals: int

    def __post_init__(self) -> None:
        validate_strict_int(
            self.num_spatial_orbitals,
            name="num_spatial_orbitals",
            minimum=1,
        )

    def to_fermionic_op(self) -> FermionicOp:
        """Return the spin magnetization operator in an orthonormal basis."""
        return _spin_z(self.num_spatial_orbitals)

    def to_qubit_hamiltonian(
        self,
        mapper: FermionToQubitMapper | None = None,
    ) -> Hamiltonian:
        """Map the magnetization operator to a Hermitian qubit observable."""
        return map_fermionic_operator(
            self.to_fermionic_op(),
            mapper,
            name="magnetization observable",
        )


@dataclass(frozen=True, slots=True, eq=False)
class AngularMomentum:
    """Factory for total spin ``S²`` in restricted or UHF orbital bases.

    ``alpha_beta_overlap=None`` selects the identity overlap appropriate for a
    shared restricted orbital basis. UHF callers must supply the actual overlap
    between the independently optimized alpha and beta orbitals.

    Args:
        num_spatial_orbitals: Positive number of spatial orbitals per spin.
        alpha_beta_overlap: Optional overlap between alpha and beta orbitals.

    Raises:
        TypeError: If a value has the wrong type.
        ValueError: If the overlap dimensions or orbital count are invalid.
    """

    num_spatial_orbitals: int
    alpha_beta_overlap: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        validate_strict_int(
            self.num_spatial_orbitals,
            name="num_spatial_orbitals",
            minimum=1,
        )
        if self.alpha_beta_overlap is None:
            overlap = np.eye(self.num_spatial_orbitals, dtype=np.float64)
        else:
            overlap = immutable_real_array(
                self.alpha_beta_overlap,
                name="alpha_beta_overlap",
            )
            if overlap.shape != (self.num_spatial_orbitals,) * 2:
                raise ValueError("alpha_beta_overlap shape must be (n, n)")
        object.__setattr__(self, "alpha_beta_overlap", overlap)

    def to_fermionic_op(self) -> FermionicOp:
        """Return ``(S+S- + S-S+)/2 + Sz²`` without numerical truncation."""
        spin_z = _spin_z(self.num_spatial_orbitals)
        assert self.alpha_beta_overlap is not None
        spin_plus = _spin_plus(
            self.num_spatial_orbitals,
            self.alpha_beta_overlap,
        )
        spin_minus = spin_plus.adjoint()
        transverse = (spin_plus * spin_minus + spin_minus * spin_plus).scaled(0.5)
        return (transverse + spin_z * spin_z).normal_order()

    def to_qubit_hamiltonian(
        self,
        mapper: FermionToQubitMapper | None = None,
    ) -> Hamiltonian:
        """Map the total-spin operator to a Hermitian qubit observable."""
        return map_fermionic_operator(
            self.to_fermionic_op(),
            mapper,
            name="total-spin observable",
        )


__all__ = ["AngularMomentum", "Magnetization", "ParticleNumber"]
