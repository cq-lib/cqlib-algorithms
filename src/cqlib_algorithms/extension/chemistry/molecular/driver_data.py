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

"""Backend-independent restricted-orbital electronic-structure input data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..electronic.integrals import ElectronicIntegrals
from ..electronic.spin_resolved_integrals import SpinResolvedElectronicIntegrals
from ..validation.numerical import (
    immutable_real_array,
    validate_finite_real,
    validate_strict_int,
)
from .specification import MolecularSpecification


@dataclass(frozen=True, slots=True, eq=False)
class ElectronicStructureData:
    """Backend-independent RHF or ROHF driver-data snapshot.

    Alpha and beta electrons share one real orthonormal spatial-orbital basis.
    The reference uses the highest-weight spin projection, so
    ``num_alpha - num_beta = multiplicity - 1``.

    Args:
        molecule: Molecular input used to generate this snapshot.
        num_alpha: Number of alpha electrons.
        num_beta: Number of beta electrons.
        orbital_energies: Shared spatial-orbital energy vector.
        molecular_orbitals: Atomic-to-molecular orbital coefficient matrix.
        hartree_fock_energy: Converged mean-field total energy.
        integrals: Restricted molecular-orbital integrals.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If electron counts, dimensions, or multiplicity disagree.
    """

    molecule: MolecularSpecification
    num_alpha: int
    num_beta: int
    orbital_energies: NDArray[np.float64]
    molecular_orbitals: NDArray[np.float64]
    hartree_fock_energy: float
    integrals: ElectronicIntegrals

    def __post_init__(self) -> None:
        if not isinstance(self.molecule, MolecularSpecification):
            raise TypeError("molecule must be a MolecularSpecification")
        validate_strict_int(self.num_alpha, name="num_alpha")
        validate_strict_int(self.num_beta, name="num_beta")
        if self.num_alpha + self.num_beta < 1:
            raise ValueError(
                "the electronic structure must contain at least one electron"
            )
        if self.num_alpha - self.num_beta != self.molecule.multiplicity - 1:
            raise ValueError(
                "num_alpha - num_beta must equal molecular multiplicity - 1"
            )
        validate_finite_real(self.hartree_fock_energy, name="hartree_fock_energy")
        if not isinstance(self.integrals, ElectronicIntegrals):
            raise TypeError("integrals must be ElectronicIntegrals")
        object.__setattr__(
            self,
            "orbital_energies",
            immutable_real_array(self.orbital_energies, name="orbital_energies"),
        )
        object.__setattr__(
            self,
            "molecular_orbitals",
            immutable_real_array(self.molecular_orbitals, name="molecular_orbitals"),
        )
        object.__setattr__(self, "hartree_fock_energy", float(self.hartree_fock_energy))
        num_orbitals = self.integrals.num_spatial_orbitals
        if self.num_alpha > num_orbitals:
            raise ValueError("num_alpha exceeds the spatial-orbital capacity")
        if self.num_beta > num_orbitals:
            raise ValueError("num_beta exceeds the spatial-orbital capacity")
        if self.orbital_energies.shape != (num_orbitals,):
            raise ValueError("orbital_energies shape must be (num_orbitals,)")
        if self.molecular_orbitals.ndim != 2:
            raise ValueError("molecular_orbitals must be a rank-2 array")
        if self.molecular_orbitals.shape[1] != num_orbitals:
            raise ValueError(
                "molecular_orbitals columns must match the number of orbitals"
            )

    @property
    def num_spatial_orbitals(self) -> int:
        """Return the number of molecular spatial orbitals."""
        return self.integrals.num_spatial_orbitals

    @property
    def num_electrons(self) -> int:
        """Return the total electron count."""
        return self.num_alpha + self.num_beta

    @property
    def is_closed_shell(self) -> bool:
        """Return whether the restricted reference is closed shell."""
        return self.num_alpha == self.num_beta


@dataclass(frozen=True, slots=True, eq=False)
class SpinResolvedElectronicStructureData:
    """Backend-independent UHF driver-data snapshot.

    Alpha and beta electrons occupy separate real orthonormal orbital bases.
    ``alpha_beta_overlap[p, q]`` stores the overlap between alpha orbital
    ``p`` and beta orbital ``q`` in the common atomic-orbital metric.

    Args:
        molecule: Molecular input used to generate this snapshot.
        num_alpha: Number of alpha electrons.
        num_beta: Number of beta electrons.
        orbital_energies_alpha: Alpha orbital energies.
        orbital_energies_beta: Beta orbital energies.
        molecular_orbitals_alpha: Alpha orbital coefficient matrix.
        molecular_orbitals_beta: Beta orbital coefficient matrix.
        alpha_beta_overlap: Cross-spin orbital overlap matrix.
        hartree_fock_energy: Converged mean-field total energy.
        integrals: Spin-resolved molecular-orbital integrals.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If electron counts, dimensions, or multiplicity disagree.
    """

    molecule: MolecularSpecification
    num_alpha: int
    num_beta: int
    orbital_energies_alpha: NDArray[np.float64]
    orbital_energies_beta: NDArray[np.float64]
    molecular_orbitals_alpha: NDArray[np.float64]
    molecular_orbitals_beta: NDArray[np.float64]
    alpha_beta_overlap: NDArray[np.float64]
    hartree_fock_energy: float
    integrals: SpinResolvedElectronicIntegrals

    def __post_init__(self) -> None:
        if not isinstance(self.molecule, MolecularSpecification):
            raise TypeError("molecule must be a MolecularSpecification")
        validate_strict_int(self.num_alpha, name="num_alpha")
        validate_strict_int(self.num_beta, name="num_beta")
        if self.num_alpha + self.num_beta < 1:
            raise ValueError(
                "the electronic structure must contain at least one electron"
            )
        if self.num_alpha - self.num_beta != self.molecule.multiplicity - 1:
            raise ValueError(
                "num_alpha - num_beta must equal molecular multiplicity - 1"
            )
        validate_finite_real(self.hartree_fock_energy, name="hartree_fock_energy")
        if not isinstance(self.integrals, SpinResolvedElectronicIntegrals):
            raise TypeError("integrals must be SpinResolvedElectronicIntegrals")

        for name in ("orbital_energies_alpha", "orbital_energies_beta"):
            object.__setattr__(
                self,
                name,
                immutable_real_array(getattr(self, name), name=name),
            )
        for name in ("molecular_orbitals_alpha", "molecular_orbitals_beta"):
            object.__setattr__(
                self,
                name,
                immutable_real_array(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "alpha_beta_overlap",
            immutable_real_array(
                self.alpha_beta_overlap,
                name="alpha_beta_overlap",
            ),
        )
        object.__setattr__(self, "hartree_fock_energy", float(self.hartree_fock_energy))

        num_orbitals = self.integrals.num_spatial_orbitals
        if self.num_alpha > num_orbitals:
            raise ValueError("num_alpha exceeds the alpha-orbital capacity")
        if self.num_beta > num_orbitals:
            raise ValueError("num_beta exceeds the beta-orbital capacity")
        expected_vector = (num_orbitals,)
        if self.orbital_energies_alpha.shape != expected_vector:
            raise ValueError("orbital_energies_alpha shape must be (num_orbitals,)")
        if self.orbital_energies_beta.shape != expected_vector:
            raise ValueError("orbital_energies_beta shape must be (num_orbitals,)")
        for name in ("molecular_orbitals_alpha", "molecular_orbitals_beta"):
            orbitals = getattr(self, name)
            if orbitals.ndim != 2:
                raise ValueError(f"{name} must be a rank-2 array")
            if orbitals.shape[1] != num_orbitals:
                raise ValueError(f"{name} columns must match the number of orbitals")
        if (
            self.molecular_orbitals_alpha.shape[0]
            != self.molecular_orbitals_beta.shape[0]
        ):
            raise ValueError("alpha and beta orbitals must use the same AO basis")
        if self.alpha_beta_overlap.shape != (num_orbitals, num_orbitals):
            raise ValueError("alpha_beta_overlap shape must be (n, n)")

    @property
    def num_spatial_orbitals(self) -> int:
        """Return the number of orbitals in each spin sector."""
        return self.integrals.num_spatial_orbitals

    @property
    def num_electrons(self) -> int:
        """Return the total electron count."""
        return self.num_alpha + self.num_beta

    @property
    def is_closed_shell(self) -> bool:
        """Return whether the alpha and beta electron counts are equal."""
        return self.num_alpha == self.num_beta


__all__ = ["ElectronicStructureData", "SpinResolvedElectronicStructureData"]
