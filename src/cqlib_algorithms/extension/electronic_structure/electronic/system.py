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

"""Validated electronic-system input for electronic-structure algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .._numeric import immutable_real_array, validate_finite_real
from .integrals import ElectronicIntegrals
from .occupation import ElectronicOccupation
from .spin_resolved_integrals import SpinResolvedElectronicIntegrals

if TYPE_CHECKING:
    from ..molecular.active_space import ActiveSpace, SpinResolvedActiveSpace
    from ..molecular.driver_data import (
        ElectronicStructureData,
        SpinResolvedElectronicStructureData,
    )


@dataclass(frozen=True, slots=True, eq=False)
class ElectronicSystem:
    """Electronic Hamiltonian data, reference occupation and orbital metadata.

    Integrals use real orthonormal restricted orbitals or separate orthonormal
    alpha/beta UHF orbitals, with interleaved spin modes. ``orbital_energies``
    is the shared restricted vector or the UHF alpha vector;
    ``orbital_energies_beta`` is present only for UHF. Orbital energies are
    optional because exact Hamiltonian mapping and Hartree--Fock initialization
    do not require canonical energies; perturbative initialization does.

    Orthonormality is an input contract of manually constructed integral
    objects. Systems created from :class:`ElectronicStructureData` inherit the
    validation performed by the originating electronic-structure driver.

    Args:
        integrals: Restricted or spin-resolved electronic integrals.
        occupation: Reference alpha/beta occupation.
        orbital_energies: Restricted or alpha orbital energies.
        hartree_fock_energy: Optional mean-field total energy.
        orbital_energies_beta: Optional beta energies for spin-resolved data.
        alpha_beta_overlap: Optional alpha/beta orbital overlap matrix.
        nuclear_repulsion_energy: Optional nuclear repulsion contribution.
        inactive_energy: Optional frozen-orbital energy contribution.

    Raises:
        TypeError: If an input has the wrong type.
        ValueError: If dimensions, spin metadata, or energy components disagree.
    """

    integrals: ElectronicIntegrals | SpinResolvedElectronicIntegrals
    occupation: ElectronicOccupation
    orbital_energies: NDArray[np.float64] | None = None
    hartree_fock_energy: float | None = None
    orbital_energies_beta: NDArray[np.float64] | None = None
    alpha_beta_overlap: NDArray[np.float64] | None = None
    nuclear_repulsion_energy: float | None = None
    inactive_energy: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.integrals,
            (ElectronicIntegrals, SpinResolvedElectronicIntegrals),
        ):
            raise TypeError(
                "integrals must be ElectronicIntegrals or "
                "SpinResolvedElectronicIntegrals"
            )
        if not isinstance(self.occupation, ElectronicOccupation):
            raise TypeError("occupation must be an ElectronicOccupation")
        if self.integrals.num_spatial_orbitals != self.occupation.num_spatial_orbitals:
            raise ValueError("integrals and occupation must use the same orbital count")

        if self.orbital_energies is not None:
            energies = immutable_real_array(
                self.orbital_energies,
                name="orbital_energies",
            )
            if energies.shape != (self.integrals.num_spatial_orbitals,):
                raise ValueError("orbital_energies shape must be (num_orbitals,)")
            object.__setattr__(self, "orbital_energies", energies)

        if self.is_spin_resolved:
            if (self.orbital_energies is None) != (self.orbital_energies_beta is None):
                raise ValueError(
                    "spin-resolved alpha and beta orbital energies must be "
                    "provided together"
                )
            if self.orbital_energies_beta is not None:
                beta_energies = immutable_real_array(
                    self.orbital_energies_beta,
                    name="orbital_energies_beta",
                )
                if beta_energies.shape != (self.integrals.num_spatial_orbitals,):
                    raise ValueError(
                        "orbital_energies_beta shape must be (num_orbitals,)"
                    )
                object.__setattr__(
                    self,
                    "orbital_energies_beta",
                    beta_energies,
                )
            if self.alpha_beta_overlap is None:
                raise ValueError("spin-resolved systems require alpha_beta_overlap")
            overlap = immutable_real_array(
                self.alpha_beta_overlap,
                name="alpha_beta_overlap",
            )
            expected_shape = (self.integrals.num_spatial_orbitals,) * 2
            if overlap.shape != expected_shape:
                raise ValueError("alpha_beta_overlap shape must be (n, n)")
            object.__setattr__(self, "alpha_beta_overlap", overlap)
        elif self.orbital_energies_beta is not None:
            raise ValueError("orbital_energies_beta requires spin-resolved integrals")
        elif self.alpha_beta_overlap is not None:
            raise ValueError("alpha_beta_overlap requires spin-resolved integrals")

        if self.hartree_fock_energy is not None:
            validate_finite_real(
                self.hartree_fock_energy,
                name="hartree_fock_energy",
            )
            object.__setattr__(
                self,
                "hartree_fock_energy",
                float(self.hartree_fock_energy),
            )

        if (self.nuclear_repulsion_energy is None) != (self.inactive_energy is None):
            raise ValueError(
                "nuclear_repulsion_energy and inactive_energy must be provided together"
            )
        if self.nuclear_repulsion_energy is not None:
            assert self.inactive_energy is not None
            validate_finite_real(
                self.nuclear_repulsion_energy,
                name="nuclear_repulsion_energy",
            )
            validate_finite_real(self.inactive_energy, name="inactive_energy")
            nuclear_repulsion = float(self.nuclear_repulsion_energy)
            inactive = float(self.inactive_energy)
            if not np.isclose(
                nuclear_repulsion + inactive,
                self.integrals.constant,
                rtol=0.0,
                atol=1e-10,
            ):
                raise ValueError(
                    "nuclear_repulsion_energy + inactive_energy must equal "
                    "the integral constant"
                )
            object.__setattr__(
                self,
                "nuclear_repulsion_energy",
                nuclear_repulsion,
            )
            object.__setattr__(self, "inactive_energy", inactive)

    @classmethod
    def from_data(
        cls,
        data: ElectronicStructureData | SpinResolvedElectronicStructureData,
        active_space: ActiveSpace | SpinResolvedActiveSpace | None = None,
    ) -> ElectronicSystem:
        """Build a restricted or spin-resolved full/active-space system.

        Args:
            data: Backend-independent electronic-structure driver data.
            active_space: Optional restricted or spin-resolved selection.

        Returns:
            Validated electronic system with transformed metadata when needed.

        Raises:
            TypeError: If the data and active-space types are incompatible.
            ValueError: If the active-space selection is physically invalid.
        """
        from ..molecular.active_space import (
            ActiveSpace,
            SpinResolvedActiveSpace,
            transform_active_space,
            transform_spin_resolved_active_space,
        )
        from ..molecular.driver_data import (
            ElectronicStructureData,
            SpinResolvedElectronicStructureData,
        )

        if not isinstance(
            data,
            (ElectronicStructureData, SpinResolvedElectronicStructureData),
        ):
            raise TypeError(
                "data must be ElectronicStructureData or "
                "SpinResolvedElectronicStructureData"
            )
        occupation = ElectronicOccupation(
            data.num_spatial_orbitals,
            data.num_alpha,
            data.num_beta,
        )
        if isinstance(data, SpinResolvedElectronicStructureData):
            if active_space is None:
                return cls(
                    data.integrals,
                    occupation,
                    data.orbital_energies_alpha,
                    data.hartree_fock_energy,
                    data.orbital_energies_beta,
                    data.alpha_beta_overlap,
                    data.integrals.constant,
                    0.0,
                )
            if isinstance(active_space, ActiveSpace):
                spin_selection = SpinResolvedActiveSpace.paired(
                    frozen_orbitals=active_space.frozen_orbitals,
                    active_orbitals=active_space.active_orbitals,
                )
            elif isinstance(active_space, SpinResolvedActiveSpace):
                spin_selection = active_space
            else:
                raise TypeError(
                    "active_space must be an ActiveSpace, "
                    "SpinResolvedActiveSpace, or None"
                )
            transformed_spin = transform_spin_resolved_active_space(
                data.integrals,
                occupation,
                spin_selection,
            )
            alpha_indices = np.asarray(
                transformed_spin.active_alpha_orbitals,
                dtype=np.intp,
            )
            beta_indices = np.asarray(
                transformed_spin.active_beta_orbitals,
                dtype=np.intp,
            )
            return cls(
                transformed_spin.integrals,
                transformed_spin.occupation,
                data.orbital_energies_alpha[alpha_indices],
                data.hartree_fock_energy,
                data.orbital_energies_beta[beta_indices],
                data.alpha_beta_overlap[np.ix_(alpha_indices, beta_indices)],
                data.integrals.constant,
                transformed_spin.integrals.constant - data.integrals.constant,
            )
        if active_space is None:
            return cls(
                data.integrals,
                occupation,
                data.orbital_energies,
                data.hartree_fock_energy,
                nuclear_repulsion_energy=data.integrals.constant,
                inactive_energy=0.0,
            )
        if not isinstance(active_space, ActiveSpace):
            raise TypeError(
                "active_space for restricted data must be an ActiveSpace or None"
            )

        transformed = transform_active_space(
            data.integrals,
            occupation,
            active_space,
        )
        active_indices = np.asarray(transformed.active_orbitals, dtype=np.intp)
        return cls(
            transformed.integrals,
            transformed.occupation,
            data.orbital_energies[active_indices],
            data.hartree_fock_energy,
            nuclear_repulsion_energy=data.integrals.constant,
            inactive_energy=transformed.integrals.constant - data.integrals.constant,
        )

    @property
    def num_spatial_orbitals(self) -> int:
        """Return the number of spatial orbitals."""
        return self.integrals.num_spatial_orbitals

    @property
    def num_spin_orbitals(self) -> int:
        """Return the number of interleaved spin-orbital modes."""
        return self.occupation.num_spin_orbitals

    @property
    def num_electrons(self) -> int:
        """Return the reference electron count."""
        return self.occupation.num_electrons

    @property
    def is_spin_resolved(self) -> bool:
        """Return whether alpha and beta use separate orbital bases."""
        return isinstance(self.integrals, SpinResolvedElectronicIntegrals)

    @property
    def orbital_energies_alpha(self) -> NDArray[np.float64] | None:
        """Return alpha or shared restricted orbital energies."""
        return self.orbital_energies


__all__ = ["ElectronicSystem"]
