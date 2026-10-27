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

"""Optional adapter for real restricted and unrestricted molecular integrals."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, Literal, cast

import numpy as np

from ..electronic.integrals import ElectronicIntegrals
from ..electronic.spin_resolved_integrals import SpinResolvedElectronicIntegrals
from .driver_data import (
    ElectronicStructureData,
    SpinResolvedElectronicStructureData,
)
from .specification import MolecularSpecification

SCFReference = Literal["auto", "rhf", "rohf", "uhf"]


def _load_pyscf() -> tuple[ModuleType, ModuleType, ModuleType]:
    try:
        return (
            import_module("pyscf.ao2mo"),
            import_module("pyscf.gto"),
            import_module("pyscf.scf"),
        )
    except ImportError as exc:
        raise ImportError(
            "PySCFDriver requires the optional 'chemistry' dependencies"
        ) from exc


class PySCFDriver:
    """Compute real restricted or unrestricted molecular-orbital integrals.

    The default ``auto`` reference preserves the restricted path: singlets use
    RHF and higher multiplicities use highest-weight ROHF. Explicit ``uhf``
    returns separate alpha and beta orbital bases and spin-resolved integrals.

    Args:
        reference: Self-consistent-field reference selection.

    Raises:
        TypeError: If ``reference`` is not a string.
        ValueError: If ``reference`` is unsupported.
    """

    __slots__ = ("_reference",)

    def __init__(self, *, reference: SCFReference = "auto") -> None:
        if not isinstance(reference, str):
            raise TypeError("reference must be a str")
        if reference not in ("auto", "rhf", "rohf", "uhf"):
            raise ValueError("reference must be 'auto', 'rhf', 'rohf', or 'uhf'")
        self._reference: SCFReference = reference

    @property
    def reference(self) -> SCFReference:
        """Return the selected self-consistent-field reference."""
        return self._reference

    def run(
        self,
        molecule: MolecularSpecification,
    ) -> ElectronicStructureData | SpinResolvedElectronicStructureData:
        """Run the selected SCF method and return independent integral data.

        Args:
            molecule: Molecular geometry, basis, charge, multiplicity, and unit.

        Returns:
            Immutable restricted or spin-resolved electronic-structure data.

        Raises:
            ImportError: If the optional chemistry dependencies are unavailable.
            TypeError: If ``molecule`` has the wrong type.
            ValueError: If the calculation does not produce valid real orbital
                data, or if RHF is requested for a non-singlet molecule.
            RuntimeError: If the SCF calculation does not converge.
        """
        if not isinstance(molecule, MolecularSpecification):
            raise TypeError("molecule must be a MolecularSpecification")

        ao2mo, gto, scf = _load_pyscf()
        geometry = [(atom.symbol, atom.coordinates) for atom in molecule.atoms]
        pyscf_molecule = cast(Any, gto).M(
            atom=geometry,
            basis=molecule.basis,
            charge=molecule.charge,
            spin=molecule.multiplicity - 1,
            unit="Angstrom" if molecule.unit == "angstrom" else "Bohr",
            verbose=0,
        )
        method = self._resolve_method(molecule)
        mean_field = getattr(cast(Any, scf), method)(pyscf_molecule)
        hartree_fock_energy = float(mean_field.kernel())
        if not bool(mean_field.converged):
            raise RuntimeError(f"{method} calculation did not converge")

        electron_counts = tuple(int(value) for value in pyscf_molecule.nelec)
        if len(electron_counts) != 2:
            raise ValueError(
                "the integral backend must return alpha/beta electron counts"
            )
        num_alpha, num_beta = electron_counts
        if num_alpha + num_beta < 1:
            raise ValueError("the molecule must contain at least one electron")
        if num_alpha - num_beta != molecule.multiplicity - 1:
            raise ValueError(
                "the integral backend electron counts do not match the multiplicity"
            )

        if method == "UHF":
            return self._build_unrestricted_data(
                cast(Any, ao2mo),
                pyscf_molecule,
                mean_field,
                molecule,
                num_alpha,
                num_beta,
                hartree_fock_energy,
            )
        return self._build_restricted_data(
            cast(Any, ao2mo),
            pyscf_molecule,
            mean_field,
            molecule,
            num_alpha,
            num_beta,
            hartree_fock_energy,
        )

    def _resolve_method(self, molecule: MolecularSpecification) -> str:
        if self._reference == "auto":
            return "RHF" if molecule.multiplicity == 1 else "ROHF"
        if self._reference == "rhf" and molecule.multiplicity != 1:
            raise ValueError("RHF reference requires molecular multiplicity 1")
        return self._reference.upper()

    @staticmethod
    def _build_restricted_data(
        ao2mo: Any,
        pyscf_molecule: Any,
        mean_field: Any,
        molecule: MolecularSpecification,
        num_alpha: int,
        num_beta: int,
        hartree_fock_energy: float,
    ) -> ElectronicStructureData:
        coefficients = np.asarray(mean_field.mo_coeff)
        if np.iscomplexobj(coefficients):
            raise ValueError("complex molecular orbitals are not supported")
        coefficients = np.asarray(coefficients, dtype=np.float64)
        if coefficients.ndim != 2:
            raise ValueError("molecular-orbital coefficients must be rank 2")
        orbital_energies = np.asarray(mean_field.mo_energy)
        occupations = np.asarray(mean_field.mo_occ)
        if np.iscomplexobj(orbital_energies) or np.iscomplexobj(occupations):
            raise ValueError("orbital energies and occupations must be real")
        orbital_energies = np.asarray(orbital_energies, dtype=np.float64)
        occupations = np.asarray(occupations, dtype=np.float64)
        num_orbitals = coefficients.shape[1]
        if orbital_energies.shape != (num_orbitals,):
            raise ValueError("orbital energies must match the molecular orbitals")
        if occupations.shape != (num_orbitals,):
            raise ValueError("orbital occupations must match the molecular orbitals")
        valid_occupations = np.isclose(
            occupations[:, None],
            np.array([0.0, 1.0, 2.0]),
            rtol=0.0,
            atol=1e-12,
        ).any(axis=1)
        if not np.all(valid_occupations):
            raise ValueError("restricted orbital occupations must be 0, 1, or 2")

        doubly_occupied = np.flatnonzero(
            np.isclose(occupations, 2.0, rtol=0.0, atol=1e-12)
        )
        singly_occupied = np.flatnonzero(
            np.isclose(occupations, 1.0, rtol=0.0, atol=1e-12)
        )
        virtual = np.flatnonzero(np.isclose(occupations, 0.0, rtol=0.0, atol=1e-12))
        if len(doubly_occupied) != num_beta:
            raise ValueError("restricted occupations disagree with num_beta")
        if len(singly_occupied) != num_alpha - num_beta:
            raise ValueError("restricted occupations disagree with num_alpha")
        order = np.concatenate((doubly_occupied, singly_occupied, virtual))
        if len(order) != num_orbitals or len(set(order.tolist())) != num_orbitals:
            raise ValueError("restricted occupations must classify every orbital")
        coefficients = coefficients[:, order]
        orbital_energies = orbital_energies[order]

        one_body = coefficients.T @ np.asarray(mean_field.get_hcore()) @ coefficients
        packed_eri = ao2mo.kernel(pyscf_molecule, coefficients)
        two_body = np.asarray(
            ao2mo.restore(1, packed_eri, num_orbitals),
            dtype=np.float64,
        )
        integrals = ElectronicIntegrals(
            constant=float(pyscf_molecule.energy_nuc()),
            one_body=one_body,
            two_body=two_body,
        )
        return ElectronicStructureData(
            molecule=molecule,
            num_alpha=num_alpha,
            num_beta=num_beta,
            orbital_energies=orbital_energies,
            molecular_orbitals=coefficients,
            hartree_fock_energy=hartree_fock_energy,
            integrals=integrals,
        )

    @staticmethod
    def _build_unrestricted_data(
        ao2mo: Any,
        pyscf_molecule: Any,
        mean_field: Any,
        molecule: MolecularSpecification,
        num_alpha: int,
        num_beta: int,
        hartree_fock_energy: float,
    ) -> SpinResolvedElectronicStructureData:
        coefficients = np.asarray(mean_field.mo_coeff)
        energies = np.asarray(mean_field.mo_energy)
        occupations = np.asarray(mean_field.mo_occ)
        if any(
            np.iscomplexobj(value) for value in (coefficients, energies, occupations)
        ):
            raise ValueError("UHF orbitals, energies, and occupations must be real")
        coefficients = np.asarray(coefficients, dtype=np.float64)
        energies = np.asarray(energies, dtype=np.float64)
        occupations = np.asarray(occupations, dtype=np.float64)
        if coefficients.ndim != 3 or coefficients.shape[0] != 2:
            raise ValueError(
                "UHF molecular-orbital coefficients must have shape (2, nao, n)"
            )
        num_orbitals = coefficients.shape[2]
        if coefficients.shape[1] < num_orbitals:
            raise ValueError("UHF molecular-orbital coefficient matrices are too short")
        if energies.shape != (2, num_orbitals):
            raise ValueError("UHF orbital energies must have shape (2, num_orbitals)")
        if occupations.shape != (2, num_orbitals):
            raise ValueError("UHF occupations must have shape (2, num_orbitals)")

        expected_counts = (num_alpha, num_beta)
        ordered_coefficients: list[np.ndarray] = []
        ordered_energies: list[np.ndarray] = []
        for spin, expected_count in enumerate(expected_counts):
            occupied = np.isclose(
                occupations[spin],
                1.0,
                rtol=0.0,
                atol=1e-12,
            )
            virtual = np.isclose(
                occupations[spin],
                0.0,
                rtol=0.0,
                atol=1e-12,
            )
            if not np.all(occupied | virtual):
                raise ValueError("UHF orbital occupations must be 0 or 1")
            occupied_indices = np.flatnonzero(occupied)
            virtual_indices = np.flatnonzero(virtual)
            if len(occupied_indices) != expected_count:
                label = "alpha" if spin == 0 else "beta"
                raise ValueError(
                    f"UHF {label} occupations disagree with electron count"
                )
            order = np.concatenate((occupied_indices, virtual_indices))
            ordered_coefficients.append(coefficients[spin][:, order])
            ordered_energies.append(energies[spin][order])

        alpha_coefficients, beta_coefficients = ordered_coefficients
        alpha_energies, beta_energies = ordered_energies
        core_source = np.asarray(mean_field.get_hcore())
        overlap_source = np.asarray(mean_field.get_ovlp())
        if np.iscomplexobj(core_source) or np.iscomplexobj(overlap_source):
            raise ValueError("UHF core and overlap matrices must be real")
        core = np.asarray(core_source, dtype=np.float64)
        overlap = np.asarray(overlap_source, dtype=np.float64)
        if core.shape != (coefficients.shape[1],) * 2:
            raise ValueError("AO core Hamiltonian shape must match UHF orbitals")
        if overlap.shape != core.shape:
            raise ValueError("AO overlap shape must match the core Hamiltonian")
        identity = np.eye(num_orbitals, dtype=np.float64)
        for label, orbital_coefficients in (
            ("alpha", alpha_coefficients),
            ("beta", beta_coefficients),
        ):
            metric = orbital_coefficients.T @ overlap @ orbital_coefficients
            if not np.allclose(metric, identity, rtol=0.0, atol=1e-8):
                raise ValueError(f"UHF {label} orbitals must be orthonormal")
        one_alpha = alpha_coefficients.T @ core @ alpha_coefficients
        one_beta = beta_coefficients.T @ core @ beta_coefficients
        alpha_beta_overlap = alpha_coefficients.T @ overlap @ beta_coefficients

        def transform(*orbital_sets: np.ndarray) -> np.ndarray:
            values = ao2mo.general(
                pyscf_molecule,
                orbital_sets,
                compact=False,
            )
            array = np.asarray(values)
            if np.iscomplexobj(array):
                raise ValueError("UHF two-electron integrals must be real")
            return np.asarray(array, dtype=np.float64).reshape((num_orbitals,) * 4)

        two_alpha = transform(
            alpha_coefficients,
            alpha_coefficients,
            alpha_coefficients,
            alpha_coefficients,
        )
        two_beta = transform(
            beta_coefficients,
            beta_coefficients,
            beta_coefficients,
            beta_coefficients,
        )
        two_alpha_beta = transform(
            alpha_coefficients,
            alpha_coefficients,
            beta_coefficients,
            beta_coefficients,
        )
        integrals = SpinResolvedElectronicIntegrals(
            constant=float(pyscf_molecule.energy_nuc()),
            one_body_alpha=one_alpha,
            one_body_beta=one_beta,
            two_body_alpha_alpha=two_alpha,
            two_body_beta_beta=two_beta,
            two_body_alpha_beta=two_alpha_beta,
        )
        return SpinResolvedElectronicStructureData(
            molecule=molecule,
            num_alpha=num_alpha,
            num_beta=num_beta,
            orbital_energies_alpha=alpha_energies,
            orbital_energies_beta=beta_energies,
            molecular_orbitals_alpha=alpha_coefficients,
            molecular_orbitals_beta=beta_coefficients,
            alpha_beta_overlap=alpha_beta_overlap,
            hartree_fock_energy=hartree_fock_energy,
            integrals=integrals,
        )


__all__ = ["PySCFDriver", "SCFReference"]
