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

"""Data, system, and adapter tests for the full-space UHF path."""

from types import SimpleNamespace

import numpy as np
import pytest

from cqlib_algorithms.extension.chemistry import (
    ActiveSpace,
    Atom,
    ElectronicOccupation,
    ElectronicSystem,
    MolecularSpecification,
    PySCFDriver,
    SpinResolvedActiveSpace,
    SpinResolvedElectronicIntegrals,
    SpinResolvedElectronicStructureData,
    hartree_fock_initial_point,
    mp2_initial_point,
)
from cqlib_algorithms.extension.chemistry.molecular import (
    pyscf_adapter as driver_module,
)


def _molecule() -> MolecularSpecification:
    return MolecularSpecification(
        atoms=(Atom("H", (0.0, 0.0, 0.0)), Atom("H", (0.0, 0.0, 0.8))),
        basis="minimal",
    )


def _same_spin_tensor(seed: int) -> np.ndarray:
    raw = np.random.default_rng(seed).normal(size=(2,) * 4)
    return (
        sum(
            (
                raw,
                raw.transpose(1, 0, 2, 3),
                raw.transpose(0, 1, 3, 2),
                raw.transpose(1, 0, 3, 2),
                raw.transpose(2, 3, 0, 1),
                raw.transpose(3, 2, 0, 1),
                raw.transpose(2, 3, 1, 0),
                raw.transpose(3, 2, 1, 0),
            )
        )
        / 8.0
    )


def _mixed_tensor() -> np.ndarray:
    raw = np.random.default_rng(702).normal(size=(2,) * 4)
    return (
        raw
        + raw.transpose(1, 0, 2, 3)
        + raw.transpose(0, 1, 3, 2)
        + raw.transpose(1, 0, 3, 2)
    ) / 4.0


def _integrals() -> SpinResolvedElectronicIntegrals:
    return SpinResolvedElectronicIntegrals(
        0.6,
        np.array([[-1.1, 0.2], [0.2, 0.4]]),
        np.array([[-0.9, -0.1], [-0.1, 0.5]]),
        _same_spin_tensor(700),
        _same_spin_tensor(701),
        _mixed_tensor(),
    )


def _data() -> SpinResolvedElectronicStructureData:
    angle = 0.23
    overlap = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return SpinResolvedElectronicStructureData(
        molecule=_molecule(),
        num_alpha=1,
        num_beta=1,
        orbital_energies_alpha=np.array([-0.8, 0.3]),
        orbital_energies_beta=np.array([-0.7, 0.4]),
        molecular_orbitals_alpha=np.eye(2),
        molecular_orbitals_beta=overlap,
        alpha_beta_overlap=overlap,
        hartree_fock_energy=-1.0,
        integrals=_integrals(),
    )


def test_spin_resolved_driver_data_and_system_are_immutable_snapshots():
    data = _data()
    system = ElectronicSystem.from_data(data)

    assert data.num_electrons == 2
    assert data.is_closed_shell
    assert system.is_spin_resolved
    assert system.occupation == ElectronicOccupation(2, 1, 1)
    assert system.orbital_energies_alpha is system.orbital_energies
    np.testing.assert_array_equal(system.orbital_energies, [-0.8, 0.3])
    np.testing.assert_array_equal(system.orbital_energies_beta, [-0.7, 0.4])
    np.testing.assert_array_equal(system.alpha_beta_overlap, data.alpha_beta_overlap)
    assert not system.alpha_beta_overlap.flags.writeable
    assert system.nuclear_repulsion_energy == data.integrals.constant
    assert system.inactive_energy == 0.0


def test_spin_resolved_system_supports_paired_and_independent_active_spaces():
    data = _data()
    active_space = ActiveSpace.full(data.num_spatial_orbitals)
    paired = ElectronicSystem.from_data(data, active_space)
    assert paired.is_spin_resolved
    np.testing.assert_array_equal(
        paired.integrals.one_body_alpha,
        data.integrals.one_body_alpha,
    )

    independent = ElectronicSystem.from_data(
        data,
        SpinResolvedActiveSpace.manual(
            frozen_beta_orbitals=(0,),
            active_alpha_orbitals=(0,),
            active_beta_orbitals=(1,),
        ),
    )
    assert independent.occupation == ElectronicOccupation(1, 1, 0)
    np.testing.assert_array_equal(independent.orbital_energies, [-0.8])
    np.testing.assert_array_equal(independent.orbital_energies_beta, [0.4])
    np.testing.assert_array_equal(
        independent.alpha_beta_overlap,
        data.alpha_beta_overlap[np.ix_([0], [1])],
    )


def test_spin_resolved_system_still_rejects_mp2_initialization():
    data = _data()

    system = ElectronicSystem.from_data(data)
    assert set(hartree_fock_initial_point(system)) == {
        "ucc_theta_0",
        "ucc_theta_1",
        "ucc_theta_2",
    }
    with pytest.raises(ValueError, match="spin-resolved"):
        mp2_initial_point(system)


def test_spin_resolved_system_requires_consistent_metadata():
    integrals = _integrals()
    occupation = ElectronicOccupation(2, 1, 1)
    with pytest.raises(ValueError, match="alpha_beta_overlap"):
        ElectronicSystem(integrals, occupation)
    with pytest.raises(ValueError, match="provided together"):
        ElectronicSystem(
            integrals,
            occupation,
            orbital_energies=np.zeros(2),
            alpha_beta_overlap=np.eye(2),
        )
    with pytest.raises(ValueError, match="shape"):
        ElectronicSystem(
            integrals,
            occupation,
            alpha_beta_overlap=np.eye(3),
        )
    with pytest.raises(ValueError, match="provided together"):
        ElectronicSystem(
            integrals,
            occupation,
            alpha_beta_overlap=np.eye(2),
            nuclear_repulsion_energy=integrals.constant,
        )
    with pytest.raises(ValueError, match="must equal"):
        ElectronicSystem(
            integrals,
            occupation,
            alpha_beta_overlap=np.eye(2),
            nuclear_repulsion_energy=0.1,
            inactive_energy=0.2,
        )


def test_driver_selects_uhf_reorders_each_spin_and_transforms_all_blocks(
    monkeypatch,
):
    angle = 0.31
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    alpha_source = np.eye(2)
    beta_source = rotation
    ao_core = np.array([[-1.0, 0.15], [0.15, 0.2]])
    ao_eri = np.arange(16, dtype=np.float64).reshape((2,) * 4)
    ao_eri = (
        sum(
            (
                ao_eri,
                ao_eri.transpose(1, 0, 2, 3),
                ao_eri.transpose(0, 1, 3, 2),
                ao_eri.transpose(1, 0, 3, 2),
                ao_eri.transpose(2, 3, 0, 1),
                ao_eri.transpose(3, 2, 0, 1),
                ao_eri.transpose(2, 3, 1, 0),
                ao_eri.transpose(3, 2, 1, 0),
            )
        )
        / 8.0
    )
    backend_molecule = SimpleNamespace(nelec=(1, 1), energy_nuc=lambda: 0.7)
    mean_field = SimpleNamespace(
        converged=True,
        kernel=lambda: -1.2,
        mo_coeff=np.stack((alpha_source, beta_source)),
        mo_energy=np.array([[0.4, -0.8], [-0.7, 0.5]]),
        mo_occ=np.array([[0.0, 1.0], [1.0, 0.0]]),
        get_hcore=lambda: ao_core,
        get_ovlp=lambda: np.eye(2),
    )
    calls = []

    def general(_molecule, orbital_sets, *, compact):
        assert compact is False
        calls.append(tuple(np.array(value, copy=True) for value in orbital_sets))
        first, second, third, fourth = orbital_sets
        return np.einsum(
            "ip,jq,kr,ls,ijkl->pqrs",
            first,
            second,
            third,
            fourth,
            ao_eri,
            optimize=True,
        )

    scf = SimpleNamespace(
        UHF=lambda value: mean_field if value is backend_molecule else None
    )
    gto = SimpleNamespace(M=lambda **kwargs: backend_molecule)
    ao2mo = SimpleNamespace(general=general)
    monkeypatch.setattr(driver_module, "_load_pyscf", lambda: (ao2mo, gto, scf))

    result = PySCFDriver(reference="uhf").run(_molecule())

    assert isinstance(result, SpinResolvedElectronicStructureData)
    expected_alpha = alpha_source[:, [1, 0]]
    expected_beta = beta_source
    np.testing.assert_allclose(result.molecular_orbitals_alpha, expected_alpha)
    np.testing.assert_allclose(result.molecular_orbitals_beta, expected_beta)
    np.testing.assert_array_equal(result.orbital_energies_alpha, [-0.8, 0.4])
    np.testing.assert_array_equal(result.orbital_energies_beta, [-0.7, 0.5])
    np.testing.assert_allclose(
        result.alpha_beta_overlap,
        expected_alpha.T @ expected_beta,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.integrals.one_body_alpha,
        expected_alpha.T @ ao_core @ expected_alpha,
    )
    np.testing.assert_allclose(
        result.integrals.one_body_beta,
        expected_beta.T @ ao_core @ expected_beta,
    )
    assert len(calls) == 3
    for actual, expected in zip(calls[0], (expected_alpha,) * 4, strict=True):
        np.testing.assert_allclose(actual, expected)
    for actual, expected in zip(calls[1], (expected_beta,) * 4, strict=True):
        np.testing.assert_allclose(actual, expected)
    for actual, expected in zip(
        calls[2],
        (expected_alpha, expected_alpha, expected_beta, expected_beta),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("reference", ["invalid", "UHF", ""])
def test_driver_rejects_unknown_reference(reference):
    with pytest.raises(ValueError, match="reference"):
        PySCFDriver(reference=reference)  # type: ignore[arg-type]


def test_driver_rejects_rhf_for_non_singlet_before_scf_execution(monkeypatch):
    molecule = MolecularSpecification(
        atoms=(Atom("H", (0.0, 0.0, 0.0)),),
        basis="minimal",
        multiplicity=2,
    )
    backend_molecule = SimpleNamespace()
    gto = SimpleNamespace(M=lambda **kwargs: backend_molecule)

    def unexpected_rhf(_molecule):
        raise AssertionError("RHF must not run for a non-singlet molecule")

    monkeypatch.setattr(
        driver_module,
        "_load_pyscf",
        lambda: (SimpleNamespace(), gto, SimpleNamespace(RHF=unexpected_rhf)),
    )
    with pytest.raises(ValueError, match="multiplicity 1"):
        PySCFDriver(reference="rhf").run(molecule)
