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

"""Boundary tests for the optional molecular integral driver."""

from types import SimpleNamespace

import numpy as np
import pytest

from cqlib_algorithms.extension.chemistry import (
    Atom,
    MolecularSpecification,
    PySCFDriver,
)
from cqlib_algorithms.extension.chemistry.molecular import (
    pyscf_adapter as driver_module,
)


def _hydrogen_molecule(*, multiplicity=1):
    return MolecularSpecification(
        atoms=(Atom("H", (0.0, 0.0, 0.0)), Atom("H", (0.0, 0.0, 0.7))),
        basis="sto-3g",
        multiplicity=multiplicity,
    )


def _helium_hydride_doublet():
    return MolecularSpecification(
        atoms=(Atom("He", (0.0, 0.0, 0.0)), Atom("H", (0.0, 0.0, 1.4))),
        basis="sto-3g",
        multiplicity=2,
    )


def test_driver_import_is_lazy(monkeypatch):
    def unavailable():
        raise ImportError("optional dependency missing")

    monkeypatch.setattr(driver_module, "_load_pyscf", unavailable)
    with pytest.raises(ImportError, match="optional dependency missing"):
        PySCFDriver().run(_hydrogen_molecule())


@pytest.mark.parametrize(
    ("multiplicity", "selected_method"),
    [(1, "RHF"), (3, "ROHF")],
)
def test_driver_selects_restricted_method_and_reports_nonconvergence(
    monkeypatch,
    multiplicity,
    selected_method,
):
    molecule = SimpleNamespace()
    mean_field = SimpleNamespace(converged=False, kernel=lambda: -1.0)
    gto = SimpleNamespace(M=lambda **kwargs: molecule)
    calls = []

    def rhf(value):
        calls.append(("RHF", value))
        return mean_field

    def rohf(value):
        calls.append(("ROHF", value))
        return mean_field

    scf = SimpleNamespace(RHF=rhf, ROHF=rohf)
    ao2mo = SimpleNamespace()
    monkeypatch.setattr(driver_module, "_load_pyscf", lambda: (ao2mo, gto, scf))

    with pytest.raises(RuntimeError, match=selected_method):
        PySCFDriver().run(_hydrogen_molecule(multiplicity=multiplicity))
    assert calls == [(selected_method, molecule)]


def test_rohf_driver_reorders_common_orbitals_and_preserves_spin_counts(monkeypatch):
    captured = {}
    molecule = SimpleNamespace(nelec=(2, 1), energy_nuc=lambda: 0.7)
    mean_field = SimpleNamespace(
        converged=True,
        kernel=lambda: -2.1,
        mo_coeff=np.eye(3),
        mo_energy=np.array([0.5, -0.2, -1.0]),
        mo_occ=np.array([0.0, 1.0, 2.0]),
        get_hcore=lambda: np.diag([0.5, -0.2, -1.0]),
        get_ovlp=lambda: np.eye(3),
    )

    def build_molecule(**kwargs):
        captured["molecule_kwargs"] = kwargs
        return molecule

    def rohf(value):
        captured["rohf_input"] = value
        return mean_field

    def rhf(value):
        raise AssertionError(f"RHF must not be used for {value!r}")

    def transform_eri(value, coefficients):
        captured["eri_molecule"] = value
        captured["eri_coefficients"] = np.array(coefficients, copy=True)
        return object()

    def restore(symmetry, packed, num_orbitals):
        assert symmetry == 1
        assert num_orbitals == 3
        return np.zeros((3,) * 4)

    gto = SimpleNamespace(M=build_molecule)
    scf = SimpleNamespace(RHF=rhf, ROHF=rohf)
    ao2mo = SimpleNamespace(kernel=transform_eri, restore=restore)
    monkeypatch.setattr(driver_module, "_load_pyscf", lambda: (ao2mo, gto, scf))

    result = PySCFDriver().run(_helium_hydride_doublet())

    assert captured["molecule_kwargs"]["spin"] == 1
    assert captured["rohf_input"] is molecule
    assert captured["eri_molecule"] is molecule
    expected_coefficients = np.eye(3)[:, [2, 1, 0]]
    np.testing.assert_array_equal(captured["eri_coefficients"], expected_coefficients)
    np.testing.assert_array_equal(result.molecular_orbitals, expected_coefficients)
    np.testing.assert_array_equal(result.orbital_energies, [-1.0, -0.2, 0.5])
    np.testing.assert_array_equal(result.integrals.one_body, np.diag([-1.0, -0.2, 0.5]))
    assert (result.num_alpha, result.num_beta) == (2, 1)
    assert not result.is_closed_shell


def test_rohf_driver_rejects_occupation_count_inconsistent_with_backend(monkeypatch):
    molecule = SimpleNamespace(nelec=(2, 1), energy_nuc=lambda: 0.0)
    mean_field = SimpleNamespace(
        converged=True,
        kernel=lambda: -1.0,
        mo_coeff=np.eye(2),
        mo_energy=np.array([-1.0, 0.2]),
        mo_occ=np.array([2.0, 0.0]),
    )
    gto = SimpleNamespace(M=lambda **kwargs: molecule)
    scf = SimpleNamespace(ROHF=lambda value: mean_field)
    monkeypatch.setattr(
        driver_module,
        "_load_pyscf",
        lambda: (SimpleNamespace(), gto, scf),
    )

    with pytest.raises(ValueError, match="num_alpha"):
        PySCFDriver().run(_helium_hydride_doublet())


def test_restricted_driver_rejects_nonorthonormal_orbitals(monkeypatch):
    molecule = SimpleNamespace(nelec=(1, 1), energy_nuc=lambda: 0.0)
    mean_field = SimpleNamespace(
        converged=True,
        kernel=lambda: -1.0,
        mo_coeff=np.eye(2),
        mo_energy=np.array([-1.0, 0.2]),
        mo_occ=np.array([2.0, 0.0]),
        get_hcore=lambda: np.eye(2),
        get_ovlp=lambda: 2.0 * np.eye(2),
    )
    gto = SimpleNamespace(M=lambda **kwargs: molecule)
    scf = SimpleNamespace(RHF=lambda value: mean_field)
    monkeypatch.setattr(
        driver_module,
        "_load_pyscf",
        lambda: (SimpleNamespace(), gto, scf),
    )

    with pytest.raises(ValueError, match="restricted orbitals must be orthonormal"):
        PySCFDriver().run(_hydrogen_molecule())


def test_driver_rejects_wrong_input_type():
    with pytest.raises(TypeError, match="MolecularSpecification"):
        PySCFDriver().run(object())  # type: ignore[arg-type]
