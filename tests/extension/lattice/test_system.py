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

"""Tests for conserved lattice sectors and the unified system input."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from cqlib_algorithms.extension.lattice import (
    FermiHubbardModel,
    FermionicSector,
    HeisenbergModel,
    IsingModel,
    Lattice,
    LatticeModelKind,
    LatticeSystem,
    LineLattice,
    SpinMagnetizationSector,
)

I = np.eye(2, dtype=np.complex128)
Z = np.diag([1.0, -1.0]).astype(np.complex128)


def _on_site(matrix: np.ndarray, site: int, num_sites: int) -> np.ndarray:
    factors = [I] * num_sites
    factors[site] = matrix
    result = factors[-1]
    for factor in reversed(factors[:-1]):
        result = np.kron(result, factor)
    return result


def _total_sz(num_sites: int) -> np.ndarray:
    return sum(
        (_on_site(Z, site, num_sites) / 2.0 for site in range(num_sites)),
        start=np.zeros((2**num_sites, 2**num_sites), dtype=np.complex128),
    )


def test_fermionic_sector_records_independent_spin_populations():
    sector = FermionicSector(num_sites=3, num_up=2, num_down=1)

    assert sector.num_particles == 3
    assert sector.num_modes == 6
    assert sector.spin_projection == 0.5
    with pytest.raises(FrozenInstanceError):
        sector.num_up = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"num_sites": True, "num_up": 0, "num_down": 0}, TypeError, "num_sites"),
        ({"num_sites": 0, "num_up": 0, "num_down": 0}, ValueError, "num_sites"),
        ({"num_sites": 2, "num_up": False, "num_down": 0}, TypeError, "num_up"),
        ({"num_sites": 2, "num_up": -1, "num_down": 0}, ValueError, "num_up"),
        ({"num_sites": 2, "num_up": 3, "num_down": 0}, ValueError, "num_up"),
        ({"num_sites": 2, "num_up": 0, "num_down": 2.0}, TypeError, "num_down"),
        ({"num_sites": 2, "num_up": 0, "num_down": -1}, ValueError, "num_down"),
        ({"num_sites": 2, "num_up": 0, "num_down": 3}, ValueError, "num_down"),
    ],
)
def test_fermionic_sector_rejects_invalid_populations(kwargs, error, message):
    with pytest.raises(error, match=message):
        FermionicSector(**kwargs)


@pytest.mark.parametrize(
    ("num_sites", "num_up", "expected_down", "expected_magnetization"),
    [
        (3, 0, 3, -1.5),
        (3, 2, 1, 0.5),
        (4, 2, 2, 0.0),
        (4, 4, 0, 2.0),
    ],
)
def test_spin_sector_uses_spin_half_magnetization_convention(
    num_sites,
    num_up,
    expected_down,
    expected_magnetization,
):
    sector = SpinMagnetizationSector(num_sites, num_up)

    assert sector.num_down == expected_down
    assert sector.magnetization == expected_magnetization


@pytest.mark.parametrize(
    ("args", "error"),
    [
        ((True, 0), TypeError),
        ((0, 0), ValueError),
        ((2, False), TypeError),
        ((2, -1), ValueError),
        ((2, 3), ValueError),
    ],
)
def test_spin_sector_rejects_invalid_populations(args, error):
    with pytest.raises(error):
        SpinMagnetizationSector(*args)


def test_fermi_hubbard_system_requires_and_exposes_a_matching_sector():
    model = FermiHubbardModel(LineLattice(3), onsite_interaction=2.0)
    sector = FermionicSector(3, 2, 1)
    system = LatticeSystem(model, sector)

    assert system.model is model
    assert system.lattice is model.lattice
    assert system.num_sites == 3
    assert system.model_kind is LatticeModelKind.FERMI_HUBBARD
    assert system.sector is sector
    assert system.conserves_z_magnetization
    assert system.fermionic_mode_layout == (
        (0, "up"),
        (0, "down"),
        (1, "up"),
        (1, "down"),
        (2, "up"),
        (2, "down"),
    )
    with pytest.raises(FrozenInstanceError):
        system.sector = None  # type: ignore[misc]


def test_spin_system_without_a_sector_accepts_nonconserving_models():
    heisenberg = LatticeSystem(
        HeisenbergModel(
            LineLattice(2),
            coupling=(1.0, 0.5, 0.2),
            external_field=(0.1, 0.0, 0.0),
        )
    )
    ising = LatticeSystem(IsingModel(LineLattice(2), transverse_field=0.3))

    assert heisenberg.model_kind is LatticeModelKind.HEISENBERG
    assert ising.model_kind is LatticeModelKind.ISING
    assert heisenberg.fermionic_mode_layout is None
    assert ising.fermionic_mode_layout is None
    assert not heisenberg.conserves_z_magnetization
    assert not ising.conserves_z_magnetization


@pytest.mark.parametrize(
    "model",
    [
        HeisenbergModel(
            LineLattice(3),
            coupling=(0.7, 0.7, 1.2),
            external_field=(0.0, 0.0, 0.4),
        ),
        IsingModel(LineLattice(3), transverse_field=0.0),
        HeisenbergModel(
            Lattice.from_nodes_and_edges(2, ()),
            coupling=(1.0, -2.0, 0.3),
        ),
    ],
)
def test_spin_sector_is_accepted_exactly_when_total_sz_is_conserved(model):
    system = LatticeSystem(model, SpinMagnetizationSector(model.lattice.num_nodes, 1))
    hamiltonian = model.qubit_hamiltonian().to_matrix()
    total_sz = _total_sz(model.lattice.num_nodes)

    assert system.conserves_z_magnetization
    np.testing.assert_allclose(
        hamiltonian @ total_sz - total_sz @ hamiltonian,
        0.0,
        rtol=0.0,
        atol=1e-14,
    )


@pytest.mark.parametrize(
    "model",
    [
        HeisenbergModel(LineLattice(2), coupling=(1.0, 0.9, 0.5)),
        HeisenbergModel(LineLattice(2), external_field=(0.1, 0.0, 0.0)),
        HeisenbergModel(LineLattice(2), external_field=(0.0, -0.1, 0.0)),
        IsingModel(LineLattice(2), transverse_field=0.2),
    ],
)
def test_spin_sector_rejects_models_that_mix_magnetization_sectors(model):
    hamiltonian = model.qubit_hamiltonian().to_matrix()
    total_sz = _total_sz(model.lattice.num_nodes)

    assert not np.allclose(
        hamiltonian @ total_sz - total_sz @ hamiltonian,
        0.0,
        rtol=0.0,
        atol=1e-14,
    )
    with pytest.raises(ValueError, match="conserves total S_z"):
        LatticeSystem(model, SpinMagnetizationSector(model.lattice.num_nodes, 1))


def test_system_rejects_wrong_model_sector_pairs_and_site_counts():
    hubbard = FermiHubbardModel(LineLattice(2))
    heisenberg = HeisenbergModel(LineLattice(2))

    with pytest.raises(TypeError, match="requires a FermionicSector"):
        LatticeSystem(hubbard)
    with pytest.raises(TypeError, match="requires a FermionicSector"):
        LatticeSystem(hubbard, SpinMagnetizationSector(2, 1))
    with pytest.raises(TypeError, match="SpinMagnetizationSector"):
        LatticeSystem(heisenberg, FermionicSector(2, 1, 1))
    with pytest.raises(ValueError, match="same number of sites"):
        LatticeSystem(hubbard, FermionicSector(3, 1, 1))
    with pytest.raises(ValueError, match="same number of sites"):
        LatticeSystem(heisenberg, SpinMagnetizationSector(3, 1))
    with pytest.raises(TypeError, match="model must be"):
        LatticeSystem(object())  # type: ignore[arg-type]
