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

"""Independent Fock-space tests for fermionic lattice observables."""

import numpy as np
import pytest
from cqlib.qis import JordanWignerMapper

from cqlib_algorithms.extension.lattice import (
    DoubleOccupation,
    FermiHubbardModel,
    FermionicMagnetization,
    Lattice,
    LineLattice,
    SiteOccupation,
    SpinDownParticleNumber,
    SpinUpParticleNumber,
    TotalParticleNumber,
)


def _expected_diagonals(num_sites: int) -> tuple[np.ndarray, ...]:
    size = 1 << (2 * num_sites)
    total = np.zeros(size)
    spin_up = np.zeros(size)
    spin_down = np.zeros(size)
    magnetization = np.zeros(size)
    site_occupations = np.zeros((num_sites, size))
    double_occupations = np.zeros((num_sites, size))
    for basis in range(size):
        for site in range(num_sites):
            up = float(bool(basis & (1 << (2 * site))))
            down = float(bool(basis & (1 << (2 * site + 1))))
            spin_up[basis] += up
            spin_down[basis] += down
            site_occupations[site, basis] = up + down
            double_occupations[site, basis] = up * down
        total[basis] = spin_up[basis] + spin_down[basis]
        magnetization[basis] = 0.5 * (spin_up[basis] - spin_down[basis])
    return (
        total,
        spin_up,
        spin_down,
        magnetization,
        site_occupations,
        double_occupations,
    )


@pytest.mark.reference
def test_all_observable_eigenvalues_match_independent_fock_counting():
    model = FermiHubbardModel(LineLattice(3), onsite_interaction=(1.0, 2.0, 3.0))
    total, spin_up, spin_down, magnetization, sites, doubles = _expected_diagonals(3)

    actual_global = (
        TotalParticleNumber(model).to_qubit_hamiltonian().to_matrix(),
        SpinUpParticleNumber(model).to_qubit_hamiltonian().to_matrix(),
        SpinDownParticleNumber(model).to_qubit_hamiltonian().to_matrix(),
        FermionicMagnetization(model).to_qubit_hamiltonian().to_matrix(),
    )
    for matrix, expected in zip(
        actual_global,
        (total, spin_up, spin_down, magnetization),
        strict=True,
    ):
        np.testing.assert_allclose(matrix, np.diag(expected), rtol=0.0, atol=1e-13)

    for site in range(model.lattice.num_nodes):
        np.testing.assert_allclose(
            SiteOccupation(model, site).to_qubit_hamiltonian().to_matrix(),
            np.diag(sites[site]),
            rtol=0.0,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            DoubleOccupation(model, site).to_qubit_hamiltonian().to_matrix(),
            np.diag(doubles[site]),
            rtol=0.0,
            atol=1e-13,
        )


def test_fermionic_operator_identities_are_exact():
    model = FermiHubbardModel(LineLattice(3))
    total = TotalParticleNumber(model).to_fermionic_op()
    spin_up = SpinUpParticleNumber(model).to_fermionic_op()
    spin_down = SpinDownParticleNumber(model).to_fermionic_op()
    site_sum = SiteOccupation(model, 0).to_fermionic_op()
    for site in range(1, model.lattice.num_nodes):
        site_sum += SiteOccupation(model, site).to_fermionic_op()

    assert total.equiv(spin_up + spin_down, atol=0.0)
    assert total.equiv(site_sum, atol=0.0)
    assert (
        FermionicMagnetization(model)
        .to_fermionic_op()
        .equiv(
            spin_up.scaled(0.5) + spin_down.scaled(-0.5),
            atol=0.0,
        )
    )


@pytest.mark.reference
def test_global_numbers_and_magnetization_commute_with_complex_hubbard_model():
    lattice = Lattice.from_nodes_and_edges(
        3,
        (
            (0, 1, -0.7 + 0.2j),
            (1, 2, 0.4 - 0.35j),
            (0, 0, 0.15),
            (2, 2, -0.3),
        ),
    )
    model = FermiHubbardModel(lattice, onsite_interaction=(1.1, -0.2, 0.8))
    hamiltonian = model.qubit_hamiltonian().to_matrix()
    observables = (
        TotalParticleNumber(model),
        SpinUpParticleNumber(model),
        SpinDownParticleNumber(model),
        FermionicMagnetization(model),
    )
    for observable in observables:
        matrix = observable.to_qubit_hamiltonian().to_matrix()
        np.testing.assert_allclose(
            hamiltonian @ matrix - matrix @ hamiltonian,
            0.0,
            rtol=0.0,
            atol=1e-12,
        )


def test_local_occupation_is_not_falsely_treated_as_conserved_under_hopping():
    model = FermiHubbardModel(LineLattice(2, edge_parameter=-0.8))
    hamiltonian = model.qubit_hamiltonian().to_matrix()
    occupation = SiteOccupation(model, 0).to_qubit_hamiltonian().to_matrix()
    double = DoubleOccupation(model, 0).to_qubit_hamiltonian().to_matrix()

    assert np.linalg.norm(hamiltonian @ occupation - occupation @ hamiltonian) > 1e-10
    assert np.linalg.norm(hamiltonian @ double - double @ hamiltonian) > 1e-10


def test_factories_delegate_total_numbers_to_the_model_and_accept_mapper():
    model = FermiHubbardModel(LineLattice(2))
    assert (
        TotalParticleNumber(model)
        .to_fermionic_op()
        .equiv(model.total_particle_number())
    )
    assert (
        SpinUpParticleNumber(model).to_fermionic_op().equiv(model.up_particle_number())
    )
    assert (
        SpinDownParticleNumber(model)
        .to_fermionic_op()
        .equiv(model.down_particle_number())
    )
    mapped = SiteOccupation(model, 1).to_qubit_hamiltonian(JordanWignerMapper())
    assert mapped.num_qubits == model.num_modes
    assert mapped.is_hermitian()


@pytest.mark.parametrize(
    "factory",
    [
        TotalParticleNumber,
        SpinUpParticleNumber,
        SpinDownParticleNumber,
        FermionicMagnetization,
    ],
)
def test_global_factories_reject_non_hubbard_models(factory):
    with pytest.raises(TypeError, match="FermiHubbardModel"):
        factory(object())


@pytest.mark.parametrize("site", [-1, 2])
@pytest.mark.parametrize("factory", [SiteOccupation, DoubleOccupation])
def test_local_factories_reject_out_of_range_sites(factory, site):
    with pytest.raises(ValueError, match="out of range"):
        factory(FermiHubbardModel(LineLattice(2)), site)


@pytest.mark.parametrize("site", [True, 0.0, "0"])
@pytest.mark.parametrize("factory", [SiteOccupation, DoubleOccupation])
def test_local_factories_reject_non_integer_sites(factory, site):
    with pytest.raises(TypeError, match="site must be an int"):
        factory(FermiHubbardModel(LineLattice(2)), site)


def test_factories_reject_invalid_mapper_contracts():
    model = FermiHubbardModel(LineLattice(1))
    with pytest.raises(TypeError, match="mapper"):
        TotalParticleNumber(model).to_qubit_hamiltonian(object())
