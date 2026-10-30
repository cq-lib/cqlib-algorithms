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

"""Independent matrix tests for spin lattice observables."""

import numpy as np
import pytest

from cqlib_algorithms.extension.lattice import (
    HeisenbergModel,
    IsingModel,
    Lattice,
    LineLattice,
    SiteMagnetization,
    SpinCorrelation,
    SpinInteractionEnergy,
    TotalMagnetization,
)

I = np.eye(2, dtype=np.complex128)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
Z = np.diag([1.0, -1.0]).astype(np.complex128)


def _on_site(matrix: np.ndarray, site: int, num_sites: int) -> np.ndarray:
    factors = [I] * num_sites
    factors[site] = matrix
    result = factors[-1]
    for factor in reversed(factors[:-1]):
        result = np.kron(result, factor)
    return result


def _two_site(
    matrix: np.ndarray,
    source: int,
    target: int,
    num_sites: int,
) -> np.ndarray:
    return _on_site(matrix, source, num_sites) @ _on_site(
        matrix,
        target,
        num_sites,
    )


@pytest.mark.reference
def test_total_and_site_magnetization_match_physical_spin_half_matrices():
    model = HeisenbergModel(LineLattice(3), coupling=(0.2, 0.2, 0.7))
    expected_sites = tuple(0.5 * _on_site(Z, site, 3) for site in range(3))
    expected_total = sum(expected_sites, np.zeros((8, 8), dtype=np.complex128))

    actual_total = TotalMagnetization(model).to_qubit_hamiltonian().to_matrix()
    np.testing.assert_allclose(actual_total, expected_total, rtol=0.0, atol=1e-14)
    for site, expected in enumerate(expected_sites):
        component = SiteMagnetization(model, site)
        np.testing.assert_allclose(
            component.to_qubit_hamiltonian().to_matrix(),
            expected,
            rtol=0.0,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            component.to_spin_op().to_matrix(),
            expected,
            rtol=0.0,
            atol=1e-14,
        )


@pytest.mark.parametrize(("axis", "matrix"), [("X", X), ("Y", Y), ("Z", Z)])
@pytest.mark.reference
def test_pauli_normalized_correlations_match_independent_matrices(axis, matrix):
    model = HeisenbergModel(LineLattice(3))
    observable = SpinCorrelation(model, 0, 2, axis)
    expected = _two_site(matrix, 0, 2, 3)

    np.testing.assert_allclose(
        observable.to_qubit_hamiltonian().to_matrix(),
        expected,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        observable.to_spin_op().to_matrix(),
        expected,
        rtol=0.0,
        atol=1e-14,
    )


@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
def test_equal_site_pauli_correlation_is_identity(axis):
    model = IsingModel(LineLattice(2))
    matrix = SpinCorrelation(model, 1, 1, axis).to_qubit_hamiltonian().to_matrix()
    np.testing.assert_allclose(matrix, np.eye(4), rtol=0.0, atol=0.0)


@pytest.mark.reference
def test_heisenberg_correlation_energy_excludes_all_local_fields():
    lattice = Lattice.from_nodes_and_edges(3, ((0, 1, 0.4), (1, 2, -0.7)))
    coupling = (0.8, -0.3, 1.2)
    fields = ((0.1, -0.2, 0.3), (0.0, 0.4, -0.5), (-0.6, 0.7, 0.0))
    model = HeisenbergModel(lattice, coupling=coupling, external_field=fields)
    axes = (X, Y, Z)
    expected = np.zeros((8, 8), dtype=np.complex128)
    for source, target, weight in ((0, 1, 0.4), (1, 2, -0.7)):
        for axis_matrix, strength in zip(axes, coupling, strict=True):
            expected += (
                weight
                * strength
                * _two_site(
                    axis_matrix,
                    source,
                    target,
                    3,
                )
            )

    actual = SpinInteractionEnergy(model).to_qubit_hamiltonian().to_matrix()
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-13)
    full = model.qubit_hamiltonian().to_matrix()
    field_matrix = np.zeros_like(full)
    for site, field in enumerate(fields):
        for axis_matrix, strength in zip(axes, field, strict=True):
            field_matrix += strength * _on_site(axis_matrix, site, 3)
    np.testing.assert_allclose(full - actual, field_matrix, rtol=0.0, atol=1e-13)


@pytest.mark.reference
def test_ising_correlation_energy_contains_only_weighted_zz_edges():
    lattice = Lattice.from_nodes_and_edges(3, ((0, 1, -0.25), (0, 2, 0.9)))
    model = IsingModel(lattice, transverse_field=(0.2, -0.1, 0.6))
    expected = -0.25 * _two_site(Z, 0, 1, 3) + 0.9 * _two_site(Z, 0, 2, 3)

    actual = SpinInteractionEnergy(model).to_qubit_hamiltonian().to_matrix()
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)


def test_empty_interaction_model_yields_width_preserving_zero_energy():
    model = IsingModel(
        Lattice.from_nodes_and_edges(2, ()),
        transverse_field=(0.2, -0.4),
    )
    observable = SpinInteractionEnergy(model).to_qubit_hamiltonian()

    assert observable.num_qubits == 2
    assert observable.num_terms == 0
    np.testing.assert_array_equal(observable.to_matrix(), np.zeros((4, 4)))


def test_total_magnetization_conservation_is_not_overclaimed():
    conserving = HeisenbergModel(
        LineLattice(3),
        coupling=(0.6, 0.6, 1.1),
        external_field=(0.0, 0.0, 0.4),
    )
    nonconserving = IsingModel(LineLattice(3), transverse_field=0.3)

    for model, should_commute in ((conserving, True), (nonconserving, False)):
        hamiltonian = model.qubit_hamiltonian().to_matrix()
        magnetization = TotalMagnetization(model).to_qubit_hamiltonian().to_matrix()
        commutator_norm = np.linalg.norm(
            hamiltonian @ magnetization - magnetization @ hamiltonian
        )
        assert bool(commutator_norm < 1e-12) is should_commute


@pytest.mark.parametrize(
    "factory",
    [TotalMagnetization, SpinInteractionEnergy],
)
def test_global_spin_factories_reject_non_spin_models(factory):
    with pytest.raises(TypeError, match="HeisenbergModel or IsingModel"):
        factory(object())


@pytest.mark.parametrize("site", [-1, 2])
def test_site_magnetization_rejects_out_of_range_sites(site):
    with pytest.raises(ValueError, match="site is out of range"):
        SiteMagnetization(IsingModel(LineLattice(2)), site)


@pytest.mark.parametrize("site", [True, 0.0, "0"])
def test_site_magnetization_rejects_non_integer_sites(site):
    with pytest.raises(TypeError, match="site must be an int"):
        SiteMagnetization(IsingModel(LineLattice(2)), site)


@pytest.mark.parametrize(("source", "target"), [(-1, 0), (0, 2), (2, 0)])
def test_spin_correlation_rejects_out_of_range_sites(source, target):
    with pytest.raises(ValueError, match="out of range"):
        SpinCorrelation(IsingModel(LineLattice(2)), source, target, "Z")


@pytest.mark.parametrize(("source", "target"), [(True, 0), (0, 1.0), ("0", 1)])
def test_spin_correlation_rejects_non_integer_sites(source, target):
    with pytest.raises(TypeError, match="must be an int"):
        SpinCorrelation(IsingModel(LineLattice(2)), source, target, "Z")


@pytest.mark.parametrize(
    ("axis", "error"), [("x", ValueError), ("A", ValueError), (1, TypeError)]
)
def test_spin_correlation_rejects_invalid_axes(axis, error):
    with pytest.raises(error, match="axis"):
        SpinCorrelation(IsingModel(LineLattice(2)), 0, 1, axis)
