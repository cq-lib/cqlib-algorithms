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

"""Scientific tests for Ising and Heisenberg lattice models."""

import numpy as np
import pytest

from cqlib_algorithms.extension.lattice import (
    HeisenbergModel,
    IsingModel,
    Lattice,
    LineLattice,
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
    left: np.ndarray,
    source: int,
    right: np.ndarray,
    target: int,
    num_sites: int,
) -> np.ndarray:
    return _on_site(left, source, num_sites) @ _on_site(right, target, num_sites)


def test_two_site_xxx_spectrum_has_singlet_and_triplet_sectors():
    model = HeisenbergModel(LineLattice(2), coupling=(1.0, 1.0, 1.0))

    eigenvalues = np.linalg.eigvalsh(model.qubit_hamiltonian().to_matrix())
    np.testing.assert_allclose(eigenvalues, (-3.0, 1.0, 1.0, 1.0), atol=1e-14)


def test_xyz_model_with_weighted_edges_and_local_fields_matches_independent_matrix():
    lattice = Lattice.from_nodes_and_edges(3, ((0, 1, 0.5), (1, 2, -1.25)))
    coupling = (0.7, -0.2, 1.1)
    fields = ((0.1, 0.2, -0.3), (-0.4, 0.0, 0.5), (0.7, -0.8, 0.9))
    model = HeisenbergModel(lattice, coupling=coupling, external_field=fields)

    expected = np.zeros((8, 8), dtype=np.complex128)
    axes = (X, Y, Z)
    for source, target, weight in ((0, 1, 0.5), (1, 2, -1.25)):
        for axis, strength in zip(axes, coupling, strict=True):
            expected += weight * strength * _two_site(axis, source, axis, target, 3)
    for site, field in enumerate(fields):
        for axis, strength in zip(axes, field, strict=True):
            expected += strength * _on_site(axis, site, 3)

    qubit_matrix = model.qubit_hamiltonian().to_matrix()
    np.testing.assert_allclose(qubit_matrix, expected, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(model.spin_operator().to_matrix(), expected, atol=1e-13)


def test_ising_is_exact_heisenberg_specialization_and_accepts_local_fields():
    lattice = Lattice.from_nodes_and_edges(3, ((0, 1, 0.3), (1, 2, -0.6)))
    fields = (0.1, -0.2, 0.7)
    ising = IsingModel(lattice, transverse_field=fields)
    heisenberg = HeisenbergModel(
        lattice,
        coupling=(0.0, 0.0, 1.0),
        external_field=tuple((field, 0.0, 0.0) for field in fields),
    )

    assert ising.spin_operator().equiv(heisenberg.spin_operator())
    np.testing.assert_allclose(
        ising.qubit_hamiltonian().to_matrix(),
        heisenberg.qubit_hamiltonian().to_matrix(),
        atol=1e-14,
    )


@pytest.mark.parametrize(
    "model",
    [
        HeisenbergModel(
            Lattice.from_nodes_and_edges(3, ((0, 1, 0.3), (1, 2, -0.5))),
            coupling=(0.7, -0.4, 1.2),
            external_field=((0.1, 0.2, 0.3), (0.0, -0.4, 0.5), (0.6, 0.0, -0.7)),
        ),
        IsingModel(
            Lattice.from_nodes_and_edges(3, ((0, 1, 0.3), (1, 2, -0.5))),
            transverse_field=(0.2, -0.3, 0.6),
        ),
    ],
)
def test_spin_model_is_exact_sum_of_interaction_and_field_operators(model):
    recomposed = model.interaction_operator() + model.field_operator()

    assert model.spin_operator().equiv(recomposed, atol=0.0)


def test_transverse_field_ising_preserves_global_spin_flip_symmetry():
    model = IsingModel(LineLattice(3), transverse_field=(0.2, -0.4, 0.7))
    hamiltonian = model.qubit_hamiltonian().to_matrix()
    parity = np.kron(X, np.kron(X, X))

    np.testing.assert_allclose(
        hamiltonian @ parity - parity @ hamiltonian,
        0.0,
        atol=1e-14,
    )


def test_uniform_field_is_broadcast_and_mutable_local_field_input_is_copied():
    model = HeisenbergModel(LineLattice(3), external_field=(0.1, 0.2, 0.3))
    local_fields = [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]
    local_model = HeisenbergModel(LineLattice(2), external_field=local_fields)
    local_fields[0][0] = 10.0

    assert model.external_fields == ((0.1, 0.2, 0.3),) * 3
    assert local_model.external_fields == ((0.1, 0.0, 0.0), (0.2, 0.0, 0.0))


def test_zero_self_loops_are_ignored_but_nonzero_self_loops_are_rejected():
    assert HeisenbergModel(LineLattice(2)).qubit_hamiltonian().num_qubits == 2
    invalid = Lattice.from_nodes_and_edges(2, ((0, 1, 1.0), (0, 0, 0.2)))
    with pytest.raises(ValueError, match="explicit field"):
        HeisenbergModel(invalid)
    with pytest.raises(ValueError, match="explicit field"):
        IsingModel(invalid)


def test_complex_spin_edge_weights_are_rejected_instead_of_silently_projected():
    lattice = Lattice.from_nodes_and_edges(2, ((0, 1, 1.0 + 0.2j),))
    with pytest.raises(ValueError, match="must be real"):
        HeisenbergModel(lattice)
    with pytest.raises(ValueError, match="must be real"):
        IsingModel(lattice)


def test_zero_spin_model_preserves_register_width():
    lattice = Lattice.from_nodes_and_edges(2, ())
    model = HeisenbergModel(
        lattice,
        coupling=(0.0, 0.0, 0.0),
        external_field=(0.0, 0.0, 0.0),
    )
    mapped = model.qubit_hamiltonian()

    assert mapped.num_qubits == 2
    assert mapped.num_terms == 0
    np.testing.assert_array_equal(mapped.to_matrix(), np.zeros((4, 4)))


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"coupling": (1.0, 2.0)}, ValueError),
        ({"coupling": (1.0, 2.0, 0.1j)}, ValueError),
        ({"coupling": (1.0, float("nan"), 3.0)}, ValueError),
        ({"external_field": (1.0, 2.0)}, ValueError),
        ({"external_field": ((1.0, 2.0, 3.0),)}, ValueError),
        ({"external_field": ((1.0, 2.0), (3.0, 4.0))}, ValueError),
    ],
)
def test_invalid_heisenberg_parameters_are_rejected(kwargs, error):
    with pytest.raises(error):
        HeisenbergModel(LineLattice(2), **kwargs)


@pytest.mark.parametrize(
    ("field", "error"),
    [
        (True, TypeError),
        ((1.0,), ValueError),
        ((1.0, 2.0, 3.0), ValueError),
        ((1.0, 0.2j), ValueError),
        ((1.0, float("inf")), ValueError),
    ],
)
def test_invalid_ising_fields_are_rejected(field, error):
    with pytest.raises(error):
        IsingModel(LineLattice(2), transverse_field=field)
