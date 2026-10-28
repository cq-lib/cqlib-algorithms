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

"""Scientific tests for the spinful Fermi--Hubbard model."""

import numpy as np
import pytest
from cqlib.qis import FermionicOp, Hamiltonian, JordanWignerMapper

from cqlib_algorithms.extension.lattice import FermiHubbardModel, Lattice, LineLattice


def _annihilation(num_modes: int, mode: int) -> np.ndarray:
    dimension = 1 << num_modes
    result = np.zeros((dimension, dimension), dtype=np.complex128)
    lower_mask = (1 << mode) - 1
    for state in range(dimension):
        if (state >> mode) & 1:
            sign = -1.0 if (state & lower_mask).bit_count() % 2 else 1.0
            result[state ^ (1 << mode), state] = sign
    return result


def _independent_hubbard_matrix(
    lattice: Lattice,
    interactions: tuple[float, ...],
) -> np.ndarray:
    num_modes = 2 * lattice.num_nodes
    annihilators = [_annihilation(num_modes, mode) for mode in range(num_modes)]
    creators = [operator.conjugate().T for operator in annihilators]
    numbers = [creators[mode] @ annihilators[mode] for mode in range(num_modes)]
    result = np.zeros((1 << num_modes, 1 << num_modes), dtype=np.complex128)
    for source, target, weight in lattice.weighted_edges:
        if source == target:
            result += weight * (numbers[2 * source] + numbers[2 * source + 1])
            continue
        for spin_offset in (0, 1):
            source_mode = 2 * source + spin_offset
            target_mode = 2 * target + spin_offset
            result += weight * creators[source_mode] @ annihilators[target_mode]
            result += (
                weight.conjugate() * creators[target_mode] @ annihilators[source_mode]
            )
    for node, interaction in enumerate(interactions):
        result += interaction * numbers[2 * node] @ numbers[2 * node + 1]
    return result


def test_one_site_spectrum_contains_empty_single_and_double_occupancy_energies():
    model = FermiHubbardModel(
        Lattice.from_nodes_and_edges(1, ((0, 0, -0.3),)),
        onsite_interaction=1.7,
    )

    matrix = model.qubit_hamiltonian().to_matrix()
    np.testing.assert_allclose(
        np.diag(matrix),
        np.array([0.0, -0.3, -0.3, 1.1]),
        atol=1e-14,
    )
    np.testing.assert_allclose(matrix, np.diag(np.diag(matrix)), atol=1e-14)


def test_complex_hopping_and_site_dependent_interactions_match_independent_fock_matrix():
    lattice = Lattice.from_nodes_and_edges(
        2,
        ((0, 1, 0.7 + 0.4j), (0, 0, -0.2), (1, 1, 0.35)),
    )
    model = FermiHubbardModel(lattice, onsite_interaction=(1.1, -0.6))

    expected = _independent_hubbard_matrix(lattice, model.onsite_interactions)
    actual = model.qubit_hamiltonian().to_matrix()
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(actual, actual.conjugate().T, atol=1e-14)


def test_two_site_half_filled_ground_energy_matches_closed_form_solution():
    hopping = 1.0
    interaction = 2.0
    model = FermiHubbardModel(
        Lattice.from_nodes_and_edges(2, ((0, 1, -hopping),)),
        onsite_interaction=interaction,
    )
    hamiltonian = model.qubit_hamiltonian().to_matrix()
    number = (
        JordanWignerMapper().map_operator(model.total_particle_number()).to_matrix()
    )
    two_particle_indices = [
        index
        for index in range(hamiltonian.shape[0])
        if np.isclose(number[index, index], 2.0)
    ]
    sector = hamiltonian[np.ix_(two_particle_indices, two_particle_indices)]
    expected_ground = 0.5 * (interaction - np.sqrt(interaction**2 + 16.0 * hopping**2))

    assert np.linalg.eigvalsh(sector)[0] == pytest.approx(expected_ground, abs=1e-13)


def test_interleaved_modes_and_particle_number_observables_are_explicit():
    model = FermiHubbardModel(LineLattice(3), onsite_interaction=2.0)

    assert model.num_modes == 6
    assert tuple(
        model.mode_index(site, spin) for site in range(3) for spin in ("up", "down")
    ) == tuple(range(6))
    expected_up = FermionicOp.number(6, 0) + FermionicOp.number(6, 2)
    expected_up = expected_up + FermionicOp.number(6, 4)
    expected_down = FermionicOp.number(6, 1) + FermionicOp.number(6, 3)
    expected_down = expected_down + FermionicOp.number(6, 5)
    assert model.up_particle_number().equiv(expected_up)
    assert model.down_particle_number().equiv(expected_down)
    assert model.total_particle_number().equiv(expected_up + expected_down)


@pytest.mark.parametrize(
    ("site", "spin", "error"),
    [
        (True, "up", TypeError),
        (-1, "up", ValueError),
        (2, "down", ValueError),
        (0, 0, TypeError),
        (0, "left", ValueError),
    ],
)
def test_mode_index_rejects_invalid_site_and_spin_labels(site, spin, error):
    model = FermiHubbardModel(LineLattice(2))

    with pytest.raises(error):
        model.mode_index(site, spin)


def test_hamiltonian_commutes_with_total_and_spin_resolved_particle_numbers():
    lattice = Lattice.from_nodes_and_edges(
        2,
        ((0, 1, -0.8 + 0.25j), (0, 0, 0.1), (1, 1, -0.2)),
    )
    model = FermiHubbardModel(lattice, onsite_interaction=(1.2, 0.9))
    hamiltonian = model.qubit_hamiltonian().to_matrix()

    for observable in (
        model.total_particle_number(),
        model.up_particle_number(),
        model.down_particle_number(),
    ):
        mapped = JordanWignerMapper().map_operator(observable).to_matrix()
        np.testing.assert_allclose(
            hamiltonian @ mapped - mapped @ hamiltonian,
            0.0,
            atol=1e-13,
        )


class _WrongWidthMapper:
    def map_operator(self, op: FermionicOp) -> Hamiltonian:
        del op
        return Hamiltonian(1)

    def num_qubits(self, num_modes: int) -> int:
        return num_modes


@pytest.mark.parametrize(
    ("interaction", "error"),
    [
        (True, TypeError),
        ((1.0,), ValueError),
        ((1.0, 2.0, 3.0), ValueError),
        ((1.0, 0.1j), ValueError),
        ((1.0, float("inf")), ValueError),
    ],
)
def test_invalid_onsite_interactions_are_rejected(interaction, error):
    with pytest.raises(error):
        FermiHubbardModel(LineLattice(2), onsite_interaction=interaction)


def test_model_validates_lattice_and_mapper_contracts():
    with pytest.raises(TypeError, match="lattice must be"):
        FermiHubbardModel(object())  # type: ignore[arg-type]
    model = FermiHubbardModel(LineLattice(2))
    with pytest.raises(TypeError, match="mapper must implement"):
        model.qubit_hamiltonian(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="inconsistent width"):
        model.qubit_hamiltonian(_WrongWidthMapper())


def test_mutable_interaction_input_is_copied_at_construction():
    interactions = [1.0, 2.0]
    model = FermiHubbardModel(LineLattice(2), onsite_interaction=interactions)
    interactions[0] = 99.0

    assert model.onsite_interactions == (1.0, 2.0)


def test_empty_one_site_model_maps_to_a_width_preserving_zero_hamiltonian():
    model = FermiHubbardModel(Lattice.from_nodes_and_edges(1, ()))
    mapped = model.qubit_hamiltonian()

    assert mapped.num_qubits == 2
    assert mapped.num_terms == 0
    np.testing.assert_array_equal(mapped.to_matrix(), np.zeros((4, 4)))
