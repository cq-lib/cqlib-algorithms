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

"""Tests for immutable lattice topology components."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from cqlib_algorithms.extension.lattice import (
    BoundaryCondition,
    HyperCubicLattice,
    Lattice,
    LineLattice,
    SquareLattice,
)


def _offsite_pairs(lattice: Lattice) -> set[tuple[int, int]]:
    return {
        (source, target)
        for source, target, _ in lattice.weighted_edges
        if source != target
    }


def test_boundary_condition_values_are_stable():
    assert BoundaryCondition.OPEN.value == "open"
    assert BoundaryCondition.PERIODIC.value == "periodic"


def test_general_lattice_copies_sorts_and_orients_edges():
    edges = [(2, 1, 2.0 + 3.0j), (0, 0, -1.0), (0, 2, 4.0)]
    lattice = Lattice.from_nodes_and_edges(3, edges)
    edges.append((1, 1, 7.0))

    assert lattice.node_indices == (0, 1, 2)
    assert lattice.num_edges == 3
    assert lattice.weighted_edges == (
        (0, 0, -1.0 + 0.0j),
        (0, 2, 4.0 + 0.0j),
        (1, 2, 2.0 - 3.0j),
    )
    assert lattice.edge_weight(2, 1) == 2.0 + 3.0j
    assert lattice.edge_weight(1, 0) is None


def test_lattice_is_frozen_and_hashable():
    lattice = Lattice.from_nodes_and_edges(2, [(0, 1, 1.0)])

    with pytest.raises(FrozenInstanceError):
        lattice.num_nodes = 3  # type: ignore[misc]
    assert hash(lattice) == hash(Lattice(2, ((0, 1, 1.0),)))


def test_weighted_adjacency_preserves_complex_orientation():
    lattice = Lattice.from_nodes_and_edges(
        3,
        [(0, 0, 1.0), (0, 1, 2.0 + 3.0j), (2, 1, -4.0 + 5.0j)],
    )

    expected = np.array(
        [
            [1.0, 2.0 + 3.0j, 0.0],
            [2.0 - 3.0j, 0.0, -4.0 - 5.0j],
            [0.0, -4.0 + 5.0j, 0.0],
        ],
        dtype=np.complex128,
    )
    actual = lattice.to_adjacency_matrix(weighted=True)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, actual.conjugate().T)


def test_binary_adjacency_tracks_declared_zero_weight_edges():
    lattice = Lattice.from_nodes_and_edges(2, [(0, 0, 0.0), (0, 1, 0.0)])

    np.testing.assert_array_equal(
        lattice.to_adjacency_matrix(),
        np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    )
    np.testing.assert_array_equal(
        lattice.to_adjacency_matrix(weighted=True),
        np.zeros((2, 2), dtype=np.complex128),
    )


def test_adjacency_constructor_round_trips_hermitian_matrix_defensively():
    matrix = np.array(
        [[1.0, 2.0 + 3.0j], [2.0 - 3.0j, -4.0]],
        dtype=np.complex128,
    )
    lattice = Lattice.from_adjacency_matrix(matrix)
    matrix[0, 1] = 99.0

    np.testing.assert_allclose(
        lattice.to_adjacency_matrix(weighted=True),
        np.array(
            [[1.0, 2.0 + 3.0j], [2.0 - 3.0j, -4.0]],
            dtype=np.complex128,
        ),
        rtol=0.0,
        atol=0.0,
    )


def test_adjacency_constructor_removes_tiny_antihermitian_roundoff():
    matrix = np.array(
        [[0.0, 2.0 + 1e-13j], [2.0 + 1e-13j, 0.0]],
        dtype=np.complex128,
    )

    lattice = Lattice.from_adjacency_matrix(matrix)

    np.testing.assert_allclose(
        lattice.to_adjacency_matrix(weighted=True),
        np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.complex128),
        rtol=0.0,
        atol=1e-15,
    )


@pytest.mark.parametrize(
    "matrix, exception",
    [
        ([], ValueError),
        ([1.0, 2.0], ValueError),
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], ValueError),
        ([[0.0, 1.0j], [1.0j, 0.0]], ValueError),
        ([[0.0, np.nan], [np.nan, 0.0]], ValueError),
        ([[0.0, np.inf], [np.inf, 0.0]], ValueError),
        ([[object()]], TypeError),
    ],
)
def test_adjacency_constructor_rejects_invalid_input(matrix, exception):
    with pytest.raises(exception):
        Lattice.from_adjacency_matrix(matrix)


@pytest.mark.parametrize("num_nodes", [True, 0, -1, 2.0])
def test_general_lattice_rejects_invalid_node_count(num_nodes):
    expected = TypeError if num_nodes in (True, 2.0) else ValueError
    with pytest.raises(expected):
        Lattice.from_nodes_and_edges(num_nodes, [])


@pytest.mark.parametrize(
    "edges, exception",
    [
        ([(0, 1)], ValueError),
        ([(0, 1, 1.0, 2.0)], ValueError),
        ([(0, 2, 1.0)], ValueError),
        ([(True, 1, 1.0)], TypeError),
        ([(0, 1, True)], TypeError),
        ([(0, 1, np.inf)], ValueError),
        ([(0, 0, 1.0j)], ValueError),
        ([(0, 1, 1.0), (1, 0, 2.0)], ValueError),
        (["edge"], TypeError),
    ],
)
def test_general_lattice_rejects_invalid_edges(edges, exception):
    with pytest.raises(exception):
        Lattice.from_nodes_and_edges(2, edges)


def test_uniform_parameters_preserve_connectivity_and_add_all_self_loops():
    original = Lattice.from_nodes_and_edges(
        3,
        [(0, 1, 2.0), (1, 2, -3.0), (1, 1, 8.0)],
    )

    uniform = original.uniform_parameters(1.0 + 2.0j, -0.5)

    assert original.edge_weight(0, 0) is None
    assert uniform.weighted_edges == (
        (0, 0, -0.5 + 0.0j),
        (0, 1, 1.0 + 2.0j),
        (1, 1, -0.5 + 0.0j),
        (1, 2, 1.0 + 2.0j),
        (2, 2, -0.5 + 0.0j),
    )


@pytest.mark.parametrize(
    "interaction, onsite, exception",
    [
        (True, 0.0, TypeError),
        (np.inf, 0.0, ValueError),
        (1.0, 1.0j, ValueError),
        (1.0, np.nan, ValueError),
    ],
)
def test_uniform_parameters_reject_invalid_weights(interaction, onsite, exception):
    lattice = Lattice.from_nodes_and_edges(2, [(0, 1, 1.0)])
    with pytest.raises(exception):
        lattice.uniform_parameters(interaction, onsite)


def test_open_line_has_nearest_neighbors_and_zero_self_loops():
    lattice = LineLattice(4)

    assert isinstance(lattice, Lattice)
    assert lattice.size == (4,)
    assert lattice.dimension == 1
    assert lattice.edge_parameter == 1.0
    assert lattice.onsite_parameter == 0.0
    assert lattice.boundary_condition is BoundaryCondition.OPEN
    assert _offsite_pairs(lattice) == {(0, 1), (1, 2), (2, 3)}
    assert {edge[:2] for edge in lattice.weighted_edges if edge[0] == edge[1]} == {
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
    }
    with pytest.raises(FrozenInstanceError):
        lattice._size = (5,)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        del lattice._size  # type: ignore[misc]


def test_periodic_line_adds_one_wrap_edge_with_conjugated_upper_weight():
    lattice = LineLattice(
        4,
        edge_parameter=2.0 + 3.0j,
        boundary_condition=BoundaryCondition.PERIODIC,
    )

    assert _offsite_pairs(lattice) == {(0, 1), (1, 2), (2, 3), (0, 3)}
    assert lattice.edge_weight(3, 0) == 2.0 + 3.0j
    assert lattice.edge_weight(0, 3) == 2.0 - 3.0j


@pytest.mark.parametrize(
    "boundary", [BoundaryCondition.OPEN, BoundaryCondition.PERIODIC]
)
def test_real_line_spectrum_matches_analytic_tight_binding_result(boundary):
    num_nodes = 7
    hopping = -1.25
    onsite = 0.3
    lattice = LineLattice(
        num_nodes,
        edge_parameter=hopping,
        onsite_parameter=onsite,
        boundary_condition=boundary,
    )

    actual = np.linalg.eigvalsh(lattice.to_adjacency_matrix(weighted=True))
    if boundary is BoundaryCondition.OPEN:
        momenta = np.arange(1, num_nodes + 1) * np.pi / (num_nodes + 1)
    else:
        momenta = 2.0 * np.pi * np.arange(num_nodes) / num_nodes
    expected = onsite + 2.0 * hopping * np.cos(momenta)

    np.testing.assert_allclose(
        np.sort(actual),
        np.sort(expected),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("num_nodes", [1, 2])
def test_short_periodic_line_does_not_create_duplicate_or_self_wrap(num_nodes):
    lattice = LineLattice(num_nodes, boundary_condition=BoundaryCondition.PERIODIC)

    assert len(_offsite_pairs(lattice)) == max(0, num_nodes - 1)


def test_square_lattice_uses_axis_zero_fastest_indexing():
    lattice = SquareLattice(2, 3)

    assert lattice.rows == 2
    assert lattice.cols == 3
    assert lattice.coordinate_to_index((0, 0)) == 0
    assert lattice.coordinate_to_index((1, 0)) == 1
    assert lattice.coordinate_to_index((0, 1)) == 2
    assert lattice.index_to_coordinate(5) == (1, 2)
    assert _offsite_pairs(lattice) == {
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (2, 4),
        (3, 5),
        (4, 5),
    }


def test_square_lattice_supports_direction_specific_parameters_and_boundaries():
    lattice = SquareLattice(
        3,
        4,
        edge_parameter=(2.0, 3.0 + 1.0j),
        onsite_parameter=-1.0,
        boundary_condition=(BoundaryCondition.PERIODIC, BoundaryCondition.OPEN),
    )

    assert lattice.edge_parameters == (2.0, 3.0 + 1.0j)
    assert lattice.boundary_conditions == (
        BoundaryCondition.PERIODIC,
        BoundaryCondition.OPEN,
    )
    assert lattice.edge_weight(2, 0) == 2.0
    assert lattice.edge_weight(0, 2) == 2.0
    assert lattice.edge_weight(0, 3) == 3.0 + 1.0j
    assert lattice.edge_weight(0, 9) is None
    assert all(lattice.edge_weight(node, node) == -1.0 for node in lattice.node_indices)


def test_three_dimensional_periodic_edge_count_matches_independent_formula():
    size = (3, 4, 2)
    conditions = (
        BoundaryCondition.PERIODIC,
        BoundaryCondition.OPEN,
        BoundaryCondition.PERIODIC,
    )
    lattice = HyperCubicLattice(size, boundary_condition=conditions)

    # Axis counts: 3*4*2, 3*(4-1)*2, and 3*4*(2-1).
    expected_offsite_edges = 24 + 18 + 12
    assert len(_offsite_pairs(lattice)) == expected_offsite_edges
    assert lattice.num_edges == expected_offsite_edges + np.prod(size)


def test_extent_two_periodic_axis_obeys_documented_simple_graph_semantics():
    open_lattice = LineLattice(2, boundary_condition=BoundaryCondition.OPEN)
    periodic_lattice = LineLattice(2, boundary_condition=BoundaryCondition.PERIODIC)

    assert _offsite_pairs(open_lattice) == {(0, 1)}
    assert periodic_lattice.weighted_edges == open_lattice.weighted_edges

    three_site_periodic = LineLattice(
        3,
        boundary_condition=BoundaryCondition.PERIODIC,
    )
    assert _offsite_pairs(three_site_periodic) == {(0, 1), (0, 2), (1, 2)}


def test_periodic_square_spectrum_is_sum_of_independent_axis_dispersions():
    rows, cols = 3, 4
    row_hopping, col_hopping = 0.7, -0.4
    onsite = 0.2
    lattice = SquareLattice(
        rows,
        cols,
        edge_parameter=(row_hopping, col_hopping),
        onsite_parameter=onsite,
        boundary_condition=BoundaryCondition.PERIODIC,
    )

    actual = np.linalg.eigvalsh(lattice.to_adjacency_matrix(weighted=True))
    expected = [
        onsite
        + 2.0 * row_hopping * np.cos(2.0 * np.pi * row_mode / rows)
        + 2.0 * col_hopping * np.cos(2.0 * np.pi * col_mode / cols)
        for row_mode in range(rows)
        for col_mode in range(cols)
    ]

    np.testing.assert_allclose(
        np.sort(actual),
        np.sort(expected),
        rtol=0.0,
        atol=1e-12,
    )


def test_coordinate_conversion_is_bijective_for_full_hypercube():
    lattice = HyperCubicLattice((2, 3, 4))

    coordinates = [lattice.index_to_coordinate(index) for index in lattice.node_indices]

    assert len(set(coordinates)) == lattice.num_nodes
    assert [lattice.coordinate_to_index(item) for item in coordinates] == list(
        lattice.node_indices
    )


@pytest.mark.parametrize(
    "size, edge_parameter, boundary_condition, onsite_parameter, exception",
    [
        ((), 1.0, BoundaryCondition.OPEN, 0.0, ValueError),
        ((2, 0), 1.0, BoundaryCondition.OPEN, 0.0, ValueError),
        ((2, True), 1.0, BoundaryCondition.OPEN, 0.0, TypeError),
        ((2, 3), (1.0,), BoundaryCondition.OPEN, 0.0, ValueError),
        ((2, 3), (1.0, np.inf), BoundaryCondition.OPEN, 0.0, ValueError),
        ((2, 3), 1.0, (BoundaryCondition.OPEN,), 0.0, ValueError),
        ((2, 3), 1.0, (BoundaryCondition.OPEN, "open"), 0.0, TypeError),
        ((2, 3), 1.0, BoundaryCondition.OPEN, 1.0j, ValueError),
    ],
)
def test_hypercubic_lattice_rejects_invalid_configuration(
    size,
    edge_parameter,
    boundary_condition,
    onsite_parameter,
    exception,
):
    with pytest.raises(exception):
        HyperCubicLattice(
            size,
            edge_parameter=edge_parameter,
            boundary_condition=boundary_condition,
            onsite_parameter=onsite_parameter,
        )


@pytest.mark.parametrize(
    "coordinate, exception",
    [
        ((0,), ValueError),
        ((0, 3), ValueError),
        ((-1, 0), ValueError),
        ((True, 0), TypeError),
        ("00", TypeError),
    ],
)
def test_coordinate_conversion_rejects_invalid_coordinate(coordinate, exception):
    lattice = HyperCubicLattice((2, 3))
    with pytest.raises(exception):
        lattice.coordinate_to_index(coordinate)


def test_public_matrix_result_is_independent_of_lattice_storage():
    lattice = LineLattice(3)
    matrix = lattice.to_adjacency_matrix(weighted=True)
    matrix[0, 1] = 100.0

    assert lattice.edge_weight(0, 1) == 1.0


def test_weighted_flag_requires_a_real_bool():
    with pytest.raises(TypeError, match="weighted must be a bool"):
        LineLattice(2).to_adjacency_matrix(weighted=1)  # type: ignore[arg-type]
