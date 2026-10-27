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

"""Independent scientific tests for Slater-determinant state preparation."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pytest
from cqlib.circuit import Circuit

from cqlib_algorithms.extension.chemistry import ElectronicOccupation
from cqlib_algorithms.operators.state_preparation import (
    HartreeFockStatePreparation,
    SlaterDeterminantStatePreparation,
)
from cqlib_algorithms.operators.state_preparation._slater_decomposition import (
    _AdjacentGivens,
    append_adjacent_givens,
)
from cqlib_algorithms.protocols import StatePreparation


def _determinant_state(coefficients: np.ndarray) -> np.ndarray:
    num_modes, num_particles = coefficients.shape
    state = np.zeros(1 << num_modes, dtype=np.complex128)
    for occupied in combinations(range(num_modes), num_particles):
        basis_index = sum(1 << mode for mode in occupied)
        state[basis_index] = np.linalg.det(coefficients[list(occupied), :])
    return state


def _assert_equal_up_to_global_phase(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float = 1e-11,
) -> None:
    overlap = np.vdot(expected, actual)
    assert abs(overlap) == pytest.approx(1.0, abs=atol)
    phase = overlap / abs(overlap)
    np.testing.assert_allclose(actual, phase * expected, rtol=0.0, atol=atol)


def _apply_ladder(state: int, mode: int, *, create: bool) -> tuple[int, float] | None:
    occupied = bool(state & (1 << mode))
    if occupied == create:
        return None
    sign = -1.0 if (state & ((1 << mode) - 1)).bit_count() % 2 else 1.0
    output = state | (1 << mode) if create else state & ~(1 << mode)
    return output, sign


def _one_particle_density_matrix(state: np.ndarray, num_modes: int) -> np.ndarray:
    """Return gamma[p,q]=<a_q† a_p> using an independent Fock action."""
    gamma = np.zeros((num_modes, num_modes), dtype=np.complex128)
    for p in range(num_modes):
        for q in range(num_modes):
            for basis, amplitude in enumerate(state):
                annihilated = _apply_ladder(basis, p, create=False)
                if annihilated is None:
                    continue
                intermediate, first_sign = annihilated
                created = _apply_ladder(intermediate, q, create=True)
                if created is None:
                    continue
                output, second_sign = created
                gamma[p, q] += (
                    np.conj(state[output]) * first_sign * second_sign * amplitude
                )
    return gamma


def _assert_prepares_slater(coefficients: np.ndarray) -> np.ndarray:
    preparation = SlaterDeterminantStatePreparation(coefficients)
    actual = preparation.build_circuit("slater").to_matrix()[:, 0]
    expected = _determinant_state(coefficients)
    _assert_equal_up_to_global_phase(actual, expected)

    num_particles = coefficients.shape[1]
    particle_number = sum(
        basis.bit_count() * abs(amplitude) ** 2
        for basis, amplitude in enumerate(actual)
    )
    assert particle_number == pytest.approx(float(num_particles), abs=1e-11)
    np.testing.assert_allclose(
        _one_particle_density_matrix(actual, coefficients.shape[0]),
        coefficients @ coefficients.conj().T,
        rtol=0.0,
        atol=1e-11,
    )
    return actual


@pytest.mark.reference
def test_adjacent_complex_givens_matches_independent_four_by_four_fock_lift():
    mixing = 0.37
    alpha = math.cos(mixing) * np.exp(0.29j)
    beta = math.sin(mixing) * np.exp(-0.61j)
    one_particle = np.array(
        ((alpha, beta), (-np.conj(beta), np.conj(alpha))),
        dtype=np.complex128,
    )
    circuit = Circuit(2)
    append_adjacent_givens(circuit, _AdjacentGivens(0, 1, alpha, beta))
    expected = np.eye(4, dtype=np.complex128)
    expected[np.ix_((1, 2), (1, 2))] = one_particle
    np.testing.assert_allclose(
        circuit.to_matrix(),
        expected,
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.reference
def test_analytic_complex_single_particle_and_nonadjacent_support():
    theta = 0.73
    phase = -0.41
    coefficients = np.array(
        [[math.cos(theta)], [0.0], [np.exp(1j * phase) * math.sin(theta)]],
        dtype=np.complex128,
    )
    _assert_prepares_slater(coefficients)


@pytest.mark.reference
def test_two_particle_complex_nonadjacent_orbitals():
    raw = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0j],
            [1.0j, 0.0],
            [0.0, -1.0],
        ],
        dtype=np.complex128,
    )
    coefficients = np.linalg.qr(raw)[0][:, :2]
    _assert_prepares_slater(coefficients)


@pytest.mark.reference
@pytest.mark.parametrize("num_modes", range(1, 6))
def test_random_complex_qr_isometries_cover_all_particle_sectors(num_modes: int):
    rng = np.random.default_rng(100 + num_modes)
    for num_particles in range(num_modes + 1):
        if num_particles == 0:
            coefficients = np.empty((num_modes, 0), dtype=np.complex128)
        else:
            raw = rng.normal(size=(num_modes, num_particles)) + 1j * rng.normal(
                size=(num_modes, num_particles)
            )
            coefficients = np.linalg.qr(raw)[0][:, :num_particles]
        _assert_prepares_slater(coefficients)


@pytest.mark.reference
def test_hartree_fock_special_case_matches_existing_preparation():
    coefficients = np.eye(6, 3, dtype=np.complex128)
    slater = SlaterDeterminantStatePreparation(coefficients).build_circuit("s")
    occupation = ElectronicOccupation(3, 2, 1)
    hartree_fock = HartreeFockStatePreparation(occupation).build_circuit("hf")
    np.testing.assert_allclose(
        slater.to_matrix()[:, 0],
        hartree_fock.to_matrix()[:, 0],
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.reference
def test_full_occupation_column_phases_and_permutations_change_only_global_phase():
    rng = np.random.default_rng(811)
    raw = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    coefficients = np.linalg.qr(raw)[0]
    original = _assert_prepares_slater(coefficients)

    phased = coefficients @ np.diag(np.exp(1j * np.array([0.2, -0.5, 0.7, 1.1])))
    permuted = coefficients[:, [2, 0, 3, 1]]
    for transformed in (phased, permuted):
        actual = _assert_prepares_slater(transformed)
        _assert_equal_up_to_global_phase(actual, original)


@pytest.mark.reference
def test_column_phase_and_odd_column_swap_have_only_global_effect():
    rng = np.random.default_rng(301)
    raw = rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
    coefficients = np.linalg.qr(raw)[0][:, :3]
    original = _assert_prepares_slater(coefficients)
    phased = coefficients @ np.diag(np.exp(1j * np.array([0.8, -0.3, 1.2])))
    swapped = coefficients[:, [1, 0, 2]]
    for transformed in (phased, swapped):
        actual = _assert_prepares_slater(transformed)
        _assert_equal_up_to_global_phase(actual, original)


@pytest.mark.reference
def test_near_degenerate_elimination_pivot_is_stable():
    epsilon = 1e-14
    coefficients = np.array(
        [
            [epsilon, 0.0],
            [math.sqrt(1.0 - epsilon**2), 0.0],
            [0.0, np.exp(0.37j)],
        ],
        dtype=np.complex128,
    )
    _assert_prepares_slater(coefficients)


def test_input_is_snapshotted_and_builds_are_independent_without_parameters():
    coefficients = np.array(
        [[1 / math.sqrt(2)], [1j / math.sqrt(2)]],
        dtype=np.complex128,
    )
    expected = _determinant_state(coefficients.copy())
    preparation = SlaterDeterminantStatePreparation(coefficients)
    coefficients[:] = np.array([[1.0], [0.0]])

    assert isinstance(preparation, StatePreparation)
    assert preparation.num_qubits() == 2
    assert preparation.num_parameters() == 0
    preparation.validate()
    first = preparation.build_circuit("a")
    second = preparation.build_circuit("b")
    _assert_equal_up_to_global_phase(second.to_matrix()[:, 0], expected)
    first.x(0)
    assert first != second
    assert not second.used_symbols


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (1.0, ValueError, "two-dimensional"),
        ([1.0, 0.0], ValueError, "two-dimensional"),
        (np.zeros((1, 1, 1)), ValueError, "two-dimensional"),
        (np.empty((0, 0)), ValueError, "at least one mode"),
        (np.zeros((2, 3)), ValueError, "must not exceed"),
        ([[True], [False]], TypeError, "bool"),
        ([[1.0], [False]], TypeError, "bool"),
        ([[1.0], [1.0, 0.0]], TypeError, "rectangular"),
        ([["invalid"]], TypeError, "rectangular complex matrix"),
        ([[np.nan]], ValueError, "finite"),
        ([[np.inf]], ValueError, "finite"),
        ([[1.0], [1.0]], ValueError, "orthonormal"),
        (
            [[1.0, 1e-6], [0.0, math.sqrt(1.0 - 1e-12)]],
            ValueError,
            "orthonormal",
        ),
    ],
)
def test_rejects_invalid_orbital_coefficients(value, error, message):
    with pytest.raises(error, match=message):
        SlaterDeterminantStatePreparation(value)


def test_prefix_must_be_a_string():
    preparation = SlaterDeterminantStatePreparation(np.eye(2, 1))
    with pytest.raises(TypeError, match="prefix must be a str"):
        preparation.build_circuit(1)  # type: ignore[arg-type]
