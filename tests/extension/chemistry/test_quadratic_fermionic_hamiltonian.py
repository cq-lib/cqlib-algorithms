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

"""Independent tests for general quadratic fermionic Hamiltonians."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from cqlib.qis import JordanWignerMapper

from cqlib_algorithms.extension.chemistry import QuadraticFermionicHamiltonian


def _ladders(num_modes: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    annihilation: list[np.ndarray] = []
    for mode in range(num_modes):
        matrix = np.zeros((1 << num_modes, 1 << num_modes), dtype=np.complex128)
        for basis in range(1 << num_modes):
            if (basis >> mode) & 1:
                sign = -1 if (basis & ((1 << mode) - 1)).bit_count() % 2 else 1
                matrix[basis ^ (1 << mode), basis] = sign
        annihilation.append(matrix)
    creation = [matrix.conj().T for matrix in annihilation]
    return creation, annihilation


def _independent_matrix(
    constant: float,
    one_body: np.ndarray,
    pairing: np.ndarray,
) -> np.ndarray:
    num_modes = one_body.shape[0]
    creation, annihilation = _ladders(num_modes)
    result = constant * np.eye(1 << num_modes, dtype=np.complex128)
    for p, q in itertools.product(range(num_modes), repeat=2):
        result += one_body[p, q] * creation[p] @ annihilation[q]
        result += 0.5 * pairing[p, q] * creation[p] @ creation[q]
        result += 0.5 * pairing[p, q].conjugate() * annihilation[q] @ annihilation[p]
    return result


def _assert_diagonalization(component: QuadraticFermionicHamiltonian) -> None:
    transformation, energies, diagonal_constant = component.diagonalize()
    num_modes = component.num_modes
    creation, annihilation = _ladders(num_modes)
    quasiparticle_creation = []
    for row in range(num_modes):
        operator = np.zeros((1 << num_modes, 1 << num_modes), dtype=np.complex128)
        for mode in range(num_modes):
            operator += transformation[row, mode] * creation[mode]
            operator += transformation[row, num_modes + mode] * annihilation[mode]
        quasiparticle_creation.append(operator)

    reconstructed = diagonal_constant * np.eye(1 << num_modes, dtype=np.complex128)
    for energy, operator in zip(energies, quasiparticle_creation):
        reconstructed += energy * operator @ operator.conj().T
    expected = _independent_matrix(
        component.constant,
        component.one_body,
        component.pairing,
    )
    np.testing.assert_allclose(reconstructed, expected, rtol=0.0, atol=2e-10)

    creation_block = transformation[:, :num_modes]
    annihilation_block = transformation[:, num_modes:]
    np.testing.assert_allclose(
        creation_block @ creation_block.conj().T
        + annihilation_block @ annihilation_block.conj().T,
        np.eye(num_modes),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        creation_block @ annihilation_block.T + annihilation_block @ creation_block.T,
        np.zeros((num_modes, num_modes)),
        rtol=0.0,
        atol=1e-12,
    )
    assert np.all(energies >= 0.0)

    bogoliubov_de_gennes = np.block(
        [
            [component.one_body, component.pairing],
            [-component.pairing.conj(), -component.one_body.conj()],
        ]
    )
    np.testing.assert_allclose(
        np.linalg.eigvalsh(bogoliubov_de_gennes),
        np.concatenate((-energies[::-1], energies)),
        rtol=0.0,
        atol=2e-10,
    )

    exact_spectrum = np.linalg.eigvalsh(expected)
    occupation_spectrum = sorted(
        diagonal_constant
        + sum(energies[index] for index in range(num_modes) if (bits >> index) & 1)
        for bits in range(1 << num_modes)
    )
    np.testing.assert_allclose(
        exact_spectrum,
        occupation_spectrum,
        rtol=0.0,
        atol=2e-10,
    )


def test_one_mode_number_conserving_analytic_spectrum():
    component = QuadraticFermionicHamiltonian(
        0.3,
        np.array([[-1.7]]),
        np.zeros((1, 1)),
    )
    transformation, energies, diagonal_constant = component.diagonalize()
    assert energies == pytest.approx([1.7], abs=1e-12)
    assert diagonal_constant == pytest.approx(-1.4, abs=1e-12)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(
            _independent_matrix(0.3, component.one_body, component.pairing)
        ),
        [-1.4, 0.3],
        rtol=0.0,
        atol=1e-12,
    )
    assert transformation.flags.writeable is False


@pytest.mark.parametrize(
    ("one_body", "pairing"),
    [
        (
            np.array([[0.4, 0.2], [0.2, -0.1]], dtype=np.complex128),
            np.array([[0.0, 0.7], [-0.7, 0.0]], dtype=np.complex128),
        ),
        (
            np.array([[0.4, 0.2j], [-0.2j, -0.1]], dtype=np.complex128),
            np.array([[0.0, 0.3 + 0.5j], [-0.3 - 0.5j, 0.0]]),
        ),
    ],
)
def test_two_mode_real_and_complex_pairing(one_body, pairing):
    component = QuadraticFermionicHamiltonian(0.17, one_body, pairing)
    expected = _independent_matrix(0.17, one_body, pairing)
    actual = JordanWignerMapper().map_operator(component.to_fermionic_op()).to_matrix()
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    _assert_diagonalization(component)


@pytest.mark.parametrize("num_modes", [2, 3, 4])
def test_random_quadratic_models_reconstruct_and_match_full_fock_spectrum(num_modes):
    rng = np.random.default_rng(9137 + num_modes)
    for _ in range(5):
        raw_one_body = rng.normal(size=(num_modes, num_modes)) + 1j * rng.normal(
            size=(num_modes, num_modes)
        )
        one_body = raw_one_body + raw_one_body.conj().T
        raw_pairing = rng.normal(size=(num_modes, num_modes)) + 1j * rng.normal(
            size=(num_modes, num_modes)
        )
        pairing = raw_pairing - raw_pairing.T
        component = QuadraticFermionicHamiltonian(-0.28, one_body, pairing)
        expected = _independent_matrix(-0.28, one_body, pairing)
        actual = (
            JordanWignerMapper().map_operator(component.to_fermionic_op()).to_matrix()
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-12)
        _assert_diagonalization(component)


def test_zero_modes_and_degenerate_modes_are_stable():
    zero = QuadraticFermionicHamiltonian(0.8, np.zeros((3, 3)), np.zeros((3, 3)))
    transformation, energies, diagonal_constant = zero.diagonalize()
    np.testing.assert_array_equal(energies, np.zeros(3))
    assert diagonal_constant == pytest.approx(0.8)
    np.testing.assert_allclose(transformation[:, :3], np.eye(3), atol=1e-12)
    _assert_diagonalization(zero)

    degenerate = QuadraticFermionicHamiltonian(
        -0.2,
        np.diag([1.5, 1.5, -1.5, -1.5]),
        np.zeros((4, 4)),
    )
    _, degenerate_energies, degenerate_constant = degenerate.diagonalize()
    np.testing.assert_allclose(
        degenerate_energies,
        [1.5, 1.5, 1.5, 1.5],
        rtol=0.0,
        atol=1e-14,
    )
    assert degenerate_constant == pytest.approx(-3.2, abs=1e-14)
    _assert_diagonalization(degenerate)

    near_zero = QuadraticFermionicHamiltonian(
        0.0,
        np.diag([1e-11, -0.8]),
        np.zeros((2, 2)),
    )
    _, energies, _ = near_zero.diagonalize()
    assert energies == pytest.approx([1e-11, 0.8], abs=1e-13)
    _assert_diagonalization(near_zero)


def test_diagonalization_preserves_independent_extreme_energy_scales():
    component = QuadraticFermionicHamiltonian(
        0.0,
        np.diag([1e12, 1.0]),
        np.zeros((2, 2)),
    )
    _, energies, diagonal_constant = component.diagonalize()
    np.testing.assert_allclose(energies, [1.0, 1e12], rtol=1e-15, atol=0.0)
    assert diagonal_constant == 0.0
    exact_spectrum = np.linalg.eigvalsh(
        _independent_matrix(0.0, component.one_body, component.pairing)
    )
    np.testing.assert_allclose(
        exact_spectrum,
        [0.0, 1.0, 1e12, 1e12 + 1.0],
        rtol=0.0,
        atol=0.0,
    )
    _assert_diagonalization(component)


def test_diagonalization_accepts_backward_stable_rotated_dynamic_range():
    angle = 0.417
    orbital_rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.complex128,
    )
    one_body = orbital_rotation @ np.diag([1.0, 1e8]) @ orbital_rotation.conj().T
    one_body = 0.5 * (one_body + one_body.conj().T)
    component = QuadraticFermionicHamiltonian(0.0, one_body, np.zeros((2, 2)))
    _, energies, diagonal_constant = component.diagonalize()
    np.testing.assert_allclose(energies, [1.0, 1e8], rtol=2e-8, atol=0.0)
    assert abs(diagonal_constant) <= 1e-7
    exact_spectrum = np.linalg.eigvalsh(
        _independent_matrix(0.0, component.one_body, component.pairing)
    )
    occupation_spectrum = np.sort(
        np.array(
            [
                diagonal_constant,
                diagonal_constant + energies[0],
                diagonal_constant + energies[1],
                diagonal_constant + energies.sum(),
            ]
        )
    )
    np.testing.assert_allclose(
        occupation_spectrum,
        exact_spectrum,
        rtol=0.0,
        atol=1e-7,
    )


def test_diagonalization_preserves_tiny_absolute_energy():
    tiny_energy = 1e-200
    component = QuadraticFermionicHamiltonian(
        0.0,
        np.array([[tiny_energy]]),
        np.zeros((1, 1)),
    )
    _, energies, diagonal_constant = component.diagonalize()
    np.testing.assert_allclose(energies, [tiny_energy], rtol=1e-15, atol=0.0)
    assert diagonal_constant == 0.0
    np.testing.assert_allclose(
        np.linalg.eigvalsh(
            _independent_matrix(0.0, component.one_body, component.pairing)
        ),
        [0.0, tiny_energy],
        rtol=1e-15,
        atol=0.0,
    )


def test_diagonalization_pairs_exact_zero_modes_without_losing_nonzero_modes():
    component = QuadraticFermionicHamiltonian(
        0.25,
        np.diag([0.0, 2.0, -2.0, 0.0]),
        np.zeros((4, 4)),
    )
    _, energies, diagonal_constant = component.diagonalize()
    np.testing.assert_array_equal(energies, [0.0, 0.0, 2.0, 2.0])
    assert diagonal_constant == pytest.approx(-1.75, abs=1e-14)
    _assert_diagonalization(component)


def test_inputs_are_immutable_snapshots_and_outputs_are_independent():
    one_body = np.array([[0.3, 0.2j], [-0.2j, -0.7]])
    pairing = np.array([[0.0, 0.4], [-0.4, 0.0]])
    component = QuadraticFermionicHamiltonian(0.1, one_body, pairing)
    one_body[:] = 9.0
    pairing[:] = 8.0
    assert component.one_body.flags.writeable is False
    assert component.pairing.flags.writeable is False
    first = component.to_fermionic_op()
    second = component.to_fermionic_op()
    assert first is not second
    first_transformation, first_energies, _ = component.diagonalize()
    second_transformation, second_energies, _ = component.diagonalize()
    assert first_transformation is not second_transformation
    assert first_energies is not second_energies
    assert first_transformation.flags.writeable is False
    assert first_energies.flags.writeable is False


def test_roundoff_sized_structure_residuals_are_canonicalized_consistently():
    one_body = np.array(
        [[1.0 + 4e-13j, 0.2 + 3e-13j], [0.2 + 1e-13j, -0.4 - 2e-13j]],
        dtype=np.complex128,
    )
    pairing = np.array(
        [[3e-13, 0.7 + 2e-13j], [-0.7 + 4e-13j, -2e-13]],
        dtype=np.complex128,
    )
    component = QuadraticFermionicHamiltonian(0.15, one_body, pairing)
    np.testing.assert_array_equal(component.one_body, component.one_body.conj().T)
    np.testing.assert_array_equal(component.pairing, -component.pairing.T)
    actual = JordanWignerMapper().map_operator(component.to_fermionic_op()).to_matrix()
    expected = _independent_matrix(
        component.constant,
        component.one_body,
        component.pairing,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(actual, actual.conj().T, rtol=0.0, atol=1e-14)
    _assert_diagonalization(component)


def test_structure_residuals_above_tolerance_are_rejected():
    with pytest.raises(ValueError, match="Hermitian"):
        QuadraticFermionicHamiltonian(
            0.0,
            np.array([[1.0 + 2e-12j]]),
            np.zeros((1, 1)),
        )
    with pytest.raises(ValueError, match="antisymmetric"):
        QuadraticFermionicHamiltonian(
            0.0,
            np.eye(2),
            np.array([[0.0, 1.0], [-1.0 + 2e-12, 0.0]]),
        )


@pytest.mark.parametrize(
    ("constant", "one_body", "pairing", "error", "message"),
    [
        (True, [[1.0]], [[0.0]], TypeError, "constant"),
        (1.0 + 0.0j, [[1.0]], [[0.0]], TypeError, "constant"),
        (float("nan"), [[1.0]], [[0.0]], ValueError, "constant"),
        (0.0, [1.0], [[0.0]], ValueError, "square"),
        (0.0, np.zeros((0, 0)), np.zeros((0, 0)), ValueError, "at least one"),
        (0.0, [[1.0, 0.0]], [[0.0]], ValueError, "square"),
        (0.0, [[1.0]], np.zeros((2, 2)), ValueError, "same square"),
        (0.0, [[1.0, 1.0], [0.0, 1.0]], np.zeros((2, 2)), ValueError, "Hermitian"),
        (0.0, np.eye(2), [[0.0, 1.0], [0.0, 0.0]], ValueError, "antisymmetric"),
        (0.0, [[True]], [[0.0]], TypeError, "booleans"),
        (0.0, [[float("inf")]], [[0.0]], ValueError, "finite"),
        (0.0, [[1.0], [1.0, 2.0]], [[0.0]], TypeError, "rectangular"),
    ],
)
def test_invalid_quadratic_inputs_are_rejected(
    constant,
    one_body,
    pairing,
    error,
    message,
):
    with pytest.raises(error, match=message):
        QuadraticFermionicHamiltonian(constant, one_body, pairing)
