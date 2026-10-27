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

"""Independent state tests for fixed pure fermionic Gaussian preparations."""

from __future__ import annotations

import numpy as np
import pytest

from cqlib_algorithms.extension.chemistry import QuadraticFermionicHamiltonian
from cqlib_algorithms.operators.state_preparation import (
    FermionicGaussianStatePreparation,
    SlaterDeterminantStatePreparation,
)


def _ladders(num_modes: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    annihilation: list[np.ndarray] = []
    for mode in range(num_modes):
        matrix = np.zeros((1 << num_modes, 1 << num_modes), dtype=np.complex128)
        for basis in range(1 << num_modes):
            if (basis >> mode) & 1:
                sign = -1 if (basis & ((1 << mode) - 1)).bit_count() % 2 else 1
                matrix[basis ^ (1 << mode), basis] = sign
        annihilation.append(matrix)
    return [matrix.conj().T for matrix in annihilation], annihilation


def _majorana_to_w(orthogonal: np.ndarray) -> np.ndarray:
    even_even = orthogonal[0::2, 0::2]
    even_odd = orthogonal[0::2, 1::2]
    odd_even = orthogonal[1::2, 0::2]
    odd_odd = orthogonal[1::2, 1::2]
    creation = 0.5 * (even_even + odd_odd + 1j * (even_odd - odd_even))
    annihilation = 0.5 * (even_even - odd_odd - 1j * (even_odd + odd_even))
    return np.concatenate((creation, annihilation), axis=1)


def _align_global_phase(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    pivot = int(np.argmax(np.abs(expected)))
    if abs(expected[pivot]) == 0.0:
        return actual
    phase = actual[pivot] / expected[pivot]
    return actual / phase


def _assert_quasiparticle_occupation(
    transformation: np.ndarray,
    occupied: tuple[int, ...],
    state: np.ndarray,
) -> None:
    num_modes = transformation.shape[0]
    creation, annihilation = _ladders(num_modes)
    for index in range(num_modes):
        b_dagger = np.zeros_like(creation[0])
        for mode in range(num_modes):
            b_dagger += transformation[index, mode] * creation[mode]
            b_dagger += transformation[index, num_modes + mode] * annihilation[mode]
        number = b_dagger @ b_dagger.conj().T
        expected = 1.0 if index in occupied else 0.0
        np.testing.assert_allclose(number @ state, expected * state, atol=2e-11)


def _covariance(state: np.ndarray, num_modes: int) -> np.ndarray:
    creation, annihilation = _ladders(num_modes)
    majoranas: list[np.ndarray] = []
    for mode in range(num_modes):
        majoranas.extend(
            (
                creation[mode] + annihilation[mode],
                1j * (creation[mode] - annihilation[mode]),
            )
        )
    covariance = np.zeros((2 * num_modes, 2 * num_modes), dtype=np.float64)
    for first in range(2 * num_modes):
        for second in range(2 * num_modes):
            commutator = (
                majoranas[first] @ majoranas[second]
                - majoranas[second] @ majoranas[first]
            )
            covariance[first, second] = float(
                np.real(0.5j * np.vdot(state, commutator @ state))
            )
    return covariance


def _w_to_majorana(transformation: np.ndarray) -> np.ndarray:
    num_modes = transformation.shape[0]
    creation = transformation[:, :num_modes]
    annihilation = transformation[:, num_modes:]
    result = np.empty((2 * num_modes, 2 * num_modes), dtype=np.float64)
    result[0::2, 0::2] = np.real(creation + annihilation)
    result[0::2, 1::2] = np.imag(creation - annihilation)
    result[1::2, 0::2] = -np.imag(creation + annihilation)
    result[1::2, 1::2] = np.real(creation - annihilation)
    return result


def test_identity_vacuum_and_full_quasiparticle_occupation():
    identity = np.eye(3, dtype=np.complex128)
    vacuum = FermionicGaussianStatePreparation(identity)
    full = FermionicGaussianStatePreparation(
        identity,
        occupied_quasiparticles=(0, 1, 2),
    )
    np.testing.assert_array_equal(
        vacuum.build_circuit("vac").to_matrix()[:, 0],
        np.eye(8, dtype=np.complex128)[:, 0],
    )
    np.testing.assert_allclose(
        full.build_circuit("full").to_matrix()[:, 0],
        np.eye(8, dtype=np.complex128)[:, 7],
        rtol=0.0,
        atol=1e-12,
    )
    assert vacuum.particle_parity == 0
    assert full.particle_parity == 1


def test_number_conserving_case_matches_existing_slater_component():
    rng = np.random.default_rng(181)
    raw = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
    orbital_unitary, _ = np.linalg.qr(raw)
    occupied = (0, 2, 4)
    gaussian = (
        FermionicGaussianStatePreparation(
            orbital_unitary,
            occupied_quasiparticles=occupied,
        )
        .build_circuit("gaussian")
        .to_matrix()[:, 0]
    )
    coefficients = orbital_unitary[np.asarray(occupied), :].T
    slater = (
        SlaterDeterminantStatePreparation(coefficients)
        .build_circuit("slater")
        .to_matrix()[:, 0]
    )
    np.testing.assert_allclose(
        _align_global_phase(gaussian, slater),
        slater,
        rtol=0.0,
        atol=1e-11,
    )


@pytest.mark.parametrize("num_modes", [2, 3, 4])
def test_random_gaussian_states_have_expected_occupation_covariance_and_parity(
    num_modes,
):
    rng = np.random.default_rng(3400 + num_modes)
    for sample in range(4):
        raw = rng.normal(size=(2 * num_modes, 2 * num_modes))
        orthogonal, _ = np.linalg.qr(raw)
        if sample % 2:
            orthogonal[:, 0] *= -1.0
        transformation = _majorana_to_w(orthogonal)
        occupied = tuple(index for index in range(num_modes) if index % 2 == sample % 2)
        component = FermionicGaussianStatePreparation(
            transformation,
            occupied_quasiparticles=occupied,
        )
        state = component.build_circuit("g").to_matrix()[:, 0]
        _assert_quasiparticle_occupation(transformation, occupied, state)

        reference_covariance = np.zeros((2 * num_modes, 2 * num_modes))
        for mode in range(num_modes):
            sign = 1.0 if mode in occupied else -1.0
            reference_covariance[2 * mode, 2 * mode + 1] = sign
            reference_covariance[2 * mode + 1, 2 * mode] = -sign
        expected_covariance = orthogonal.T @ reference_covariance @ orthogonal
        np.testing.assert_allclose(
            _covariance(state, num_modes),
            expected_covariance,
            rtol=0.0,
            atol=2e-11,
        )

        support = np.flatnonzero(np.abs(state) > 1e-11)
        assert support.size > 0
        assert all(
            index.bit_count() % 2 == component.particle_parity for index in support
        )


@pytest.mark.parametrize("num_modes", [2, 3, 4])
def test_diagonalized_quadratic_hamiltonian_states_are_exact_eigenstates(num_modes):
    rng = np.random.default_rng(509 + num_modes)
    raw_one_body = rng.normal(size=(num_modes, num_modes)) + 1j * rng.normal(
        size=(num_modes, num_modes)
    )
    one_body = raw_one_body + raw_one_body.conj().T
    raw_pairing = rng.normal(size=(num_modes, num_modes)) + 1j * rng.normal(
        size=(num_modes, num_modes)
    )
    pairing = raw_pairing - raw_pairing.T
    hamiltonian = QuadraticFermionicHamiltonian(0.37, one_body, pairing)
    transformation, energies, diagonal_constant = hamiltonian.diagonalize()

    creation, annihilation = _ladders(num_modes)
    matrix = 0.37 * np.eye(1 << num_modes, dtype=np.complex128)
    for p in range(num_modes):
        for q in range(num_modes):
            matrix += one_body[p, q] * creation[p] @ annihilation[q]
            matrix += 0.5 * pairing[p, q] * creation[p] @ creation[q]
            matrix += (
                0.5 * pairing[p, q].conjugate() * annihilation[q] @ annihilation[p]
            )

    occupations = tuple(
        tuple(index for index in range(num_modes) if (bits >> index) & 1)
        for bits in range(1 << num_modes)
    )
    for occupied in occupations:
        state = (
            FermionicGaussianStatePreparation(
                transformation,
                occupied_quasiparticles=occupied,
            )
            .build_circuit("eigen")
            .to_matrix()[:, 0]
        )
        eigenvalue = diagonal_constant + sum(energies[index] for index in occupied)
        np.testing.assert_allclose(matrix @ state, eigenvalue * state, atol=3e-10)


def test_defensive_copy_repeated_build_no_parameters_and_prefix_validation():
    orthogonal = np.eye(4)
    angle = 0.38
    orthogonal[np.ix_([0, 3], [0, 3])] = [
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)],
    ]
    transformation = _majorana_to_w(orthogonal)
    component = FermionicGaussianStatePreparation(transformation)
    transformation[:] = 0.0
    assert component.num_parameters() == 0
    assert component.num_qubits() == 2
    assert not component.build_circuit("g").used_symbols
    first = component.build_circuit("g")
    second = component.build_circuit("g")
    first.x(0)
    assert not np.allclose(first.to_matrix(), second.to_matrix())
    with pytest.raises(TypeError, match="prefix"):
        component.build_circuit(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("occupied", "error", "message"),
    [
        ("0", TypeError, "sequence"),
        ([True], TypeError, "only ints"),
        ([0.0], TypeError, "only ints"),
        ([-1], ValueError, "out-of-range"),
        ([2], ValueError, "out-of-range"),
        ([1, 0], ValueError, "strictly increasing"),
        ([0, 0], ValueError, "strictly increasing"),
    ],
)
def test_invalid_quasiparticle_occupations_are_rejected(occupied, error, message):
    with pytest.raises(error, match=message):
        FermionicGaussianStatePreparation(
            np.eye(2),
            occupied_quasiparticles=occupied,
        )
