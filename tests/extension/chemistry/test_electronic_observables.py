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

"""Independent Fock-space checks for electronic observables."""

import numpy as np
import pytest

from cqlib_algorithms.extension.chemistry import (
    AngularMomentum,
    Magnetization,
    ParticleNumber,
)


def _ladder_matrix(num_modes: int, mode: int, *, create: bool) -> np.ndarray:
    size = 1 << num_modes
    matrix = np.zeros((size, size), dtype=np.complex128)
    for basis in range(size):
        occupied = bool(basis & (1 << mode))
        if occupied == create:
            continue
        sign = -1.0 if (basis & ((1 << mode) - 1)).bit_count() % 2 else 1.0
        output = basis | (1 << mode) if create else basis & ~(1 << mode)
        matrix[output, basis] = sign
    return matrix


def _reference_observables(num_spatial_orbitals: int) -> tuple[np.ndarray, ...]:
    num_modes = 2 * num_spatial_orbitals
    size = 1 << num_modes
    number = np.zeros((size, size), dtype=np.complex128)
    spin_z = np.zeros_like(number)
    spin_plus = np.zeros_like(number)
    for spatial in range(num_spatial_orbitals):
        alpha = 2 * spatial
        beta = alpha + 1
        create_alpha = _ladder_matrix(num_modes, alpha, create=True)
        annihilate_alpha = _ladder_matrix(num_modes, alpha, create=False)
        create_beta = _ladder_matrix(num_modes, beta, create=True)
        annihilate_beta = _ladder_matrix(num_modes, beta, create=False)
        alpha_number = create_alpha @ annihilate_alpha
        beta_number = create_beta @ annihilate_beta
        number += alpha_number + beta_number
        spin_z += 0.5 * (alpha_number - beta_number)
        spin_plus += create_alpha @ annihilate_beta
    spin_minus = spin_plus.conj().T
    spin_squared = (
        0.5 * (spin_plus @ spin_minus + spin_minus @ spin_plus) + spin_z @ spin_z
    )
    return number, spin_z, spin_squared


def _reference_spin_squared(overlap: np.ndarray) -> np.ndarray:
    num_spatial_orbitals = overlap.shape[0]
    num_modes = 2 * num_spatial_orbitals
    size = 1 << num_modes
    spin_z = np.zeros((size, size), dtype=np.complex128)
    spin_plus = np.zeros_like(spin_z)
    for spatial in range(num_spatial_orbitals):
        alpha = 2 * spatial
        beta = alpha + 1
        spin_z += 0.5 * (
            _ladder_matrix(num_modes, alpha, create=True)
            @ _ladder_matrix(num_modes, alpha, create=False)
            - _ladder_matrix(num_modes, beta, create=True)
            @ _ladder_matrix(num_modes, beta, create=False)
        )
    for alpha in range(num_spatial_orbitals):
        for beta in range(num_spatial_orbitals):
            spin_plus += (
                overlap[alpha, beta]
                * _ladder_matrix(num_modes, 2 * alpha, create=True)
                @ _ladder_matrix(num_modes, 2 * beta + 1, create=False)
            )
    spin_minus = spin_plus.conj().T
    return 0.5 * (spin_plus @ spin_minus + spin_minus @ spin_plus) + spin_z @ spin_z


@pytest.mark.parametrize("num_spatial_orbitals", [1, 2, 3])
def test_observable_matrices_match_independent_fock_space_references(
    num_spatial_orbitals,
):
    expected = _reference_observables(num_spatial_orbitals)
    actual = (
        ParticleNumber(num_spatial_orbitals).to_qubit_hamiltonian().to_matrix(),
        Magnetization(num_spatial_orbitals).to_qubit_hamiltonian().to_matrix(),
        AngularMomentum(num_spatial_orbitals).to_qubit_hamiltonian().to_matrix(),
    )
    for matrix, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(matrix, reference, rtol=0.0, atol=1e-12)


def test_total_spin_distinguishes_singlet_and_triplet_sectors():
    matrix = AngularMomentum(2).to_qubit_hamiltonian().to_matrix()
    closed_shell = np.eye(16)[:, 0b0011]
    triplet_m_plus = np.eye(16)[:, 0b0101]

    assert np.vdot(closed_shell, matrix @ closed_shell).real == pytest.approx(0.0)
    assert np.vdot(triplet_m_plus, matrix @ triplet_m_plus).real == pytest.approx(2.0)


@pytest.mark.reference
def test_total_spin_uses_alpha_beta_orbital_overlap():
    angle = 0.37
    overlap = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    expected = _reference_spin_squared(overlap)
    actual = AngularMomentum(2, overlap).to_qubit_hamiltonian().to_matrix()

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    identity_result = AngularMomentum(2).to_qubit_hamiltonian().to_matrix()
    assert not np.allclose(actual, identity_result, rtol=0.0, atol=1e-8)


def test_total_spin_overlap_is_snapshotted_and_validated():
    overlap = np.eye(2)
    observable = AngularMomentum(2, overlap)
    overlap[0, 0] = 4.0
    np.testing.assert_array_equal(observable.alpha_beta_overlap, np.eye(2))
    assert not observable.alpha_beta_overlap.flags.writeable

    with pytest.raises(ValueError, match="shape"):
        AngularMomentum(2, np.eye(3))
    with pytest.raises(ValueError, match="finite"):
        AngularMomentum(2, np.full((2, 2), np.nan))


@pytest.mark.parametrize("factory", [ParticleNumber, Magnetization, AngularMomentum])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_observable_factories_reject_invalid_orbital_counts(factory, value):
    error = TypeError if isinstance(value, (bool, float, str)) else ValueError
    with pytest.raises(error):
        factory(value)


def test_observable_factories_reject_invalid_mappers():
    with pytest.raises(TypeError, match="mapper"):
        ParticleNumber(1).to_qubit_hamiltonian(object())  # type: ignore[arg-type]
