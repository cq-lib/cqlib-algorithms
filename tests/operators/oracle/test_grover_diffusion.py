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

"""Tests for the Grover diffusion operator."""

from __future__ import annotations

import numpy as np
import pytest
from cqlib import Circuit, Qubit

from cqlib_algorithms.operators.oracle import GroverDiffusion


def _diffusion_reference(num_qubits: int) -> np.ndarray:
    dimension = 1 << num_qubits
    return 2.0 * np.ones(
        (dimension, dimension), dtype=np.complex128
    ) / dimension - np.eye(dimension, dtype=np.complex128)


@pytest.mark.parametrize("num_qubits", [1, 2, 3, 4])
def test_grover_diffusion_matches_mean_inversion_matrix(num_qubits):
    matrix = GroverDiffusion(num_qubits).build_circuit().to_matrix()

    np.testing.assert_allclose(
        matrix, _diffusion_reference(num_qubits), rtol=0.0, atol=1e-12
    )


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
def test_grover_diffusion_is_hermitian_unitary_and_involutory(num_qubits):
    matrix = GroverDiffusion(num_qubits).build_circuit().to_matrix()
    identity = np.eye(1 << num_qubits)

    np.testing.assert_allclose(matrix, matrix.conj().T, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(matrix.conj().T @ matrix, identity, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(matrix @ matrix, identity, rtol=0.0, atol=1e-12)


def test_one_qubit_diffusion_is_exact_pauli_x_not_negative_x():
    np.testing.assert_allclose(
        GroverDiffusion(1).build_circuit().to_matrix(),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        rtol=0.0,
        atol=1e-12,
    )


def test_build_returns_fresh_independent_circuits():
    diffusion = GroverDiffusion(2)
    first = diffusion.build_circuit()
    second = diffusion.build_circuit()

    assert first is not second
    assert first == second
    second_size = len(second.operations)
    first.x(0)
    assert len(second.operations) == second_size


def test_grover_diffusion_composes_onto_non_contiguous_qubits():
    diffusion = GroverDiffusion(2).build_circuit()
    destination = Circuit([7, 2])

    destination.compose(diffusion, [Qubit(7), Qubit(2)])

    np.testing.assert_allclose(
        destination.to_matrix([7, 2]),
        _diffusion_reference(2),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("num_qubits", [True, 1.5, "2"])
def test_grover_diffusion_rejects_non_integer_counts(num_qubits):
    with pytest.raises(TypeError, match="num_qubits must be an integer"):
        GroverDiffusion(num_qubits)  # type: ignore[arg-type]


@pytest.mark.parametrize("num_qubits", [0, -1])
def test_grover_diffusion_rejects_non_positive_counts(num_qubits):
    with pytest.raises(ValueError, match="greater than zero"):
        GroverDiffusion(num_qubits)
