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

"""Tests for exact one-ancilla block encodings via unitary dilation."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from cqlib_algorithms.operators.oracle import BlockEncoding, UnitaryDilation


def _projected_block(
    unitary: np.ndarray,
    *,
    system_dimension: int,
) -> np.ndarray:
    return np.asarray(
        unitary[:system_dimension, :system_dimension],
        dtype=np.complex128,
    )


def _random_square_matrix(dimension: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(
        rng.normal(size=(dimension, dimension))
        + 1j * rng.normal(size=(dimension, dimension)),
        dtype=np.complex128,
    )


def test_unitary_dilation_is_a_standard_exact_one_ancilla_block_encoding():
    matrix = np.array(
        [
            [2.0, 0.0],
            [0.0, 0.5],
        ],
        dtype=np.complex128,
    )
    component = UnitaryDilation(matrix)

    assert isinstance(component, BlockEncoding)
    assert component.alpha == pytest.approx(2.0)
    assert component.ancilla_qubits == 1
    assert component.epsilon == pytest.approx(0.0)
    assert component.matrix_shape == (2, 2)
    assert component.system_qubits == 1
    assert component.total_qubits == 2
    assert component.spec == pytest.approx((2.0, 1, 0.0))
    assert component.system_register_indices == [0]
    assert component.ancilla_register_indices == [1]


def test_unitary_dilation_auto_alpha_uses_the_spectral_norm():
    matrix = np.array(
        [
            [0.0, 0.0],
            [0.0, -0.25],
        ],
        dtype=np.complex128,
    )

    component = UnitaryDilation(matrix, validate=False)

    assert component.alpha == pytest.approx(0.25)
    np.testing.assert_allclose(
        component.normalized_matrix,
        matrix / 0.25,
        rtol=0.0,
        atol=0.0,
    )


def test_unitary_dilation_uses_alpha_one_for_the_zero_matrix():
    matrix = np.zeros((2, 2), dtype=np.complex128)

    component = UnitaryDilation(matrix, validate=False)

    assert component.alpha == pytest.approx(1.0)
    np.testing.assert_allclose(
        component.normalized_matrix,
        matrix,
        rtol=0.0,
        atol=0.0,
    )


def test_unitary_dilation_projected_block_recovers_matrix_over_alpha():
    matrix = np.array(
        [
            [0.5, 0.1j],
            [-0.1j, 0.25],
        ],
        dtype=np.complex128,
    )
    component = UnitaryDilation(matrix, validate=False)

    circuit_matrix = component.build_circuit().to_matrix()
    projected = _projected_block(circuit_matrix, system_dimension=matrix.shape[0])

    np.testing.assert_allclose(
        projected,
        matrix / component.alpha,
        rtol=0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        component.dilation_unitary,
        circuit_matrix,
        rtol=0.0,
        atol=1e-10,
    )


def test_unitary_dilation_recovers_arbitrary_sixteen_dimensional_matrix_over_alpha():
    matrix = _random_square_matrix(16, seed=123)
    spectral_norm = float(np.linalg.svd(matrix, compute_uv=False)[0])
    alpha = 2.0 * spectral_norm
    component = UnitaryDilation(matrix, alpha=alpha, validate=False)

    projected = _projected_block(
        component.build_circuit().to_matrix(),
        system_dimension=matrix.shape[0],
    )

    np.testing.assert_allclose(
        projected,
        matrix / alpha,
        rtol=0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        alpha * projected,
        matrix,
        rtol=0.0,
        atol=1e-10,
    )


def test_unitary_dilation_accepts_explicit_larger_alpha():
    matrix = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.25],
        ],
        dtype=np.complex128,
    )
    component = UnitaryDilation(matrix, alpha=4.0, validate=False)

    assert component.alpha == pytest.approx(4.0)
    np.testing.assert_allclose(
        component.normalized_matrix,
        matrix / 4.0,
        rtol=0.0,
        atol=0.0,
    )

    projected = _projected_block(
        component.build_circuit().to_matrix(),
        system_dimension=matrix.shape[0],
    )
    np.testing.assert_allclose(
        component.alpha * projected,
        matrix,
        rtol=0.0,
        atol=1e-10,
    )


def test_unitary_dilation_build_adjoint_circuit_matches_conjugate_transpose():
    matrix = np.array(
        [
            [0.5, 0.1j],
            [-0.1j, 0.25],
        ],
        dtype=np.complex128,
    )
    component = UnitaryDilation(matrix, validate=False)

    circuit_matrix = np.asarray(component.build_circuit().to_matrix(), dtype=np.complex128)
    adjoint_matrix = np.asarray(
        component.build_adjoint_circuit().to_matrix(),
        dtype=np.complex128,
    )

    np.testing.assert_allclose(
        adjoint_matrix,
        circuit_matrix.conj().T,
        rtol=0.0,
        atol=1e-12,
    )


def test_unitary_dilation_circuit_is_unitary_and_inverts_with_its_adjoint():
    matrix = np.array(
        [
            [0.5, 0.1j],
            [-0.1j, 0.25],
        ],
        dtype=np.complex128,
    )
    component = UnitaryDilation(matrix, validate=False)

    circuit_matrix = np.asarray(component.build_circuit().to_matrix(), dtype=np.complex128)
    adjoint_matrix = np.asarray(
        component.build_adjoint_circuit().to_matrix(),
        dtype=np.complex128,
    )
    identity = np.eye(circuit_matrix.shape[0], dtype=np.complex128)

    np.testing.assert_allclose(
        circuit_matrix.conj().T @ circuit_matrix,
        identity,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        circuit_matrix @ adjoint_matrix,
        identity,
        rtol=0.0,
        atol=1e-12,
    )


def test_unitary_dilation_properties_are_keyword_only_and_defensive():
    matrix = np.eye(2, dtype=np.complex128)
    signature = inspect.signature(UnitaryDilation)
    component = UnitaryDilation(matrix, validate=False)

    assert signature.parameters["alpha"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["atol"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["validate"].kind is inspect.Parameter.KEYWORD_ONLY
    assert component.validates_reconstruction is False
    assert component.atol == pytest.approx(1e-10)

    matrix_copy = component.matrix
    normalized_copy = component.normalized_matrix
    dilation_copy = component.dilation_unitary
    matrix_copy[0, 0] = 0.0
    normalized_copy[0, 0] = 0.0
    dilation_copy[0, 0] = 0.0

    np.testing.assert_allclose(component.matrix, matrix, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        component.normalized_matrix,
        matrix,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        component.dilation_unitary,
        component.build_circuit().to_matrix(),
        rtol=0.0,
        atol=1e-10,
    )

    with pytest.raises(TypeError):
        UnitaryDilation(matrix, 1.0)  # type: ignore[misc]


def test_unitary_dilation_rejects_invalid_inputs():
    with pytest.raises(TypeError, match="matrix must be a numpy.ndarray"):
        UnitaryDilation([[1.0, 0.0], [0.0, 1.0]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="matrix must be square"):
        UnitaryDilation(np.ones((2, 3), dtype=np.complex128))
    with pytest.raises(ValueError, match="matrix dimension must be a power of two"):
        UnitaryDilation(np.eye(3, dtype=np.complex128))
    with pytest.raises(ValueError, match="matrix must contain only finite entries"):
        UnitaryDilation(np.array([[1.0, np.inf], [0.0, 1.0]], dtype=np.complex128))
    with pytest.raises(TypeError, match="alpha must be a real finite scalar"):
        UnitaryDilation(np.eye(2, dtype=np.complex128), alpha=True)
    with pytest.raises(ValueError, match="alpha must be greater than zero"):
        UnitaryDilation(np.eye(2, dtype=np.complex128), alpha=0.0)
    with pytest.raises(
        ValueError,
        match="alpha must be at least the spectral norm of matrix",
    ):
        UnitaryDilation(np.eye(2, dtype=np.complex128), alpha=0.5)
    with pytest.raises(TypeError, match="atol must be a real finite scalar"):
        UnitaryDilation(np.eye(2, dtype=np.complex128), atol=True)
    with pytest.raises(ValueError, match="atol must be greater than or equal to zero"):
        UnitaryDilation(np.eye(2, dtype=np.complex128), atol=-1e-10)
    with pytest.raises(TypeError, match="validate must be a bool"):
        UnitaryDilation(np.eye(2, dtype=np.complex128), validate=1)  # type: ignore[arg-type]
