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

"""Tests for Dicke-state oracle building blocks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cqlib_algorithms.operators.oracle import (
    DickeUnitary,
    SplitCycleShift,
    WeightDistributionBlock,
)

_ATOL = 1e-12


def _unary_index(num_qubits: int, ones: int) -> int:
    """Return the little-endian basis index of ``|0^(n-ones) 1^ones>``."""
    if ones == 0:
        return 0
    return ((1 << ones) - 1) << (num_qubits - ones)


def _basis_state(num_qubits: int, basis_index: int) -> np.ndarray:
    """Return one computational basis state as a dense vector."""
    state = np.zeros(1 << num_qubits, dtype=np.complex128)
    state[basis_index] = 1.0
    return state


def _nonzero_indices(state: np.ndarray) -> np.ndarray:
    """Return indices whose amplitudes are numerically nonzero."""
    return np.flatnonzero(np.abs(state) > _ATOL)


def _aligned_state(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Align one global phase before comparing two state vectors."""
    nonzero = _nonzero_indices(expected)
    if nonzero.size == 0:
        return actual

    anchor = int(nonzero[0])
    phase_ratio = actual[anchor] / expected[anchor]
    if abs(phase_ratio) <= _ATOL:
        return actual
    return actual * np.conjugate(phase_ratio / abs(phase_ratio))


def _assert_state_close_up_to_global_phase(
    actual: np.ndarray, expected: np.ndarray
) -> None:
    """Assert that two states agree up to one global phase."""
    np.testing.assert_allclose(
        _aligned_state(actual, expected),
        expected,
        rtol=0.0,
        atol=_ATOL,
    )


def _expected_split_cycle_shift_state(n: int, k: int, ell: int) -> np.ndarray:
    """Return the analytic SCS output for one unary input weight."""
    num_qubits = k + 1
    if ell == 0:
        return _basis_state(num_qubits, 0)
    if ell == num_qubits:
        return _basis_state(num_qubits, (1 << num_qubits) - 1)

    state = np.zeros(1 << num_qubits, dtype=np.complex128)
    stay_index = _unary_index(num_qubits, ell)
    shifted_index = ((1 << ell) - 1) << (k - ell)
    state[stay_index] = math.sqrt(ell / n)
    state[shifted_index] = math.sqrt((n - ell) / n)
    return state


def _expected_weight_distribution_state(
    num_qubits: int, left_width: int, ell: int
) -> np.ndarray:
    """Return the analytic WDB output for one unary input weight."""
    state = np.zeros(1 << num_qubits, dtype=np.complex128)
    right_width = num_qubits - left_width
    normalization = math.comb(num_qubits, ell)
    for left_weight in range(ell + 1):
        numerator = math.comb(left_width, left_weight) * math.comb(
            right_width, ell - left_weight
        )
        amplitude = math.sqrt(numerator / normalization)
        basis_index = _unary_index(left_width, left_weight) + (
            _unary_index(right_width, ell - left_weight) << left_width
        )
        state[basis_index] = amplitude
    return state


def _expected_dicke_state(num_qubits: int, weight: int) -> np.ndarray:
    """Return the ideal Dicke state of one fixed Hamming weight."""
    state = np.zeros(1 << num_qubits, dtype=np.complex128)
    amplitude = 1.0 / math.sqrt(math.comb(num_qubits, weight))
    for basis_index in range(1 << num_qubits):
        if basis_index.bit_count() == weight:
            state[basis_index] = amplitude
    return state


@pytest.mark.parametrize(
    ("n", "k"),
    [(2, 1), (4, 2), (6, 3), (8, 5)],
)
def test_split_cycle_shift_matches_reference_on_unary_inputs(n: int, k: int) -> None:
    matrix = SplitCycleShift(n, k).build_circuit().to_matrix()

    for ell in range(k + 1):
        actual = np.asarray(
            matrix[:, _unary_index(k + 1, ell)], dtype=np.complex128
        ).reshape(-1)
        expected = _expected_split_cycle_shift_state(n, k, ell)

        _assert_state_close_up_to_global_phase(actual, expected)
        if ell == 0:
            assert _nonzero_indices(actual).tolist() == [0]
            continue

        assert len(_nonzero_indices(actual)) == 2
        assert sorted(index.bit_count() for index in _nonzero_indices(actual)) == [
            ell,
            ell,
        ]

    all_ones_actual = np.asarray(matrix[:, (1 << (k + 1)) - 1], dtype=np.complex128)
    all_ones_expected = _expected_split_cycle_shift_state(n, k, k + 1)
    _assert_state_close_up_to_global_phase(all_ones_actual, all_ones_expected)


@pytest.mark.parametrize(("n", "k"), [(2, 1), (4, 2), (5, 3)])
def test_split_cycle_shift_is_unitary(n: int, k: int) -> None:
    matrix = np.asarray(
        SplitCycleShift(n, k).build_circuit().to_matrix(), dtype=np.complex128
    )
    identity = np.eye(1 << (k + 1), dtype=np.complex128)
    np.testing.assert_allclose(
        matrix.conj().T @ matrix,
        identity,
        rtol=0.0,
        atol=_ATOL,
    )


@pytest.mark.parametrize("n", [True, 1.5, "2"])
def test_split_cycle_shift_rejects_invalid_n_type(n: object) -> None:
    with pytest.raises(TypeError, match="n must be an integer"):
        SplitCycleShift(n, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize("k", [True, 1.5, "1"])
def test_split_cycle_shift_rejects_invalid_k_type(k: object) -> None:
    with pytest.raises(TypeError, match="k must be an integer"):
        SplitCycleShift(2, k)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("n", "k", "message"),
    [
        (0, 1, "n must be greater than zero"),
        (-1, 1, "n must be greater than zero"),
        (2, 0, "k must be greater than zero"),
        (2, -1, "k must be greater than zero"),
        (2, 3, "k must not exceed n"),
    ],
)
def test_split_cycle_shift_rejects_invalid_value_ranges(
    n: int, k: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SplitCycleShift(n, k)


@pytest.mark.parametrize(
    ("num_qubits", "left_width", "k"),
    [(4, 2, 1), (5, 2, 2), (6, 3, 2), (8, 4, 3)],
)
def test_weight_distribution_block_matches_hypergeometric_reference(
    num_qubits: int, left_width: int, k: int
) -> None:
    matrix = (
        WeightDistributionBlock(num_qubits, left_width, k).build_circuit().to_matrix()
    )

    for ell in range(k + 1):
        source_index = _unary_index(num_qubits, ell)
        actual = np.asarray(matrix[:, source_index], dtype=np.complex128).reshape(-1)
        expected = _expected_weight_distribution_state(num_qubits, left_width, ell)

        _assert_state_close_up_to_global_phase(actual, expected)
        assert len(_nonzero_indices(actual)) == ell + 1
        assert {
            basis_index.bit_count() for basis_index in _nonzero_indices(actual)
        } == {ell}


@pytest.mark.parametrize(("num_qubits", "left_width", "k"), [(4, 2, 1), (6, 3, 2)])
def test_weight_distribution_block_is_unitary(
    num_qubits: int, left_width: int, k: int
) -> None:
    matrix = np.asarray(
        WeightDistributionBlock(num_qubits, left_width, k).build_circuit().to_matrix(),
        dtype=np.complex128,
    )
    identity = np.eye(1 << num_qubits, dtype=np.complex128)
    np.testing.assert_allclose(
        matrix.conj().T @ matrix,
        identity,
        rtol=0.0,
        atol=_ATOL,
    )


@pytest.mark.parametrize("num_qubits", [True, 4.0, "4"])
def test_weight_distribution_block_rejects_invalid_n_type(num_qubits: object) -> None:
    with pytest.raises(TypeError, match="n must be an integer"):
        WeightDistributionBlock(num_qubits, 2, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize("left_width", [True, 2.0, "2"])
def test_weight_distribution_block_rejects_invalid_m_type(left_width: object) -> None:
    with pytest.raises(TypeError, match="m must be an integer"):
        WeightDistributionBlock(4, left_width, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize("k", [True, 1.0, "1"])
def test_weight_distribution_block_rejects_invalid_k_type(k: object) -> None:
    with pytest.raises(TypeError, match="k must be an integer"):
        WeightDistributionBlock(4, 2, k)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("num_qubits", "left_width", "k", "message"),
    [
        (0, 1, 1, "n must be greater than zero"),
        (4, 0, 1, "m must be greater than zero"),
        (4, 2, 0, "k must be greater than zero"),
        (4, 4, 1, "m must be less than n"),
        (4, 1, 2, "k must not exceed m"),
        (5, 4, 2, "k must not exceed n - m"),
    ],
)
def test_weight_distribution_block_rejects_invalid_value_ranges(
    num_qubits: int, left_width: int, k: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        WeightDistributionBlock(num_qubits, left_width, k)


@pytest.mark.parametrize(("num_qubits", "k"), [(2, 1), (4, 2), (6, 3), (7, 3)])
def test_dicke_unitary_matches_uniform_weight_subspace_reference(
    num_qubits: int, k: int
) -> None:
    matrix = DickeUnitary(num_qubits, k).build_circuit().to_matrix()

    for ell in range(k + 1):
        source_index = _unary_index(num_qubits, ell)
        actual = np.asarray(matrix[:, source_index], dtype=np.complex128).reshape(-1)
        expected = _expected_dicke_state(num_qubits, ell)

        _assert_state_close_up_to_global_phase(actual, expected)
        assert len(_nonzero_indices(actual)) == math.comb(num_qubits, ell)
        assert {
            basis_index.bit_count() for basis_index in _nonzero_indices(actual)
        } == {ell}


@pytest.mark.parametrize("num_qubits", [2, 3, 5, 8])
def test_dicke_unitary_produces_w_state_for_weight_one(num_qubits: int) -> None:
    matrix = DickeUnitary(num_qubits, 1).build_circuit().to_matrix()
    source_index = _unary_index(num_qubits, 1)
    actual = np.asarray(matrix[:, source_index], dtype=np.complex128).reshape(-1)

    expected_indices = {1 << qubit for qubit in range(num_qubits)}
    actual_indices = set(_nonzero_indices(actual).tolist())

    assert actual_indices == expected_indices
    np.testing.assert_allclose(
        np.abs(actual[sorted(expected_indices)]),
        np.full(num_qubits, 1.0 / math.sqrt(num_qubits)),
        rtol=0.0,
        atol=_ATOL,
    )


@pytest.mark.parametrize(("num_qubits", "k"), [(2, 1), (4, 2)])
def test_dicke_unitary_is_unitary(num_qubits: int, k: int) -> None:
    matrix = np.asarray(
        DickeUnitary(num_qubits, k).build_circuit().to_matrix(),
        dtype=np.complex128,
    )
    identity = np.eye(1 << num_qubits, dtype=np.complex128)
    np.testing.assert_allclose(
        matrix.conj().T @ matrix,
        identity,
        rtol=0.0,
        atol=_ATOL,
    )


@pytest.mark.parametrize("num_qubits", [True, 2.0, "2"])
def test_dicke_unitary_rejects_invalid_n_type(num_qubits: object) -> None:
    with pytest.raises(TypeError, match="n must be an integer"):
        DickeUnitary(num_qubits, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize("k", [True, 1.0, "1"])
def test_dicke_unitary_rejects_invalid_k_type(k: object) -> None:
    with pytest.raises(TypeError, match="k must be an integer"):
        DickeUnitary(2, k)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("num_qubits", "k", "message"),
    [
        (0, 0, "n must be greater than zero"),
        (-1, 0, "n must be greater than zero"),
        (2, -1, "k must satisfy 0 <= k <= n"),
        (2, 3, "k must satisfy 0 <= k <= n"),
    ],
)
def test_dicke_unitary_rejects_invalid_value_ranges(
    num_qubits: int, k: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DickeUnitary(num_qubits, k)
