"""Tests for Dicke-state preparation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cqlib_algorithms.operators.state_preparation import DickeStatePreparation

_ATOL = 1e-12


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


def _expected_dicke_state(num_qubits: int, weight: int) -> np.ndarray:
    """Return the ideal Dicke state of one fixed Hamming weight."""
    state = np.zeros(1 << num_qubits, dtype=np.complex128)
    amplitude = 1.0 / math.sqrt(math.comb(num_qubits, weight))
    for basis_index in range(1 << num_qubits):
        if basis_index.bit_count() == weight:
            state[basis_index] = amplitude
    return state


@pytest.mark.parametrize(
    ("num_qubits", "weight"),
    [(1, 0), (2, 1), (4, 2), (5, 2), (5, 3), (6, 3), (7, 3)],
)
def test_dicke_state_preparation_matches_reference_state(
    num_qubits: int, weight: int
) -> None:
    preparation = DickeStatePreparation(num_qubits, weight)
    circuit = preparation.build_circuit("dicke")
    actual = np.asarray(circuit.to_matrix()[:, 0], dtype=np.complex128).reshape(-1)
    expected = _expected_dicke_state(num_qubits, weight)

    preparation.validate()
    assert preparation.num_qubits() == num_qubits
    assert preparation.num_parameters() == 0
    _assert_state_close_up_to_global_phase(actual, expected)
    assert len(_nonzero_indices(actual)) == math.comb(num_qubits, weight)
    assert {
        basis_index.bit_count() for basis_index in _nonzero_indices(actual)
    } == {weight}
    np.testing.assert_allclose(
        np.sum(np.abs(actual) ** 2),
        1.0,
        rtol=0.0,
        atol=_ATOL,
    )


@pytest.mark.parametrize("num_qubits", [2, 3, 5, 8])
def test_dicke_state_preparation_produces_w_state_for_weight_one(
    num_qubits: int,
) -> None:
    actual = np.asarray(
        DickeStatePreparation(num_qubits, 1).build_circuit("w").to_matrix()[:, 0],
        dtype=np.complex128,
    ).reshape(-1)

    expected_indices = {1 << qubit for qubit in range(num_qubits)}
    actual_indices = set(_nonzero_indices(actual).tolist())

    assert actual_indices == expected_indices
    np.testing.assert_allclose(
        np.abs(actual[list(expected_indices)]),
        np.full(num_qubits, 1.0 / math.sqrt(num_qubits)),
        rtol=0.0,
        atol=_ATOL,
    )


def test_dicke_state_preparation_builds_independent_circuits() -> None:
    preparation = DickeStatePreparation(4, 2)
    first = preparation.build_circuit("a")
    second = preparation.build_circuit("b")

    assert first is not second
    second_size = len(second.operations)
    first.x(0)
    assert len(second.operations) == second_size


def test_dicke_state_preparation_rejects_invalid_prefix_type() -> None:
    with pytest.raises(TypeError, match="prefix must be a str"):
        DickeStatePreparation(4, 2).build_circuit(1)  # type: ignore[arg-type]


@pytest.mark.parametrize("num_qubits", [True, 4.0, "4", None])
def test_dicke_state_preparation_rejects_invalid_qubit_count_type(
    num_qubits: object,
) -> None:
    with pytest.raises(TypeError, match="num_qubits must be an int"):
        DickeStatePreparation(num_qubits, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize("num_qubits", [0, -1])
def test_dicke_state_preparation_rejects_invalid_qubit_count_value(
    num_qubits: int,
) -> None:
    with pytest.raises(ValueError, match="num_qubits must be at least 1"):
        DickeStatePreparation(num_qubits, 0)


@pytest.mark.parametrize("weight", [True, 1.0, "1", None])
def test_dicke_state_preparation_rejects_invalid_weight_type(weight: object) -> None:
    with pytest.raises(TypeError, match="weight must be an int"):
        DickeStatePreparation(4, weight)  # type: ignore[arg-type]


@pytest.mark.parametrize("weight", [-1, 5])
def test_dicke_state_preparation_rejects_invalid_weight_value(weight: int) -> None:
    with pytest.raises(
        ValueError, match="weight must satisfy 0 <= weight <= num_qubits"
    ):
        DickeStatePreparation(4, weight)
