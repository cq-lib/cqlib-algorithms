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

"""Correctness and boundary tests for quantum mutual information."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from cqlib import Circuit
from cqlib.qis import metrics
from cqlib.qis.state import DensityMatrix, Statevector

from cqlib_algorithms.operators.information import mutual_information


def _density_matrix(matrix: np.ndarray) -> DensityMatrix:
    dimension = matrix.shape[0]
    num_qubits = dimension.bit_length() - 1
    return DensityMatrix.from_density_matrix(num_qubits, matrix.reshape(-1))


def _basis_index(
    kept_value: int,
    environment_value: int,
    kept: Sequence[int],
    environment: Sequence[int],
) -> int:
    value = 0
    for local_index, qubit in enumerate(kept):
        value |= ((kept_value >> local_index) & 1) << qubit
    for local_index, qubit in enumerate(environment):
        value |= ((environment_value >> local_index) & 1) << qubit
    return value


def _reduced_density_matrix(
    matrix: np.ndarray,
    num_qubits: int,
    kept: Sequence[int],
) -> np.ndarray:
    kept = tuple(sorted(kept))
    environment = tuple(index for index in range(num_qubits) if index not in kept)
    reduced = np.zeros((1 << len(kept), 1 << len(kept)), dtype=complex)
    for row in range(reduced.shape[0]):
        for column in range(reduced.shape[1]):
            for environment_value in range(1 << len(environment)):
                full_row = _basis_index(row, environment_value, kept, environment)
                full_column = _basis_index(
                    column,
                    environment_value,
                    kept,
                    environment,
                )
                reduced[row, column] += matrix[full_row, full_column]
    return reduced


def _entropy(matrix: np.ndarray, base: float) -> float:
    eigenvalues = np.linalg.eigvalsh(matrix)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    nonzero = eigenvalues[eigenvalues > 1e-15]
    return float(-np.sum(nonzero * np.log(nonzero)) / np.log(base))


def _mutual_information_reference(
    matrix: np.ndarray,
    num_qubits: int,
    subsystem_a: Sequence[int],
    base: float,
) -> float:
    subsystem_b = tuple(
        index for index in range(num_qubits) if index not in subsystem_a
    )
    return (
        _entropy(
            _reduced_density_matrix(matrix, num_qubits, subsystem_a),
            base,
        )
        + _entropy(
            _reduced_density_matrix(matrix, num_qubits, subsystem_b),
            base,
        )
        - _entropy(matrix, base)
    )


def test_product_state_has_zero_mutual_information():
    state = Statevector.from_state(2, [1.0, 0.0, 0.0, 0.0])

    assert mutual_information(state, [0]) == pytest.approx(0.0, abs=1e-14)


def test_bell_state_has_two_bits_of_mutual_information():
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    state = Statevector.from_circuit(circuit)

    assert mutual_information(state, [0]) == pytest.approx(2.0, abs=1e-12)
    assert mutual_information(state, [1]) == pytest.approx(2.0, abs=1e-12)


def test_classically_correlated_mixed_state_has_one_bit():
    matrix = np.diag([0.5, 0.0, 0.0, 0.5]).astype(complex)
    state = _density_matrix(matrix)

    assert mutual_information(state, (0,)) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("base", [2.0, 10.0, np.e])
@pytest.mark.parametrize("subsystem_a", [(0,), (1,), (2,), (0, 2)])
def test_random_mixed_state_matches_independent_reference(base, subsystem_a):
    rng = np.random.default_rng(7301)
    raw = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    matrix = raw @ raw.conj().T
    matrix /= np.trace(matrix)
    state = _density_matrix(matrix)

    expected = _mutual_information_reference(matrix, 3, subsystem_a, base)

    assert mutual_information(state, subsystem_a, base=base) == pytest.approx(
        expected,
        abs=1e-12,
    )


def test_partition_order_does_not_change_result():
    circuit = Circuit(3)
    circuit.h(0)
    circuit.cx(0, 2)
    state = Statevector.from_circuit(circuit)

    assert mutual_information(state, [0, 2]) == pytest.approx(
        mutual_information(state, [2, 0]),
        abs=1e-12,
    )


def test_state_is_not_modified():
    rng = np.random.default_rng(119)
    raw = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    matrix = raw @ raw.conj().T
    matrix /= np.trace(matrix)
    state = _density_matrix(matrix)
    before = state.data

    mutual_information(state, [0])

    np.testing.assert_array_equal(state.data, before)


def test_non_finite_core_entropy_is_rejected(monkeypatch):
    state = Statevector.from_state(2, [1.0, 0.0, 0.0, 0.0])
    values = iter((0.0, 0.0, float("nan")))
    monkeypatch.setattr(metrics, "entropy", lambda _state: next(values))

    with pytest.raises(ArithmeticError, match="non-finite"):
        mutual_information(state, [0])


def test_materially_negative_result_is_rejected(monkeypatch):
    state = Statevector.from_state(2, [1.0, 0.0, 0.0, 0.0])
    values = iter((0.0, 0.0, 1.0))
    monkeypatch.setattr(metrics, "entropy", lambda _state: next(values))

    with pytest.raises(ArithmeticError, match="negative beyond numerical roundoff"):
        mutual_information(state, [0])


def test_roundoff_sized_negative_result_is_clamped(monkeypatch):
    state = Statevector.from_state(2, [1.0, 0.0, 0.0, 0.0])
    values = iter((0.0, 0.0, np.finfo(float).eps))
    monkeypatch.setattr(metrics, "entropy", lambda _state: next(values))

    assert mutual_information(state, [0]) == 0.0


@pytest.mark.parametrize("state", [None, object(), [1.0, 0.0]])
def test_invalid_state_type_is_rejected(state):
    with pytest.raises(TypeError, match="Statevector or DensityMatrix"):
        mutual_information(state, [0])


def test_single_qubit_state_cannot_be_bipartitioned():
    state = Statevector.from_state(1, [1.0, 0.0])

    with pytest.raises(ValueError, match="at least two qubits"):
        mutual_information(state, [0])


@pytest.mark.parametrize("subsystem_a", ["0", b"0", {0}])
def test_partition_must_be_a_sequence(subsystem_a):
    state = Statevector.from_state(2, [1.0, 0.0, 0.0, 0.0])

    with pytest.raises(TypeError, match="sequence"):
        mutual_information(state, subsystem_a)


@pytest.mark.parametrize("subsystem_a", [[False], [0.0], ["0"]])
def test_partition_indices_must_be_integers(subsystem_a):
    state = Statevector.from_state(2, [1.0, 0.0, 0.0, 0.0])

    with pytest.raises(TypeError, match="integer qubit indices"):
        mutual_information(state, subsystem_a)


@pytest.mark.parametrize(
    ("subsystem_a", "message"),
    [
        ([], "must not be empty"),
        ([0, 0], "must be unique"),
        ([-1], "must be in range"),
        ([2], "must be in range"),
        ([0, 1], "proper subset"),
    ],
)
def test_invalid_partition_values_are_rejected(subsystem_a, message):
    state = Statevector.from_state(2, [1.0, 0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match=message):
        mutual_information(state, subsystem_a)


@pytest.mark.parametrize("base", [None, True, "2", complex(2.0, 0.0)])
def test_base_must_be_real(base):
    state = Statevector.from_state(2, [1.0, 0.0, 0.0, 0.0])

    with pytest.raises(TypeError, match="real number"):
        mutual_information(state, [0], base=base)


@pytest.mark.parametrize("base", [float("nan"), float("inf"), -float("inf")])
def test_base_must_be_finite(base):
    state = Statevector.from_state(2, [1.0, 0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="finite"):
        mutual_information(state, [0], base=base)


@pytest.mark.parametrize("base", [-2.0, 0.0, 0.5, 1.0])
def test_base_must_be_greater_than_one(base):
    state = Statevector.from_state(2, [1.0, 0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="greater than one"):
        mutual_information(state, [0], base=base)
