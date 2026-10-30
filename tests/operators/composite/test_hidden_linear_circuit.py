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

"""Tests for the hidden-linear-function circuit component."""

from __future__ import annotations

import numpy as np
import pytest

from cqlib_algorithms.operators.composite import HiddenLinearFunctionCircuit


def _bits(value: int, width: int) -> tuple[int, ...]:
    return tuple((value >> qubit) & 1 for qubit in range(width))


def _quadratic_value(matrix: tuple[tuple[int, ...], ...], value: int) -> int:
    bits = _bits(value, len(matrix))
    diagonal = sum(matrix[index][index] * bits[index] for index in range(len(matrix)))
    interactions = 2 * sum(
        matrix[row][column] * bits[row] * bits[column]
        for row in range(len(matrix))
        for column in range(row + 1, len(matrix))
    )
    return (diagonal + interactions) % 4


def _reference_matrix(matrix: tuple[tuple[int, ...], ...]) -> np.ndarray:
    width = len(matrix)
    dimension = 1 << width
    single_hadamard = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    hadamard = np.array([[1.0]])
    for _ in range(width):
        hadamard = np.kron(single_hadamard, hadamard)
    phases = np.array(
        [(1j) ** _quadratic_value(matrix, value) for value in range(dimension)]
    )
    return hadamard @ np.diag(phases) @ hadamard


def _is_hidden_linear_solution(
    matrix: tuple[tuple[int, ...], ...], solution: int
) -> bool:
    width = len(matrix)
    solution_bits = _bits(solution, width)
    for value in range(1 << width):
        bits = _bits(value, width)
        in_kernel = all(
            sum(matrix[row][column] * bits[column] for column in range(width)) % 2 == 0
            for row in range(width)
        )
        if not in_kernel:
            continue
        linear_value = 2 * (
            sum(solution_bits[index] * bits[index] for index in range(width)) % 2
        )
        if _quadratic_value(matrix, value) != linear_value:
            return False
    return True


def _symmetric_matrix(width: int, pattern: int) -> tuple[tuple[int, ...], ...]:
    matrix = [[0] * width for _ in range(width)]
    bit = 0
    for row in range(width):
        for column in range(row, width):
            value = (pattern >> bit) & 1
            matrix[row][column] = value
            matrix[column][row] = value
            bit += 1
    return tuple(tuple(row) for row in matrix)


@pytest.mark.parametrize(
    "matrix",
    [
        ((0,),),
        ((1,),),
        ((0, 1), (1, 0)),
        ((1, 1), (1, 0)),
        ((1, 1, 0), (1, 0, 1), (0, 1, 1)),
        ((0, 1, 0, 1), (1, 1, 1, 0), (0, 1, 0, 1), (1, 0, 1, 1)),
    ],
)
def test_circuit_matches_independent_quadratic_phase_matrix(matrix):
    component = HiddenLinearFunctionCircuit(matrix)

    np.testing.assert_allclose(
        component.build_circuit().to_matrix(),
        _reference_matrix(matrix),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.reference
def test_all_instances_up_to_three_qubits_return_exact_hidden_linear_solutions():
    for width in range(1, 4):
        independent_entries = width * (width + 1) // 2
        for pattern in range(1 << independent_entries):
            matrix = _symmetric_matrix(width, pattern)
            state = (
                HiddenLinearFunctionCircuit(matrix).build_circuit().to_matrix()[:, 0]
            )
            probabilities = np.abs(state) ** 2
            observed = {
                value
                for value, probability in enumerate(probabilities)
                if probability > 1e-12
            }
            expected = {
                value
                for value in range(1 << width)
                if _is_hidden_linear_solution(matrix, value)
            }

            assert observed == expected
            np.testing.assert_allclose(
                probabilities[list(expected)],
                np.full(len(expected), 1.0 / len(expected)),
                rtol=0.0,
                atol=1e-12,
            )


def test_numpy_input_is_copied_and_canonicalized():
    source = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.int64)

    component = HiddenLinearFunctionCircuit(source)
    source[0, 0] = 0

    assert component.adjacency_matrix == (
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
    )
    assert component.num_qubits == 3


def test_boolean_matrix_is_accepted():
    component = HiddenLinearFunctionCircuit(
        np.array([[True, False], [False, True]], dtype=np.bool_)
    )

    assert component.adjacency_matrix == ((1, 0), (0, 1))


def test_build_returns_fresh_independent_unmeasured_circuits():
    component = HiddenLinearFunctionCircuit(((0, 1), (1, 0)))

    first = component.build_circuit()
    second = component.build_circuit()

    assert first is not second
    assert first == second
    operation_count = len(second.operations)
    first.x(0)
    assert len(second.operations) == operation_count
    assert not second.classical_vars
    assert not second.classical_values


def test_component_is_slotted_and_configuration_is_read_only():
    component = HiddenLinearFunctionCircuit(((0,),))

    assert not hasattr(component, "__dict__")
    with pytest.raises(AttributeError):
        component.adjacency_matrix = ((1,),)  # type: ignore[misc]


@pytest.mark.parametrize("value", [None, 1, "01"])
def test_non_array_like_input_is_rejected(value):
    with pytest.raises(TypeError, match="two-dimensional array-like"):
        HiddenLinearFunctionCircuit(value)


@pytest.mark.parametrize(
    "matrix",
    [
        [],
        np.empty((0, 0), dtype=int),
    ],
)
def test_empty_matrix_is_rejected(matrix):
    with pytest.raises(ValueError, match="must not be empty"):
        HiddenLinearFunctionCircuit(matrix)


def test_non_two_dimensional_numpy_input_is_rejected():
    with pytest.raises(ValueError, match="two-dimensional"):
        HiddenLinearFunctionCircuit(np.array([0, 1]))


@pytest.mark.parametrize(
    "matrix",
    [
        [[0, 1], [1]],
        [[0, 1, 0], [1, 0, 1]],
    ],
)
def test_non_square_matrix_is_rejected(matrix):
    with pytest.raises(ValueError, match="square"):
        HiddenLinearFunctionCircuit(matrix)


def test_non_sequence_row_is_rejected():
    with pytest.raises(TypeError, match="rows must be sequences"):
        HiddenLinearFunctionCircuit([0])


@pytest.mark.parametrize("entry", [0.0, 1.0, 1j, "1", None])
def test_non_integer_entry_is_rejected(entry):
    with pytest.raises(TypeError, match="entries must be integers"):
        HiddenLinearFunctionCircuit([[entry]])


@pytest.mark.parametrize("entry", [-1, 2, 3])
def test_non_binary_integer_entry_is_rejected(entry):
    with pytest.raises(ValueError, match="entries must be binary"):
        HiddenLinearFunctionCircuit([[entry]])


def test_non_symmetric_matrix_is_rejected():
    with pytest.raises(ValueError, match="must be symmetric"):
        HiddenLinearFunctionCircuit([[0, 1], [0, 0]])


def test_input_matrix_is_not_modified():
    matrix = [[1, 1], [1, 0]]
    snapshot = [row.copy() for row in matrix]

    HiddenLinearFunctionCircuit(matrix)

    assert matrix == snapshot


def test_basis_index_uses_qubit_zero_as_least_significant_bit():
    matrix = ((1, 0, 0), (0, 0, 0), (0, 0, 0))
    actual = HiddenLinearFunctionCircuit(matrix).build_circuit().to_matrix()
    most_significant_diagonal = ((0, 0, 0), (0, 0, 0), (0, 0, 1))

    np.testing.assert_allclose(
        actual,
        _reference_matrix(matrix),
        rtol=0.0,
        atol=1e-12,
    )
    assert not np.allclose(actual, _reference_matrix(most_significant_diagonal))
