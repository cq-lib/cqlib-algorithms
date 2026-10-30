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

"""Tests for bit-flip-oracle circuit construction."""

from __future__ import annotations

import itertools
import math
from dataclasses import is_dataclass

import numpy as np
import pytest
from cqlib import Circuit, Qubit

from cqlib_algorithms.operators.oracle import (
    BitFlipOracle,
    BooleanFunction,
    BooleanVectorFunction,
    PhaseOracle,
)


def _bit_flip_reference(function: BooleanFunction) -> np.ndarray:
    num_inputs = function.num_variables
    input_dimension = 1 << num_inputs
    dimension = 2 * input_dimension
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for column in range(dimension):
        input_state = column & (input_dimension - 1)
        output = column >> num_inputs
        output ^= int(function.truth_table[input_state])
        row = input_state | (output << num_inputs)
        matrix[row, column] = 1.0
    return matrix


def _vector_bit_flip_reference(function: BooleanVectorFunction) -> np.ndarray:
    num_inputs = function.num_variables
    input_dimension = 1 << num_inputs
    dimension = 1 << (num_inputs + function.num_outputs)
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for column in range(dimension):
        input_state = column & (input_dimension - 1)
        output_state = column >> num_inputs
        function_value = sum(
            int(value) << index
            for index, value in enumerate(function.truth_table[input_state])
        )
        row = input_state | ((output_state ^ function_value) << num_inputs)
        matrix[row, column] = 1.0
    return matrix


@pytest.mark.parametrize(
    "marked_states",
    [[], [0], [1], [2], [0, 3], [0, 1, 2, 3]],
)
def test_bit_flip_oracle_matches_independent_permutation(marked_states):
    function = BooleanFunction.from_marked_states(2, marked_states)

    actual = BitFlipOracle(function).build_circuit().to_matrix()

    np.testing.assert_allclose(
        actual, _bit_flip_reference(function), rtol=0.0, atol=1e-12
    )


def test_bit_flip_oracle_one_input_boundary_is_exact():
    function = BooleanFunction.from_marked_states(1, [0])

    np.testing.assert_allclose(
        BitFlipOracle(function).build_circuit().to_matrix(),
        _bit_flip_reference(function),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("num_outputs", [2, 3])
def test_vector_bit_flip_matches_every_small_input_and_output_state(num_outputs):
    rows = [
        tuple(bool((state >> output) & 1) for output in range(num_outputs))
        for state in (0, 1, 2, 3)
    ]
    function = BooleanVectorFunction.from_truth_table(rows)

    actual = BitFlipOracle(function).build_circuit().to_matrix()

    np.testing.assert_allclose(
        actual, _vector_bit_flip_reference(function), rtol=0.0, atol=1e-12
    )


def test_vector_output_zero_maps_to_q_n_and_constant_outputs_are_optimized():
    function = BooleanVectorFunction.from_truth_table([(True, False), (True, False)])
    circuit = BitFlipOracle(function).build_circuit()

    assert [operation.name for operation in circuit.operations] == ["X"]
    assert [qubit.index for qubit in circuit.operations[0].qubits] == [1]
    np.testing.assert_allclose(
        circuit.to_matrix(), _vector_bit_flip_reference(function), rtol=0.0, atol=1e-12
    )


def test_repeated_vector_outputs_flip_each_corresponding_output():
    scalar = BooleanFunction.from_marked_states(1, [1])
    function = BooleanVectorFunction.from_functions((scalar, scalar))

    np.testing.assert_allclose(
        BitFlipOracle(function).build_circuit().to_matrix(),
        _vector_bit_flip_reference(function),
        rtol=0.0,
        atol=1e-12,
    )


def test_two_output_bit_flip_exhausts_all_one_input_functions():
    for flattened_table in itertools.product((False, True), repeat=4):
        rows = (flattened_table[:2], flattened_table[2:])
        function = BooleanVectorFunction.from_truth_table(rows)

        np.testing.assert_allclose(
            BitFlipOracle(function).build_circuit().to_matrix(),
            _vector_bit_flip_reference(function),
            rtol=0.0,
            atol=1e-12,
        )


def test_bit_flip_register_order_uses_input_q0_as_lsb_and_output_last():
    function = BooleanFunction.from_marked_states(2, [1])
    matrix = BitFlipOracle(function).build_circuit().to_matrix()

    for output in (0, 1):
        column = 1 | (output << 2)
        expected_row = 1 | ((output ^ 1) << 2)
        assert matrix[expected_row, column] == 1.0
        assert np.count_nonzero(matrix[:, column]) == 1


def test_bit_flip_oracle_is_hermitian_unitary_and_involutory():
    matrix = (
        BitFlipOracle(BooleanFunction.from_marked_states(3, [0, 3, 6]))
        .build_circuit()
        .to_matrix()
    )
    identity = np.eye(16)

    np.testing.assert_allclose(matrix, matrix.conj().T, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(matrix.conj().T @ matrix, identity, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(matrix @ matrix, identity, rtol=0.0, atol=1e-12)


def test_output_minus_state_produces_phase_kickback():
    function = BooleanFunction.from_marked_states(2, [1, 3])
    oracle = BitFlipOracle(function).build_circuit()

    for input_state in range(4):
        prepared = Circuit(3)
        for qubit in range(2):
            if (input_state >> qubit) & 1:
                prepared.x(qubit)
        prepared.x(2)
        prepared.h(2)
        prepared.compose(oracle)

        actual = prepared.to_matrix()[:, 0]
        expected = np.zeros(8, dtype=np.complex128)
        phase = -1.0 if function.truth_table[input_state] else 1.0
        expected[input_state] = phase / math.sqrt(2)
        expected[input_state | 4] = -phase / math.sqrt(2)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_selected_vector_outputs_kick_back_parity_and_unselected_plus_does_not():
    function = BooleanVectorFunction.from_truth_table(
        [(False, True), (True, True), (True, False), (False, False)]
    )
    output_mask = 0b01
    bit_flip = BitFlipOracle(function).build_circuit()
    phase = PhaseOracle(function, output_mask=output_mask).build_circuit()

    for input_state in range(4):
        actual = Circuit(4)
        reference = Circuit(4)
        for circuit in (actual, reference):
            for qubit in range(2):
                if (input_state >> qubit) & 1:
                    circuit.x(qubit)
            circuit.x(2)
            circuit.h(2)
            circuit.h(3)
        actual.compose(bit_flip)
        reference.compose(phase, [0, 1])

        np.testing.assert_allclose(
            actual.to_matrix()[:, 0],
            reference.to_matrix()[:, 0],
            rtol=0.0,
            atol=1e-12,
        )


def test_build_returns_fresh_independent_circuits():
    oracle = BitFlipOracle(BooleanFunction.from_marked_states(2, [2]))
    first = oracle.build_circuit()
    second = oracle.build_circuit()

    assert first is not second
    assert first == second
    second_size = len(second.operations)
    first.x(0)
    assert len(second.operations) == second_size


def test_bit_flip_oracle_composes_positionally_onto_non_contiguous_qubits():
    function = BooleanFunction.from_marked_states(2, [2])
    oracle = BitFlipOracle(function).build_circuit()
    destination = Circuit([7, 2, 9])

    destination.compose(oracle, [Qubit(7), Qubit(2), Qubit(9)])

    np.testing.assert_allclose(
        destination.to_matrix([7, 2, 9]),
        _bit_flip_reference(function),
        rtol=0.0,
        atol=1e-12,
    )


def test_vector_bit_flip_composes_positionally_onto_non_contiguous_qubits():
    function = BooleanVectorFunction.from_marked_state_sets(1, [[0], [1]])
    oracle = BitFlipOracle(function).build_circuit()
    destination = Circuit([7, 2, 9])

    destination.compose(oracle, [Qubit(7), Qubit(2), Qubit(9)])

    np.testing.assert_allclose(
        destination.to_matrix([7, 2, 9]),
        _vector_bit_flip_reference(function),
        rtol=0.0,
        atol=1e-12,
    )


def test_bit_flip_oracle_rejects_non_boolean_function():
    with pytest.raises(TypeError, match="function must be a BooleanFunction"):
        BitFlipOracle(object())  # type: ignore[arg-type]


def test_bit_flip_oracle_is_a_slotted_component_with_read_only_configuration():
    function = BooleanFunction.from_marked_states(1, [1])
    oracle = BitFlipOracle(function)

    assert not is_dataclass(oracle)
    assert not hasattr(oracle, "__dict__")
    assert oracle.function is function
    with pytest.raises(AttributeError):
        oracle.function = BooleanFunction.from_marked_states(1, [0])  # type: ignore[misc]
