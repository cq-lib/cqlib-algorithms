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

"""Tests for phase-oracle circuit construction."""

from __future__ import annotations

import inspect
import itertools
from dataclasses import is_dataclass

import numpy as np
import pytest

from cqlib import Circuit, Qubit
from cqlib_algorithms.operators.oracle import (
    BooleanFunction,
    BooleanVectorFunction,
    PhaseOracle,
)


def _phase_reference(function: BooleanFunction) -> np.ndarray:
    return np.diag(
        np.array(
            [-1.0 if value else 1.0 for value in function.truth_table],
            dtype=np.complex128,
        )
    )


def _vector_phase_reference(
    function: BooleanVectorFunction, output_mask: int
) -> np.ndarray:
    diagonal = []
    for row in function.truth_table:
        parity = (
            sum(
                int(value)
                for output, value in enumerate(row)
                if (output_mask >> output) & 1
            )
            % 2
        )
        diagonal.append(-1.0 if parity else 1.0)
    return np.diag(np.asarray(diagonal, dtype=np.complex128))


@pytest.mark.parametrize(
    "marked_states",
    [[], [0], [1], [2], [0, 3], [0, 1, 2, 3]],
)
def test_phase_oracle_matches_exact_truth_table_diagonal(marked_states):
    function = BooleanFunction.from_marked_states(2, marked_states)

    actual = PhaseOracle(function).build_circuit().to_matrix()

    np.testing.assert_allclose(actual, _phase_reference(function), rtol=0.0, atol=1e-12)


def test_phase_oracle_one_qubit_boundary_is_exact():
    function = BooleanFunction.from_marked_states(1, [0])

    np.testing.assert_allclose(
        PhaseOracle(function).build_circuit().to_matrix(),
        np.diag([-1.0, 1.0]),
        rtol=0.0,
        atol=1e-12,
    )


def test_phase_oracle_uses_qubit_zero_as_truth_table_lsb():
    function = BooleanFunction.from_expression(
        "low & ~high", variable_order=("low", "high")
    )

    assert function.marked_states == (1,)
    np.testing.assert_allclose(
        PhaseOracle(function).build_circuit().to_matrix(),
        np.diag([1.0, -1.0, 1.0, 1.0]),
        rtol=0.0,
        atol=1e-12,
    )


def test_constant_false_is_identity_and_constant_true_is_exact_minus_identity():
    false_function = BooleanFunction.from_truth_table([False, False])
    true_function = BooleanFunction.from_truth_table([True, True])

    np.testing.assert_allclose(
        PhaseOracle(false_function).build_circuit().to_matrix(),
        np.eye(2),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        PhaseOracle(true_function).build_circuit().to_matrix(),
        -np.eye(2),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("output_mask", [0b001, 0b010, 0b100, 0b111])
def test_vector_phase_oracle_matches_masked_parity(output_mask):
    function = BooleanVectorFunction.from_truth_table(
        [
            (False, False, True),
            (True, False, True),
            (False, True, False),
            (True, True, False),
        ]
    )

    actual = PhaseOracle(function, output_mask=output_mask).build_circuit().to_matrix()

    np.testing.assert_allclose(
        actual,
        _vector_phase_reference(function, output_mask),
        rtol=0.0,
        atol=1e-12,
    )


def test_masked_phase_uses_parity_not_logical_or():
    function = BooleanVectorFunction.from_truth_table([(False, False), (True, True)])

    matrix = PhaseOracle(function, output_mask=0b11).build_circuit().to_matrix()

    np.testing.assert_allclose(matrix, np.eye(2), rtol=0.0, atol=1e-12)
    assert any(any(row) for row in function.truth_table)


def test_two_output_phase_exhausts_all_one_input_functions_and_masks():
    for flattened_table in itertools.product((False, True), repeat=4):
        rows = (flattened_table[:2], flattened_table[2:])
        function = BooleanVectorFunction.from_truth_table(rows)

        for output_mask in (0b01, 0b10, 0b11):
            np.testing.assert_allclose(
                PhaseOracle(
                    function,
                    output_mask=output_mask,
                )
                .build_circuit()
                .to_matrix(),
                _vector_phase_reference(function, output_mask),
                rtol=0.0,
                atol=1e-12,
            )


def test_effective_constant_parity_uses_exact_global_phase():
    function = BooleanVectorFunction.from_truth_table([(True, False), (True, True)])

    np.testing.assert_allclose(
        PhaseOracle(function, output_mask=0b01).build_circuit().to_matrix(),
        -np.eye(2),
        rtol=0.0,
        atol=1e-12,
    )


def test_phase_oracle_is_hermitian_unitary_and_involutory():
    matrix = (
        PhaseOracle(BooleanFunction.from_marked_states(3, [1, 4, 7]))
        .build_circuit()
        .to_matrix()
    )
    identity = np.eye(8)

    np.testing.assert_allclose(matrix, matrix.conj().T, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(matrix.conj().T @ matrix, identity, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(matrix @ matrix, identity, rtol=0.0, atol=1e-12)


def test_build_returns_fresh_independent_circuits():
    oracle = PhaseOracle(BooleanFunction.from_marked_states(2, [2]))
    first = oracle.build_circuit()
    second = oracle.build_circuit()

    assert first is not second
    assert first == second
    second_size = len(second.operations)
    first.x(0)
    assert len(second.operations) == second_size


def test_phase_oracle_composes_positionally_onto_non_contiguous_qubits():
    oracle = PhaseOracle(BooleanFunction.from_marked_states(2, [1])).build_circuit()
    destination = Circuit([7, 2])

    destination.compose(oracle, [Qubit(7), Qubit(2)])

    np.testing.assert_allclose(
        destination.to_matrix([7, 2]),
        np.diag([1.0, -1.0, 1.0, 1.0]),
        rtol=0.0,
        atol=1e-12,
    )


def test_vector_phase_oracle_is_fresh_and_composes_onto_non_contiguous_qubits():
    function = BooleanVectorFunction.from_marked_state_sets(2, [[1], [2]])
    oracle = PhaseOracle(function, output_mask=0b11)
    first = oracle.build_circuit()
    second = oracle.build_circuit()
    destination = Circuit([7, 2])

    destination.compose(first, [Qubit(7), Qubit(2)])

    assert first is not second
    assert first == second
    np.testing.assert_allclose(
        destination.to_matrix([7, 2]),
        _vector_phase_reference(function, 0b11),
        rtol=0.0,
        atol=1e-12,
    )


def test_phase_oracle_rejects_non_boolean_function():
    with pytest.raises(TypeError, match="function must be a BooleanFunction"):
        PhaseOracle(object())  # type: ignore[arg-type]


def test_phase_oracle_mask_is_keyword_only_and_configuration_is_immutable():
    function = BooleanFunction.from_marked_states(1, [1])
    signature = inspect.signature(PhaseOracle)
    oracle = PhaseOracle(function)

    assert signature.parameters["output_mask"].kind is inspect.Parameter.KEYWORD_ONLY
    assert not is_dataclass(oracle)
    assert oracle.output_mask == 1
    assert not hasattr(oracle, "__dict__")
    with pytest.raises(TypeError):
        PhaseOracle(function, 1)  # type: ignore[misc]
    with pytest.raises(AttributeError):
        oracle.output_mask = 0  # type: ignore[misc]
    with pytest.raises(AttributeError):
        oracle.function = BooleanFunction.from_marked_states(1, [0])  # type: ignore[misc]


@pytest.mark.parametrize("output_mask", [True, 1.5, "1"])
def test_phase_oracle_rejects_non_integer_masks(output_mask):
    function = BooleanFunction.from_marked_states(1, [1])
    with pytest.raises(TypeError, match="output_mask must be an integer"):
        PhaseOracle(function, output_mask=output_mask)  # type: ignore[arg-type]


@pytest.mark.parametrize("output_mask", [0, 2, -1])
def test_scalar_phase_oracle_rejects_masks_other_than_one(output_mask):
    function = BooleanFunction.from_marked_states(1, [1])
    with pytest.raises(ValueError, match="must be 1"):
        PhaseOracle(function, output_mask=output_mask)


def test_vector_phase_oracle_requires_explicit_in_range_nonzero_mask():
    function = BooleanVectorFunction.from_marked_state_sets(1, [[0], [1]])

    with pytest.raises(ValueError, match="required"):
        PhaseOracle(function)
    for output_mask in (0, -1, 4):
        with pytest.raises(ValueError, match="select one or more existing outputs"):
            PhaseOracle(function, output_mask=output_mask)
