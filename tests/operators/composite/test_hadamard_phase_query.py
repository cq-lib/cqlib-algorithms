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

"""Tests for the Hadamard phase-query composite operator."""

from __future__ import annotations

import numpy as np
import pytest

from cqlib_algorithms.operators.composite import HadamardPhaseQuery
from cqlib_algorithms.operators.oracle import (
    BooleanFunction,
    BooleanVectorFunction,
)


def _reference_query_matrix(signs):
    dimension = len(signs)
    num_qubits = dimension.bit_length() - 1
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    transform = hadamard
    for _ in range(num_qubits - 1):
        transform = np.kron(hadamard, transform)
    return transform @ np.diag(signs) @ transform


@pytest.mark.parametrize(
    "marked_states",
    [(), (0,), (0, 1, 2), (1, 4, 7), tuple(range(8))],
)
def test_scalar_query_matches_independent_h_diagonal_h_matrix(marked_states):
    function = BooleanFunction.from_marked_states(3, marked_states)
    signs = [-1.0 if state in marked_states else 1.0 for state in range(1 << 3)]

    circuit = HadamardPhaseQuery(function).build_circuit()

    np.testing.assert_allclose(
        circuit.to_matrix(),
        _reference_query_matrix(signs),
        rtol=0.0,
        atol=1e-12,
    )


def test_vector_query_uses_selected_output_parity():
    function = BooleanVectorFunction.from_truth_table(
        [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ]
    )
    query = HadamardPhaseQuery(function, output_mask=0b11)
    signs = [1.0, -1.0, -1.0, 1.0]

    np.testing.assert_allclose(
        query.build_circuit().to_matrix(),
        _reference_query_matrix(signs),
        rtol=0.0,
        atol=1e-12,
    )
    assert query.function is function
    assert query.output_mask == 0b11
    assert query.num_qubits == 2


def test_scalar_default_mask_is_canonicalized_to_one():
    function = BooleanFunction.from_marked_states(1, (1,))

    query = HadamardPhaseQuery(function)

    assert query.output_mask == 1


def test_build_returns_fresh_independent_circuits():
    query = HadamardPhaseQuery(BooleanFunction.from_marked_states(2, (1,)))

    first = query.build_circuit()
    second = query.build_circuit()

    assert first is not second
    assert first == second
    original_size = len(second.operations)
    first.x(0)
    assert len(second.operations) == original_size


def test_component_is_slotted_and_configuration_is_read_only():
    function = BooleanFunction.from_marked_states(1, (1,))
    query = HadamardPhaseQuery(function)

    assert not hasattr(query, "__dict__")
    with pytest.raises(AttributeError):
        query.function = function  # type: ignore[misc]


@pytest.mark.parametrize("function", [None, "f", 2])
def test_invalid_function_is_rejected_by_phase_oracle(function):
    with pytest.raises(TypeError, match="BooleanFunction or BooleanVectorFunction"):
        HadamardPhaseQuery(function)  # type: ignore[arg-type]


def test_vector_function_requires_a_valid_output_mask():
    function = BooleanVectorFunction.from_truth_table([(False, False), (True, True)])

    with pytest.raises(ValueError, match="output_mask is required"):
        HadamardPhaseQuery(function)
    with pytest.raises(ValueError, match="select one or more existing outputs"):
        HadamardPhaseQuery(function, output_mask=4)
