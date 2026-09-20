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

"""Tests for immutable multi-output Boolean functions."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from cqlib_algorithms.operators.oracle import BooleanFunction, BooleanVectorFunction


def test_from_functions_preserves_output_order_and_allows_duplicates():
    first = BooleanFunction.from_marked_states(2, [1, 3])
    second = BooleanFunction.from_marked_states(2, [2])
    function = BooleanVectorFunction.from_functions([first, second, first])

    assert function.num_variables == 2
    assert function.num_outputs == 3
    assert function.variable_order == ("x0", "x1")
    assert function.outputs == (first, second, first)
    assert function.marked_states == ((1, 3), (2,), (1, 3))
    assert function.truth_table == (
        (False, False, False),
        (True, False, True),
        (False, True, False),
        (True, False, True),
    )


def test_marked_state_sets_deduplicate_and_map_outputs_in_order():
    function = BooleanVectorFunction.from_marked_state_sets(2, [[0, 0, 3], [1], []])

    assert function.marked_states == ((0, 3), (1,), ())
    assert function.truth_table == (
        (True, False, False),
        (False, True, False),
        (False, False, False),
        (True, False, False),
    )


def test_truth_table_is_input_major_and_preserves_variable_order():
    rows = [
        [False, True],
        [True, True],
        [False, False],
        [True, False],
    ]
    function = BooleanVectorFunction.from_truth_table(
        rows, variable_order=("low", "high")
    )

    assert function.variable_order == ("low", "high")
    assert function.truth_table == tuple(tuple(row) for row in rows)
    assert function.marked_states == ((1, 3), (0, 1))


def test_expressions_use_global_first_occurrence_order():
    function = BooleanVectorFunction.from_expressions(("b & ~a", "c ^ b", "a | c"))

    assert function.variable_order == ("b", "a", "c")
    assert function.num_outputs == 3
    for state, row in enumerate(function.truth_table):
        b = bool(state & 1)
        a = bool(state & 2)
        c = bool(state & 4)
        assert row == (b and not a, c != b, a or c)


def test_explicit_expression_order_allows_constants_and_unused_variables():
    function = BooleanVectorFunction.from_expressions(
        ("True", "x", "False"), variable_order=("unused", "x")
    )

    assert function.truth_table == (
        (True, False, False),
        (True, False, False),
        (True, True, False),
        (True, True, False),
    )


def test_all_constant_expressions_require_explicit_nonempty_order():
    with pytest.raises(ValueError, match="require an explicit variable_order"):
        BooleanVectorFunction.from_expressions(("True", "False"))
    with pytest.raises(ValueError, match="at least one variable"):
        BooleanVectorFunction.from_expressions(("True", "False"), variable_order=())


def test_all_nested_input_containers_are_defensively_snapshotted():
    first_states = [0]
    second_states = [1]
    state_sets = [first_states, second_states]
    from_states = BooleanVectorFunction.from_marked_state_sets(1, state_sets)

    first_row = [False, True]
    second_row = [True, False]
    rows = [first_row, second_row]
    from_table = BooleanVectorFunction.from_truth_table(rows)
    expressions = ["x", "~x"]
    variable_order = ["x"]
    from_expressions = BooleanVectorFunction.from_expressions(
        expressions, variable_order=variable_order
    )

    first_states.append(1)
    second_states.clear()
    state_sets.clear()
    first_row[0] = True
    second_row[1] = True
    rows.clear()
    expressions.clear()
    variable_order[0] = "changed"

    assert from_states.marked_states == ((0,), (1,))
    assert from_table.truth_table == ((False, True), (True, False))
    assert from_expressions.variable_order == ("x",)
    assert from_expressions.truth_table == ((False, True), (True, False))


def test_direct_constructor_snapshots_output_container():
    first = BooleanFunction.from_marked_states(1, [0])
    second = BooleanFunction.from_marked_states(1, [1])
    outputs = [first, second]
    function = BooleanVectorFunction(outputs)  # type: ignore[arg-type]

    outputs.clear()

    assert function.outputs == (first, second)
    assert not hasattr(function, "__dict__")
    with pytest.raises(FrozenInstanceError):
        function._outputs = ()  # type: ignore[misc]


def test_vector_function_is_declared_final():
    assert BooleanVectorFunction.__final__ is True


@pytest.mark.parametrize("outputs", [[], [BooleanFunction.from_marked_states(1, [])]])
def test_vector_rejects_fewer_than_two_outputs(outputs):
    with pytest.raises(ValueError, match="at least two outputs"):
        BooleanVectorFunction.from_functions(outputs)


def test_vector_rejects_non_functions_and_mismatched_variable_orders():
    first = BooleanFunction.from_truth_table([False, True], variable_order=("a",))
    second = BooleanFunction.from_truth_table([False, True], variable_order=("b",))
    with pytest.raises(TypeError, match="only BooleanFunction"):
        BooleanVectorFunction.from_functions([first, object()])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="same variable_order"):
        BooleanVectorFunction.from_functions([first, second])


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [[False, True]],
        [[False, True], [True]],
        [[False], [True]],
        [[False, True], [True, 0]],
        [False, True],
        "rows",
    ],
)
def test_truth_table_rejects_invalid_shapes_and_values(rows):
    with pytest.raises((TypeError, ValueError)):
        BooleanVectorFunction.from_truth_table(rows)  # type: ignore[arg-type]


@pytest.mark.parametrize("num_variables", [True, 0, -1, 1.5])
def test_marked_state_sets_reject_invalid_variable_counts(num_variables):
    with pytest.raises((TypeError, ValueError)):
        BooleanVectorFunction.from_marked_state_sets(
            num_variables,  # type: ignore[arg-type]
            [[], []],
        )


@pytest.mark.parametrize("state_sets", [[[0], [4]], [[0], [True]], [[], object()]])
def test_marked_state_sets_reject_invalid_nested_values(state_sets):
    with pytest.raises((TypeError, ValueError)):
        BooleanVectorFunction.from_marked_state_sets(2, state_sets)  # type: ignore[arg-type]


def test_expressions_reject_missing_variables_bad_syntax_and_non_sequences():
    with pytest.raises(ValueError, match="contain every expression variable"):
        BooleanVectorFunction.from_expressions(
            ("a", "b"), variable_order=("a", "unused")
        )
    with pytest.raises(ValueError):
        BooleanVectorFunction.from_expressions(("a", "f(a)"))
    with pytest.raises(TypeError):
        BooleanVectorFunction.from_expressions("a")  # type: ignore[arg-type]
