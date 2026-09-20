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

"""Tests for immutable Boolean-function oracle specifications."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from cqlib_algorithms.operators.oracle import BooleanFunction, PhaseOracle
from cqlib_algorithms.operators.oracle.boolean_function import (
    _canonicalize_variable_order,
    _validate_num_variables,
)


def test_variable_count_validation_returns_none():
    assert _validate_num_variables(2) is None


def test_variable_order_canonicalization_preserves_input_and_copies_snapshot():
    names = ["b", "a"]
    snapshot = _canonicalize_variable_order(names, 2)
    assert names == ["b", "a"]
    assert snapshot == ("b", "a")
    names[0] = "changed"
    assert snapshot == ("b", "a")
    assert _canonicalize_variable_order(None, 2) == ("x0", "x1")


def test_marked_states_are_sorted_deduplicated_and_use_default_names():
    function = BooleanFunction.from_marked_states(3, [5, 1, 5])

    assert function.num_variables == 3
    assert function.variable_order == ("x0", "x1", "x2")
    assert function.marked_states == (1, 5)
    assert function.truth_table == (
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
    )


def test_empty_and_constant_true_functions_are_representable():
    empty = BooleanFunction.from_marked_states(2, [])
    constant_true = BooleanFunction.from_truth_table([True, True])

    assert empty.marked_states == ()
    assert empty.truth_table == (False,) * 4
    assert constant_true.marked_states == (0, 1)


def test_truth_table_preserves_explicit_lsb_to_msb_variable_order():
    function = BooleanFunction.from_truth_table(
        [False, True, True, False], variable_order=("low", "high")
    )

    assert function.variable_order == ("low", "high")
    assert function.marked_states == (1, 2)


def test_expression_default_order_is_first_left_to_right_occurrence():
    function = BooleanFunction.from_expression("b & ~a")

    assert function.variable_order == ("b", "a")
    assert function.marked_states == (1,)


def test_expression_explicit_order_determines_integer_bit_positions():
    function = BooleanFunction.from_expression("b & ~a", variable_order=("a", "b"))

    assert function.variable_order == ("a", "b")
    assert function.marked_states == (2,)


def test_expression_explicit_order_allows_unused_variables_and_constants():
    function = BooleanFunction.from_expression(
        "b", variable_order=("unused", "b", "also_unused")
    )
    constant = BooleanFunction.from_expression("True", variable_order=("low", "high"))

    assert function.variable_order == ("unused", "b", "also_unused")
    assert function.marked_states == (2, 3, 6, 7)
    assert constant.truth_table == (True, True, True, True)
    assert function.num_outputs == 1


def test_expression_supports_all_whitelisted_operators_and_constants():
    function = BooleanFunction.from_expression("(~a & b) | (a ^ False)")

    assert function.variable_order == ("a", "b")
    assert function.truth_table == (False, True, True, True)


@pytest.mark.parametrize("num_variables", [True, 1.5, "2"])
def test_marked_states_reject_non_integer_variable_counts(num_variables):
    with pytest.raises(TypeError, match="num_variables must be an integer"):
        BooleanFunction.from_marked_states(num_variables, [])


@pytest.mark.parametrize("num_variables", [0, -1])
def test_marked_states_reject_non_positive_variable_counts(num_variables):
    with pytest.raises(ValueError, match="greater than zero"):
        BooleanFunction.from_marked_states(num_variables, [])


@pytest.mark.parametrize("state", [True, 1.5, "1"])
def test_marked_states_reject_non_integer_states(state):
    with pytest.raises(TypeError, match="marked states must be integers"):
        BooleanFunction.from_marked_states(2, [state])


@pytest.mark.parametrize("state", [-1, 4])
def test_marked_states_reject_out_of_range_states(state):
    with pytest.raises(ValueError, match=r"0 <= state < 2\*\*num_variables"):
        BooleanFunction.from_marked_states(2, [state])


@pytest.mark.parametrize(
    "values",
    [[], [True], [False, True, False], [False] * 6],
)
def test_truth_table_rejects_invalid_lengths(values):
    with pytest.raises(ValueError, match="power of two greater than one"):
        BooleanFunction.from_truth_table(values)


@pytest.mark.parametrize(
    "values",
    [[0, 1], [False, 1], "FT", iter([False, True])],
)
def test_truth_table_rejects_non_boolean_sequences(values):
    with pytest.raises(TypeError):
        BooleanFunction.from_truth_table(values)


@pytest.mark.parametrize(
    "variable_order",
    ["ab", ("a",), ("a", "a"), ("a", 1), ("a-b", "b")],
)
def test_truth_table_rejects_invalid_variable_orders(variable_order):
    with pytest.raises((TypeError, ValueError)):
        BooleanFunction.from_truth_table(
            [False, False, False, True], variable_order=variable_order
        )


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "True",
        "a and b",
        "not a",
        "a == b",
        "a + b",
        "f(a)",
        "a.value",
        "a[0]",
        "1",
        "-a",
    ],
)
def test_expression_rejects_unsupported_or_variable_free_syntax(expression):
    with pytest.raises(ValueError):
        BooleanFunction.from_expression(expression)


def test_expression_rejects_incomplete_empty_or_duplicate_explicit_orders():
    with pytest.raises(ValueError, match="contain every expression variable"):
        BooleanFunction.from_expression("a ^ b", variable_order=("a", "c"))
    with pytest.raises(ValueError, match="at least one variable"):
        BooleanFunction.from_expression("a", variable_order=())
    with pytest.raises(ValueError, match="duplicate"):
        BooleanFunction.from_expression("a ^ b", variable_order=("a", "a"))


def test_value_is_frozen_slotted_and_public_views_are_tuples():
    function = BooleanFunction.from_marked_states(1, [1])

    assert not hasattr(function, "__dict__")
    assert isinstance(function.variable_order, tuple)
    assert isinstance(function.truth_table, tuple)
    assert isinstance(function.marked_states, tuple)
    with pytest.raises(FrozenInstanceError):
        function._truth_table = (True, False)  # type: ignore[misc]


def test_direct_construction_defensively_copies_mutable_sequences():
    variable_order = ["x0"]
    truth_table = [False, True]
    function = BooleanFunction(
        variable_order,
        truth_table,  # type: ignore[arg-type]
    )
    expected_oracle = np.diag([1.0, -1.0])

    variable_order[0] = "changed"
    truth_table[1] = False

    assert function.variable_order == ("x0",)
    assert function.truth_table == (False, True)
    np.testing.assert_allclose(
        PhaseOracle(function).build_circuit().to_matrix(),
        expected_oracle,
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("variable_order", "truth_table", "message"),
    [
        ("x0", (False, True), "variable_order must be a sequence"),
        (("x0",), "FT", "truth table must be a sequence"),
        (("x0",), object(), "truth table must be a sequence"),
    ],
)
def test_direct_construction_rejects_invalid_container_types(
    variable_order, truth_table, message
):
    with pytest.raises(TypeError, match=message):
        BooleanFunction(variable_order, truth_table)  # type: ignore[arg-type]
