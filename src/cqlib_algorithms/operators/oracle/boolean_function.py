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

"""Immutable Boolean-function values shared by oracle operators."""

from __future__ import annotations

import ast
import keyword
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


def _default_variable_order(num_variables: int) -> tuple[str, ...]:
    return tuple(f"x{index}" for index in range(num_variables))


def _validate_num_variables(num_variables: int) -> None:
    """Check the variable count without transforming or returning it."""
    if isinstance(num_variables, bool) or not isinstance(num_variables, int):
        raise TypeError("num_variables must be an integer")
    if num_variables <= 0:
        raise ValueError("num_variables must be greater than zero")


def _normalize_variable_order(variable_order: Sequence[str]) -> tuple[str, ...]:
    if isinstance(variable_order, (str, bytes)) or not isinstance(
        variable_order, Sequence
    ):
        raise TypeError("variable_order must be a sequence of variable names")

    order = tuple(variable_order)
    if any(not isinstance(name, str) for name in order):
        raise TypeError("variable_order entries must be strings")
    if any(not name.isidentifier() or keyword.iskeyword(name) for name in order):
        raise ValueError("variable_order entries must be valid non-keyword identifiers")
    if len(set(order)) != len(order):
        raise ValueError("variable_order must not contain duplicate names")
    return order


def _canonicalize_variable_order(
    variable_order: Sequence[str] | None,
    num_variables: int,
) -> tuple[str, ...]:
    """Return a validated snapshot, generating a default order if omitted."""
    if variable_order is None:
        return _default_variable_order(num_variables)
    order = _normalize_variable_order(variable_order)
    if len(order) != num_variables:
        raise ValueError("variable_order length must equal the number of variables")
    return order


def _collect_expression_variables(node: ast.AST, variables: list[str]) -> None:
    if isinstance(node, ast.Expression):
        _collect_expression_variables(node.body, variables)
        return
    if isinstance(node, ast.Name):
        if node.id not in variables:
            variables.append(node.id)
        return
    if isinstance(node, ast.Constant):
        if type(node.value) is not bool:
            raise ValueError("expression constants must be True or False")
        return
    if isinstance(node, ast.UnaryOp):
        match node.op:
            case ast.Invert():
                _collect_expression_variables(node.operand, variables)
                return
            case _:
                raise ValueError("expression unary operator must be '~'")
    if isinstance(node, ast.BinOp):
        match node.op:
            case ast.BitAnd() | ast.BitOr() | ast.BitXor():
                _collect_expression_variables(node.left, variables)
                _collect_expression_variables(node.right, variables)
                return
            case _:
                raise ValueError("expression binary operators must be '&', '|', or '^'")
    raise ValueError(
        "expression may contain only variables, True, False, '~', '&', '|', '^', "
        "and parentheses"
    )


def _evaluate_expression(node: ast.AST, assignment: Mapping[str, bool]) -> bool:
    if isinstance(node, ast.Expression):
        return _evaluate_expression(node.body, assignment)
    if isinstance(node, ast.Name):
        return assignment[node.id]
    if isinstance(node, ast.Constant):
        return bool(node.value)
    if isinstance(node, ast.UnaryOp):
        return not _evaluate_expression(node.operand, assignment)
    if isinstance(node, ast.BinOp):
        left = _evaluate_expression(node.left, assignment)
        right = _evaluate_expression(node.right, assignment)
        if isinstance(node.op, ast.BitAnd):
            return left and right
        if isinstance(node.op, ast.BitOr):
            return left or right
        return left != right
    raise AssertionError("expression was not validated")


def _parse_expression(expression: str) -> tuple[ast.AST, tuple[str, ...]]:
    if not isinstance(expression, str):
        raise TypeError("expression must be a string")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as error:
        raise ValueError("expression is not valid supported syntax") from error

    discovered: list[str] = []
    _collect_expression_variables(tree, discovered)
    return tree, tuple(discovered)


@dataclass(frozen=True, slots=True)
class BooleanFunction:
    """A finite single-output Boolean function with explicit variable order.

    The truth-table index is the input integer ``x``. ``variable_order[i]``
    corresponds to bit ``i`` of ``x`` and therefore to oracle qubit ``q_i``;
    qubit zero is the least-significant bit.

    Construct values with :meth:`from_marked_states`, :meth:`from_truth_table`,
    or :meth:`from_expression`. The underscored dataclass fields are internal
    validated storage; the public views are immutable tuples.
    """

    _variable_order: tuple[str, ...]
    _truth_table: tuple[bool, ...]

    def __post_init__(self) -> None:
        if isinstance(self._variable_order, (str, bytes)) or not isinstance(
            self._variable_order, Sequence
        ):
            raise TypeError("variable_order must be a sequence of variable names")
        if isinstance(self._truth_table, (str, bytes)) or not isinstance(
            self._truth_table, Sequence
        ):
            raise TypeError("truth table must be a sequence of booleans")

        variable_order = tuple(self._variable_order)
        truth_table = tuple(self._truth_table)
        object.__setattr__(self, "_variable_order", variable_order)
        object.__setattr__(self, "_truth_table", truth_table)

        length = len(truth_table)
        if length <= 1 or length & (length - 1):
            raise ValueError(
                "truth table length must be a power of two greater than one"
            )
        if any(type(value) is not bool for value in truth_table):
            raise TypeError("truth table entries must be booleans")
        expected_variables = length.bit_length() - 1
        validated_order = _canonicalize_variable_order(
            variable_order, expected_variables
        )
        object.__setattr__(self, "_variable_order", validated_order)

    @classmethod
    def from_marked_states(
        cls,
        num_variables: int,
        marked_states: Iterable[int],
    ) -> BooleanFunction:
        """Construct a function that is true exactly on ``marked_states``.

        Duplicate marked states are combined. State integers use the ordinary
        binary convention with variable and qubit zero as the LSB.

        Raises:
            TypeError: If the variable count or a marked state has wrong type.
            ValueError: If the count is non-positive or a state is out of range.
        """
        _validate_num_variables(num_variables)
        try:
            states = tuple(marked_states)
        except TypeError as error:
            raise TypeError("marked_states must be an iterable of integers") from error

        dimension = 1 << num_variables
        unique_states: set[int] = set()
        for state in states:
            if isinstance(state, bool) or not isinstance(state, int):
                raise TypeError("marked states must be integers")
            if not 0 <= state < dimension:
                raise ValueError(
                    "marked states must satisfy 0 <= state < 2**num_variables"
                )
            unique_states.add(state)

        table = tuple(index in unique_states for index in range(dimension))
        return cls(_default_variable_order(num_variables), table)

    @classmethod
    def from_truth_table(
        cls,
        values: Sequence[bool],
        *,
        variable_order: Sequence[str] | None = None,
    ) -> BooleanFunction:
        """Construct a function from values indexed by the input integer.

        Raises:
            TypeError: If ``values`` is not a sequence or contains non-booleans.
            ValueError: If its length is not a power of two greater than one,
                or if ``variable_order`` is invalid.
        """
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError("values must be a sequence of booleans")
        table = tuple(values)
        if any(type(value) is not bool for value in table):
            raise TypeError("truth table entries must be booleans")
        length = len(table)
        if length <= 1 or length & (length - 1):
            raise ValueError(
                "truth table length must be a power of two greater than one"
            )

        num_variables = length.bit_length() - 1
        order = _canonicalize_variable_order(variable_order, num_variables)
        return cls(order, table)

    @classmethod
    def from_expression(
        cls,
        expression: str,
        *,
        variable_order: Sequence[str] | None = None,
    ) -> BooleanFunction:
        """Construct a function from a restricted Boolean expression.

        Supported syntax consists only of variable names, ``True``, ``False``,
        parentheses, and the Boolean operators ``~``, ``&``, ``^``, and ``|``.
        The expression is parsed and interpreted through an AST whitelist; it
        is never evaluated as Python code. With no explicit order, variables
        are ordered by their first left-to-right occurrence. Construction
        materializes the complete truth table and therefore uses ``O(2**n)``
        time and space for ``n`` variables; it is intended for exact
        construction at small to moderate sizes.

        Raises:
            TypeError: If ``expression`` or ``variable_order`` has wrong type.
            ValueError: If the expression syntax, variables, or order is invalid.
        """
        tree, discovered = _parse_expression(expression)

        if variable_order is None:
            if not discovered:
                raise ValueError(
                    "a constant expression requires an explicit variable_order"
                )
            order = tuple(discovered)
        else:
            order = _normalize_variable_order(variable_order)
            if not order:
                raise ValueError("variable_order must contain at least one variable")
            if not set(discovered).issubset(order):
                raise ValueError(
                    "variable_order must contain every expression variable"
                )

        table = tuple(
            _evaluate_expression(
                tree,
                {name: bool((state >> index) & 1) for index, name in enumerate(order)},
            )
            for state in range(1 << len(order))
        )
        return cls(order, table)

    @property
    def num_variables(self) -> int:
        """Return the number of input variables."""
        return len(self._variable_order)

    @property
    def num_outputs(self) -> int:
        """Return one for this scalar Boolean function."""
        return 1

    @property
    def variable_order(self) -> tuple[str, ...]:
        """Return variable names in LSB-to-MSB order."""
        return self._variable_order

    @property
    def truth_table(self) -> tuple[bool, ...]:
        """Return values indexed by the input integer."""
        return self._truth_table

    @property
    def marked_states(self) -> tuple[int, ...]:
        """Return sorted input integers for which the function is true."""
        return tuple(index for index, value in enumerate(self._truth_table) if value)
