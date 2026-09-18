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

"""Immutable multi-output Boolean functions for oracle operators."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import final

from .boolean_function import (
    BooleanFunction,
    _canonicalize_variable_order,
    _evaluate_expression,
    _normalize_variable_order,
    _parse_expression,
    _validate_num_variables,
)


@final
@dataclass(frozen=True, slots=True)
class BooleanVectorFunction:
    """A finite Boolean vector function with two or more outputs.

    Every scalar output has the same variable order. ``output[i]`` is mapped
    to oracle output qubit ``q[n + i]``. The public truth table is input-major:
    ``truth_table[x][i]`` is output ``i`` for input integer ``x``. Marked-state
    sets are output-major.
    """

    _outputs: tuple[BooleanFunction, ...]

    def __post_init__(self) -> None:
        if isinstance(self._outputs, (str, bytes)):
            raise TypeError("outputs must be an iterable of BooleanFunction values")
        try:
            outputs = tuple(self._outputs)
        except TypeError as error:
            raise TypeError(
                "outputs must be an iterable of BooleanFunction values"
            ) from error
        if len(outputs) < 2:
            raise ValueError("a BooleanVectorFunction requires at least two outputs")
        if any(not isinstance(output, BooleanFunction) for output in outputs):
            raise TypeError("outputs must contain only BooleanFunction values")

        variable_order = outputs[0].variable_order
        if any(output.variable_order != variable_order for output in outputs[1:]):
            raise ValueError("all outputs must have exactly the same variable_order")
        object.__setattr__(self, "_outputs", outputs)

    @classmethod
    def from_functions(
        cls, outputs: Iterable[BooleanFunction]
    ) -> BooleanVectorFunction:
        """Construct a vector function from compatible scalar outputs."""
        if isinstance(outputs, (str, bytes)):
            raise TypeError("outputs must be an iterable of BooleanFunction values")
        try:
            snapshot = tuple(outputs)
        except TypeError as error:
            raise TypeError(
                "outputs must be an iterable of BooleanFunction values"
            ) from error
        return cls(snapshot)

    @classmethod
    def from_marked_state_sets(
        cls,
        num_variables: int,
        marked_states_by_output: Iterable[Iterable[int]],
    ) -> BooleanVectorFunction:
        """Construct outputs from their marked computational-basis states."""
        _validate_num_variables(num_variables)
        if isinstance(marked_states_by_output, (str, bytes)):
            raise TypeError("marked_states_by_output must be an iterable")
        try:
            state_sets = tuple(tuple(states) for states in marked_states_by_output)
        except TypeError as error:
            raise TypeError(
                "marked_states_by_output must contain iterable state sets"
            ) from error
        return cls(
            tuple(
                BooleanFunction.from_marked_states(num_variables, states)
                for states in state_sets
            )
        )

    @classmethod
    def from_truth_table(
        cls,
        rows: Sequence[Sequence[bool]],
        variable_order: Sequence[str] | None = None,
    ) -> BooleanVectorFunction:
        """Construct from an input-major rectangular Boolean truth table."""
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise TypeError("rows must be a sequence of Boolean rows")
        row_snapshot: list[tuple[bool, ...]] = []
        for row in rows:
            if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
                raise TypeError("each truth-table row must be a sequence")
            row_snapshot.append(tuple(row))
        table = tuple(row_snapshot)
        dimension = len(table)
        if dimension <= 1 or dimension & (dimension - 1):
            raise ValueError(
                "truth table row count must be a power of two greater than one"
            )

        num_outputs = len(table[0])
        if num_outputs < 2:
            raise ValueError("a BooleanVectorFunction requires at least two outputs")
        if any(len(row) != num_outputs for row in table):
            raise ValueError("truth table rows must have equal length")
        if any(type(value) is not bool for row in table for value in row):
            raise TypeError("truth table entries must be booleans")

        num_variables = dimension.bit_length() - 1
        order = _canonicalize_variable_order(variable_order, num_variables)
        return cls(
            tuple(
                BooleanFunction(order, tuple(row[index] for row in table))
                for index in range(num_outputs)
            )
        )

    @classmethod
    def from_expressions(
        cls,
        expressions: Sequence[str],
        variable_order: Sequence[str] | None = None,
    ) -> BooleanVectorFunction:
        """Construct outputs from restricted Boolean expressions.

        Without an explicit order, variables use their first occurrence over
        all expressions. An explicit non-empty order may contain variables not
        referenced by any expression. Construction materializes every scalar
        truth table and therefore takes ``O(m * 2**n)`` time and space.
        """
        if isinstance(expressions, (str, bytes)) or not isinstance(
            expressions, Sequence
        ):
            raise TypeError("expressions must be a sequence of strings")
        expression_snapshot = tuple(expressions)
        if len(expression_snapshot) < 2:
            raise ValueError("a BooleanVectorFunction requires at least two outputs")

        parsed: list[tuple[ast.AST, tuple[str, ...]]] = []
        discovered: list[str] = []
        for expression in expression_snapshot:
            tree, local_variables = _parse_expression(expression)
            parsed.append((tree, local_variables))
            for name in local_variables:
                if name not in discovered:
                    discovered.append(name)

        if variable_order is None:
            if not discovered:
                raise ValueError(
                    "constant expressions require an explicit variable_order"
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

        outputs = []
        for tree, _ in parsed:
            table = tuple(
                _evaluate_expression(
                    tree,
                    {
                        name: bool((state >> index) & 1)
                        for index, name in enumerate(order)
                    },
                )
                for state in range(1 << len(order))
            )
            outputs.append(BooleanFunction(order, table))
        return cls(tuple(outputs))

    @property
    def num_variables(self) -> int:
        """Return the number of input variables."""
        return self._outputs[0].num_variables

    @property
    def num_outputs(self) -> int:
        """Return the number of output bits."""
        return len(self._outputs)

    @property
    def variable_order(self) -> tuple[str, ...]:
        """Return variable names in LSB-to-MSB order."""
        return self._outputs[0].variable_order

    @property
    def outputs(self) -> tuple[BooleanFunction, ...]:
        """Return scalar outputs in output-qubit order."""
        return self._outputs

    @property
    def truth_table(self) -> tuple[tuple[bool, ...], ...]:
        """Return the input-major truth table ``table[x][i]``."""
        return tuple(
            tuple(output.truth_table[state] for output in self._outputs)
            for state in range(1 << self.num_variables)
        )

    @property
    def marked_states(self) -> tuple[tuple[int, ...], ...]:
        """Return marked states for each output bit."""
        return tuple(output.marked_states for output in self._outputs)
