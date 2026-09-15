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
"""Half adder wrapper.

This module exposes the public half adder interface of the arithmetic package.
It keeps register a unchanged, updates register b with the lower n bits of the
sum, and exposes a carry out bit through a fixed local layout. The class is a
thin wrapper around the concrete adder implementations in this package.
"""

from __future__ import annotations

from cqlib import Circuit

from cqlib_algorithms.operators.arithmetic.adders._draper_qft_adder import (
    build_draper_qft_adder,
)
from cqlib_algorithms.operators.arithmetic.adders._ripple_carry_adder import (
    build_ripple_carry_adder,
)


class HalfAdder:
    """Half adder with a fixed local qubit layout.

    Parameters
        num_state_qubits Positive integer width n of each input register.
        synthesis String synthesis strategy name. Accepted values are
            ripple_carry and draper_qft. Default ripple_carry.
        name Optional human readable circuit name defaulting to HalfAdder.

    Attributes
        num_state_qubits Stored register width n.
        synthesis Selected synthesis strategy name.
        a_register_indices Local qubit indices of the read only a register.
        b_register_indices Local qubit indices of the sum register which
            holds the lower n bits of a+b.
        carry_in_index Always None for HalfAdder.
        carry_out_index Local qubit index of the carry out bit.
        helper_index Local qubit index of the helper wire.
        num_qubits Total number of qubits, equal to 2 n + 2.
        name Human readable name.
    """

    def __init__(
        self,
        num_state_qubits: int,
        synthesis: str = "ripple_carry",
        *,
        name: str | None = None,
    ) -> None:
        if not isinstance(num_state_qubits, int) or num_state_qubits < 1:
            raise ValueError(
                "num_state_qubits must be an integer >= 1, got "
                f"{num_state_qubits!r}."
            )
        if synthesis not in {"ripple_carry", "draper_qft"}:
            raise ValueError(
                f"Unsupported HalfAdder synthesis strategy {synthesis!r}. "
                "Expected one of ripple_carry draper_qft."
            )
        if name is not None and not isinstance(name, str):
            raise ValueError(
                f"name must be a string or None, got {name!r}."
            )

        self._num_state_qubits = int(num_state_qubits)
        self._synthesis: str = synthesis
        self._name: str = name or "HalfAdder"
        n = self._num_state_qubits
        self._num_qubits: int = 2 * n + 2
        self._a_register_indices: list[int] = list(range(n))
        self._b_register_indices: list[int] = list(range(n, 2 * n))
        self._carry_in_index: int | None = None
        self._carry_out_index: int = 2 * n
        self._helper_index: int = 2 * n + 1

    @property
    def name(self) -> str:
        """Human readable name stored on construction."""
        return self._name

    @property
    def num_qubits(self) -> int:
        """Total number of qubits allocated by the built circuit."""
        return self._num_qubits

    @property
    def num_state_qubits(self) -> int:
        """Width n of each of the two operand registers."""
        return self._num_state_qubits

    @property
    def synthesis(self) -> str:
        """Name of the synthesis strategy selected at construction time."""
        return self._synthesis

    @property
    def a_register_indices(self) -> list[int]:
        """Local qubit indices of the read only a register."""
        return list(self._a_register_indices)

    @property
    def b_register_indices(self) -> list[int]:
        """Local qubit indices of the sum register holding lower n bits."""
        return list(self._b_register_indices)

    @property
    def carry_in_index(self) -> int | None:
        """HalfAdder does not expose a carry in wire."""
        return self._carry_in_index

    @property
    def carry_out_index(self) -> int:
        """Local qubit index of the carry out bit."""
        return self._carry_out_index

    @property
    def helper_index(self) -> int:
        """Local qubit index of the helper wire."""
        return self._helper_index

    def build_circuit(self) -> Circuit:
        """Return a fresh Circuit implementing this half adder instance."""
        n = self._num_state_qubits
        total = self._num_qubits
        circuit = Circuit(total)
        if self._synthesis == "ripple_carry":
            sub = build_ripple_carry_adder(n, kind="half")
            circuit.compose(sub, list(range(total)))
            return circuit
        draper_circuit = build_draper_qft_adder(n, kind="half")
        mapping = (
            list(self._a_register_indices)
            + list(self._b_register_indices)
            + [self._carry_out_index]
        )
        circuit.compose(draper_circuit, mapping)
        return circuit

