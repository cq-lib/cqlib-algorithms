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
"""Full adder wrapper.

This module exposes the public full adder interface of the arithmetic package.
It preserves the carry input and register a, updates register b with the lower
n bits of c_in plus a plus b, and exposes the carry out bit through a fixed
local layout. The class is a thin wrapper around the concrete adder
implementations in this package.
"""

from __future__ import annotations

from cqlib import Circuit

from cqlib_algorithms.operators.arithmetic.adders._ripple_carry_adder import (
    build_ripple_carry_adder,
)


class FullAdder:
    """Full adder with a fixed local qubit layout.

    Parameters
        num_state_qubits Positive integer width n of each input register.
        synthesis String synthesis strategy name. Only ripple_carry is
            currently supported.
        name Optional human readable circuit name defaulting to FullAdder.

    Attributes
        num_state_qubits Stored register width n.
        synthesis Selected synthesis strategy name.
        a_register_indices Local qubit indices of the read only a register.
        b_register_indices Local qubit indices of the n wide sum register
            containing the lower n bits of c_in + a + b.
        carry_in_index Local index of the incoming carry bit.
        carry_out_index Local index of the outgoing carry bit.
        num_qubits Total qubit count, equal to 2n + 2.
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
        if synthesis != "ripple_carry":
            raise ValueError(
                f"Unsupported FullAdder synthesis strategy {synthesis!r}. "
                "Expected ripple_carry."
            )
        if name is not None and not isinstance(name, str):
            raise ValueError(
                f"name must be a string or None, got {name!r}."
            )

        self._num_state_qubits = int(num_state_qubits)
        self._synthesis: str = synthesis
        self._name: str = name or "FullAdder"
        n = self._num_state_qubits
        self._num_qubits: int = 2 * n + 2
        self._carry_in_index: int = 0
        self._a_register_indices: list[int] = list(range(1, n + 1))
        self._b_register_indices: list[int] = list(range(n + 1, 2 * n + 1))
        self._carry_out_index: int = 2 * n + 1

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
        """Sum register indices holding the lower n bits of the result."""
        return list(self._b_register_indices)

    @property
    def carry_in_index(self) -> int:
        """Local qubit index of the incoming carry bit."""
        return self._carry_in_index

    @property
    def carry_out_index(self) -> int:
        """Local qubit index of the outgoing carry bit."""
        return self._carry_out_index

    def build_circuit(self) -> Circuit:
        """Return a fresh Circuit implementing this full adder instance."""
        total = self._num_qubits
        circuit = Circuit(total)
        sub = build_ripple_carry_adder(self._num_state_qubits, kind="full")
        circuit.compose(sub, list(range(total)))
        return circuit

