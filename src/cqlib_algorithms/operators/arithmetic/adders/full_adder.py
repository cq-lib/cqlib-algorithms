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

This module exposes the public full adder interface of the arithmetic
package.  It preserves the carry input and register ``a``, updates register
``b`` with the lower ``n`` bits of ``c_in + a + b``, and exposes the carry
out bit through a fixed local layout.  The class is a thin compatibility
subclass of the unified
:class:`cqlib_algorithms.operators.arithmetic.adders._adder.Adder` with
``kind`` locked to ``full``.
"""

from __future__ import annotations

from typing import Literal

from cqlib import Circuit

from ._adder import Adder


class FullAdder(Adder):
    """Full adder with a fixed local qubit layout.

    Parameters
        num_state_qubits Positive integer width ``n`` of each input register.
        synthesis String synthesis strategy name. Only ``ripple_carry`` is
            currently supported.
        name Optional human readable circuit name defaulting to FullAdder.

    Attributes
        num_state_qubits Stored register width ``n``.
        synthesis Selected synthesis strategy name.
        a_register_indices Local qubit indices of the read only ``a`` register.
        b_register_indices Local qubit indices of the ``n`` wide sum register
            containing the lower ``n`` bits of the result.
        carry_in_index Local index of the incoming carry bit.
        carry_out_index Local index of the outgoing carry bit.
        num_qubits Total qubit count, equal to ``2n + 2``.
        name Human readable name.
    """

    def __init__(
        self,
        num_state_qubits: int,
        synthesis: Literal["ripple_carry"] = "ripple_carry",
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            num_state_qubits,
            kind="full",
            synthesis=synthesis,
            name=name or "FullAdder",
        )

    def build_circuit(self) -> Circuit:
        """Return a fresh Circuit implementing this full adder instance."""
        return super().build_circuit()
