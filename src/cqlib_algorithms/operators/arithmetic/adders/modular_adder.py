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
"""Modular adder wrapper.

This module exposes the public modular adder interface of the arithmetic
package.  It keeps register ``a`` unchanged and updates register ``b`` with
the value of ``a + b mod 2 ** n`` through a fixed local layout.  The class
is a thin compatibility subclass of the unified
:class:`cqlib_algorithms.operators.arithmetic.adders._adder.Adder` with
``kind`` locked to ``modular``.
"""

from __future__ import annotations

from cqlib import Circuit

from ._adder import Adder, AdderSynthesis


class ModularAdder(Adder):
    """Modular adder with a fixed local qubit layout.

    Parameters
        num_state_qubits Positive integer width ``n`` of each input register.
        synthesis String synthesis strategy name. Accepted values are
            ``ripple_carry`` and ``draper_qft``. Default ``ripple_carry``.
        name Optional human readable circuit name defaulting to ModularAdder.

    Attributes
        num_state_qubits Stored register width ``n``.
        synthesis Selected synthesis strategy name.
        a_register_indices Local qubit indices of the read only ``a`` register.
        b_register_indices Local qubit indices of the register holding the
            modular sum ``a + b mod 2^n`` on output.
        carry_in_index Always ``None`` for :class:`ModularAdder`.
        carry_out_index Always ``None`` for :class:`ModularAdder`.
        helper_index Local qubit index of the helper wire.
        num_qubits Total qubit count, equal to ``2n + 1``.
        name Human readable name.
    """

    def __init__(
        self,
        num_state_qubits: int,
        synthesis: AdderSynthesis = "ripple_carry",
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            num_state_qubits,
            kind="modular",
            synthesis=synthesis,
            name=name or "ModularAdder",
        )

    def build_circuit(self) -> Circuit:
        """Return a fresh Circuit implementing this modular adder instance."""
        return super().build_circuit()
