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
package. It keeps register a unchanged and updates register b with the value of
a plus b modulo 2 to the power n through a fixed local layout. The class is a
thin wrapper around the concrete adder implementations in this package.
"""

from __future__ import annotations

from cqlib import Circuit

from cqlib_algorithms.operators.arithmetic.adders.ripple_carry_adder import (
    RippleCarryAdder,
)
from cqlib_algorithms.operators.arithmetic.adders.draper_qft_adder import (
    DraperQFTAdder,
)


class ModularAdder:
    """Modular adder with a fixed local qubit layout.

    Parameters
        num_state_qubits Positive integer width n of each input register.
        synthesis String synthesis strategy name. Accepted values are
            ripple_carry and draper_qft. Default ripple_carry.
        name Optional human readable circuit name defaulting to ModularAdder.

    Attributes
        num_state_qubits Stored register width n.
        synthesis Selected synthesis strategy name.
        a_register_indices Local qubit indices of the read only a register.
        b_register_indices Local qubit indices of the register holding the
            modular sum a + b mod 2^n on output.
        carry_in_index Always None for ModularAdder.
        carry_out_index Always None for ModularAdder.
        helper_index Local qubit index of the helper wire.
        num_qubits Total qubit count, equal to 2n + 1.
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
                f"Unsupported ModularAdder synthesis strategy {synthesis!r}. "
                "Expected one of ripple_carry draper_qft."
            )

        self._num_state_qubits = int(num_state_qubits)
        self._synthesis: str = synthesis
        self._name: str = name or "ModularAdder"

        reference = RippleCarryAdder(self._num_state_qubits, kind="modular")
        self._num_qubits: int = reference.num_qubits
        self._a_register_indices: list[int] = reference.a_register_indices
        self._b_register_indices: list[int] = reference.b_register_indices
        assert reference.helper_index is not None
        self._carry_in_index: int | None = None
        self._carry_out_index: int | None = None
        self._helper_index: int = reference.helper_index
        self._ref_modular: RippleCarryAdder = reference

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
        """Local qubit indices of the modular sum register."""
        return list(self._b_register_indices)

    @property
    def carry_in_index(self) -> int | None:
        """ModularAdder does not expose a carry in wire."""
        return self._carry_in_index

    @property
    def carry_out_index(self) -> int | None:
        """ModularAdder does not expose a carry out wire."""
        return self._carry_out_index

    @property
    def helper_index(self) -> int:
        """Local qubit index of the helper wire."""
        return self._helper_index

    def build_circuit(self) -> Circuit:
        """Return a fresh Circuit implementing this modular adder instance."""
        n = self._num_state_qubits
        total = self._num_qubits
        circuit = Circuit(total)
        if self._synthesis == "ripple_carry":
            sub = self._ref_modular.build_circuit()
            circuit.compose(sub, list(range(total)))
            return circuit
        draper = DraperQFTAdder(n, kind="modular")
        draper_circuit = draper.build_circuit()
        a_wires = list(self._a_register_indices)
        b_wires = list(self._b_register_indices)
        mapping = a_wires + b_wires
        circuit.compose(draper_circuit, mapping)
        return circuit




if __name__ == "__main__":
    import numpy as np

    from cqlib.qis.state import Statevector
    from cqlib.visualization import draw_figure

    adder = ModularAdder(3, synthesis="draper_qft")
    test_circuit = Circuit(adder.num_qubits)

    a_value = 5
    b_value = 6

    for bit, qubit in enumerate(adder.a_register_indices):
        if (a_value >> bit) & 1:
            test_circuit.x(qubit)
    for bit, qubit in enumerate(adder.b_register_indices):
        if (b_value >> bit) & 1:
            test_circuit.x(qubit)

    test_circuit.compose(
        adder.build_circuit(),
        list(range(adder.num_qubits)),
    )

    # figure_path = "modular_adder_demo.svg"
    # draw_figure(test_circuit, output_path=figure_path)
    # print(f"circuit figure saved to {figure_path}")

    statevector = Statevector.from_circuit(test_circuit)
    probabilities = np.asarray(statevector.probabilities(), dtype=np.float64)
    outcome = int(np.argmax(probabilities))

    sum_value = 0
    for bit, qubit in enumerate(adder.b_register_indices):
        if (outcome >> qubit) & 1:
            sum_value |= 1 << bit

    print(
        "verification:",
        f"{a_value} + {b_value}",
        f"-> low_bits={sum_value}",
    )

