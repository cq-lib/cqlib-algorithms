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
"""Draper QFT adder circuits.

This module implements the concrete Fourier basis adders used by the arithmetic
package. It provides the modular and half forms of the Draper construction and
updates register b in place by composing a forward QFT, controlled phase
rotations, and an inverse QFT. Register a remains unchanged and all local
registers follow the shared LSB first convention.

References
Thomas G Draper, Addition on a Quantum Computer, 2000,
arXiv quant ph 0008033.

Don Coppersmith, An Approximate Fourier Transform Useful in Quantum
Factoring, 2002, arXiv quant ph 0201067.
"""

from __future__ import annotations

from typing import Literal

from cqlib import Circuit
from cqlib.circuit import Parameter
from cqlib.circuit.gates import StandardGate

from cqlib_algorithms.transforms import QFT as TransformQFT


DraperAdderKind = Literal["modular", "half"]


_TWO_PI: float = 2.0 * 3.141592653589793


def _controlled_phase(
    circuit: Circuit,
    control: int,
    target: int,
    lam: float | Parameter,
) -> None:
    """Append CPHASE(lambda) matching transforms qft primitives."""
    gate = StandardGate.Phase(lam).control(1)
    circuit.append_mc_gate(gate, [control, target])


def _append_draper_schedule(
    circ: Circuit,
    n: int,
    *,
    kind: DraperAdderKind,
    approximation_degree: int | None,
) -> None:
    """Append the Draper QFT schedule onto circ."""
    a_start = 0
    b_start = n
    carry_out_index = None if kind == "modular" else 2 * n
    sum_wires = list(range(b_start, b_start + n))
    if carry_out_index is not None:
        sum_wires.append(carry_out_index)
    qft_size = len(sum_wires)

    if approximation_degree is None:
        qft_degree = 0
    else:
        if (
            not isinstance(approximation_degree, int)
            or approximation_degree < 0
        ):
            raise ValueError(
                "approximation_degree must be None or a non-negative "
                f"integer, got {approximation_degree!r}."
            )
        max_kept = int(approximation_degree)
        qft_degree = max(0, qft_size - 1 - max_kept) if qft_size >= 1 else 0
        if qft_degree >= qft_size:
            qft_degree = qft_size - 1

    fwd_qft_circ: Circuit = TransformQFT(
        qft_size,
        approximation_degree=qft_degree,
        do_swaps=False,
        inverse=False,
    ).build_circuit()
    circ.compose(fwd_qft_circ, sum_wires)

    for l in range(n):
        b_l = b_start + l
        for k in range(l + 1):
            a_k = a_start + k
            lam: float | Parameter = _TWO_PI / (2 ** (l - k + 1))
            _controlled_phase(circ, a_k, b_l, lam)

    if carry_out_index is not None:
        for j in range(n):
            lam = 3.141592653589793 / (2 ** (j + 1))
            control = a_start + (n - j - 1)
            _controlled_phase(circ, control, carry_out_index, lam)

    inv_qft_circ: Circuit = TransformQFT(
        qft_size,
        approximation_degree=qft_degree,
        do_swaps=False,
        inverse=True,
    ).build_circuit()
    circ.compose(inv_qft_circ, sum_wires)


class DraperQFTAdder:
    """Draper QFT adder for modular or half addition.

    Parameters
        num_state_qubits Width n of each input register, must be at least 1.
        kind Either modular or half. The modular variant maps
            |a>|b> to |a>|a+b mod 2**n>. The half variant maps
            |a>|b>|0> to |a>|low_n_bits(a+b)>|carry_out>.
        approximation_degree Optional truncation threshold forwarded to the
            internal forward and inverse QFT calls. None means use the exact
            QFT. Non-negative integers are interpreted as the longest
            binary distance whose controlled rotation is retained inside each
            QFT block.
        name Optional human readable name, defaults to DraperQFTAdder.

    Attributes
        num_state_qubits Width n of each of the two registers.
        kind Selected adder variant.
        approximation_degree Approximate QFT threshold in use, None when the
            exact transform is used.
        a_register_indices Local wire indices of the read only a register.
        b_register_indices Local wire indices of the b register.
        carry_out_index Local carry output index for the half variant,
            otherwise None.
    """

    def __init__(
        self,
        num_state_qubits: int,
        *,
        kind: DraperAdderKind = "modular",
        approximation_degree: int | None = None,
        name: str | None = None,
    ) -> None:
        if not isinstance(num_state_qubits, int) or num_state_qubits < 1:
            raise ValueError(
                "num_state_qubits must be an integer >= 1, got "
                f"{num_state_qubits!r}."
            )
        if kind not in {"modular", "half"}:
            raise ValueError(
                f"kind must be one of modular half, got {kind!r}."
            )
        if approximation_degree is not None and (
            not isinstance(approximation_degree, int)
            or approximation_degree < 0
        ):
            raise ValueError(
                "approximation_degree must be None or a non-negative "
                f"integer, got {approximation_degree!r}."
            )

        self._num_state_qubits = int(num_state_qubits)
        self._kind: DraperAdderKind = kind
        self._approximation_degree = (
            None if approximation_degree is None else int(approximation_degree)
        )
        self._name: str = name or "DraperQFTAdder"

        n = self._num_state_qubits
        self._num_qubits: int = 2 * n if kind == "modular" else 2 * n + 1
        self._a_register_indices: list[int] = list(range(n))
        self._b_register_indices: list[int] = list(range(n, 2 * n))
        self._carry_out_index: int | None = None if kind == "modular" else 2 * n

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human readable name used when building the circuit."""
        return self._name

    @property
    def num_qubits(self) -> int:
        """Total qubit count of the constructed circuit."""
        return self._num_qubits

    @property
    def num_state_qubits(self) -> int:
        """Width n of each of the two input registers."""
        return self._num_state_qubits

    @property
    def approximation_degree(self) -> int | None:
        """Approximate QFT threshold, None for the exact transform."""
        return self._approximation_degree

    @property
    def kind(self) -> DraperAdderKind:
        """Selected Draper adder kind."""
        return self._kind

    @property
    def a_register_indices(self) -> list[int]:
        """Local wire indices of the read only a register."""
        return list(self._a_register_indices)

    @property
    def b_register_indices(self) -> list[int]:
        """Local wire indices of the b sum register."""
        return list(self._b_register_indices)

    @property
    def carry_out_index(self) -> int | None:
        """Local carry out index for half mode, otherwise None."""
        return self._carry_out_index

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------

    def build_circuit(self) -> Circuit:
        """Return a fresh Circuit instance implementing this adder."""
        circ = Circuit(self._num_qubits)
        _append_draper_schedule(
            circ,
            self._num_state_qubits,
            kind=self._kind,
            approximation_degree=self._approximation_degree,
        )
        return circ




if __name__ == "__main__":
    import numpy as np

    from cqlib.qis.state import Statevector
    from cqlib.visualization import draw_figure

    adder = DraperQFTAdder(3, kind="modular")
    test_circuit = Circuit(adder.num_qubits)

    a_value = 3
    b_value = 4

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

    # figure_path = "draper_qft_adder_demo.svg"
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

