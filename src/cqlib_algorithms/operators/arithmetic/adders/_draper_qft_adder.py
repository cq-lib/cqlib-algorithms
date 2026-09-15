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


def build_draper_qft_adder(
    num_state_qubits: int,
    *,
    kind: DraperAdderKind = "modular",
    approximation_degree: int | None = None,
) -> Circuit:
    """Return a fresh Circuit instance implementing this adder."""
    num_qubits = (
        2 * num_state_qubits
        if kind == "modular"
        else 2 * num_state_qubits + 1
    )
    circ = Circuit(num_qubits)
    _append_draper_schedule(
        circ,
        num_state_qubits,
        kind=kind,
        approximation_degree=approximation_degree,
    )
    return circ
