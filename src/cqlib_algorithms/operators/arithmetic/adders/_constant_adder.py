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
"""Draper constant adder circuits.

This module implements the concrete classical constant version of the Draper
adder. A fixed integer value is compiled into phase rotations on a single
Fourier basis register and the result is written back in place under the LSB
first convention used by this package.

References
----------
[1] T. G. Draper, "Addition on a Quantum Computer," 2000.
    arXiv:quant-ph/0008033.
"""

from __future__ import annotations

import math

from cqlib import Circuit
from cqlib_algorithms.transforms import QFT as TransformQFT


def build_draper_constant_adder(
    num_state_qubits: int,
    value: int,
) -> Circuit:
    """Build a new circuit implementing |b> -> |b + value mod 2^n>."""
    n = num_state_qubits
    circuit = Circuit(n)
    wires = list(range(n))

    forward_qft = TransformQFT(
        n,
        approximation_degree=0,
        do_swaps=False,
        inverse=False,
    ).build_circuit()
    circuit.compose(forward_qft, wires)

    for target_bit, qubit in enumerate(wires):
        angle = 0.0
        for const_bit in range(target_bit + 1):
            if (value >> const_bit) & 1:
                angle += (2.0 * math.pi) / (
                    2 ** (target_bit - const_bit + 1)
                )
        if angle != 0.0:
            circuit.phase(qubit, angle)

    inverse_qft = TransformQFT(
        n,
        approximation_degree=0,
        do_swaps=False,
        inverse=True,
    ).build_circuit()
    circuit.compose(inverse_qft, wires)
    return circuit
