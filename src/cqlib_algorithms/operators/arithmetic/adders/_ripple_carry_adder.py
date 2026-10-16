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
"""Ripple carry adder circuits.

This module implements the concrete ripple carry adders used by the arithmetic
package. It provides the Cuccaro MAJ and UMA primitives together with the
half, full, and modular schedules used by the public wrapper classes. The
circuits follow the shared LSB first register convention of this package.

References
----------
[1] S. A. Cuccaro, T. G. Draper, S. A. Kutin, and D. P. Moulton,
    "A New Quantum Ripple Carry Addition Circuit," 2004.
    arXiv:quant-ph/0410184.
[2] V. Vedral, A. Barenco, and A. Ekert, "Quantum Networks for Elementary
    Arithmetic Operations," 1996.
    arXiv:quant-ph/9511018.
"""

from __future__ import annotations

from typing import Literal

from cqlib import Circuit

AdderKind = Literal["half", "full", "modular"]


def _append_maj(
    circuit: Circuit,
    a_index: int,
    b_index: int,
    c_index: int,
) -> None:
    """Append the Cuccaro MAJ primitive."""
    circuit.cx(c_index, b_index)
    circuit.cx(c_index, a_index)
    circuit.ccx(a_index, b_index, c_index)


def _append_uma(
    circuit: Circuit,
    a_index: int,
    b_index: int,
    c_index: int,
) -> None:
    """Append the Cuccaro UMA primitive."""
    circuit.ccx(a_index, b_index, c_index)
    circuit.cx(c_index, a_index)
    circuit.cx(a_index, b_index)


def build_ripple_carry_adder(
    num_state_qubits: int,
    kind: AdderKind = "half",
) -> Circuit:
    """Build the CDKM adder circuit for the selected kind."""
    n = num_state_qubits
    if kind == "full":
        num_qubits = 2 * n + 2
        cin_index = 0
        a_indices = list(range(1, n + 1))
        b_indices = list(range(n + 1, 2 * n + 1))
        cout_index = 2 * n + 1
        circuit = Circuit(num_qubits)

        _append_maj(circuit, cin_index, b_indices[0], a_indices[0])
        for i in range(n - 1):
            _append_maj(
                circuit,
                a_indices[i],
                b_indices[i + 1],
                a_indices[i + 1],
            )

        circuit.cx(a_indices[n - 1], cout_index)

        for i in range(n - 2, -1, -1):
            _append_uma(
                circuit,
                a_indices[i],
                b_indices[i + 1],
                a_indices[i + 1],
            )

        _append_uma(circuit, cin_index, b_indices[0], a_indices[0])
        return circuit

    if kind == "half":
        num_qubits = 2 * n + 2
        a_indices = list(range(n))
        b_indices = list(range(n, 2 * n))
        cout_index = 2 * n
        helper_index = 2 * n + 1
        circuit = Circuit(num_qubits)

        _append_maj(circuit, helper_index, b_indices[0], a_indices[0])
        for i in range(n - 1):
            _append_maj(
                circuit,
                a_indices[i],
                b_indices[i + 1],
                a_indices[i + 1],
            )

        circuit.cx(a_indices[n - 1], cout_index)

        for i in range(n - 2, -1, -1):
            _append_uma(
                circuit,
                a_indices[i],
                b_indices[i + 1],
                a_indices[i + 1],
            )

        _append_uma(circuit, helper_index, b_indices[0], a_indices[0])
        return circuit

    num_qubits = 2 * n + 1
    a_indices = list(range(n))
    b_indices = list(range(n, 2 * n))
    helper_index = 2 * n
    circuit = Circuit(num_qubits)

    if n == 1:
        circuit.cx(a_indices[0], b_indices[0])
        return circuit

    _append_maj(circuit, helper_index, b_indices[0], a_indices[0])
    for i in range(n - 2):
        _append_maj(
            circuit,
            a_indices[i],
            b_indices[i + 1],
            a_indices[i + 1],
        )

    circuit.cx(a_indices[n - 2], b_indices[n - 1])

    for i in range(n - 3, -1, -1):
        _append_uma(
            circuit,
            a_indices[i],
            b_indices[i + 1],
            a_indices[i + 1],
        )

    _append_uma(circuit, helper_index, b_indices[0], a_indices[0])
    circuit.cx(a_indices[n - 1], b_indices[n - 1])
    return circuit
