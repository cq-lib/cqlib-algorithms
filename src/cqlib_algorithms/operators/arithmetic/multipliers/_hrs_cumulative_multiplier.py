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
"""HRS cumulative multiplier circuits.

This module implements a minimal cumulative multiplier based on controlled
Cuccaro ripple-carry stages. The local circuit layout follows the arithmetic
package convention with LSB-first register indexing.

References
Martin Haener, Martin Roetteler, Krysta M Svore,
Optimizing Quantum Circuits for Arithmetic, 2018, arXiv:1805.12445.

Steven A Cuccaro, Thomas G Draper, Samuel A Kutin, David Petrie Moulton,
A New Quantum Ripple Carry Addition Circuit, 2004, arXiv:quant-ph/0410184.
"""

from __future__ import annotations

from cqlib.circuit.gates import StandardGate

from cqlib import Circuit


def _accumulator_window_indices(
    accumulator_indices: list[int],
    bit: int,
    width: int,
) -> list[int]:
    """Return the width-sized accumulator window for one multiplier stage."""
    return list(accumulator_indices[bit : bit + width])


def _append_controlled_cx(
    circuit: Circuit,
    control_index: int,
    source_index: int,
    target_index: int,
) -> None:
    """Append a CNOT controlled by a global enable qubit."""
    gate = StandardGate.X.control(2)
    circuit.append_mc_gate(
        gate,
        [control_index, source_index, target_index],
    )


def _append_controlled_ccx(
    circuit: Circuit,
    control_index: int,
    source_a_index: int,
    source_b_index: int,
    target_index: int,
) -> None:
    """Append a Toffoli controlled by a global enable qubit."""
    gate = StandardGate.X.control(3)
    circuit.append_mc_gate(
        gate,
        [control_index, source_a_index, source_b_index, target_index],
    )


def _append_controlled_maj(
    circuit: Circuit,
    control_index: int,
    a_index: int,
    b_index: int,
    c_index: int,
) -> None:
    """Append the controlled Cuccaro MAJ primitive."""
    _append_controlled_cx(circuit, control_index, c_index, b_index)
    _append_controlled_cx(circuit, control_index, c_index, a_index)
    _append_controlled_ccx(
        circuit,
        control_index,
        a_index,
        b_index,
        c_index,
    )


def _append_controlled_uma(
    circuit: Circuit,
    control_index: int,
    a_index: int,
    b_index: int,
    c_index: int,
) -> None:
    """Append the controlled Cuccaro UMA primitive."""
    _append_controlled_ccx(
        circuit,
        control_index,
        a_index,
        b_index,
        c_index,
    )
    _append_controlled_cx(circuit, control_index, c_index, a_index)
    _append_controlled_cx(circuit, control_index, a_index, b_index)


def _build_controlled_ripple_stage(
    circuit: Circuit,
    *,
    control_index: int,
    a_indices: list[int],
    window_indices: list[int],
    carry_out_index: int,
    helper_index: int,
) -> None:
    """Append one controlled half-adder stage into the product register."""
    n = len(a_indices)

    _append_controlled_maj(
        circuit,
        control_index,
        helper_index,
        window_indices[0],
        a_indices[0],
    )
    for i in range(n - 1):
        _append_controlled_maj(
            circuit,
            control_index,
            a_indices[i],
            window_indices[i + 1],
            a_indices[i + 1],
        )

    _append_controlled_cx(
        circuit,
        control_index,
        a_indices[n - 1],
        carry_out_index,
    )

    for i in range(n - 2, -1, -1):
        _append_controlled_uma(
            circuit,
            control_index,
            a_indices[i],
            window_indices[i + 1],
            a_indices[i + 1],
        )

    _append_controlled_uma(
        circuit,
        control_index,
        helper_index,
        window_indices[0],
        a_indices[0],
    )


def build_hrs_cumulative_multiplier(num_state_qubits: int) -> Circuit:
    """Build and return the internal HRS cumulative multiplier circuit."""
    n = num_state_qubits
    num_qubits = 4 * n + 1
    a_indices = list(range(n))
    b_indices = list(range(n, 2 * n))
    product_indices = list(range(2 * n, 4 * n))
    helper_index = 4 * n

    circuit = Circuit(num_qubits)
    for bit, control_index in enumerate(b_indices):
        window_indices = _accumulator_window_indices(product_indices, bit, n)
        carry_out_index = product_indices[bit + n]
        _build_controlled_ripple_stage(
            circuit,
            control_index=control_index,
            a_indices=a_indices,
            window_indices=window_indices,
            carry_out_index=carry_out_index,
            helper_index=helper_index,
        )
    return circuit
