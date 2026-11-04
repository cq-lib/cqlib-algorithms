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
"""Draper QFT multiplier circuits.

This module implements a minimal QFT-based multiplier using a single forward
QFT on the product register, a schedule of doubly controlled phase rotations,
and a final inverse QFT. The local circuit layout follows the arithmetic
package convention with LSB-first register indexing.

References
----------
[1] L. Ruiz-Perez and J. C. Garcia-Escartin, "Quantum arithmetic with the
    Quantum Fourier Transform," 2017.
    arXiv:1411.5949.
[2] T. G. Draper, "Addition on a Quantum Computer," 2000.
    arXiv:quant-ph/0008033.
"""

from __future__ import annotations

import math

from cqlib import Circuit
from cqlib.circuit.gates import StandardGate

from cqlib_algorithms.operators.composite.qft import QFT

_TWO_PI = 2.0 * math.pi


def _qft_degree(
    register_size: int,
    approximation_degree: int | None,
) -> int:
    """Translate an optional kept-distance hint into QFT degree."""
    if approximation_degree is None:
        return 0
    qft_degree = max(0, register_size - 1 - int(approximation_degree))
    if qft_degree >= register_size:
        qft_degree = register_size - 1
    return qft_degree


def _append_double_controlled_phase(
    circuit: Circuit,
    control_a: int,
    control_b: int,
    target: int,
    lam: float,
) -> None:
    """Append a doubly controlled phase rotation."""
    gate = StandardGate.Phase(lam).control(2)
    circuit.append_mc_gate(gate, [control_a, control_b, target])


def _append_qft_multiplier_schedule(
    circuit: Circuit,
    *,
    a_indices: list[int],
    b_indices: list[int],
    product_indices: list[int],
) -> None:
    """Append the doubly controlled phase schedule in the Fourier basis."""
    for i, control_a in enumerate(a_indices):
        for j, control_b in enumerate(b_indices):
            weight = i + j
            for l in range(weight, len(product_indices)):
                lam = _TWO_PI / (2 ** (l - weight + 1))
                _append_double_controlled_phase(
                    circuit,
                    control_a,
                    control_b,
                    product_indices[l],
                    lam,
                )


def build_draper_qft_multiplier(
    num_state_qubits: int,
    approximation_degree: int | None = None,
) -> Circuit:
    """Build and return the internal Draper QFT multiplier circuit."""
    n = num_state_qubits
    a_indices = list(range(n))
    b_indices = list(range(n, 2 * n))
    product_indices = list(range(2 * n, 4 * n))
    circuit = Circuit(4 * n)

    qft_degree = _qft_degree(len(product_indices), approximation_degree)
    forward_qft = QFT(
        len(product_indices),
        approximation_degree=qft_degree,
        do_swaps=False,
        inverse=False,
    ).build_circuit()
    circuit.compose(forward_qft, product_indices)

    _append_qft_multiplier_schedule(
        circuit,
        a_indices=a_indices,
        b_indices=b_indices,
        product_indices=product_indices,
    )

    inverse_qft = QFT(
        len(product_indices),
        approximation_degree=qft_degree,
        do_swaps=False,
        inverse=True,
    ).build_circuit()
    circuit.compose(inverse_qft, product_indices)
    return circuit
