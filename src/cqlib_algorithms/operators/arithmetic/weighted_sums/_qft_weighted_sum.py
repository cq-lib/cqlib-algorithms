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
"""QFT based weighted sum synthesis (Ruiz-Perez 2017).

This module implements the quantum Fourier transform based construction of
the weighted sum operator described in Section 6 of Ruiz-Perez and
Garcia-Escartin [1]_.  The operator maps

.. math::

    |x_0, x_1, \\dots, x_{n-1}\\rangle |0\\rangle
        \\mapsto
    |x_0, x_1, \\dots, x_{n-1}\\rangle
        \\left| \\sum_{i=0}^{n-1} w_i \\cdot x_i \\right\\rangle,

where :math:`w_i \\in \\mathbb{N}` are classical per-qubit weights and the
result is stored LSB first in the :math:`s`-qubit accumulator initialised to
:math:`|0\\rangle`.

The overall structure mirrors the original Draper QFT adder [2]_:
1. Apply a forward QFT (without final swaps) to the accumulator register.
2. For every pair :math:`(i, l)` of a state qubit and an accumulator qubit
   apply the single-qubit controlled rotation

   .. math::

       \\text{CP}(\\theta_{i,l}) \\quad\\text{with}\\quad
       \\theta_{i,l} = \\frac{2\\pi \\cdot w_i}{2^{l+1}}

   with :math:`x_i` serving as the control.  This is exactly the per-bit
   schedule used by Ruiz-Perez to accumulate the binary expansion of the
   weighted sum into the QFT phase of the accumulator.
3. Apply the inverse QFT to the accumulator to map the phases back into the
   computational basis.

The whole construction uses no ancillary qubits and the total circuit width
is therefore :math:`n + s` where :math:`s` is the accumulator width
determined by :math:`1 + \\lfloor \\log_2(\\sum w_i) \\rfloor`.

References
----------
[1] L. Ruiz-Perez and J. C. Garcia-Escartin, "Quantum arithmetic with the
    quantum Fourier transform," Quantum Information Processing, 16(6), 153,
    2017.
    arXiv:1411.5949.
[2] T. G. Draper, "Addition on a quantum computer," 2000.
    arXiv:quant-ph/0008033.
"""

from __future__ import annotations

from cqlib.circuit.gates import StandardGate

from cqlib import Circuit
from cqlib_algorithms.transforms import QFT as TransformQFT

_TWO_PI: float = 2.0 * 3.141592653589793


def _append_controlled_phase(
    circuit: Circuit,
    control: int,
    target: int,
    lam: float,
) -> None:
    """Append a single-controlled phase rotation matching the Draper style."""

    gate = StandardGate.Phase(lam).control(1)
    circuit.append_mc_gate(gate, [control, target])


def build_qft_weighted_sum(
    weights: list[int],
    approximation_degree: int | None = None,
) -> Circuit:
    """Return a Circuit implementing the weighted sum via QFT phase addition.

    Parameters
        weights Classical per-qubit weight assigned to each of the ``n``
            state qubits.  Every entry is interpreted as a non-negative
            integer and ``len(weights) == n`` is required.
        approximation_degree Optional QFT approximation threshold forwarded
            to the forward and inverse Fourier transforms.  When ``None``
            the exact QFT is used (default).

    Returns
        A fresh ``Circuit`` of width ``n + s`` where the first ``n`` qubits
        carry the read-only ``|x>`` state and the following ``s`` qubits
        hold the LSB-first binary representation of
        ``sum(w_i * x_i)``.  The caller is responsible for initialising the
        accumulator qubits to ``|0>``.
    """

    n = len(weights)
    total_weight = sum(weights)
    if total_weight == 0:
        s = 1
    else:
        s = total_weight.bit_length()
    num_qubits = n + s
    circuit = Circuit(num_qubits)

    state_indices = list(range(n))
    sum_indices = list(range(n, n + s))

    if approximation_degree is None:
        qft_degree = 0
    else:
        max_kept = int(approximation_degree)
        if s >= 1:
            qft_degree = max(0, s - 1 - max_kept)
            if qft_degree >= s:
                qft_degree = s - 1
        else:
            qft_degree = 0

    fwd_qft_circ: Circuit = TransformQFT(
        s,
        approximation_degree=qft_degree,
        do_swaps=False,
        inverse=False,
    ).build_circuit()
    circuit.compose(fwd_qft_circ, sum_indices)

    for i in range(n):
        w_i = weights[i]
        if w_i == 0:
            continue
        control = state_indices[i]
        for l in range(s):
            denominator = 1 << (l + 1)
            lam = _TWO_PI * w_i / denominator
            if abs(lam) < 1e-14:
                continue
            target = sum_indices[l]
            _append_controlled_phase(circuit, control, target, lam)

    inv_qft_circ: Circuit = TransformQFT(
        s,
        approximation_degree=qft_degree,
        do_swaps=False,
        inverse=True,
    ).build_circuit()
    circuit.compose(inv_qft_circ, sum_indices)

    return circuit
