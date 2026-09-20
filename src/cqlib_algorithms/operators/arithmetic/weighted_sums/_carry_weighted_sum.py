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
"""Computational basis weighted sum via carry-chain multi-controlled X gates.

This module implements the multiqubit weighted sum operator in the
computational basis following the two-route design document
``weighted_sum_two_routes.md`` (Route 1: Weighted sum via classical carry
schedule).  The operator maps

.. math::

    |x_0, x_1, \\dots, x_{n-1}\\rangle |0, \\dots, 0\\rangle
        \\mapsto
    |x_0, x_1, \\dots, x_{n-1}\\rangle
        \\left| \\sum_{i=0}^{n-1} w_i \\cdot x_i \\right\\rangle,

where the LSB-first accumulator of width

.. math::

    s = 1 + \\lfloor \\log_2(\\sum w_i) \\rfloor

stores the classical weighted sum.

Construction
------------
The synthesis mirrors a classical ripple-carry adder where each qubit of the
result is obtained by XOR-ing the weighted-population parity of the
corresponding bit-plane with the carry propagated from the previous bit.
For every bit level :math:`j \\in [0, s)` and every input qubit the
algorithm branches on the ``j``-th bit of the classical weight ``w_i`` and
conditionally applies

* a CNOT between the state qubit and the ``j``-th sum bit,
* a three-controlled Toffoli (C3X) that combines ``state_i``, ``carry_in_j``
  and a previously computed partial sum bit to produce ``carry_out_j``, and
* additional controlled operations that propagate the carry forward when the
  weight bit is zero.

The three-controlled Toffoli gates required by the middle steps (
:math:`j \\in [1, s-2]`) are decomposed into a linear CCX v-chain following
Barenco *et al.* [1]_ Lemma 7.3, which uses a single dirty ancilla and four
CCX gates applied in forward / reverse order so that the helper returns to
its original computational basis state.

References
----------
.. [1] Barenco, A., Bennett, C. H., Cleve, R., DiVincenzo, D. P.,
       Margolus, N., Shor, P., Sleator, T., Smolin, J. A., and
       Weinfurter, H. (1995).  Elementary gates for quantum computation.
       *Physical Review A*, 52(5), 3457.
"""

from __future__ import annotations

from collections.abc import Sequence

from cqlib import Circuit


def _append_c3x_dirty(
    circuit: Circuit,
    controls: Sequence[int],
    target: int,
    helper: int,
) -> None:
    """Append a CCX-v-chain decomposition of a 3-controlled Toffoli.

    The helper qubit is assumed to be initialised to ``|0>`` in the
    computational basis and is guaranteed to be restored to ``|0>`` after
    the four CCX gates (Barenco v-chain).
    """

    c0, c1, c2 = controls
    circuit.ccx(c0, c1, helper)
    circuit.ccx(c2, helper, target)
    circuit.ccx(c0, c1, helper)
    circuit.ccx(c2, helper, target)


def _bit(value: int, position: int) -> int:
    """Return the ``position``-th LSB bit of ``value`` (0 or 1)."""

    return (value >> position) & 1


def build_carry_weighted_sum(weights: list[int]) -> Circuit:
    """Return a Circuit implementing the weighted sum via carry-chain MCX.

    Parameters
        weights Classical per-qubit weight assigned to each of the ``n``
            state qubits.  Every entry is interpreted as a non-negative
            integer and ``len(weights) == n`` is required.

    Returns
        A fresh ``Circuit`` with layout

        * state qubits: ``[0, n)``
        * sum qubits (LSB first): ``[n, n + s)``
        * intermediate carry qubits (one per level ``1..s-1``):
          ``[n + s, n + 2s - 1)`` (``s - 1`` qubits total)
        * control helper (present only when ``s > 2``): ``n + 2s - 2``

        On output the sum register contains the LSB-first binary
        representation of ``sum(w_i * x_i)`` and every carry / helper
        ancilla is guaranteed to be restored to its initial ``|0>`` state.
    """

    n = len(weights)
    total_weight = sum(weights)
    if total_weight == 0:
        s = 1
    else:
        s = total_weight.bit_length()

    num_carries = max(s - 1, 0)
    need_ctrl = s > 2
    num_qubits = n + s + num_carries + (1 if need_ctrl else 0)
    circuit = Circuit(num_qubits)

    state_indices = list(range(n))
    sum_indices = list(range(n, n + s))
    if num_carries > 0:
        carry_indices = list(range(n + s, n + s + num_carries))
    else:
        carry_indices = []
    control_index: int | None = (
        n + s + num_carries if need_ctrl else None
    )

    for i in range(n):
        w_i = weights[i]
        if w_i == 0:
            continue
        state_bit = state_indices[i]

        for j in range(s):
            w_bit = _bit(w_i, j)
            sum_bit = sum_indices[j]
            if j < s - 1:
                carry_out = carry_indices[j]
            else:
                carry_out = None
            if j >= 1:
                carry_in = carry_indices[j - 1]
            else:
                carry_in = None

            if s == 1:
                if w_bit == 1:
                    circuit.cx(state_bit, sum_bit)
                continue

            if j == 0:
                if w_bit == 1:
                    assert carry_out is not None
                    circuit.ccx(state_bit, sum_bit, carry_out)
                    circuit.cx(state_bit, sum_bit)
                continue

            if j == s - 1:
                assert carry_in is not None
                if w_bit == 1:
                    circuit.cx(state_bit, sum_bit)
                    circuit.ccx(state_bit, carry_in, sum_bit)
                else:
                    circuit.ccx(state_bit, carry_in, sum_bit)
                continue

            assert carry_in is not None
            assert carry_out is not None
            assert control_index is not None

            if w_bit == 1:
                circuit.x(sum_bit)
                circuit.x(carry_in)
                _append_c3x_dirty(
                    circuit,
                    [state_bit, sum_bit, carry_in],
                    carry_out,
                    control_index,
                )
                circuit.cx(state_bit, carry_out)
                circuit.x(sum_bit)
                circuit.x(carry_in)
                circuit.cx(state_bit, sum_bit)
                circuit.ccx(state_bit, carry_in, sum_bit)
            else:
                _append_c3x_dirty(
                    circuit,
                    [state_bit, sum_bit, carry_in],
                    carry_out,
                    control_index,
                )
                circuit.ccx(state_bit, carry_in, sum_bit)

        for j in range(s - 1, -1, -1):
            w_bit = _bit(w_i, j)
            if s == 1:
                continue
            if j == 0:
                if w_bit == 1:
                    carry_out = carry_indices[0]
                    sum_bit = sum_indices[0]
                    circuit.x(sum_bit)
                    circuit.ccx(state_bit, sum_bit, carry_out)
                    circuit.x(sum_bit)
                continue
            if j == s - 1:
                continue
            carry_in = carry_indices[j - 1]
            carry_out = carry_indices[j]
            sum_bit = sum_indices[j]
            assert control_index is not None

            if w_bit == 1:
                circuit.x(carry_in)
                _append_c3x_dirty(
                    circuit,
                    [state_bit, sum_bit, carry_in],
                    carry_out,
                    control_index,
                )
                circuit.cx(state_bit, carry_out)
                circuit.x(carry_in)
            else:
                circuit.x(sum_bit)
                _append_c3x_dirty(
                    circuit,
                    [state_bit, sum_bit, carry_in],
                    carry_out,
                    control_index,
                )
                circuit.x(sum_bit)

    return circuit
