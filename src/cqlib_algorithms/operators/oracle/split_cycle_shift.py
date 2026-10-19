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

r"""Split-and-cyclic-shift circuit construction.

This module implements the ``SCS_{n,k}`` building block introduced in the
deterministic Dicke-state construction of Bärtschi and Eidenbenz. The operator
acts on ``k + 1`` local qubits and, for each unary state
``|0>^(k + 1 - l) |1>^l`` with ``1 <= l <= k``, performs

.. math::

    |0\rangle^{k + 1 - l}|1\rangle^l \mapsto
    \sqrt{l / n} |0\rangle^{k + 1 - l}|1\rangle^l
    + \sqrt{(n - l) / n} |0\rangle^{k - l}|1\rangle^l|0\rangle,

while leaving ``|0...0>`` and ``|1...1>`` unchanged.

The implementation follows the gate decomposition described in Subsection 2.2
of the paper: one two-qubit block for ``l = 1`` and ``k - 1`` three-qubit
blocks for ``l = 2, ..., k``.

References
----------
.. [1] Bärtschi, A., and Eidenbenz, S. (2019). Deterministic Preparation of
       Dicke States. In *Fundamentals of Computation Theory (FCT 2019)*,
       LNCS 11651, pp. 126-139. Springer.
.. [2] Bärtschi, A., and Eidenbenz, S. (2022). Deterministic Preparation of
       Dicke States. *Quantum Information Processing*, 21, 282.
"""

from __future__ import annotations

import math

from cqlib.circuit.gates import StandardGate

from cqlib import Circuit


class SplitCycleShift:
    r"""Construct the exact local ``SCS_{n,k}`` operator.

    The returned circuit acts on the local qubit set ``0..k``. Local qubit
    ``k`` is the rightmost site in the unary basis used by the SCS definition.

    Args:
        n: Global Dicke-construction parameter controlling the amplitudes.
        k: Local unary-weight bound. The circuit acts on ``k + 1`` qubits.

    Raises:
        TypeError: If ``n`` or ``k`` is not an integer or is a Boolean.
        ValueError: If ``n`` or ``k`` is not positive, or if ``k > n``.
    """

    __slots__ = ("_k", "_n")

    _n: int
    _k: int

    def __init__(self, n: int, k: int) -> None:
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError("n must be an integer")
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer")
        if n <= 0:
            raise ValueError("n must be greater than zero")
        if k <= 0:
            raise ValueError("k must be greater than zero")
        if k > n:
            raise ValueError("k must not exceed n")
        self._n = n
        self._k = k

    @property
    def n(self) -> int:
        r"""Return the global SCS amplitude parameter."""
        return self._n

    @property
    def k(self) -> int:
        r"""Return the local unary-weight bound."""
        return self._k

    @property
    def num_qubits(self) -> int:
        r"""Return the local circuit width ``k + 1``."""
        return self.k + 1

    def build_circuit(self) -> Circuit:
        r"""Build and return a fresh ``SCS_{n,k}`` circuit."""
        circuit = Circuit(self.num_qubits)
        right = self.k

        circuit.cx(right - 1, right)
        circuit.append_mc_gate(
            StandardGate.RY(self._rotation_angle(1)).control(1),
            [right, right - 1],
        )
        circuit.cx(right - 1, right)

        for ell in range(2, self.k + 1):
            left = self.k - ell
            middle = left + 1

            circuit.cx(left, right)
            circuit.append_mc_gate(
                StandardGate.RY(self._rotation_angle(ell)).control(2),
                [middle, right, left],
            )
            circuit.cx(left, right)

        return circuit

    def _rotation_angle(self, ell: int) -> float:
        r"""Return ``2 arccos(sqrt(ell / n))`` for one SCS block."""
        return float(2.0 * math.acos(math.sqrt(ell / self.n)))
