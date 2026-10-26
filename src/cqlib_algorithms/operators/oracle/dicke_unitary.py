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

r"""Bartschi-Eidenbenz unary-to-Dicke unitary construction.

This module implements the family of unitaries ``U_{n,k}`` from the
deterministic Dicke-state construction of Bärtschi and Eidenbenz. For every
``l`` with ``0 <= l <= k``, the operator satisfies

.. math::

    U_{n,k} |0\rangle^{n - l} |1\rangle^l = |D_l^n\rangle,

where ``|D_l^n>`` is the ``n``-qubit Dicke state of Hamming weight ``l``.

The implementation follows the recursive construction proved in Lemma 2 of the
paper:

.. math::

    U_{n,k} = (U_{n - 1, \min(k, n - 1)} \otimes I)
        (I^{\otimes n - \min(k + 1, n)} \otimes SCS_{n, \min(k, n - 1)}).

This formulation is equivalent to the product expression in the paper and is
implemented by composing :class:`SplitCycleShift` blocks on the right-aligned
suffix that carries the unary input.

References
----------
.. [1] Bärtschi, A., and Eidenbenz, S. (2019). Deterministic Preparation of
       Dicke States. In *Fundamentals of Computation Theory (FCT 2019)*,
       LNCS 11651, pp. 126-139. Springer.
.. [2] Bärtschi, A., and Eidenbenz, S. (2022). Deterministic Preparation of
       Dicke States. *Quantum Information Processing*, 21, 282.
"""

from __future__ import annotations

from cqlib import Circuit

from .split_cycle_shift import SplitCycleShift


class DickeUnitary:
    r"""Construct the unary-to-Dicke unitary ``U_{n,k}``.

    The operator acts on ``n`` qubits. For each right-aligned unary input
    ``|0>^(n - l) |1>^l`` with ``0 <= l <= k``, the output is the exact Dicke
    state ``|D_l^n>``.

    Args:
        n: Total number of qubits.
        k: Maximum Dicke weight covered by the unitary.

    Raises:
        TypeError: If ``n`` or ``k`` is not an integer or is a Boolean.
        ValueError: If ``n`` is not positive, or if ``k`` lies outside
            ``0 <= k <= n``.
    """

    def __init__(self, n: int, k: int) -> None:
        self.validate(n, k)
        self._n = n
        self._k = k

    @staticmethod
    def validate(n: int, k: int) -> None:
        r"""Validate the unary-to-Dicke unitary parameters."""
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError("n must be an integer")
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer")
        if n <= 0:
            raise ValueError("n must be greater than zero")
        if not 0 <= k <= n:
            raise ValueError("k must satisfy 0 <= k <= n")

    @property
    def n(self) -> int:
        r"""Return the total circuit width."""
        return self._n

    @property
    def k(self) -> int:
        r"""Return the maximum Dicke weight covered by the unitary."""
        return self._k

    @property
    def num_qubits(self) -> int:
        r"""Return the number of qubits acted on by the unitary."""
        return self.n

    def build_circuit(self) -> Circuit:
        r"""Build and return a fresh circuit for ``U_{n,k}``."""
        return self._build_recursive_circuit(self.n, self.k)

    @classmethod
    def _build_recursive_circuit(cls, num_qubits: int, max_weight: int) -> Circuit:
        r"""Recursively construct ``U_{n,k}`` from the paper recurrence."""
        circuit = Circuit(num_qubits)
        if max_weight == 0 or num_qubits == 1:
            return circuit

        local_k = min(max_weight, num_qubits - 1)
        scs = SplitCycleShift(num_qubits, local_k).build_circuit()
        scs_positions = list(range(num_qubits - local_k - 1, num_qubits))
        circuit.compose(scs, scs_positions)

        recursive = cls._build_recursive_circuit(num_qubits - 1, local_k)
        circuit.compose(recursive, list(range(num_qubits - 1)))
        return circuit
