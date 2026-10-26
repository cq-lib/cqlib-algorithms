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

r"""Gate-level implementation of the Weight Distribution Block.

This module provides the documented Step 2 and Step 4 subcircuits together
with the full Weight Distribution Block (WDB) operator. The active subcircuits
act on two ``k``-qubit windows arranged as ``A[0:k]`` followed by ``B[k:2k]``
in paper order, while the full WDB embeds those windows into the rightmost
``k`` qubits of the actual ``m``-qubit left register and ``(n-m)``-qubit
right register.

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


def _validate_positive_int(value: int, *, name: str) -> None:
    r"""Validate one positive integer parameter."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def _binomial(n: int, k: int) -> int:
    r"""Return the binomial coefficient with out-of-range protection."""
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def _unary_index(num_qubits: int, ones: int) -> int:
    r"""Return the basis index of ``|0^(n-ones) 1^ones>`` in paper order."""
    if ones == 0:
        return 0
    return ((1 << ones) - 1) << (num_qubits - ones)


def _one_hot_index(num_qubits: int, location: int) -> int:
    r"""Return the basis index of ``|0^(k-l) 1 0^(l-1)>`` in paper order."""
    if location < 1 or location > num_qubits:
        raise ValueError("location must satisfy 1 <= location <= num_qubits")
    return 1 << (num_qubits - location)


class WeightDistributionBlockControlledAddition:
    r"""Construct Step 2 of the Weight Distribution Block.

    The circuit acts on two ``k``-qubit registers:

    * register ``A`` starts in ``|0^k>``
    * register ``B`` stores ``ell`` in one-hot form
      ``|0^(k-ell) 1 0^(ell-1)>``

    For each ``1 <= ell <= k``, the circuit leaves ``B`` unchanged and maps
    ``A`` to the unary superposition

    .. math::

        \frac{1}{\sqrt{\binom{n}{\ell}}}
        \sum_{i=0}^{\ell}
        \sqrt{\binom{m}{i}\binom{n-m}{\ell-i}}
        |0^{k-i}1^i\rangle_A.

    The implementation follows the documented staircase of one-hot-controlled
    ``RY`` rotations. At level ``j``, the target is the ``j``-th unary bit of
    ``A`` and, for ``j > 1``, the previous unary bit provides the chain
    control.

    Args:
        n: Total Dicke-state width.
        m: Width of the left split.
        k: Maximum supported Hamming weight.

    Raises:
        TypeError: If any argument is not an integer or is a Boolean.
        ValueError: If the numeric ranges are invalid.
    """

    def __init__(self, n: int, m: int, k: int) -> None:
        self.validate(n, m, k)
        self._n = n
        self._m = m
        self._k = k

    @staticmethod
    def validate(n: int, m: int, k: int) -> None:
        r"""Validate the controlled-addition parameters."""
        _validate_positive_int(n, name="n")
        _validate_positive_int(m, name="m")
        _validate_positive_int(k, name="k")
        if m >= n:
            raise ValueError("m must be less than n")
        if k > n:
            raise ValueError("k must not exceed n")

    @property
    def n(self) -> int:
        r"""Return the total Dicke-state width."""
        return self._n

    @property
    def m(self) -> int:
        r"""Return the width of the left split."""
        return self._m

    @property
    def k(self) -> int:
        r"""Return the supported Hamming-weight bound."""
        return self._k

    @property
    def num_qubits(self) -> int:
        r"""Return the active circuit width ``2 * k``."""
        return 2 * self.k

    def build_circuit(self) -> Circuit:
        r"""Build and return a fresh controlled-addition circuit."""
        circuit = Circuit(self.num_qubits)
        for ell in range(1, self.k + 1):
            self._append_ell_staircase(circuit, ell)
        return circuit

    def _append_ell_staircase(self, circuit: Circuit, ell: int) -> None:
        r"""Append the rotation staircase selected by one-hot value ``ell``."""
        one_hot_control = self.k + (self.k - ell)

        for level in range(1, ell + 1):
            suffix_sum = self._suffix_sum(ell, level - 1)
            if suffix_sum == 0:
                continue

            branch_weight = self._weight_term(ell, level - 1)
            cosine = math.sqrt(branch_weight / suffix_sum)
            theta = 2.0 * math.acos(max(0.0, min(1.0, cosine)))
            target = self.k - level

            if level == 1:
                circuit.append_mc_gate(
                    StandardGate.RY(theta).control(1),
                    [one_hot_control, target],
                )
                continue

            chain_control = target + 1
            circuit.append_mc_gate(
                StandardGate.RY(theta).control(2),
                [one_hot_control, chain_control, target],
            )

    def _weight_term(self, ell: int, left_weight: int) -> int:
        r"""Return the unnormalized hypergeometric term."""
        return _binomial(self.m, left_weight) * _binomial(
            self.n - self.m,
            ell - left_weight,
        )

    def _suffix_sum(self, ell: int, start_weight: int) -> int:
        r"""Return the suffix sum used by the rotation-angle formula."""
        return sum(
            self._weight_term(ell, left_weight)
            for left_weight in range(start_weight, ell + 1)
        )


class WeightDistributionBlockUnarySubtraction:
    r"""Construct Step 4 of the Weight Distribution Block.

    The circuit acts on two ``k``-qubit unary registers:

    * register ``A`` stores ``l`` as ``|0^(k-l) 1^l>``
    * register ``B`` stores ``i`` as ``|0^(k-i) 1^i>``

    On the intended unary subspace ``0 <= i <= l <= k``, the operator performs

    .. math::

        |0^{k-l}1^l\rangle_A |0^{k-i}1^i\rangle_B
        \mapsto
        |0^{k+i-l}1^{l-i}\rangle_A |0^{k-i}1^i\rangle_B.

    Each active bit of ``B`` controls one prefix block on ``A`` consisting of
    a controlled staircase of adjacent SWAPs followed by a CNOT erasure.

    Args:
        k: Unary register width for both registers.

    Raises:
        TypeError: If ``k`` is not an integer or is a Boolean.
        ValueError: If ``k`` is not positive.
    """

    def __init__(self, k: int) -> None:
        self.validate(k)
        self._k = k

    @staticmethod
    def validate(k: int) -> None:
        r"""Validate the unary register width."""
        _validate_positive_int(k, name="k")

    @property
    def k(self) -> int:
        r"""Return the unary register width."""
        return self._k

    @property
    def num_qubits(self) -> int:
        r"""Return the active circuit width ``2 * k``."""
        return 2 * self.k

    def build_circuit(self) -> Circuit:
        r"""Build and return a fresh unary-subtraction circuit."""
        circuit = Circuit(self.num_qubits)
        for prefix_length in range(1, self.k + 1):
            self._append_controlled_prefix_block(circuit, prefix_length)
        return circuit

    def _append_controlled_prefix_block(
        self, circuit: Circuit, prefix_length: int
    ) -> None:
        r"""Append one controlled ``R`` + ``E`` prefix block."""
        control = self.k + prefix_length - 1
        prefix_start = 0
        prefix_head = prefix_length - 1

        # Right-rotate the active prefix of A by one position.
        for qubit in range(prefix_head - 1, prefix_start - 1, -1):
            circuit.append_mc_gate(
                StandardGate.SWAP.control(1),
                [control, qubit, qubit + 1],
            )

        # Erase the new prefix head after the rotation.
        circuit.cx(control, prefix_start)


class WeightDistributionBlock:
    r"""Construct the full Weight Distribution Block ``WDB_k^{n,m}``.

    The operator acts on ``n`` qubits split into a left register of width
    ``m`` and a right register of width ``n - m``. For every ``0 <= ell <= k``,
    it satisfies

    .. math::

        WDB_k^{n,m} |0^{n-ell}1^ell\rangle
        =
        \frac{1}{\sqrt{\binom{n}{\ell}}}
        \sum_{i=0}^{\ell}
        \sqrt{\binom{m}{i}\binom{n-m}{\ell-i}}
        |0^{m-i}1^i\rangle
        \otimes
        |0^{(n-m)-(\ell-i)}1^{\ell-i}\rangle.

    The implementation follows the four-step construction described by
    Bärtschi and Eidenbenz [1]_ [2]_:

    1. convert the active unary tail of the right register to one-hot form,
    2. run the controlled-addition staircase on the active ``k+k`` window,
    3. uncompute the one-hot encoding,
    4. subtract the left unary weight from the right unary weight.

    The rightmost ``k`` qubits of each register carry the non-trivial action;
    all leading qubits remain untouched.

    Args:
        n: Total circuit width.
        m: Width of the left register.
        k: Maximum supported Hamming weight.

    Raises:
        TypeError: If any argument is not an integer or is a Boolean.
        ValueError: If the numeric ranges are invalid.

    References
    ----------
    .. [1] Bärtschi, A., and Eidenbenz, S. (2019). Deterministic Preparation
           of Dicke States. In *Fundamentals of Computation Theory (FCT 2019)*,
           LNCS 11651, pp. 126-139. Springer.
    .. [2] Bärtschi, A., and Eidenbenz, S. (2022). Deterministic Preparation
           of Dicke States. *Quantum Information Processing*, 21, 282.
    """

    def __init__(self, n: int, m: int, k: int) -> None:
        self.validate(n, m, k)
        self._n = n
        self._m = m
        self._k = k

    @staticmethod
    def validate(n: int, m: int, k: int) -> None:
        r"""Validate the Weight Distribution Block parameters."""
        _validate_positive_int(n, name="n")
        _validate_positive_int(m, name="m")
        _validate_positive_int(k, name="k")
        if m >= n:
            raise ValueError("m must be less than n")
        if k > m:
            raise ValueError("k must not exceed m")
        if k > n - m:
            raise ValueError("k must not exceed n - m")

    @property
    def n(self) -> int:
        r"""Return the total circuit width."""
        return self._n

    @property
    def m(self) -> int:
        r"""Return the width of the left register."""
        return self._m

    @property
    def k(self) -> int:
        r"""Return the supported Hamming-weight bound."""
        return self._k

    @property
    def num_qubits(self) -> int:
        r"""Return the total circuit width."""
        return self.n

    def build_circuit(self) -> Circuit:
        r"""Build and return a fresh circuit for ``WDB_k^{n,m}``."""
        circuit = Circuit(self.num_qubits)
        if self.k == 0:
            return circuit

        circuit.compose(
            self._build_unary_to_one_hot_circuit(),
            self._right_active_positions(),
        )
        circuit.compose(
            WeightDistributionBlockControlledAddition(self.n, self.m, self.k)
            .build_circuit(),
            self._active_positions(),
        )
        circuit.compose(
            self._build_one_hot_to_unary_circuit(),
            self._right_active_positions(),
        )
        circuit.compose(
            WeightDistributionBlockUnarySubtraction(self.k).build_circuit(),
            self._subtraction_positions(),
        )
        return circuit

    def _left_active_positions(self) -> list[int]:
        r"""Return the active ``k``-qubit window inside the left register."""
        return list(range(self.m - self.k, self.m))

    def _right_active_positions(self) -> list[int]:
        r"""Return the active ``k``-qubit window inside the right register."""
        return list(range(self.n - self.k, self.n))

    def _active_positions(self) -> list[int]:
        r"""Return the active WDB window in ``A``-then-``B`` order."""
        return self._left_active_positions() + self._right_active_positions()

    def _subtraction_positions(self) -> list[int]:
        r"""Return the active Step 4 window in ``right``-then-``left`` order."""
        return self._right_active_positions() + self._left_active_positions()

    def _build_unary_to_one_hot_circuit(self) -> Circuit:
        r"""Build the Step 1 unary-to-one-hot CNOT chain."""
        circuit = Circuit(self.k)
        for qubit in range(self.k - 2, -1, -1):
            circuit.cx(qubit, qubit + 1)
        return circuit

    def _build_one_hot_to_unary_circuit(self) -> Circuit:
        r"""Build the Step 3 one-hot-to-unary inverse CNOT chain."""
        circuit = Circuit(self.k)
        for qubit in range(self.k - 1):
            circuit.cx(qubit, qubit + 1)
        return circuit
