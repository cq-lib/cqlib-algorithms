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

"""Long phase-matched diffusion operator construction.

This module implements the phase-matched diffusion operator used in the
Long variant of Grover search.  For a real phase ``phi`` and the uniform
superposition ``|s> = H**n |0...0>``, the operator is

.. math::

    D_{\\phi} = H^{\\otimes n} S_0(\\phi) H^{\\otimes n},

with

.. math::

    S_0(\\phi) = I + (e^{i\\phi} - 1)|0\\ldots 0\\rangle\\langle 0\\ldots 0|.

When ``phi = pi``, this reduces to ``I - 2|s><s|``, which is equal to the
standard Grover diffusion operator up to a global minus sign.

The helper formulas :meth:`GroverLong.get_beta`,
:meth:`GroverLong.get_it_num`, and :meth:`GroverLong.get_phi` follow the
phase-matching convention used by the legacy prototype:

.. math::

    \\beta = \\arcsin\\left(\\sqrt{M/N}\\right),

.. math::

    j = \\left\\lfloor \\frac{\\pi/2 - \\beta}{2\\beta} \\right\\rfloor,

.. math::

    \\phi = 2 \\arcsin\\left(\\frac{\\sin(\\pi / (4j + 6))}{\\sin(\\beta)}\\right).

References
----------
[1] G. L. Long, "Grover algorithm with zero theoretical failure rate,"
    Physical Review A, 64(2), 022307, 2001.
[2] G. L. Long, Y. S. Li, W. L. Zhang, and L. Niu, "Phase matching in
    quantum searching," Physics Letters A, 262(1), 27-34, 1999.
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
from cqlib.circuit.gates import StandardGate

from cqlib import Circuit


class GroverLong:
    """Construct the Long diffusion operator with an arbitrary phase.

    For a real phase ``phi`` and the uniform superposition ``|s> = H**n|0>``,
    this component returns the circuit for

    .. math::

        D_{\\phi} = H^{\\otimes n} S_0(\\phi) H^{\\otimes n},

    where

    .. math::

        S_0(\\phi) = I + (e^{i\\phi} - 1)|0\\ldots 0\\rangle\\langle 0\\ldots 0|.

    When ``phi = pi``, the operator is ``I - 2|s><s|``, which equals the
    standard Grover diffusion operator up to a global minus sign.

    Args:
        num_qubits: Positive number of reflected qubits.
        phase: Reflection phase ``phi`` in radians.

    Raises:
        TypeError: If ``num_qubits`` is not an integer or is a Boolean, or if
            ``phase`` is not a real finite scalar.
        ValueError: If ``num_qubits`` is not positive.
    """

    __slots__ = ("_num_qubits", "_phase")

    _num_qubits: int
    _phase: float

    def __init__(self, num_qubits: int, *, phase: float) -> None:
        if isinstance(num_qubits, bool) or not isinstance(num_qubits, int):
            raise TypeError("num_qubits must be an integer")
        if num_qubits <= 0:
            raise ValueError("num_qubits must be greater than zero")
        if isinstance(phase, bool) or not isinstance(phase, Real):
            raise TypeError("phase must be a real finite scalar")
        phase_float = float(phase)
        if not math.isfinite(phase_float):
            raise TypeError("phase must be a real finite scalar")

        self._num_qubits = num_qubits
        self._phase = phase_float

    @property
    def num_qubits(self) -> int:
        """Return the number of diffusion qubits."""
        return self._num_qubits

    @property
    def phase(self) -> float:
        """Return the configured phase in radians."""
        return self._phase

    def build_circuit(self) -> Circuit:
        """Build and return a fresh Long diffusion circuit."""
        circuit = Circuit(self.num_qubits)
        for qubit in range(self.num_qubits):
            circuit.h(qubit)
        circuit.compose(self._build_zero_state_phase_rotation())
        for qubit in range(self.num_qubits):
            circuit.h(qubit)
        return circuit

    def _build_zero_state_phase_rotation(self) -> Circuit:
        """Return ``S_0(phi)`` on the local ``num_qubits`` register."""
        circuit = Circuit(self.num_qubits)
        for qubit in range(self.num_qubits):
            circuit.x(qubit)

        if self.num_qubits == 1:
            circuit.append_gate(StandardGate.Phase(self.phase), [0])
        else:
            phase_gate = StandardGate.Phase(self.phase).control(
                self.num_qubits - 1
            )
            circuit.append_mc_gate(phase_gate, list(range(self.num_qubits)))

        for qubit in range(self.num_qubits - 1, -1, -1):
            circuit.x(qubit)
        return circuit

    @staticmethod
    def get_beta(num_find: int, num_total: int) -> float:
        """Return ``beta = arcsin(sqrt(num_find / num_total))``.

        Args:
            num_find: Number of marked states.
            num_total: Total number of candidate states.

        Raises:
            TypeError: If integer arguments are not valid integers.
            ValueError: If counts are non-positive or ``num_find > num_total``.
        """
        if isinstance(num_find, bool) or not isinstance(num_find, int):
            raise TypeError("num_find must be an integer")
        if isinstance(num_total, bool) or not isinstance(num_total, int):
            raise TypeError("num_total must be an integer")
        if num_find <= 0:
            raise ValueError("num_find must be greater than zero")
        if num_total <= 0:
            raise ValueError("num_total must be greater than zero")
        if num_find > num_total:
            raise ValueError("num_find must not exceed num_total")
        return float(np.arcsin(np.sqrt(num_find / num_total)))

    @staticmethod
    def get_it_num(beta: float) -> int:
        """Return the Long-operator iteration count.

        The adopted convention matches the legacy prototype:

        .. math::

            j = \\lfloor (\\pi/2 - \\beta) / (2\\beta) \\rfloor.

        Raises:
            TypeError: If ``beta`` is not a real finite scalar.
            ValueError: If ``beta`` is outside ``(0, pi/2]``.
        """
        if isinstance(beta, bool) or not isinstance(beta, Real):
            raise TypeError("beta must be a real finite scalar")
        beta_float = float(beta)
        if not math.isfinite(beta_float):
            raise TypeError("beta must be a real finite scalar")
        if not 0.0 < beta_float <= math.pi / 2.0:
            raise ValueError("beta must satisfy 0 < beta <= pi / 2")
        ratio = (math.pi / 2.0 - beta_float) / (2.0 * beta_float)
        return int(np.floor(ratio + 1e-12))

    @staticmethod
    def get_phi(it_num: int, beta: float) -> float:
        """Return the phase-matched Long angle.

        The adopted convention matches the legacy prototype:

        .. math::

            \\phi = 2 \\arcsin\\left(
                \\frac{\\sin(\\pi / (4j + 6))}{\\sin(\\beta)}
            \\right).

        Raises:
            TypeError: If inputs are not valid scalar types.
            ValueError: If ``it_num`` is negative or ``beta`` is outside
                ``(0, pi/2]``.
        """
        if isinstance(it_num, bool) or not isinstance(it_num, int):
            raise TypeError("it_num must be an integer")
        if it_num < 0:
            raise ValueError("it_num must be non-negative")
        if isinstance(beta, bool) or not isinstance(beta, Real):
            raise TypeError("beta must be a real finite scalar")
        beta_float = float(beta)
        if not math.isfinite(beta_float):
            raise TypeError("beta must be a real finite scalar")
        if not 0.0 < beta_float <= math.pi / 2.0:
            raise ValueError("beta must satisfy 0 < beta <= pi / 2")

        numerator = math.sin(math.pi / (4 * it_num + 6))
        denominator = math.sin(beta_float)
        ratio = numerator / denominator
        if abs(ratio) > 1.0:
            ratio = max(-1.0, min(1.0, ratio))
        return float(2.0 * math.asin(ratio))

    @classmethod
    def get_phi_total(cls, num_find: int, num_total: int) -> float:
        """Return the phase derived from ``num_find`` and ``num_total``."""
        beta = cls.get_beta(num_find, num_total)
        it_num = cls.get_it_num(beta)
        return cls.get_phi(it_num, beta)


if __name__ == "__main__":
    operator = GroverLong(3, phase=math.pi)
    print(operator.build_circuit().to_matrix())
