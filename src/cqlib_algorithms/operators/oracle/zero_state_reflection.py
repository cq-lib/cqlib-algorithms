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

"""Reflection about the all-zero computational-basis state.

References
----------
[1] G. Brassard, P. Hoyer, M. Mosca, and A. Tapp, "Quantum amplitude
    amplification and estimation," Contemporary Mathematics, 305, 53-74,
    2002.
    arXiv:quant-ph/0005055.
"""

from __future__ import annotations

import math

from cqlib import Circuit
from cqlib.circuit.gates import StandardGate


class ZeroStateReflection:
    """Construct ``R0 = 2|0...0><0...0| - I`` exactly.

    The global phase is part of this contract and is retained explicitly; the
    returned circuit is not merely equivalent up to global phase.

    Args:
        num_qubits: Positive number of reflected qubits.

    Raises:
        TypeError: If ``num_qubits`` is not an integer or is a Boolean.
        ValueError: If ``num_qubits`` is not positive.
    """

    __slots__ = ("_num_qubits",)

    _num_qubits: int

    def __init__(self, num_qubits: int) -> None:
        if isinstance(num_qubits, bool) or not isinstance(num_qubits, int):
            raise TypeError("num_qubits must be an integer")
        if num_qubits <= 0:
            raise ValueError("num_qubits must be greater than zero")
        self._num_qubits = num_qubits

    @property
    def num_qubits(self) -> int:
        """Return the number of reflected qubits."""
        return self._num_qubits

    def build_circuit(self) -> Circuit:
        """Build and return a fresh exact zero-state reflection circuit."""
        circuit = Circuit(self.num_qubits)
        for qubit in range(self.num_qubits):
            circuit.x(qubit)

        if self.num_qubits == 1:
            circuit.z(0)
        else:
            controlled_z = StandardGate.Z.control(self.num_qubits - 1)
            circuit.append_mc_gate(controlled_z, list(range(self.num_qubits)))

        for qubit in range(self.num_qubits - 1, -1, -1):
            circuit.x(qubit)
        circuit.set_global_phase(math.pi)
        return circuit
