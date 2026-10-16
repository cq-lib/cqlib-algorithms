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

"""Grover diffusion-operator circuit construction.

References
----------
[1] L. K. Grover, "A fast quantum mechanical algorithm for database search,"
    Proceedings of the 28th Annual ACM Symposium on Theory of Computing,
    212-219, 1996. DOI: 10.1145/237814.237866.
"""

from __future__ import annotations

from cqlib import Circuit

from .zero_state_reflection import ZeroStateReflection


class GroverDiffusion:
    """Construct ``D = 2|s><s| - I`` for the uniform superposition ``|s>``.

    Equivalently, ``D = H**n R0 H**n`` where ``R0`` is the exact
    :class:`ZeroStateReflection`. This component contains no oracle.

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
        """Return the number of diffusion qubits."""
        return self._num_qubits

    def build_circuit(self) -> Circuit:
        """Build and return a fresh Grover diffusion circuit."""
        circuit = Circuit(self.num_qubits)
        for qubit in range(self.num_qubits):
            circuit.h(qubit)
        circuit.compose(ZeroStateReflection(self.num_qubits).build_circuit())
        for qubit in range(self.num_qubits):
            circuit.h(qubit)
        return circuit
