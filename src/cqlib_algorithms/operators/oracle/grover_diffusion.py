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

"""Grover diffusion-operator circuit construction."""

from __future__ import annotations

from dataclasses import dataclass

from cqlib import Circuit

from .zero_state_reflection import ZeroStateReflection


@dataclass(frozen=True, slots=True)
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

    num_qubits: int

    def __post_init__(self) -> None:
        if isinstance(self.num_qubits, bool) or not isinstance(self.num_qubits, int):
            raise TypeError("num_qubits must be an integer")
        if self.num_qubits <= 0:
            raise ValueError("num_qubits must be greater than zero")

    def build_circuit(self) -> Circuit:
        """Build and return a fresh Grover diffusion circuit."""
        circuit = Circuit(self.num_qubits)
        for qubit in range(self.num_qubits):
            circuit.h(qubit)
        circuit.compose(ZeroStateReflection(self.num_qubits).build_circuit())
        for qubit in range(self.num_qubits):
            circuit.h(qubit)
        return circuit
