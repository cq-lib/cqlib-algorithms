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

"""Binary MERA-inspired variational circuit.

References
----------
[1] G. Vidal, "Class of quantum many-body states that can be efficiently
    simulated," Physical Review Letters, 101, 110501, 2008.
    DOI: 10.1103/PhysRevLett.101.110501.
[2] E. Grant et al., "Hierarchical quantum classifiers," npj Quantum
    Information, 4, 65, 2018.
    DOI: 10.1038/s41534-018-0116-9.
"""

from __future__ import annotations

from typing import final

from cqlib.circuit import Circuit, Parameter

from ._tensor_network import append_real_two_qubit_block, mera_scales
from ._validation import validate_positive_qubit_count
from .base_ansatz import BaseAnsatz


@final
class MERAAnsatz(BaseAnsatz):
    """Build a complete binary MERA-inspired unitary circuit.

    Each scale applies boundary disentangler blocks before pairwise
    isometry-position blocks, retaining each isometry pair's right qubit for
    the next scale. The shared blocks are unitary circuit elements rather than
    an API for arbitrary non-unitary MERA tensors.
    """

    __slots__ = ("_num_qubits", "_root", "_scales")

    def __init__(self, num_qubits: int) -> None:
        self._validate_width(num_qubits)
        self._num_qubits = num_qubits
        self._scales, self._root = mera_scales(num_qubits)
        self.validate()

    @staticmethod
    def _validate_width(num_qubits: int) -> None:
        validate_positive_qubit_count(num_qubits)
        if num_qubits < 2:
            raise ValueError("num_qubits must be at least two")
        if num_qubits & (num_qubits - 1):
            raise ValueError("num_qubits must be a power of two")

    def _validate_configuration(self) -> None:
        """Validate the width and frozen multiscale topology."""
        self._validate_width(self._num_qubits)
        expected_scales, expected_root = mera_scales(self._num_qubits)
        if self._scales != expected_scales or self._root != expected_root:
            raise ValueError("MERA topology is inconsistent with num_qubits")

    def _build_circuit(self, prefix: str) -> Circuit:
        """Build each scale in disentangler-before-isometry order."""
        circuit = Circuit(self._num_qubits)
        parameter_index = 0
        for disentanglers, isometries in self._scales:
            for edge in disentanglers:
                parameter_index = append_real_two_qubit_block(
                    circuit, edge, prefix, parameter_index
                )
            for edge in isometries:
                parameter_index = append_real_two_qubit_block(
                    circuit, edge, prefix, parameter_index
                )
        circuit.ry(self._root, Parameter(f"{prefix}_{parameter_index}"))
        return circuit

    def _parameter_count(self) -> int:
        """Return two parameters per block plus one root rotation."""
        levels = self._num_qubits.bit_length() - 1
        return 4 * self._num_qubits - 3 - 2 * levels

    def _qubit_count(self) -> int:
        """Return the configured register width."""
        return self._num_qubits
