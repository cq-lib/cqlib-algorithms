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

"""Tree-tensor-network-inspired variational circuit.

References
----------
[1] E. Grant et al., "Hierarchical quantum classifiers," npj Quantum
    Information, 4, 65, 2018.
    DOI: 10.1038/s41534-018-0116-9.
"""

from __future__ import annotations

from typing import final

from cqlib.circuit import Circuit, Parameter

from ._tensor_network import append_real_two_qubit_block, ttn_levels
from ._validation import validate_positive_qubit_count
from .base_ansatz import BaseAnsatz


@final
class TTNAnsatz(BaseAnsatz):
    """Build a binary reduction tree with a trainable root rotation.

    At every level, active qubits are paired left-to-right, the right qubit of
    each pair is retained, and an unmatched final qubit is carried forward.
    All physical qubits remain in the circuit; retention only defines later
    connectivity.
    """

    __slots__ = ("_levels", "_num_qubits", "_root")

    def __init__(self, num_qubits: int) -> None:
        self._validate_width(num_qubits)
        self._num_qubits = num_qubits
        self._levels, self._root = ttn_levels(num_qubits)
        self.validate()

    @staticmethod
    def _validate_width(num_qubits: int) -> None:
        validate_positive_qubit_count(num_qubits)
        if num_qubits < 2:
            raise ValueError("num_qubits must be at least two")

    def _validate_configuration(self) -> None:
        """Validate the width and frozen reduction topology."""
        self._validate_width(self._num_qubits)
        expected_levels, expected_root = ttn_levels(self._num_qubits)
        if self._levels != expected_levels or self._root != expected_root:
            raise ValueError("TTN topology is inconsistent with num_qubits")

    def _build_circuit(self, prefix: str) -> Circuit:
        """Build levels left-to-right and append the final root rotation."""
        circuit = Circuit(self._num_qubits)
        parameter_index = 0
        for level in self._levels:
            for edge in level:
                parameter_index = append_real_two_qubit_block(
                    circuit, edge, prefix, parameter_index
                )
        circuit.ry(self._root, Parameter(f"{prefix}_{parameter_index}"))
        return circuit

    def _parameter_count(self) -> int:
        """Return two parameters per reduction block plus one root rotation."""
        return 2 * (self._num_qubits - 1) + 1

    def _qubit_count(self) -> int:
        """Return the configured register width."""
        return self._num_qubits
