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

"""Matrix-product-state-inspired variational circuit."""

from __future__ import annotations

from typing import final

from cqlib.circuit import Circuit

from ._tensor_network import append_real_two_qubit_block, mps_edges
from ._validation import validate_positive_qubit_count
from .base_ansatz import BaseAnsatz


@final
class MPSAnsatz(BaseAnsatz):
    """Build a left-to-right nearest-neighbor two-qubit-block sweep.

    Each edge ``(q, q + 1)`` uses two independent ``RY`` parameters followed
    by ``CX(q, q + 1)``. The class exposes an MPS-inspired circuit topology;
    it does not import arbitrary classical MPS tensors or bond dimensions.

    Args:
        num_qubits: Circuit width of at least two.

    Raises:
        TypeError: If ``num_qubits`` is not an integer.
        ValueError: If ``num_qubits`` is less than two.
    """

    __slots__ = ("_edges", "_num_qubits")

    def __init__(self, num_qubits: int) -> None:
        self._validate_width(num_qubits)
        self._num_qubits = num_qubits
        self._edges = mps_edges(num_qubits)
        self.validate()

    @staticmethod
    def _validate_width(num_qubits: int) -> None:
        validate_positive_qubit_count(num_qubits)
        if num_qubits < 2:
            raise ValueError("num_qubits must be at least two")

    def _validate_configuration(self) -> None:
        """Validate the width and frozen chain topology."""
        self._validate_width(self._num_qubits)
        if self._edges != mps_edges(self._num_qubits):
            raise ValueError("MPS edge topology is inconsistent with num_qubits")

    def _build_circuit(self, prefix: str) -> Circuit:
        """Build the chain in edge and gate-appearance order."""
        circuit = Circuit(self._num_qubits)
        parameter_index = 0
        for edge in self._edges:
            parameter_index = append_real_two_qubit_block(
                circuit, edge, prefix, parameter_index
            )
        return circuit

    def _parameter_count(self) -> int:
        """Return two parameters for every chain edge."""
        return 2 * len(self._edges)

    def _qubit_count(self) -> int:
        """Return the configured register width."""
        return self._num_qubits
