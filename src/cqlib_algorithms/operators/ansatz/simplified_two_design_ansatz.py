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

"""Deterministic simplified two-design variational ansatz.

References
----------
[1] M. Cerezo et al., "Cost function dependent barren plateaus in shallow
    parametrized quantum circuits," Nature Communications, 12, 1791, 2021.
    DOI: 10.1038/s41467-021-21728-w.
"""

from __future__ import annotations

from typing import final

from cqlib.circuit import Circuit, Parameter

from ._validation import validate_positive_qubit_count, validate_repetitions
from .base_ansatz import BaseAnsatz


@final
class SimplifiedTwoDesignAnsatz(BaseAnsatz):
    """Build trainable RY layers around even/odd nearest-neighbor CZ blocks.

    The circuit starts with one RY on every qubit. Each repetition then
    visits even edges followed by odd edges; every edge contributes CZ and
    one new RY parameter on each endpoint. The deterministic architecture is
    motivated by simplified two-design constructions; finite instances are
    not claimed to form exact unitary two-designs.

    Args:
        num_qubits: Positive circuit width.
        reps: Nonnegative brickwork repetition count.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If width or repetitions are invalid.
    """

    __slots__ = ("_num_qubits", "_reps")

    def __init__(self, num_qubits: int, *, reps: int = 1) -> None:
        validate_positive_qubit_count(num_qubits)
        validate_repetitions(reps)
        self._num_qubits = num_qubits
        self._reps = reps
        self.validate()

    def _validate_configuration(self) -> None:
        """Validate the immutable architecture dimensions."""
        validate_positive_qubit_count(self._num_qubits)
        validate_repetitions(self._reps)

    def _build_circuit(self, prefix: str) -> Circuit:
        """Build the circuit with parameters in gate-appearance order."""
        circuit = Circuit(self._num_qubits)
        parameter_index = 0
        for qubit in range(self._num_qubits):
            circuit.ry(qubit, Parameter(f"{prefix}_{parameter_index}"))
            parameter_index += 1

        for _ in range(self._reps):
            for start in (0, 1):
                for left in range(start, self._num_qubits - 1, 2):
                    right = left + 1
                    circuit.cz(left, right)
                    circuit.ry(left, Parameter(f"{prefix}_{parameter_index}"))
                    parameter_index += 1
                    circuit.ry(right, Parameter(f"{prefix}_{parameter_index}"))
                    parameter_index += 1
        return circuit

    def _parameter_count(self) -> int:
        """Return initial and edge-local rotation parameter slots."""
        return self._num_qubits + 2 * self._reps * (self._num_qubits - 1)

    def _qubit_count(self) -> int:
        """Return the configured register width."""
        return self._num_qubits
