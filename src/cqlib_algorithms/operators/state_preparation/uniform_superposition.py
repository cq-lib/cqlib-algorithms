"""Uniform-superposition state preparation."""

from __future__ import annotations

from typing import final

from cqlib.circuit import Circuit

from ...protocols.state_preparation import StatePreparation
from ._state_synthesis import validate_positive_qubit_count, validate_prefix


@final
class UniformSuperposition(StatePreparation):
    r"""Prepare the uniform state :math:`2^{-n/2}\sum_x |x\rangle`."""

    __slots__ = ("_num_qubits",)

    def __init__(self, num_qubits: int) -> None:
        self._num_qubits = validate_positive_qubit_count(num_qubits)

    def validate(self) -> None:
        """Validate the fixed component configuration."""
        validate_positive_qubit_count(self._num_qubits)

    def build_circuit(self, prefix: str) -> Circuit:
        """Return a new circuit containing one Hadamard per qubit."""
        validate_prefix(prefix)
        circuit = Circuit(self._num_qubits)
        for qubit in range(self._num_qubits):
            circuit.h(qubit)
        return circuit

    def num_parameters(self) -> int:
        """Return zero because the prepared state is fixed."""
        return 0

    def num_qubits(self) -> int:
        """Return the configured register width."""
        return self._num_qubits
