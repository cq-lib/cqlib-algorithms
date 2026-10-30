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

"""User-supplied circuit state preparation."""

from __future__ import annotations

import copy
from typing import final

from cqlib.circuit import Circuit, CircuitError

from ...protocols.state_preparation import StatePreparation
from ._state_synthesis import validate_prefix


@final
class UserCircuitPreparation(StatePreparation):
    """Snapshot an invertible user circuit as a state preparation.

    Source qubits are mapped by insertion position onto contiguous qubits
    ``0..n-1``. The snapshot preserves executable operations, live symbolic
    parameters, and global phase. Classical or otherwise non-invertible
    operations are rejected because a state-preparation component must be
    composable with inverse-based algorithms.

    Args:
        circuit: Positive-width, statically invertible circuit to snapshot.

    Raises:
        TypeError: If ``circuit`` is not a Circuit.
        ValueError: If it is empty-width, invalid, or non-invertible.
    """

    __slots__ = ("_circuit", "_num_parameters")

    def __init__(self, circuit: Circuit) -> None:
        if not isinstance(circuit, Circuit):
            raise TypeError("circuit must be a Circuit")
        if circuit.num_qubits < 1:
            raise ValueError("circuit must contain at least one qubit")

        source = copy.copy(circuit)
        source.validate()
        normalized = Circuit(source.num_qubits)
        normalized.compose(source, list(range(source.num_qubits)))
        try:
            normalized.inverse()
        except CircuitError as error:
            raise ValueError(
                "circuit must contain only statically invertible operations"
            ) from error
        self._circuit = normalized
        self._num_parameters = len(normalized.used_symbols)

    def validate(self) -> None:
        """Validate the snapshot and its static invertibility."""
        self._circuit.validate()
        self._circuit.inverse()

    def build_circuit(self, prefix: str) -> Circuit:
        """Return an independent snapshot copy; ``prefix`` does not rename it."""
        validate_prefix(prefix)
        return copy.copy(self._circuit)

    def num_parameters(self) -> int:
        """Return the number of symbols used by executable circuit IR."""
        return self._num_parameters

    def num_qubits(self) -> int:
        """Return the normalized snapshot register width."""
        return self._circuit.num_qubits
