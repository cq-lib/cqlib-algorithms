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

"""Shared validation helpers for ansatz components."""

from __future__ import annotations

import copy
from collections.abc import Sequence

from cqlib.circuit.ansatz import EntanglementTopology
from cqlib.circuit.gates import StandardGate

_ROTATION_GATES = (StandardGate.RX, StandardGate.RY, StandardGate.RZ)
_ENTANGLEMENT_GATES = (StandardGate.CX, StandardGate.CY, StandardGate.CZ)


def validate_positive_qubit_count(num_qubits: int) -> int:
    """Return a validated positive qubit count."""
    if isinstance(num_qubits, bool) or not isinstance(num_qubits, int):
        raise TypeError("num_qubits must be an integer")
    if num_qubits <= 0:
        raise ValueError("num_qubits must be greater than zero")
    return num_qubits


def validate_repetitions(reps: int) -> int:
    """Return a validated non-negative repetition count."""
    if isinstance(reps, bool) or not isinstance(reps, int):
        raise TypeError("reps must be an integer")
    if reps < 0:
        raise ValueError("reps must be non-negative")
    return reps


def validate_prefix(prefix: str) -> None:
    """Validate a circuit-parameter prefix."""
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a string")


def validate_boolean(value: bool, name: str) -> bool:
    """Return a validated Boolean option."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def validate_rotation_gates(
    gates: Sequence[StandardGate],
) -> tuple[StandardGate, ...]:
    """Validate and freeze unbound one-parameter rotation gates."""
    if isinstance(gates, (str, bytes)) or not isinstance(gates, Sequence):
        raise TypeError("rotation_gates must be a sequence of StandardGate values")
    frozen = tuple(gates)
    for gate in frozen:
        if not isinstance(gate, StandardGate):
            raise TypeError("rotation_gates must contain only StandardGate values")
        if gate not in _ROTATION_GATES:
            raise ValueError("rotation_gates may contain only unbound RX, RY, or RZ")
    return frozen


def validate_entanglement_gate(gate: StandardGate) -> StandardGate:
    """Validate an unbound Core two-qubit entangling gate."""
    if not isinstance(gate, StandardGate):
        raise TypeError("entanglement_gate must be a StandardGate")
    if gate not in _ENTANGLEMENT_GATES:
        raise ValueError("entanglement_gate must be an unbound CX, CY, or CZ")
    return gate


def resolve_entanglement(
    entanglement: EntanglementTopology | None,
) -> EntanglementTopology:
    """Validate and snapshot a topology, using linear connectivity by default."""
    if entanglement is None:
        return EntanglementTopology.linear()
    if not isinstance(entanglement, EntanglementTopology):
        raise TypeError("entanglement must be an EntanglementTopology or None")
    return copy.copy(entanglement)
