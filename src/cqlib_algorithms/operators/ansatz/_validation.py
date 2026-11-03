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


def validate_boolean(value: bool, name: str) -> None:
    """Validate a Boolean option."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


def snapshot_rotation_gates(
    gates: Sequence[StandardGate],
) -> tuple[StandardGate, ...]:
    """Validate RX/RY/RZ templates and snapshot their sequence as a tuple.

    The tuple isolates the container from later edits to the caller's list;
    this is not a deep copy or a claim that arbitrary gate objects are frozen.
    Bound rotations and unsupported gate templates are rejected.
    """
    if isinstance(gates, (str, bytes)) or not isinstance(gates, Sequence):
        raise TypeError("rotation_gates must be a sequence of StandardGate values")
    frozen = tuple(gates)
    for gate in frozen:
        if not isinstance(gate, StandardGate):
            raise TypeError("rotation_gates must contain only StandardGate values")
        if gate not in _ROTATION_GATES:
            raise ValueError("rotation_gates may contain only unbound RX, RY, or RZ")
    return frozen


def validate_entanglement_gate(gate: StandardGate) -> None:
    """Validate an unbound Core two-qubit entangling gate."""
    if not isinstance(gate, StandardGate):
        raise TypeError("entanglement_gate must be a StandardGate")
    if gate not in _ENTANGLEMENT_GATES:
        raise ValueError("entanglement_gate must be an unbound CX, CY, or CZ")


def resolve_entanglement(
    entanglement: EntanglementTopology | None,
) -> EntanglementTopology:
    """Validate and snapshot a topology, using linear connectivity by default."""
    if entanglement is None:
        return EntanglementTopology.linear()
    if not isinstance(entanglement, EntanglementTopology):
        raise TypeError("entanglement must be an EntanglementTopology or None")
    return copy.copy(entanglement)
