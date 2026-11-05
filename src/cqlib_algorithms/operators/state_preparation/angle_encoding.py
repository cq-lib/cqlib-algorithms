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

"""Single-qubit angle encoding."""

from __future__ import annotations

import copy
from typing import final

from cqlib.circuit import Circuit
from cqlib.circuit.ansatz import AngleEncoding as CoreAngleEncoding
from cqlib.circuit.gates import StandardGate

from ..._utils.numeric import validate_positive_qubit_count
from ..._utils.parameters import validate_parameter_prefix
from ...protocols.state_preparation import StatePreparation

_SUPPORTED_ROTATION_GATES = (
    StandardGate.RX,
    StandardGate.RY,
    StandardGate.RZ,
    StandardGate.Phase,
)


@final
class AngleEncoding(StatePreparation):
    """Encode one symbolic feature per qubit with a Core rotation gate.

    ``rotation_gate`` must be the unbound ``RX``, ``RY``, ``RZ``, or
    ``Phase`` gate definition. Circuit construction and symbol naming are
    delegated to :class:`cqlib.circuit.ansatz.AngleEncoding`.

    Args:
        num_qubits: Positive number of encoded features and qubits.
        rotation_gate: Unbound Core RX, RY, RZ, or Phase gate definition.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the width or gate choice is invalid.
    """

    __slots__ = ("_inner", "_num_qubits", "_rotation_gate")

    def __init__(self, num_qubits: int, rotation_gate: StandardGate) -> None:
        validate_positive_qubit_count(num_qubits)
        self._num_qubits = num_qubits
        if not isinstance(rotation_gate, StandardGate):
            raise TypeError("rotation_gate must be a StandardGate")
        if rotation_gate not in _SUPPORTED_ROTATION_GATES:
            raise ValueError(
                "rotation_gate must be an unbound RX, RY, RZ, or Phase gate"
            )
        self._rotation_gate = copy.copy(rotation_gate)
        self._inner = CoreAngleEncoding(self._num_qubits, self._rotation_gate)
        self.validate()

    def validate(self) -> None:
        """Validate the delegated Core angle encoding."""
        self._inner.validate()

    def build_circuit(self, prefix: str) -> Circuit:
        """Return a new symbolic angle-encoding circuit."""
        validate_parameter_prefix(prefix)
        return self._inner.build_circuit(prefix)

    def num_parameters(self) -> int:
        """Return one logical feature slot per qubit."""
        return self._inner.num_parameters()

    def num_qubits(self) -> int:
        """Return the configured register width."""
        return self._inner.num_qubits()
