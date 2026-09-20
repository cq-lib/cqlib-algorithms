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

"""Hardware-efficient variational ansatz."""

from __future__ import annotations

from typing import final

from cqlib.circuit import Circuit
from cqlib.circuit.ansatz import EntanglementTopology
from cqlib.circuit.gates import StandardGate

from ...protocols.ansatz import Ansatz
from .two_local_ansatz import TwoLocalAnsatz


@final
class HardwareEfficientAnsatz(Ansatz):
    """A digital hardware-efficient ansatz with RZ-RX-RZ Euler blocks.

    This class freezes one useful hardware-efficient architecture; it is not a
    universal definition of hardware-efficient ansatze. Parameters are ordered
    by rotation layer, qubit, then RZ, RX, RZ position.
    """

    __slots__ = ("_inner",)

    def __init__(
        self,
        num_qubits: int,
        *,
        reps: int = 3,
        entanglement: EntanglementTopology | None = None,
        entanglement_gate: StandardGate = StandardGate.CX,
        skip_final_rotation_layer: bool = False,
    ) -> None:
        self._inner = TwoLocalAnsatz(
            num_qubits,
            reps=reps,
            rotation_gates=(StandardGate.RZ, StandardGate.RX, StandardGate.RZ),
            entanglement_gate=entanglement_gate,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation_layer,
        )
        self.validate()

    def validate(self) -> None:
        """Validate the delegated two-local configuration."""
        self._inner.validate()

    def build_circuit(self, prefix: str) -> Circuit:
        """Return a new RZ-RX-RZ hardware-efficient circuit."""
        return self._inner.build_circuit(prefix)

    def num_parameters(self) -> int:
        """Return the number of Euler-angle parameter slots."""
        return self._inner.num_parameters()

    def num_qubits(self) -> int:
        """Return the configured register width."""
        return self._inner.num_qubits()
