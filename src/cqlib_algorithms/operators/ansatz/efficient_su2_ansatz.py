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

"""Efficient SU(2) variational ansatz."""

from __future__ import annotations

from typing import final

from cqlib.circuit.ansatz import EntanglementTopology
from cqlib.circuit.gates import StandardGate

from .base_ansatz import BaseAnsatz
from .two_local_ansatz import TwoLocalAnsatz


@final
class EfficientSU2Ansatz(BaseAnsatz):
    """Two-local ansatz specialized to RY/RZ rotations and CX entanglers.

    Parameters are ordered by rotation layer, qubit, then RY followed by RZ.
    """

    __slots__ = ()

    def __init__(
        self,
        num_qubits: int,
        *,
        reps: int = 3,
        entanglement: EntanglementTopology | None = None,
        skip_final_rotation_layer: bool = False,
    ) -> None:
        super().__init__(
            TwoLocalAnsatz(
                num_qubits,
                reps=reps,
                rotation_gates=(StandardGate.RY, StandardGate.RZ),
                entanglement_gate=StandardGate.CX,
                entanglement=entanglement,
                skip_final_rotation_layer=skip_final_rotation_layer,
            )
        )
        self.validate()
