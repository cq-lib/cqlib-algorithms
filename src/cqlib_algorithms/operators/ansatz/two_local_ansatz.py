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

"""General two-local variational ansatz."""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from cqlib.circuit.ansatz import EntanglementTopology, TwoLocal
from cqlib.circuit.gates import StandardGate

from ._validation import (
    resolve_entanglement,
    snapshot_rotation_gates,
    validate_boolean,
    validate_entanglement_gate,
    validate_positive_qubit_count,
    validate_repetitions,
)
from .base_ansatz import BaseAnsatz


@final
class TwoLocalAnsatz(BaseAnsatz):
    """Alternate local rotations with topology-defined entanglement layers.

    Parameter slots are ordered by rotation layer, qubit, and then position in
    ``rotation_gates``. Each of the first ``reps`` rotation layers is followed
    by an entanglement layer. A final rotation layer is included unless
    ``skip_final_rotation_layer`` is true.
    """

    __slots__ = ()

    def __init__(
        self,
        num_qubits: int,
        *,
        reps: int = 3,
        rotation_gates: Sequence[StandardGate] = (StandardGate.RY,),
        entanglement_gate: StandardGate = StandardGate.CX,
        entanglement: EntanglementTopology | None = None,
        skip_final_rotation_layer: bool = False,
    ) -> None:
        validate_positive_qubit_count(num_qubits)
        validate_repetitions(reps)
        rotations = snapshot_rotation_gates(rotation_gates)
        validate_entanglement_gate(entanglement_gate)
        topology = resolve_entanglement(entanglement)
        validate_boolean(skip_final_rotation_layer, "skip_final_rotation_layer")

        inner = (
            TwoLocal(num_qubits)
            .reps(reps)
            .rotation_gates(list(rotations))
            .entanglement_gate(entanglement_gate)
            .entanglement(topology)
            .skip_final_rotation_layer(skip_final_rotation_layer)
        )
        super().__init__(inner)
        self.validate()
