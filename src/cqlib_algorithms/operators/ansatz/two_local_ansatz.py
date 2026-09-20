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

from cqlib.circuit import Circuit
from cqlib.circuit.ansatz import EntanglementTopology, TwoLocal
from cqlib.circuit.gates import StandardGate

from ...protocols.ansatz import Ansatz
from ._validation import (
    resolve_entanglement,
    validate_boolean,
    validate_entanglement_gate,
    validate_positive_qubit_count,
    validate_prefix,
    validate_repetitions,
    validate_rotation_gates,
)


@final
class TwoLocalAnsatz(Ansatz):
    """Alternate local rotations with topology-defined entanglement layers.

    Parameter slots are ordered by rotation layer, qubit, and then position in
    ``rotation_gates``. Each of the first ``reps`` rotation layers is followed
    by an entanglement layer. A final rotation layer is included unless
    ``skip_final_rotation_layer`` is true.
    """

    __slots__ = ("_inner",)

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
        width = validate_positive_qubit_count(num_qubits)
        repetitions = validate_repetitions(reps)
        rotations = validate_rotation_gates(rotation_gates)
        entangler = validate_entanglement_gate(entanglement_gate)
        topology = resolve_entanglement(entanglement)
        skip_final = validate_boolean(
            skip_final_rotation_layer, "skip_final_rotation_layer"
        )

        self._inner = (
            TwoLocal(width)
            .reps(repetitions)
            .rotation_gates(list(rotations))
            .entanglement_gate(entangler)
            .entanglement(topology)
            .skip_final_rotation_layer(skip_final)
        )
        self.validate()

    def validate(self) -> None:
        """Validate the delegated Core two-local configuration."""
        self._inner.validate()

    def build_circuit(self, prefix: str) -> Circuit:
        """Return a new two-local circuit with ``prefix``-named parameters."""
        validate_prefix(prefix)
        return self._inner.build_circuit(prefix)

    def num_parameters(self) -> int:
        """Return the number of logical rotation-parameter slots."""
        return self._inner.num_parameters()

    def num_qubits(self) -> int:
        """Return the configured register width."""
        return self._inner.num_qubits()
