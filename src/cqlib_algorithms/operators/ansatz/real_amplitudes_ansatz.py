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

"""Real-amplitudes variational ansatz."""

from __future__ import annotations

from typing import final

from cqlib.circuit.ansatz import EntanglementTopology
from cqlib.circuit.gates import StandardGate

from .base_ansatz import _DelegatingAnsatz
from .two_local_ansatz import TwoLocalAnsatz


@final
class RealAmplitudesAnsatz(_DelegatingAnsatz):
    """Two-local ansatz specialized to RY rotations and CX entanglers.

    All generated gate matrices are real. Parameters are ordered by rotation
    layer and qubit.

    Args:
        num_qubits: Positive circuit width.
        reps: Nonnegative number of entangling repetitions.
        entanglement: Core topology, or ``None`` for the default.
        skip_final_rotation_layer: Whether to omit the final RY layer.

    Raises:
        TypeError: If an option has the wrong type.
        ValueError: If width, repetitions, or topology are invalid.
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
                rotation_gates=(StandardGate.RY,),
                entanglement_gate=StandardGate.CX,
                entanglement=entanglement,
                skip_final_rotation_layer=skip_final_rotation_layer,
            )
        )
        self.validate()
