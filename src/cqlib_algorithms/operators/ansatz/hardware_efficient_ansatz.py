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

"""Hardware-efficient variational ansatz.

References
----------
[1] A. Kandala et al., "Hardware-efficient variational quantum eigensolver
    for small molecules and quantum magnets," Nature, 549, 242-246, 2017.
    DOI: 10.1038/nature23879.
"""

from __future__ import annotations

from typing import final

from cqlib.circuit.ansatz import EntanglementTopology
from cqlib.circuit.gates import StandardGate

from .base_ansatz import BaseAnsatz
from .two_local_ansatz import TwoLocalAnsatz


@final
class HardwareEfficientAnsatz(BaseAnsatz):
    """A digital hardware-efficient ansatz with RZ-RX-RZ Euler blocks.

    This class freezes one useful hardware-efficient architecture; it is not a
    universal definition of hardware-efficient ansatze. Parameters are ordered
    by rotation layer, qubit, then RZ, RX, RZ position.
    """

    __slots__ = ()

    def __init__(
        self,
        num_qubits: int,
        *,
        reps: int = 3,
        entanglement: EntanglementTopology | None = None,
        entanglement_gate: StandardGate = StandardGate.CX,
        skip_final_rotation_layer: bool = False,
    ) -> None:
        super().__init__(
            TwoLocalAnsatz(
                num_qubits,
                reps=reps,
                rotation_gates=(
                    StandardGate.RZ,
                    StandardGate.RX,
                    StandardGate.RZ,
                ),
                entanglement_gate=entanglement_gate,
                entanglement=entanglement,
                skip_final_rotation_layer=skip_final_rotation_layer,
            )
        )
        self.validate()
