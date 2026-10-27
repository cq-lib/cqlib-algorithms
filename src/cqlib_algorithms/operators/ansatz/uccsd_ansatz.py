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

"""Alpha/beta-number-conserving ordered factorized UCCSD ansatz."""

from __future__ import annotations

from typing import final

from cqlib.qis import (
    FermionToQubitMapper,
    JordanWignerMapper,
    OccupationStateMapper,
)

from ...extension.chemistry import (
    ElectronicOccupation,
    generate_uccsd_excitations,
)
from ...protocols.state_preparation import StatePreparation
from ..state_preparation.hartree_fock_state_preparation import (
    HartreeFockStatePreparation,
)
from .base_ansatz import BaseAnsatz
from .ucc_ansatz import UCCAnsatz


@final
class UCCSDAnsatz(BaseAnsatz):
    """Build alpha/beta-number-conserving singles and doubles as UCC factors.

    Args:
        occupation: Reference electronic occupation defining the excitation pool.
        mapper: Fermion-to-qubit mapper; Jordan--Wigner when omitted.
        initial_state: Optional fixed reference; Hartree--Fock when omitted.

    Raises:
        TypeError: If an argument has the wrong type or the mapper cannot map
            occupation states for the default reference.
        ValueError: If no UCCSD excitation exists or mapped evolution is invalid.
    """

    __slots__ = ()

    def __init__(
        self,
        occupation: ElectronicOccupation,
        *,
        mapper: FermionToQubitMapper | None = None,
        initial_state: StatePreparation | None = None,
    ) -> None:
        if not isinstance(occupation, ElectronicOccupation):
            raise TypeError("occupation must be an ElectronicOccupation")
        selected_mapper = JordanWignerMapper() if mapper is None else mapper
        if not isinstance(selected_mapper, FermionToQubitMapper):
            raise TypeError("mapper must implement FermionToQubitMapper")

        excitations = generate_uccsd_excitations(occupation)
        if not excitations:
            raise ValueError("occupation must produce at least one UCCSD excitation")

        state = initial_state
        if state is None:
            if not isinstance(selected_mapper, OccupationStateMapper):
                raise TypeError(
                    "mapper must implement OccupationStateMapper when initial_state is None"
                )
            state = HartreeFockStatePreparation(occupation, selected_mapper)
        inner = UCCAnsatz(
            excitations,
            mapper=selected_mapper,
            initial_state=state,
        )
        super().__init__(inner)
        self.validate()


__all__ = ["UCCSDAnsatz"]
