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

"""Fixed generalized spin-adapted paired-UCCSD ansatz."""

from __future__ import annotations

from typing import final

from cqlib.qis import (
    FermionToQubitMapper,
    JordanWignerMapper,
    OccupationStateMapper,
)

from ...extension.chemistry import (
    ElectronicOccupation,
    generate_puccsd_generators,
)
from ...protocols.state_preparation import StatePreparation
from ..state_preparation.hartree_fock_state_preparation import (
    HartreeFockStatePreparation,
)
from .base_ansatz import BaseAnsatz
from .ucc_ansatz import UCCAnsatz


@final
class PUCCSDAnsatz(BaseAnsatz):
    """Build generalized spin-adapted singles and paired doubles.

    Args:
        occupation: Nonempty, non-full closed-shell reference occupation.
        mapper: Fermion-to-qubit mapper; Jordan--Wigner when omitted.
        initial_state: Optional fixed reference; Hartree--Fock when omitted.

    Raises:
        TypeError: If an argument has the wrong type or the mapper cannot map
            occupation states for the default reference.
        ValueError: If the occupation or mapped generator pool is invalid.
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
        if not (
            1
            <= occupation.num_alpha
            == occupation.num_beta
            < occupation.num_spatial_orbitals
        ):
            raise ValueError(
                "occupation must be a nonempty, non-full closed shell occupation"
            )
        selected_mapper = JordanWignerMapper() if mapper is None else mapper
        if not isinstance(selected_mapper, FermionToQubitMapper):
            raise TypeError("mapper must implement FermionToQubitMapper")

        generators = generate_puccsd_generators(occupation)
        if not generators:
            raise ValueError(
                "occupation must produce at least one paired-UCCSD generator"
            )

        state = initial_state
        if state is None:
            if not isinstance(selected_mapper, OccupationStateMapper):
                raise TypeError(
                    "mapper must implement OccupationStateMapper when initial_state is None"
                )
            state = HartreeFockStatePreparation(occupation, selected_mapper)
        super().__init__(
            UCCAnsatz(
                generators,
                mapper=selected_mapper,
                initial_state=state,
            )
        )
        self.validate()


__all__ = ["PUCCSDAnsatz"]
