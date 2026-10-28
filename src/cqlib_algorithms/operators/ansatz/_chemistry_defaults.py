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

"""Shared mapper and reference-state choices for chemistry ansatze."""

from cqlib.qis import (
    FermionToQubitMapper,
    JordanWignerMapper,
    OccupationStateMapper,
)

from ...extension.chemistry import ElectronicOccupation
from ...protocols.state_preparation import StatePreparation
from ..state_preparation.hartree_fock_state_preparation import (
    HartreeFockStatePreparation,
)


def validate_electronic_occupation(occupation: ElectronicOccupation) -> None:
    """Validate an electronic reference occupation."""
    if not isinstance(occupation, ElectronicOccupation):
        raise TypeError("occupation must be an ElectronicOccupation")


def validate_nonempty_nonfull_closed_shell(
    occupation: ElectronicOccupation,
) -> None:
    """Validate a closed-shell reference with occupied and virtual orbitals."""
    validate_electronic_occupation(occupation)
    if not (
        1
        <= occupation.num_alpha
        == occupation.num_beta
        < occupation.num_spatial_orbitals
    ):
        raise ValueError(
            "occupation must be a nonempty, non-full closed shell occupation"
        )


def select_fermion_mapper(
    mapper: FermionToQubitMapper | None,
) -> FermionToQubitMapper:
    """Return a validated mapper, defaulting to Jordan--Wigner."""
    selected = JordanWignerMapper() if mapper is None else mapper
    if not isinstance(selected, FermionToQubitMapper):
        raise TypeError("mapper must implement FermionToQubitMapper")
    return selected


def select_hartree_fock_reference(
    occupation: ElectronicOccupation,
    mapper: FermionToQubitMapper,
    initial_state: StatePreparation | None,
) -> StatePreparation:
    """Return the supplied state or construct the default Hartree--Fock state."""
    if initial_state is not None:
        return initial_state
    if not isinstance(mapper, OccupationStateMapper):
        raise TypeError(
            "mapper must implement OccupationStateMapper when initial_state is None"
        )
    return HartreeFockStatePreparation(occupation, mapper)
