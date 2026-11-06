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

"""Hartree--Fock computational-basis state preparation."""

from __future__ import annotations

from typing import final

from cqlib.circuit import Circuit
from cqlib.qis import JordanWignerMapper, OccupationStateMapper

from ...extension.electronic_structure import ElectronicOccupation
from ...protocols.state_preparation import StatePreparation
from .basis_state_initialization import BasisStateInitialization


@final
class HartreeFockStatePreparation(StatePreparation):
    """Prepare a mapped index-ordered reference as a computational basis state.

    Args:
        occupation: Index-ordered alpha/beta reference occupation.
        mapper: Occupation-state mapper; Jordan--Wigner when omitted.

    Raises:
        TypeError: If the occupation, mapper, or mapped bits have the wrong type.
        ValueError: If the mapper returns an empty occupation.
    """

    __slots__ = ("_basis_preparation", "_mapped_occupation")

    def __init__(
        self,
        occupation: ElectronicOccupation,
        mapper: OccupationStateMapper | None = None,
    ) -> None:
        if not isinstance(occupation, ElectronicOccupation):
            raise TypeError("occupation must be an ElectronicOccupation")
        selected_mapper = JordanWignerMapper() if mapper is None else mapper
        if not isinstance(selected_mapper, OccupationStateMapper):
            raise TypeError("mapper must implement OccupationStateMapper")

        mapped = selected_mapper.map_occupation(
            tuple(int(value) for value in occupation.spin_orbital_occupation)
        )
        if type(mapped) is not list:
            raise TypeError("mapper must return a list of bool values")
        if not mapped:
            raise ValueError("mapper must return a non-empty occupation")
        if any(type(value) is not bool for value in mapped):
            raise TypeError("mapper must return a list of bool values")

        self._mapped_occupation = tuple(mapped)
        basis_index = sum(
            int(occupied) << qubit
            for qubit, occupied in enumerate(self._mapped_occupation)
        )
        self._basis_preparation = BasisStateInitialization(
            len(self._mapped_occupation), basis_index
        )

    def validate(self) -> None:
        """Validate the frozen mapped occupation and delegated preparation."""
        if not self._mapped_occupation:
            raise ValueError("mapped occupation must not be empty")
        if any(type(value) is not bool for value in self._mapped_occupation):
            raise TypeError("mapped occupation must contain only bool values")
        self._basis_preparation.validate()

    def build_circuit(self, prefix: str) -> Circuit:
        """Return an independent mapped Hartree--Fock preparation circuit."""
        return self._basis_preparation.build_circuit(prefix)

    def num_parameters(self) -> int:
        """Return zero because the mapped occupation is fixed."""
        return 0

    def num_qubits(self) -> int:
        """Return the mapped register width."""
        return len(self._mapped_occupation)


__all__ = ["HartreeFockStatePreparation"]
