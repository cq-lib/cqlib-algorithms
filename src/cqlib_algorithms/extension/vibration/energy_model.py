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

"""Vibrational Hamiltonians in a finite modal basis."""

from __future__ import annotations

from dataclasses import dataclass

from cqlib.qis import Hamiltonian

from ..many_body import ModalSpace, OneHotVibrationalMapper, VibrationalOp
from ..many_body._mapping_validation import snapshot_real_hamiltonian
from .modal_integrals import ModalIntegrals


@dataclass(frozen=True, slots=True)
class VibrationalEnergy:
    """Hermitian vibrational Hamiltonian with optional coupling truncation.

    Args:
        integrals: Hermitian finite-modal integrals.
        truncation_order: Optional maximum number of coupled modes per term.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the truncation order is outside the modal space.
    """

    integrals: ModalIntegrals
    truncation_order: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.integrals, ModalIntegrals):
            raise TypeError("integrals must be ModalIntegrals")
        if self.truncation_order is not None:
            if type(self.truncation_order) is not int:
                raise TypeError("truncation_order must be an int or None")
            if (
                self.truncation_order < 1
                or self.truncation_order > self.integrals.modal_space.num_modes
            ):
                raise ValueError(
                    "truncation_order must be between 1 and the number of modes"
                )

    @property
    def modal_space(self) -> ModalSpace:
        """Return the finite modal space of the Hamiltonian."""
        return self.integrals.modal_space

    @property
    def effective_order(self) -> int:
        """Return the largest coupling order retained by this model."""
        if self.truncation_order is None:
            return self.integrals.max_order
        return min(self.truncation_order, self.integrals.max_order)

    def to_vibrational_op(self) -> VibrationalOp:
        """Build the constant plus all retained modal-coupling terms."""
        if self.truncation_order is None:
            return self.integrals.to_vibrational_op()
        return self.integrals.truncated(self.truncation_order).to_vibrational_op()

    def to_qubit_hamiltonian(
        self,
        mapper: OneHotVibrationalMapper | None = None,
    ) -> Hamiltonian:
        """Map the vibrational Hamiltonian to the one-hot qubit register."""
        selected_mapper = OneHotVibrationalMapper() if mapper is None else mapper
        if not isinstance(selected_mapper, OneHotVibrationalMapper):
            raise TypeError("mapper must be a OneHotVibrationalMapper or None")
        mapped = selected_mapper.map_operator(self.to_vibrational_op())
        return snapshot_real_hamiltonian(
            mapped,
            expected_width=self.modal_space.num_qubits,
            name="vibrational Hamiltonian",
        )


__all__ = ["VibrationalEnergy"]
