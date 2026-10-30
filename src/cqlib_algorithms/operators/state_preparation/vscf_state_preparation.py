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

"""Vibrational self-consistent-field reference-state preparation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from cqlib.circuit import Circuit

from ...extension.vibration import ModalSpace
from ...protocols.state_preparation import StatePreparation
from .basis_state_initialization import BasisStateInitialization


def _copy_reference_modals(
    modal_space: ModalSpace,
    reference_modals: Sequence[int] | None,
) -> tuple[int, ...]:
    if reference_modals is None:
        return (0,) * modal_space.num_modes
    if isinstance(reference_modals, (str, bytes)) or not isinstance(
        reference_modals, Sequence
    ):
        raise TypeError("reference_modals must be a sequence of integers or None")
    copied = tuple(reference_modals)
    modal_space.modal_basis_index(copied)
    return copied


@final
class VSCFStatePreparation(StatePreparation):
    """Prepare one modal per mode in the direct one-hot encoding.

    The default reference occupies modal zero of every mode. An explicit
    reference may select any one valid modal per mode. The component contains
    no variational parameters.

    Args:
        modal_space: Finite modal basis and one-hot register layout.
        reference_modals: Occupied modal per mode; modal zero when omitted.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the reference configuration is outside the modal space.
    """

    __slots__ = ("_basis_state", "_modal_space", "_reference_modals")

    def __init__(
        self,
        modal_space: ModalSpace,
        reference_modals: Sequence[int] | None = None,
    ) -> None:
        if not isinstance(modal_space, ModalSpace):
            raise TypeError("modal_space must be a ModalSpace")
        self._modal_space = modal_space
        self._reference_modals = _copy_reference_modals(
            modal_space,
            reference_modals,
        )
        self._basis_state = BasisStateInitialization(
            modal_space.num_qubits,
            modal_space.encoded_basis_index(self._reference_modals),
        )
        self.validate()

    @property
    def modal_space(self) -> ModalSpace:
        """Return the immutable finite modal space."""
        return self._modal_space

    @property
    def reference_modals(self) -> tuple[int, ...]:
        """Return the selected modal of each mode."""
        return self._reference_modals

    @property
    def basis_index(self) -> int:
        """Return the encoded one-hot computational-basis index."""
        return self._modal_space.encoded_basis_index(self._reference_modals)

    def validate(self) -> None:
        """Validate the one-hot reference and delegated basis preparation."""
        self._modal_space.modal_basis_index(self._reference_modals)
        if not self._modal_space.is_physical_basis_index(self.basis_index):
            raise ValueError("reference must occupy exactly one modal per mode")
        self._basis_state.validate()

    def build_circuit(self, prefix: str) -> Circuit:
        """Return an independent one-hot VSCF preparation circuit."""
        return self._basis_state.build_circuit(prefix)

    def num_parameters(self) -> int:
        """Return zero because the reference state is fixed."""
        return 0

    def num_qubits(self) -> int:
        """Return one qubit per modal."""
        return self._modal_space.num_qubits


__all__ = ["VSCFStatePreparation"]
