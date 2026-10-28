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

"""VSCF-relative single- and double-excitation UVCC ansatz."""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from ...extension.vibration import (
    ModalSpace,
    generate_vibrational_double_excitations,
    generate_vibrational_single_excitations,
)
from ...protocols.state_preparation import StatePreparation
from ..state_preparation.vscf_state_preparation import VSCFStatePreparation
from .base_ansatz import _DelegatingAnsatz
from .uvcc_ansatz import UVCCAnsatz


@final
class UVCCSDAnsatz(_DelegatingAnsatz):
    """Build VSCF-relative vibrational single and double excitation factors.

    Args:
        modal_space: Finite modal basis defining the excitation pool.
        reference_modals: Occupied modal per mode; modal zero when omitted.
        reps: Non-negative repetition count.
        initial_state: Optional fixed reference; VSCF when omitted.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the reference, repetition count, or excitation pool is invalid.
    """

    __slots__ = ()

    def __init__(
        self,
        modal_space: ModalSpace,
        *,
        reference_modals: Sequence[int] | None = None,
        reps: int = 1,
        initial_state: StatePreparation | None = None,
    ) -> None:
        if not isinstance(modal_space, ModalSpace):
            raise TypeError("modal_space must be a ModalSpace")
        excitations = (
            *generate_vibrational_single_excitations(modal_space, reference_modals),
            *generate_vibrational_double_excitations(modal_space, reference_modals),
        )
        if not excitations:
            raise ValueError("modal_space must produce at least one UVCCSD excitation")

        state = initial_state
        if state is None:
            state = VSCFStatePreparation(modal_space, reference_modals)
        inner = UVCCAnsatz(
            excitations,
            reps=reps,
            initial_state=state,
        )
        super().__init__(inner)
        self.validate()


__all__ = ["UVCCSDAnsatz"]
