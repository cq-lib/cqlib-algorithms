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

"""Modal-occupation observables for one-hot vibrational systems."""

from __future__ import annotations

from dataclasses import dataclass

from cqlib.qis import Hamiltonian

from ..many_body import OneHotVibrationalMapper, VibrationalLadder, VibrationalOp
from .modal_space import ModalSpace


def _validate_modal_space(modal_space: ModalSpace) -> None:
    """Validate a modal-space value."""
    if not isinstance(modal_space, ModalSpace):
        raise TypeError("modal_space must be a ModalSpace")


def _select_mapper(
    mapper: OneHotVibrationalMapper | None,
) -> OneHotVibrationalMapper:
    selected = OneHotVibrationalMapper() if mapper is None else mapper
    if not isinstance(selected, OneHotVibrationalMapper):
        raise TypeError("mapper must be a OneHotVibrationalMapper or None")
    return selected


@dataclass(frozen=True, slots=True)
class ModalOccupation:
    r"""Factory for :math:`n_{m,k}=a^\dagger_{m,k}a_{m,k}`.

    Args:
        modal_space: Finite modal basis.
        mode: Vibrational-mode index.
        modal: Modal index within ``mode``.

    Raises:
        TypeError: If the modal space or an index has the wrong type.
        ValueError: If an index is outside the modal space.
    """

    modal_space: ModalSpace
    mode: int
    modal: int

    def __post_init__(self) -> None:
        _validate_modal_space(self.modal_space)
        self.modal_space.qubit_index(self.mode, self.modal)

    def to_vibrational_op(self) -> VibrationalOp:
        """Return the selected modal number operator."""
        return VibrationalOp.from_terms(
            self.modal_space,
            (
                (
                    (
                        VibrationalLadder("+", self.mode, self.modal),
                        VibrationalLadder("-", self.mode, self.modal),
                    ),
                    1.0,
                ),
            ),
        )

    def to_qubit_hamiltonian(
        self,
        mapper: OneHotVibrationalMapper | None = None,
    ) -> Hamiltonian:
        """Map the occupation projector to the one-hot qubit register."""
        return _select_mapper(mapper).map_operator(self.to_vibrational_op())


@dataclass(frozen=True, slots=True)
class OccupiedModals:
    r"""Factory for every mode occupancy :math:`N_m=\sum_k n_{m,k}`.

    Physical one-hot states satisfy ``N_m = 1`` for every mode.

    Args:
        modal_space: Finite modal basis.

    Raises:
        TypeError: If ``modal_space`` has the wrong type.
    """

    modal_space: ModalSpace

    def __post_init__(self) -> None:
        _validate_modal_space(self.modal_space)

    def for_mode(self, mode: int) -> VibrationalOp:
        """Return the total modal occupation of one mode."""
        if type(mode) is not int:
            raise TypeError("mode must be an int")
        if mode < 0 or mode >= self.modal_space.num_modes:
            raise ValueError("mode is outside the modal space")
        return VibrationalOp.from_terms(
            self.modal_space,
            (
                (
                    (
                        VibrationalLadder("+", mode, modal),
                        VibrationalLadder("-", mode, modal),
                    ),
                    1.0,
                )
                for modal in range(self.modal_space.num_modals[mode])
            ),
        )

    def to_vibrational_ops(self) -> tuple[VibrationalOp, ...]:
        """Return one total-occupation observable per mode."""
        return tuple(self.for_mode(mode) for mode in range(self.modal_space.num_modes))

    def to_qubit_hamiltonians(
        self,
        mapper: OneHotVibrationalMapper | None = None,
    ) -> tuple[Hamiltonian, ...]:
        """Map all mode-occupation observables in mode order."""
        selected_mapper = _select_mapper(mapper)
        return tuple(
            selected_mapper.map_operator(operator)
            for operator in self.to_vibrational_ops()
        )


@dataclass(frozen=True, slots=True)
class OneHotLeakage:
    r"""Factory for strict one-hot leakage :math:`L_m=(N_m-I)^2`.

    ``L_m`` is positive semidefinite on the full encoded register and has
    eigenvalue ``(r-1)^2`` when mode ``m`` contains ``r`` occupied modal
    qubits. Therefore ``<L_m> = 0`` holds exactly when the state has complete
    support in the one-hot sector of that mode. In contrast, ``<N_m> = 1`` is
    only a necessary mean-occupation condition.

    Args:
        modal_space: Finite modal basis.

    Raises:
        TypeError: If ``modal_space`` has the wrong type.
    """

    modal_space: ModalSpace

    def __post_init__(self) -> None:
        _validate_modal_space(self.modal_space)

    def for_mode(self, mode: int) -> VibrationalOp:
        """Return the squared one-hot constraint violation of one mode."""
        occupancy = OccupiedModals(self.modal_space).for_mode(mode)
        residual = occupancy - VibrationalOp.identity(self.modal_space)
        return residual @ residual

    def to_vibrational_ops(self) -> tuple[VibrationalOp, ...]:
        """Return one strict leakage observable per mode."""
        return tuple(self.for_mode(mode) for mode in range(self.modal_space.num_modes))

    def to_qubit_hamiltonians(
        self,
        mapper: OneHotVibrationalMapper | None = None,
    ) -> tuple[Hamiltonian, ...]:
        """Map all strict leakage observables in mode order."""
        selected_mapper = _select_mapper(mapper)
        return tuple(
            selected_mapper.map_operator(operator)
            for operator in self.to_vibrational_ops()
        )


__all__ = ["ModalOccupation", "OccupiedModals", "OneHotLeakage"]
