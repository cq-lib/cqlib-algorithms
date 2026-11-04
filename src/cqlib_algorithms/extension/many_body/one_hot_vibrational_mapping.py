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

"""Exact one-hot mapping of finite-modal vibrational operators."""

from __future__ import annotations

from dataclasses import dataclass

from cqlib.qis import Hamiltonian

from ._qubit_mapping import map_one_hot_vibrational_terms
from .modal_space import ModalSpace
from .vibrational_operator import VibrationalOp


@dataclass(frozen=True, slots=True)
class OneHotVibrationalMapper:
    """Map every modal ladder to a qubit raising or lowering operator."""

    def map_operator(self, operator: VibrationalOp) -> Hamiltonian:
        """Return the exact qubit representation of ``operator``."""
        if not isinstance(operator, VibrationalOp):
            raise TypeError("operator must be a VibrationalOp")

        return map_one_hot_vibrational_terms(
            operator.modal_space.num_qubits,
            operator.terms,
            operator.modal_space.qubit_index,
        )

    @staticmethod
    def num_qubits(modal_space: ModalSpace) -> int:
        """Return one qubit per modal."""
        if not isinstance(modal_space, ModalSpace):
            raise TypeError("modal_space must be a ModalSpace")
        return modal_space.num_qubits


__all__ = ["OneHotVibrationalMapper"]
