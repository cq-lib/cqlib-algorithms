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

from cqlib.qis import Hamiltonian, Pauli, PauliString, Phase

from .modal_space import ModalSpace
from .vibrational_operator import VibrationalLadder, VibrationalOp


def _local_pauli(width: int, qubit: int, pauli: Pauli) -> PauliString:
    result = PauliString(width)
    result.set_pauli(qubit, pauli)
    return result


def _ladder_expansion(
    modal_space: ModalSpace,
    ladder: VibrationalLadder,
) -> tuple[tuple[PauliString, complex], tuple[PauliString, complex]]:
    qubit = modal_space.qubit_index(ladder.mode, ladder.modal)
    y_coefficient = -0.5j if ladder.kind == "+" else 0.5j
    return (
        (_local_pauli(modal_space.num_qubits, qubit, Pauli.x()), 0.5),
        (_local_pauli(modal_space.num_qubits, qubit, Pauli.y()), y_coefficient),
    )


def _phase_free_copy(pauli: PauliString) -> PauliString:
    copied = pauli.copy()
    copied.phase = Phase.plus()
    return copied


@dataclass(frozen=True, slots=True)
class OneHotVibrationalMapper:
    """Map every modal ladder to a qubit raising or lowering operator."""

    def map_operator(self, operator: VibrationalOp) -> Hamiltonian:
        """Return the exact qubit representation of ``operator``."""
        if not isinstance(operator, VibrationalOp):
            raise TypeError("operator must be a VibrationalOp")

        width = operator.modal_space.num_qubits
        aggregated: dict[tuple[int, int], tuple[PauliString, complex]] = {}
        for term, coefficient in operator.terms:
            expansion: tuple[tuple[PauliString, complex], ...] = (
                (PauliString(width), coefficient),
            )
            for ladder in term:
                next_expansion: list[tuple[PauliString, complex]] = []
                for left_pauli, left_coefficient in expansion:
                    for right_pauli, right_coefficient in _ladder_expansion(
                        operator.modal_space,
                        ladder,
                    ):
                        product = left_pauli.copy()
                        product *= right_pauli
                        next_expansion.append(
                            (product, left_coefficient * right_coefficient)
                        )
                expansion = tuple(next_expansion)

            for pauli, expanded_coefficient in expansion:
                effective = expanded_coefficient * pauli.phase.to_complex()
                key = (pauli.x_mask, pauli.z_mask)
                phase_free = _phase_free_copy(pauli)
                previous = aggregated.get(key, (phase_free, 0.0j))[1]
                aggregated[key] = (phase_free, previous + effective)

        terms = tuple(
            (pauli, coefficient)
            for _, (pauli, coefficient) in sorted(aggregated.items())
            if coefficient != 0.0
        )
        return Hamiltonian.from_terms(width, terms)

    @staticmethod
    def num_qubits(modal_space: ModalSpace) -> int:
        """Return one qubit per modal."""
        if not isinstance(modal_space, ModalSpace):
            raise TypeError("modal_space must be a ModalSpace")
        return modal_space.num_qubits


__all__ = ["OneHotVibrationalMapper"]
