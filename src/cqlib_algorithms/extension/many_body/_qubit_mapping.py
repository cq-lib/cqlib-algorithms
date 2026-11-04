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

"""Pure qubit-mapping kernels for sparse many-body terms."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Literal, Protocol

from cqlib.qis import Hamiltonian, Pauli, PauliString, Phase


class _VibrationalLadderLike(Protocol):
    """Structural ladder data consumed by the private mapping kernel."""

    @property
    def kind(self) -> Literal["+", "-"]:
        """Return the ladder direction."""

    @property
    def mode(self) -> int:
        """Return the mode index."""

    @property
    def modal(self) -> int:
        """Return the modal index."""


def map_spin_half_terms(
    num_spins: int,
    terms: Iterable[tuple[Iterable[tuple[Literal["X", "Y", "Z"], int]], complex]],
) -> Hamiltonian:
    """Map validated spin-half terms using ``S^alpha = sigma_alpha / 2``."""
    pauli_by_axis = {
        "X": Pauli.x(),
        "Y": Pauli.y(),
        "Z": Pauli.z(),
    }
    mapped_terms: list[tuple[PauliString, complex]] = []
    for term, coefficient in terms:
        pauli_string = PauliString(num_spins)
        mapped_coefficient = coefficient
        for axis, index in term:
            local = PauliString(num_spins)
            local.set_pauli(index, pauli_by_axis[axis])
            pauli_string *= local
            mapped_coefficient *= 0.5
        mapped_terms.append((pauli_string, mapped_coefficient))
    return Hamiltonian.from_terms(num_spins, mapped_terms)


def _local_pauli(width: int, qubit: int, pauli: Pauli) -> PauliString:
    result = PauliString(width)
    result.set_pauli(qubit, pauli)
    return result


def _ladder_expansion(
    width: int,
    qubit_index: Callable[[int, int], int],
    ladder: _VibrationalLadderLike,
) -> tuple[tuple[PauliString, complex], tuple[PauliString, complex]]:
    qubit = qubit_index(ladder.mode, ladder.modal)
    y_coefficient = -0.5j if ladder.kind == "+" else 0.5j
    return (
        (_local_pauli(width, qubit, Pauli.x()), 0.5),
        (_local_pauli(width, qubit, Pauli.y()), y_coefficient),
    )


def _phase_free_copy(pauli: PauliString) -> PauliString:
    copied = pauli.copy()
    copied.phase = Phase.plus()
    return copied


def map_one_hot_vibrational_terms(
    num_qubits: int,
    terms: Iterable[tuple[Iterable[_VibrationalLadderLike], complex]],
    qubit_index: Callable[[int, int], int],
) -> Hamiltonian:
    """Map validated finite-modal ladder terms to one-hot qubit operators."""
    aggregated: dict[tuple[int, int], tuple[PauliString, complex]] = {}
    for term, coefficient in terms:
        expansion: tuple[tuple[PauliString, complex], ...] = (
            (PauliString(num_qubits), coefficient),
        )
        for ladder in term:
            next_expansion: list[tuple[PauliString, complex]] = []
            for left_pauli, left_coefficient in expansion:
                for right_pauli, right_coefficient in _ladder_expansion(
                    num_qubits,
                    qubit_index,
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

    mapped_terms = tuple(
        (pauli, coefficient)
        for _, (pauli, coefficient) in sorted(aggregated.items())
        if coefficient != 0.0
    )
    return Hamiltonian.from_terms(num_qubits, mapped_terms)
