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

"""Hamiltonian variational ansatz built from commuting operator blocks.

References
----------
[1] D. Wecker, M. B. Hastings, and M. Troyer, "Progress towards practical
    quantum variational algorithms," Physical Review A, 92, 042303, 2015.
    DOI: 10.1103/PhysRevA.92.042303.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from cqlib.circuit.ansatz import EvolutionStrategy
from cqlib.qis import Hamiltonian

from ..measurement import CommutingObservableGrouping
from ._validation import copy_hermitian_hamiltonian, validate_repetitions
from .base_ansatz import BaseAnsatz
from .evolved_operator_ansatz import EvolvedOperatorAnsatz


def _automatic_blocks(hamiltonian: Hamiltonian) -> tuple[Hamiltonian, ...]:
    """Partition one Hamiltonian into stable globally commuting blocks."""
    if hamiltonian.num_qubits <= 0:
        raise ValueError("hamiltonian must act on at least one qubit")
    snapshot = copy_hermitian_hamiltonian(hamiltonian, name="hamiltonian")
    snapshot_terms = tuple(snapshot.terms)
    terms = tuple(pauli for pauli, _ in snapshot_terms)
    groups = CommutingObservableGrouping().group(terms)

    blocks: list[Hamiltonian] = []
    for group in groups:
        block = Hamiltonian(snapshot.num_qubits)
        for index in group:
            pauli, coefficient = snapshot_terms[index]
            block.add_term(pauli.copy(), coefficient)
        blocks.append(block)
    return tuple(blocks)


def _explicit_blocks(
    hamiltonians: Sequence[Hamiltonian],
) -> tuple[Hamiltonian, ...]:
    """Validate and snapshot caller-provided commuting blocks."""
    candidates = tuple(hamiltonians)
    if not candidates:
        raise ValueError("hamiltonian blocks must not be empty")

    blocks: list[Hamiltonian] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Hamiltonian):
            raise TypeError("hamiltonian blocks must contain only Hamiltonian values")
        if candidate.num_qubits <= 0:
            raise ValueError("hamiltonian blocks must act on at least one qubit")
        block = copy_hermitian_hamiltonian(
            candidate,
            name=f"hamiltonian[{index}]",
        )
        if not block.all_terms_commute():
            raise ValueError("each explicit Hamiltonian block must commute internally")
        blocks.append(block)

    width = blocks[0].num_qubits
    if any(block.num_qubits != width for block in blocks[1:]):
        raise ValueError("all Hamiltonian blocks must act on the same number of qubits")
    return tuple(blocks)


@final
class HamiltonianVariationalAnsatz(BaseAnsatz):
    """Build repeated exact evolutions of commuting Hamiltonian blocks.

    A single Hamiltonian is partitioned by stable first-fit global
    commutation. A sequence is interpreted as an explicit ordered block
    decomposition; every block must be internally commuting, while distinct
    blocks need not commute. Evolution and parameter ordering are delegated
    to :class:`EvolvedOperatorAnsatz`.

    Args:
        hamiltonian: Hermitian Hamiltonian or ordered commuting blocks.
        reps: Positive number of block-sequence repetitions.

    Raises:
        TypeError: If an input has the wrong type.
        ValueError: If widths, Hermiticity, commutation, or repetitions are invalid.
    """

    __slots__ = ()

    def __init__(
        self,
        hamiltonian: Hamiltonian | Sequence[Hamiltonian],
        *,
        reps: int = 1,
    ) -> None:
        validate_repetitions(reps)
        if isinstance(hamiltonian, Hamiltonian):
            blocks = _automatic_blocks(hamiltonian)
        elif isinstance(hamiltonian, Sequence) and not isinstance(
            hamiltonian, (str, bytes)
        ):
            blocks = _explicit_blocks(hamiltonian)
        else:
            raise TypeError(
                "hamiltonian must be a Hamiltonian or a sequence of Hamiltonians"
            )

        inner = EvolvedOperatorAnsatz(
            blocks,
            reps=reps,
            strategy=EvolutionStrategy.exact(),
            remove_identity_operators=True,
        )
        super().__init__(inner)
        self.validate()
