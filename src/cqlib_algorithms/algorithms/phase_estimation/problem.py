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

"""Problem definition shared by quantum phase-estimation algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field

from cqlib import Circuit

from ..._utils.phase_estimation import (
    snapshot_phase_preparation,
    snapshot_static_unitary_circuit,
    validate_static_unitary_circuit,
)
from ...operators.state_preparation import UserCircuitPreparation
from ...protocols.state_preparation import StatePreparation


@dataclass(frozen=True, slots=True, init=False)
class PhaseEstimationProblem:
    """A unitary and input-state preparation for phase estimation.

    If the prepared state is a superposition of eigenstates, standard QPE
    samples the corresponding eigenphases according to their squared overlap.
    Iterative QPE instead requires preparation of one eigenstate because its
    classically fed-forward rounds execute as separate circuits.

    Args:
        unitary: Fully bound static unitary circuit with its absolute global phase.
            All target qubits, including any ancillas, belong to the estimated unitary.
        state_preparation: Fully bound StatePreparation for the system register,
            built once with prefix ``"state_preparation"``. Both inputs are snapshotted.
            The stored preparation remains a component; ``preparation_circuit`` exposes
            its execution snapshot and must remain unchanged.

    Raises:
        TypeError: If either input has an unsupported type.
        ValueError: If the circuits are invalid, parameterized, or incompatible.
    """

    unitary: Circuit
    state_preparation: StatePreparation
    _preparation_circuit: Circuit = field(init=False, repr=False)

    __hash__ = None  # type: ignore[assignment]

    def __init__(self, unitary: Circuit, state_preparation: StatePreparation) -> None:
        unitary = snapshot_static_unitary_circuit(unitary, "unitary")
        preparation = snapshot_phase_preparation(state_preparation)
        if unitary.num_qubits != preparation.num_qubits:
            raise ValueError(
                "unitary and state_preparation must have the same qubit count"
            )
        object.__setattr__(self, "unitary", unitary)
        object.__setattr__(
            self, "state_preparation", UserCircuitPreparation(preparation)
        )
        object.__setattr__(self, "_preparation_circuit", preparation)

    @property
    def preparation_circuit(self) -> Circuit:
        """Return the stored preparation snapshot, which must remain unchanged."""
        return self._preparation_circuit

    @property
    def num_qubits(self) -> int:
        """Return the system-register width."""
        return int(self.unitary.num_qubits)

    def validate(self) -> None:
        """Revalidate the shallowly frozen problem before execution."""
        validate_static_unitary_circuit(self.unitary, "unitary")
        validate_static_unitary_circuit(
            self.preparation_circuit,
            "state_preparation",
        )
        if self.unitary.num_qubits != self.preparation_circuit.num_qubits:
            raise ValueError(
                "unitary and state_preparation must have the same qubit count"
            )


def _validate_problem(problem: object) -> None:
    if not isinstance(problem, PhaseEstimationProblem):
        raise TypeError("problem must be a PhaseEstimationProblem")
    problem.validate()


__all__ = ["PhaseEstimationProblem"]
