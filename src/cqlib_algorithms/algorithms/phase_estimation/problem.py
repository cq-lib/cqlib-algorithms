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

from dataclasses import dataclass

from cqlib import Circuit

from ...operators.composite.phase_estimation_circuit import (
    _snapshot_static_unitary_circuit,
    _validate_static_unitary_circuit,
)


@dataclass(frozen=True, slots=True)
class PhaseEstimationProblem:
    """A unitary and input-state preparation for phase estimation.

    If the prepared state is a superposition of eigenstates, standard QPE
    samples the corresponding eigenphases according to their squared overlap.
    Iterative QPE instead requires preparation of one eigenstate because its
    classically fed-forward rounds execute as separate circuits.

    Args:
        unitary: Fully bound static unitary whose eigenphase is estimated.
        state_preparation: Fully bound circuit preparing the system state.

    Raises:
        TypeError: If an input is not a Circuit.
        ValueError: If the circuits are invalid, parameterized, or incompatible.
    """

    unitary: Circuit
    state_preparation: Circuit

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        unitary = _snapshot_static_unitary_circuit(self.unitary, "unitary")
        state_preparation = _snapshot_static_unitary_circuit(
            self.state_preparation,
            "state_preparation",
        )
        if unitary.num_qubits != state_preparation.num_qubits:
            raise ValueError(
                "unitary and state_preparation must have the same qubit count"
            )
        object.__setattr__(self, "unitary", unitary)
        object.__setattr__(self, "state_preparation", state_preparation)

    @property
    def num_qubits(self) -> int:
        """Return the system-register width."""
        return int(self.unitary.num_qubits)

    def validate(self) -> None:
        """Revalidate the shallowly frozen problem before execution."""
        _validate_static_unitary_circuit(self.unitary, "unitary")
        _validate_static_unitary_circuit(
            self.state_preparation,
            "state_preparation",
        )
        if self.unitary.num_qubits != self.state_preparation.num_qubits:
            raise ValueError(
                "unitary and state_preparation must have the same qubit count"
            )


def _validate_problem(problem: object) -> PhaseEstimationProblem:
    if not isinstance(problem, PhaseEstimationProblem):
        raise TypeError("problem must be a PhaseEstimationProblem")
    problem.validate()
    return problem


__all__ = ["PhaseEstimationProblem"]
