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

"""Problem definition for amplitude-estimation algorithms."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from cqlib import Circuit

from ..._utils.circuit import (
    snapshot_bound_static_unitary_circuit,
    validate_bound_static_unitary_circuit,
)


@dataclass(frozen=True, slots=True)
class AmplitudeEstimationProblem:
    """A state preparation and Grover iterate for amplitude estimation.

    Source circuits are defensively copied and normalized by qubit insertion
    position onto contiguous qubits ``0..n-1``. ``objective_qubits`` contains
    positions in that normalized system register. A Sampler reports a measured
    objective bitstring with its rightmost bit corresponding to the first
    objective qubit; ``is_good_state`` must classify that string.

    The caller is responsible for ensuring that ``grover_operator`` has the
    phase-rotation semantics associated with ``state_preparation`` and the
    good-state predicate. This object validates circuit mechanics, not that
    semantic relationship.

    Args:
        state_preparation: Fully bound unitary ``A`` that prepares the state.
        grover_operator: Fully bound Grover iterate ``Q`` on the same register.
        objective_qubits: Unique system-register positions used by the predicate.
        is_good_state: Callable that classifies measured objective bitstrings.

    Raises:
        TypeError: If a circuit, qubit index, or predicate has the wrong type.
        ValueError: If the circuits or objective-qubit selection are invalid.
    """

    state_preparation: Circuit
    grover_operator: Circuit
    objective_qubits: tuple[int, ...]
    is_good_state: Callable[[str], bool]

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        state_preparation = snapshot_bound_static_unitary_circuit(
            self.state_preparation, "state_preparation"
        )
        grover_operator = snapshot_bound_static_unitary_circuit(
            self.grover_operator, "grover_operator"
        )
        if state_preparation.num_qubits != grover_operator.num_qubits:
            raise ValueError(
                "state_preparation and grover_operator must have the same qubit count"
            )
        if not isinstance(self.objective_qubits, tuple):
            raise TypeError("objective_qubits must be a tuple")
        for qubit in self.objective_qubits:
            if isinstance(qubit, bool) or not isinstance(qubit, int):
                raise TypeError("objective_qubits must contain only integers")
            if qubit < 0 or qubit >= state_preparation.num_qubits:
                raise ValueError(
                    "objective_qubits entries must be valid system positions"
                )
        if not self.objective_qubits:
            raise ValueError("objective_qubits must not be empty")
        if len(set(self.objective_qubits)) != len(self.objective_qubits):
            raise ValueError("objective_qubits must not contain duplicates")
        if not callable(self.is_good_state):
            raise TypeError("is_good_state must be callable")

        object.__setattr__(self, "state_preparation", state_preparation)
        object.__setattr__(self, "grover_operator", grover_operator)

    @property
    def num_qubits(self) -> int:
        """Return the system-register width."""
        return int(self.state_preparation.num_qubits)

    def validate(self) -> None:
        """Revalidate the shallowly frozen problem before execution."""
        validate_bound_static_unitary_circuit(
            self.state_preparation, "state_preparation"
        )
        validate_bound_static_unitary_circuit(self.grover_operator, "grover_operator")
        if self.state_preparation.num_qubits != self.grover_operator.num_qubits:
            raise ValueError(
                "state_preparation and grover_operator must have the same qubit count"
            )
        if any(
            qubit < 0 or qubit >= self.state_preparation.num_qubits
            for qubit in self.objective_qubits
        ):
            raise ValueError("objective_qubits entries must be valid system positions")


def _validate_problem(problem: object) -> None:
    if not isinstance(problem, AmplitudeEstimationProblem):
        raise TypeError("problem must be an AmplitudeEstimationProblem")
    problem.validate()


__all__ = ["AmplitudeEstimationProblem"]
