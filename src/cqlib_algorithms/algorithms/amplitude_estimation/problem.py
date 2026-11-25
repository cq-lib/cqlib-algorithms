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
from copy import deepcopy
from dataclasses import dataclass, field

from cqlib import Circuit

from ..._utils.circuit import (
    snapshot_bound_static_unitary_circuit,
    validate_bound_static_unitary_circuit,
)
from ..._utils.state_preparation import build_state_preparation_circuit
from ...operators.composite.grover_operator import GroverOperator
from ...operators.state_preparation import UserCircuitPreparation
from ...protocols.state_preparation import StatePreparation


@dataclass(frozen=True, slots=True, init=False)
class AmplitudeEstimationProblem:
    """A state preparation and Grover iterate for amplitude estimation.

    Components are built at construction with preparation prefix ``"ae"``.
    ``state_preparation`` stores a fixed preparation and ``grover_operator`` an
    independent component copy. ``preparation_circuit`` and ``iteration_circuit``
    expose the execution snapshots and must remain unchanged. The two circuits are
    independent snapshots with qubits mapped in insertion order to
    contiguous positions ``0..n-1``. ``objective_qubits`` contains
    positions in that normalized system register. A Sampler reports a measured
    objective bitstring with its rightmost bit corresponding to the first
    objective qubit; ``is_good_state`` must classify that string.

    Supply a Grover iteration built for the same preparation and good-state
    subspace. Standard AE reads the phase register. The power-sampling AE
    methods use ``is_good_state`` to classify objective-register samples.

    Args:
        state_preparation: Component for the fully bound unitary ``A`` that
            prepares the state.
        grover_operator: Grover iteration using the same register, preparation,
            and marked subspace as the predicate.
        objective_qubits: Unique system-register positions used by the predicate.
        is_good_state: Callable that classifies measured objective bitstrings.

    Raises:
        TypeError: If a component, qubit index, or predicate has the wrong type.
        ValueError: If the circuits or objective-qubit selection are invalid.
    """

    state_preparation: StatePreparation
    grover_operator: GroverOperator
    objective_qubits: tuple[int, ...]
    is_good_state: Callable[[str], bool]
    _preparation_circuit: Circuit = field(init=False, repr=False)
    _iteration_circuit: Circuit = field(init=False, repr=False)

    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self,
        state_preparation: StatePreparation,
        grover_operator: GroverOperator,
        objective_qubits: tuple[int, ...],
        is_good_state: Callable[[str], bool],
    ) -> None:
        preparation = snapshot_bound_static_unitary_circuit(
            build_state_preparation_circuit(state_preparation, prefix="ae"),
            "state_preparation",
        )
        if not isinstance(grover_operator, GroverOperator):
            raise TypeError("grover_operator must be a GroverOperator")
        iteration = grover_operator.build_circuit()
        if not isinstance(iteration, Circuit):
            raise TypeError("GroverOperator.build_circuit must return a Circuit")
        if iteration.num_qubits != grover_operator.num_qubits:
            raise ValueError(
                "GroverOperator circuit width must match its declared width"
            )
        iteration = snapshot_bound_static_unitary_circuit(iteration, "grover_operator")
        if preparation.num_qubits != iteration.num_qubits:
            raise ValueError(
                "state_preparation and grover_operator must have the same qubit count"
            )
        if not isinstance(objective_qubits, tuple):
            raise TypeError("objective_qubits must be a tuple")
        for qubit in objective_qubits:
            if isinstance(qubit, bool) or not isinstance(qubit, int):
                raise TypeError("objective_qubits must contain only integers")
            if qubit < 0 or qubit >= preparation.num_qubits:
                raise ValueError(
                    "objective_qubits entries must be valid system positions"
                )
        if not objective_qubits:
            raise ValueError("objective_qubits must not be empty")
        if len(set(objective_qubits)) != len(objective_qubits):
            raise ValueError("objective_qubits must not contain duplicates")
        if not callable(is_good_state):
            raise TypeError("is_good_state must be callable")

        object.__setattr__(
            self, "state_preparation", UserCircuitPreparation(preparation)
        )
        object.__setattr__(self, "grover_operator", deepcopy(grover_operator))
        object.__setattr__(self, "_preparation_circuit", preparation)
        object.__setattr__(self, "_iteration_circuit", iteration)
        object.__setattr__(self, "objective_qubits", objective_qubits)
        object.__setattr__(self, "is_good_state", is_good_state)

    @property
    def preparation_circuit(self) -> Circuit:
        """Return the stored preparation snapshot, which must remain unchanged."""
        return self._preparation_circuit

    @property
    def iteration_circuit(self) -> Circuit:
        """Return the stored iteration snapshot, which must remain unchanged."""
        return self._iteration_circuit

    @property
    def num_qubits(self) -> int:
        """Return the system-register width."""
        return int(self.preparation_circuit.num_qubits)

    def validate(self) -> None:
        """Revalidate the shallowly frozen problem before execution."""
        validate_bound_static_unitary_circuit(
            self.preparation_circuit, "state_preparation"
        )
        validate_bound_static_unitary_circuit(self.iteration_circuit, "grover_operator")
        if self.preparation_circuit.num_qubits != self.iteration_circuit.num_qubits:
            raise ValueError(
                "state_preparation and grover_operator must have the same qubit count"
            )
        if any(
            qubit < 0 or qubit >= self.preparation_circuit.num_qubits
            for qubit in self.objective_qubits
        ):
            raise ValueError("objective_qubits entries must be valid system positions")


def _validate_problem(problem: object) -> None:
    if not isinstance(problem, AmplitudeEstimationProblem):
        raise TypeError("problem must be an AmplitudeEstimationProblem")
    problem.validate()


__all__ = ["AmplitudeEstimationProblem"]
