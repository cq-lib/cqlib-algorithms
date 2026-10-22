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

"""Variational evolution of an ordered sequence of Hermitian operators."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import final

from cqlib.circuit import Circuit
from cqlib.circuit.ansatz import EvolutionStrategy, PauliEvolutionAnsatz
from cqlib.qis import Hamiltonian

from ._validation import (
    copy_hermitian_hamiltonian,
    validate_boolean,
    validate_hermitian_hamiltonian,
    validate_repetitions,
)
from .base_ansatz import BaseAnsatz


def _snapshot_operators(
    operators: Hamiltonian | Sequence[Hamiltonian],
) -> tuple[Hamiltonian, ...]:
    """Return a validated, equal-width snapshot of an operator sequence."""
    if isinstance(operators, Hamiltonian):
        candidates: tuple[Hamiltonian, ...] = (operators,)
    elif isinstance(operators, Sequence) and not isinstance(operators, (str, bytes)):
        candidates = tuple(operators)
        if not candidates:
            raise ValueError("operators must not be empty")
    else:
        raise TypeError("operators must be a Hamiltonian or a sequence of Hamiltonians")

    snapshots: list[Hamiltonian] = []
    for index, operator in enumerate(candidates):
        if not isinstance(operator, Hamiltonian):
            raise TypeError("operators must contain only Hamiltonian values")
        if operator.num_qubits <= 0:
            raise ValueError("operators must act on at least one qubit")
        snapshots.append(
            copy_hermitian_hamiltonian(operator, name=f"operators[{index}]")
        )

    width = snapshots[0].num_qubits
    if any(operator.num_qubits != width for operator in snapshots[1:]):
        raise ValueError("all operators must act on the same number of qubits")
    return tuple(snapshots)


def _is_identity_hamiltonian(operator: Hamiltonian) -> bool:
    """Return whether every simplified Pauli term is the identity."""
    return all(pauli.x_mask == 0 and pauli.z_mask == 0 for pauli, _ in operator.terms)


@final
class EvolvedOperatorAnsatz(BaseAnsatz):
    """Evolve an ordered sequence of independently parameterized operators.

    For operators ``O_0, ..., O_(m-1)``, each repetition appends their
    evolutions in index order. Parameter names are repetition-major and then
    operator-major: ``{prefix}_0, {prefix}_1, ...``.

    Pure identity operators are removed by default because they contribute
    only a parameter-dependent global phase. Set
    ``remove_identity_operators=False`` when that exact phase is relevant.
    Removing every supplied operator produces an identity circuit with the
    original register width and no parameters.

    Args:
        operators: One Hamiltonian or an ordered nonempty Hamiltonian sequence.
        reps: Positive number of ordered evolution repetitions.
        strategy: Optional Cqlib evolution strategy for noncommuting terms.
        remove_identity_operators: Whether to drop pure identity blocks.

    Raises:
        TypeError: If operators or options have invalid types.
        ValueError: If widths, Hermiticity, repetitions, or blocks are invalid.
    """

    __slots__ = (
        "_num_qubits",
        "_operators",
        "_remove_identity_operators",
        "_reps",
        "_strategy",
    )

    def __init__(
        self,
        operators: Hamiltonian | Sequence[Hamiltonian],
        *,
        reps: int = 1,
        strategy: EvolutionStrategy | None = None,
        remove_identity_operators: bool = True,
    ) -> None:
        validate_repetitions(reps)
        if strategy is not None and not isinstance(strategy, EvolutionStrategy):
            raise TypeError("strategy must be an EvolutionStrategy or None")
        validate_boolean(remove_identity_operators, "remove_identity_operators")

        snapshots = _snapshot_operators(operators)
        self._num_qubits = snapshots[0].num_qubits
        self._operators = tuple(
            operator
            for operator in snapshots
            if not (remove_identity_operators and _is_identity_hamiltonian(operator))
        )
        self._reps = reps
        self._strategy = copy.copy(
            EvolutionStrategy.auto(1) if strategy is None else strategy
        )
        self._remove_identity_operators = remove_identity_operators
        self.validate()

    def _validate_configuration(self) -> None:
        """Validate the frozen operator and evolution configuration."""
        validate_repetitions(self._reps)
        validate_boolean(
            self._remove_identity_operators,
            "remove_identity_operators",
        )
        if not isinstance(self._strategy, EvolutionStrategy):
            raise TypeError("strategy must be an EvolutionStrategy")
        for operator in self._operators:
            if operator.num_qubits != self._num_qubits:
                raise ValueError("all operators must retain their original width")
            validate_hermitian_hamiltonian(operator, name="operator")
            (
                PauliEvolutionAnsatz(operator)
                .with_strategy(copy.copy(self._strategy))
                .validate()
            )

    def _build_circuit(self, prefix: str) -> Circuit:
        """Build the repeated operator evolutions in stable time order."""
        circuit = Circuit(self._num_qubits)
        parameter_index = 0
        for _ in range(self._reps):
            for operator in self._operators:
                evolution = (
                    PauliEvolutionAnsatz(operator)
                    .with_strategy(copy.copy(self._strategy))
                    .with_time_param_name(f"{prefix}_{parameter_index}")
                    .build_circuit(prefix)
                )
                circuit.compose(evolution, circuit.qubits)
                parameter_index += 1
        return circuit

    def _parameter_count(self) -> int:
        """Return one parameter per retained operator and repetition."""
        return self._reps * len(self._operators)

    def _qubit_count(self) -> int:
        """Return the shared operator width."""
        return self._num_qubits
