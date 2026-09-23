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

"""Quantum approximate optimization composed from reusable components."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import final

from cqlib import Circuit
from cqlib.circuit.ansatz import EvolutionStrategy
from cqlib.qis import Hamiltonian

from ..operators.ansatz import QAOACostEvolution, QAOAMixerEvolution
from ..operators.ansatz._validation import copy_hermitian_hamiltonian
from ..operators.state_preparation import UniformSuperposition
from ..protocols.ansatz import Ansatz
from ..protocols.estimator import Estimator
from ..protocols.optimizer import Optimizer, OptimizerResult
from ..protocols.state_preparation import StatePreparation
from .vqe import VQE, VQEResult


def _validate_reps(reps: int) -> None:
    if isinstance(reps, bool) or not isinstance(reps, int):
        raise TypeError("reps must be an integer")
    if reps <= 0:
        raise ValueError("reps must be greater than zero")


def _copy_cost_operator(cost_operator: Hamiltonian) -> Hamiltonian:
    if not isinstance(cost_operator, Hamiltonian):
        raise TypeError("cost_operator must be a Hamiltonian")
    if cost_operator.num_qubits <= 0:
        raise ValueError("cost_operator must act on at least one qubit")

    snapshot = copy_hermitian_hamiltonian(cost_operator, name="cost_operator")
    if any(pauli.x_mask != 0 for pauli, _ in snapshot.terms):
        raise ValueError("cost_operator must be diagonal in the computational basis")
    return snapshot


@final
class _QAOAAnsatz(Ansatz):
    """Compose an initial state and alternating cost and mixer layers."""

    __slots__ = (
        "_cost_layer",
        "_initial_state",
        "_mixer_layer",
        "_num_qubits",
        "_reps",
    )

    def __init__(
        self,
        cost_operator: Hamiltonian,
        reps: int,
        *,
        mixer_operator: Hamiltonian | None,
        initial_state: StatePreparation | None,
        cost_strategy: EvolutionStrategy | None,
        mixer_strategy: EvolutionStrategy | None,
    ) -> None:
        _validate_reps(reps)
        self._num_qubits = cost_operator.num_qubits
        self._reps = reps
        self._initial_state = (
            UniformSuperposition(self._num_qubits)
            if initial_state is None
            else initial_state
        )
        if self._initial_state.num_qubits() != self._num_qubits:
            raise ValueError("initial_state and cost_operator must have the same width")

        self._cost_layer = QAOACostEvolution(
            cost_operator,
            strategy=cost_strategy,
        )
        self._mixer_layer = QAOAMixerEvolution(
            self._num_qubits,
            mixer_operator=mixer_operator,
            strategy=mixer_strategy,
        )
        self.validate()

    def validate(self) -> None:
        """Validate all composed components and fixed-state requirements."""
        self._initial_state.validate()
        self._cost_layer.validate()
        self._mixer_layer.validate()
        if self._initial_state.num_qubits() != self._num_qubits:
            raise ValueError("initial_state and cost_operator must have the same width")
        if self._cost_layer.num_qubits() != self._num_qubits:
            raise ValueError("cost evolution width changed after construction")
        if self._mixer_layer.num_qubits() != self._num_qubits:
            raise ValueError("mixer evolution width changed after construction")

        initial_circuit = self._initial_state.build_circuit("qaoa_initial")
        initial_circuit.validate()
        if initial_circuit.num_qubits != self._num_qubits:
            raise ValueError("initial_state circuit has an inconsistent width")
        if initial_circuit.used_symbols:
            raise ValueError("initial_state must not contain active symbols")

    def build_circuit(self, prefix: str) -> Circuit:
        """Build a fresh circuit in initial, cost, mixer time order."""
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a str")
        self.validate()
        circuit = self._initial_state.build_circuit(f"{prefix}_initial")
        for layer in range(self._reps):
            layer_prefix = f"{prefix}_{layer}"
            circuit.compose(self._cost_layer.build_circuit(layer_prefix))
            circuit.compose(self._mixer_layer.build_circuit(layer_prefix))
        circuit.validate()
        return circuit

    def num_parameters(self) -> int:
        """Return two logical angles per alternating layer."""
        return 2 * self._reps

    def num_qubits(self) -> int:
        """Return the cost-register width."""
        return self._num_qubits


@dataclass(frozen=True, slots=True)
class QAOAResult:
    """Maximum cost and the complete delegated variational result.

    ``optimal_value`` is the expectation of the original cost operator.
    ``optimizer_result.optimal_value`` is its negative because the delegated
    optimizer minimizes the negated cost objective.
    """

    vqe_result: VQEResult = field(repr=False)

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.vqe_result, VQEResult):
            raise TypeError("vqe_result must be a VQEResult")

    @property
    def optimal_value(self) -> float:
        """Return the maximum expectation of the original cost operator."""
        return -self.vqe_result.eigenvalue

    @property
    def optimal_parameters(self) -> Mapping[str, float]:
        """Return the immutable optimal parameter mapping."""
        return self.vqe_result.optimal_parameters

    @property
    def optimal_circuit(self) -> Circuit:
        """Return the independent, fully bound optimal circuit."""
        return self.vqe_result.optimal_circuit

    @property
    def optimizer_result(self) -> OptimizerResult:
        """Return metadata for minimizing the negated cost objective."""
        return self.vqe_result.optimizer_result


@final
class QAOA:
    """Maximize a diagonal classical cost with alternating evolutions.

    Each of ``reps`` layers applies ``exp(-i gamma H_C)`` followed by
    ``exp(-i beta H_M)``. The default initial state is the uniform
    superposition and the default mixer is the transverse field ``sum_i X_i``.
    A custom initial state must be fixed and a custom Hermitian mixer must act
    on the cost register.

    Optimization is delegated to :class:`VQE` by minimizing ``-H_C``. The
    injected optimizer continues to own any gradient implementation.
    """

    __slots__ = (
        "_cost_strategy",
        "_estimator",
        "_initial_state",
        "_mixer_operator",
        "_mixer_strategy",
        "_optimizer",
        "_prefix",
        "_reps",
    )

    def __init__(
        self,
        estimator: Estimator,
        optimizer: Optimizer,
        *,
        reps: int = 1,
        mixer_operator: Hamiltonian | None = None,
        initial_state: StatePreparation | None = None,
        cost_strategy: EvolutionStrategy | None = None,
        mixer_strategy: EvolutionStrategy | None = None,
        prefix: str = "qaoa",
    ) -> None:
        if not isinstance(estimator, Estimator):
            raise TypeError("estimator must be an Estimator")
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be an Optimizer")
        _validate_reps(reps)
        if mixer_operator is not None and not isinstance(mixer_operator, Hamiltonian):
            raise TypeError("mixer_operator must be a Hamiltonian or None")
        if initial_state is not None and not isinstance(
            initial_state, StatePreparation
        ):
            raise TypeError("initial_state must be a StatePreparation or None")
        if cost_strategy is not None and not isinstance(
            cost_strategy, EvolutionStrategy
        ):
            raise TypeError("cost_strategy must be an EvolutionStrategy or None")
        if mixer_strategy is not None and not isinstance(
            mixer_strategy, EvolutionStrategy
        ):
            raise TypeError("mixer_strategy must be an EvolutionStrategy or None")
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a str")

        self._estimator = estimator
        self._optimizer = optimizer
        self._reps = reps
        self._mixer_operator = (
            None
            if mixer_operator is None
            else copy_hermitian_hamiltonian(mixer_operator, name="mixer_operator")
        )
        self._initial_state = initial_state
        self._cost_strategy = (
            None if cost_strategy is None else copy.copy(cost_strategy)
        )
        self._mixer_strategy = (
            None if mixer_strategy is None else copy.copy(mixer_strategy)
        )
        self._prefix = prefix

    @property
    def estimator(self) -> Estimator:
        """Return the injected expectation estimator."""
        return self._estimator

    @property
    def optimizer(self) -> Optimizer:
        """Return the injected classical optimizer."""
        return self._optimizer

    @property
    def reps(self) -> int:
        """Return the number of alternating layers."""
        return self._reps

    @property
    def prefix(self) -> str:
        """Return the parameter-name prefix."""
        return self._prefix

    def compute_maximum(
        self,
        cost_operator: Hamiltonian,
        initial_point: Mapping[str, float],
        *,
        shots: int | None = None,
    ) -> QAOAResult:
        """Optimize and return the maximum reachable cost expectation.

        The cost must be a non-empty, finite Hermitian Hamiltonian containing
        only ``I`` and ``Z`` Pauli factors. Its computational-basis diagonal
        therefore defines a classical objective. The initial point must bind
        exactly ``prefix_{k}_gamma`` and ``prefix_{k}_beta`` for every layer
        ``k``, in that order.

        The result is the final optimizer iterate and does not assert global
        optimality. ``shots`` is passed unchanged through the delegated
        estimator request.
        """
        cost_snapshot = _copy_cost_operator(cost_operator)
        ansatz = _QAOAAnsatz(
            cost_snapshot,
            self._reps,
            mixer_operator=self._mixer_operator,
            initial_state=self._initial_state,
            cost_strategy=self._cost_strategy,
            mixer_strategy=self._mixer_strategy,
        )
        objective = cost_snapshot.copy()
        objective.scale(-1.0)
        vqe_result = VQE(
            self._estimator,
            ansatz,
            self._optimizer,
            prefix=self._prefix,
        ).compute_minimum_eigenvalue(objective, initial_point, shots=shots)
        return QAOAResult(vqe_result)
