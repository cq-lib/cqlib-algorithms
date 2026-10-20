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

"""Quantum approximate optimization composed from reusable components.

References
----------
[1] E. Farhi, J. Goldstone, and S. Gutmann, "A quantum approximate
    optimization algorithm," 2014.
    arXiv:1411.4028.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Literal

from cqlib import Circuit
from cqlib.circuit.ansatz import EvolutionStrategy
from cqlib.qis import Hamiltonian

from ...operators.ansatz import QAOACostEvolution, QAOAMixerEvolution
from ...operators.ansatz._validation import copy_hermitian_hamiltonian
from ...operators.state_preparation import UniformSuperposition
from ...protocols.ansatz import Ansatz
from ...protocols.estimator import Estimator, EstimatorRequest
from ...protocols.optimizer import Optimizer, OptimizerRequest, OptimizerResult
from ...protocols.state_preparation import StatePreparation
from .result import VQAResult

QAOAResult = VQAResult


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


class QAOA:
    """Optimize a diagonal classical cost with alternating evolutions.

    Each of ``reps`` layers applies ``exp(-i gamma H_C)`` followed by a
    one-parameter mixer unitary. Hamiltonian mixers use
    ``exp(-i beta H_M)``; circuit mixers define their own ordered unitary.
    The default initial state is the uniform superposition and the default
    mixer is the transverse field ``sum_i X_i``. A custom initial state must
    be fixed. A custom mixer may be supplied either as a Hermitian Hamiltonian
    or as a one-parameter ``Ansatz`` following the layer's ``{prefix}_beta``
    naming contract.

    The primary entry point minimizes H_C; compute_maximum explicitly
    minimizes -H_C while preserving the original cost evolution. The
    injected optimizer owns the update rule and any gradient implementation.
    Subclasses may override build_circuit to construct related variants.
    """

    __slots__ = (
        "_cost_strategy",
        "_estimator",
        "_initial_state",
        "_mixer",
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
        mixer: Ansatz | None = None,
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
        if mixer is not None and not isinstance(mixer, Ansatz):
            raise TypeError("mixer must be an Ansatz or None")
        if mixer_operator is not None and not isinstance(mixer_operator, Hamiltonian):
            raise TypeError("mixer_operator must be a Hamiltonian or None")
        if mixer is not None and mixer_operator is not None:
            raise ValueError("mixer and mixer_operator are mutually exclusive")
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
        if mixer is not None and mixer_strategy is not None:
            raise ValueError("mixer_strategy cannot be used with a circuit mixer")
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a str")

        self._estimator = estimator
        self._optimizer = optimizer
        self._reps = reps
        self._mixer = mixer
        if mixer is not None:
            mixer.validate()
            if mixer.num_parameters() != 1:
                raise ValueError("mixer must declare exactly one parameter")
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

    def build_circuit(self, cost_operator: Hamiltonian) -> Circuit:
        """Build a fresh initial-state, cost, mixer circuit without optimizing.

        The cost must be non-empty, finite, Hermitian and computational-basis
        diagonal. Active symbols follow layer order: prefix_k_gamma,
        prefix_k_beta; degenerate evolutions may omit inactive symbols.
        Subclasses may override this construction.
        """
        cost = _copy_cost_operator(cost_operator)
        width = cost.num_qubits
        initial_state = (
            UniformSuperposition(width)
            if self._initial_state is None
            else self._initial_state
        )
        initial_state.validate()
        if initial_state.num_qubits() != width:
            raise ValueError("initial_state and cost_operator must have the same width")
        circuit = initial_state.build_circuit(f"{self._prefix}_initial")
        if not isinstance(circuit, Circuit):
            raise TypeError("initial_state must build a cqlib.Circuit")
        circuit.validate()
        if circuit.num_qubits != width:
            raise ValueError("initial_state circuit has an inconsistent width")
        if circuit.used_symbols:
            raise ValueError("initial_state must not contain active symbols")
        circuit = copy.copy(circuit)
        cost_layer = QAOACostEvolution(cost, strategy=self._cost_strategy)
        mixer_layer = None
        if self._mixer is None:
            mixer_layer = QAOAMixerEvolution(
                width,
                mixer_operator=self._mixer_operator,
                strategy=self._mixer_strategy,
            )
        else:
            self._mixer.validate()
            if self._mixer.num_qubits() != width:
                raise ValueError("mixer and cost_operator must have the same width")
            if self._mixer.num_parameters() != 1:
                raise ValueError("mixer must declare exactly one parameter")
        for layer in range(self._reps):
            layer_prefix = f"{self._prefix}_{layer}"
            circuit.compose(cost_layer.build_circuit(layer_prefix))
            if self._mixer is None:
                assert mixer_layer is not None
                mixer_circuit = mixer_layer.build_circuit(layer_prefix)
            else:
                mixer_circuit = self._mixer.build_circuit(layer_prefix)
                if not isinstance(mixer_circuit, Circuit):
                    raise TypeError("mixer must build a cqlib.Circuit")
                mixer_circuit.validate()
                if mixer_circuit.num_qubits != width:
                    raise ValueError("mixer circuit has an inconsistent width")
                expected = {f"{layer_prefix}_beta"}
                if set(mixer_circuit.used_symbols) != expected:
                    raise ValueError(
                        "mixer circuit must use exactly the layer beta parameter"
                    )
            circuit.compose(mixer_circuit)
        circuit.validate()
        return circuit

    def compute_minimum_eigenvalue(
        self,
        cost_operator: Hamiltonian,
        initial_point: Mapping[str, float],
        *,
        shots: int | None = None,
    ) -> VQAResult:
        """Minimize the original cost expectation and return the final iterate.

        Initial bindings must exactly cover the built circuit's active symbols.
        Shots and objective evaluation are delegated to the supplied components.
        No global optimum or exact eigenstate is guaranteed.

        Raises:
            TypeError: If input types or the optimizer response are invalid.
            ValueError: If the cost, bindings, shots or response violates its
                component contract.
        """
        return self._compute(cost_operator, initial_point, shots, "minimize")

    def compute_maximum(
        self,
        cost_operator: Hamiltonian,
        initial_point: Mapping[str, float],
        *,
        shots: int | None = None,
    ) -> VQAResult:
        """Maximize original cost by minimizing its negative objective.

        Cost evolution still uses the original Hamiltonian. The returned
        value has the original sign, while optimizer metadata retains the
        minimized negative value. Input constraints match the minimum entry
        point. This preserves the previous maximum-solving operation.
        """
        return self._compute(cost_operator, initial_point, shots, "maximize")

    def _compute(
        self,
        cost_operator: Hamiltonian,
        initial_point: Mapping[str, float],
        shots: int | None,
        objective_sense: Literal["minimize", "maximize"],
    ) -> VQAResult:
        cost = _copy_cost_operator(cost_operator)
        circuit = self.build_circuit(cost.copy())
        if not isinstance(circuit, Circuit):
            raise TypeError("build_circuit must return a cqlib.Circuit")
        circuit.validate()
        parameter_names = tuple(circuit.used_symbols)
        objective = cost.copy()
        if objective_sense == "maximize":
            objective.scale(-1.0)
        request = OptimizerRequest(
            EstimatorRequest(circuit, objective, initial_point, shots),
            parameter_names,
        )
        results = self._optimizer.run(self._estimator, [request])
        if not isinstance(results, list):
            raise TypeError("optimizer must return a list of OptimizerResult values")
        if len(results) != 1:
            raise ValueError("optimizer must return exactly one result for QAOA")
        result = results[0]
        if not isinstance(result, OptimizerResult):
            raise TypeError("optimizer must return OptimizerResult values")
        if result.parameter_names != parameter_names:
            raise ValueError(
                "optimizer result parameter names must match the QAOA request"
            )
        parameters = dict(
            zip(result.parameter_names, result.optimal_parameters, strict=True)
        )
        optimal_circuit = circuit.assign_parameters(parameters)
        value = result.optimal_value
        if objective_sense == "maximize":
            value = -value
        return VQAResult(
            value,
            parameters,
            optimal_circuit,
            result,
            objective_sense=objective_sense,
        )
