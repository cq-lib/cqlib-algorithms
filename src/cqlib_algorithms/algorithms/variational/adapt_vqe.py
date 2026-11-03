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

"""Adaptive variational eigensolving from a Hermitian generator pool.

References
----------
[1] H. R. Grimsley et al., "An adaptive variational algorithm for exact
    molecular simulations on a quantum computer," Nature Communications 10,
    3007, 2019. DOI: 10.1038/s41467-019-10988-2.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import final

import numpy as np
from cqlib import Circuit
from cqlib.circuit.ansatz import EvolutionStrategy
from cqlib.qis import Hamiltonian, PauliString

from ..._utils.hamiltonian import (
    snapshot_nonzero_hermitian_hamiltonian,
)
from ..._utils.numeric import (
    normalize_finite_real,
    normalize_nonnegative_real,
    normalize_positive_real,
)
from ..._utils.state_preparation import snapshot_fixed_initial_state
from ...operators.ansatz import EvolvedOperatorAnsatz
from ...protocols import Ansatz, Estimator, EstimatorRequest, Optimizer
from ...protocols.estimator import EstimatorResult, require_estimator_results
from ...protocols.sampler import spawn_seeds
from ...protocols.state_preparation import StatePreparation
from .result import VQAResult
from .vqe import VQE, VQEResult


class AdaptVQETermination(Enum):
    """Termination reason of an adaptive variational run."""

    GRADIENT_CONVERGED = "gradient_converged"
    ENERGY_CONVERGED = "energy_converged"
    MAX_ITERATIONS = "max_iterations"
    CYCLIC_SEQUENCE = "cyclic_sequence"


@dataclass(frozen=True, slots=True, repr=False)
class AdaptVQEResult(VQAResult):
    """Final variational state and adaptive-construction diagnostics.

    ``eigenvalue_history`` contains the reference-state value followed by one
    value for every accepted generator. ``gradient_norms`` records every full
    pool norm evaluated, including the final convergence or iteration-limit
    check. Selected pool indices preserve circuit time order.

    Args:
        optimal_value: Final accepted target expectation.
        optimal_parameters: Final accepted bindings in optimizer order.
        optimal_circuit: Fully bound circuit for the final accepted state.
        optimizer_result: Optimizer metadata for the final accepted state.
        selected_operator_indices: Pool indices in circuit time order.
        gradient_norms: Full-pool L2 norms evaluated during adaptation.
        eigenvalue_history: Reference value followed by each accepted value.
        final_max_gradient: Largest absolute component in the final pool
            gradient.
        termination: Reason the adaptive loop stopped.
        objective_sense: Objective direction inherited from ``VQAResult``.

    Raises:
        TypeError: If adaptive metadata has an invalid type.
        ValueError: If metadata is non-finite or internally inconsistent.
    """

    selected_operator_indices: tuple[int, ...]
    gradient_norms: tuple[float, ...]
    eigenvalue_history: tuple[float, ...]
    final_max_gradient: float
    termination: AdaptVQETermination

    __hash__ = None

    def __post_init__(self) -> None:
        VQAResult.__post_init__(self)
        if not isinstance(self.selected_operator_indices, tuple):
            raise TypeError("selected_operator_indices must be a tuple")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in self.selected_operator_indices
        ):
            raise ValueError("selected operator indices must be non-negative integers")
        if not isinstance(self.gradient_norms, tuple):
            raise TypeError("gradient_norms must be a tuple")
        if not isinstance(self.eigenvalue_history, tuple):
            raise TypeError("eigenvalue_history must be a tuple")
        norms = tuple(
            normalize_nonnegative_real(value, "gradient norm")
            for value in self.gradient_norms
        )
        if not norms:
            raise ValueError("gradient_norms must not be empty")
        energies = tuple(
            normalize_finite_real(value, "eigenvalue history entry")
            for value in self.eigenvalue_history
        )
        if len(energies) != len(self.selected_operator_indices) + 1:
            raise ValueError(
                "eigenvalue_history must contain the reference value and one "
                "entry per selected operator"
            )
        if not energies or energies[-1] != self.optimal_value:
            raise ValueError("final eigenvalue history entry must match optimal_value")
        object.__setattr__(self, "gradient_norms", norms)
        object.__setattr__(self, "eigenvalue_history", energies)
        final_max_gradient = normalize_nonnegative_real(
            self.final_max_gradient, "final_max_gradient"
        )
        if final_max_gradient > norms[-1]:
            raise ValueError("final_max_gradient must not exceed the final pool norm")
        object.__setattr__(self, "final_max_gradient", final_max_gradient)
        if not isinstance(self.termination, AdaptVQETermination):
            raise TypeError("termination must be an AdaptVQETermination")

    @property
    def num_iterations(self) -> int:
        """Return the number of accepted adaptive generators."""
        return len(self.selected_operator_indices)


def _snapshot_operator(
    operator: Hamiltonian | PauliString,
    *,
    name: str,
) -> Hamiltonian:
    """Validate and return a Hermitian Hamiltonian snapshot."""
    if isinstance(operator, PauliString):
        operator = Hamiltonian.from_pauli(operator.copy())
    if not isinstance(operator, Hamiltonian):
        raise TypeError(f"{name} must be a Hamiltonian or PauliString")
    return snapshot_nonzero_hermitian_hamiltonian(operator, name=name)


def _snapshot_pool(
    operator_pool: Sequence[Hamiltonian | PauliString],
) -> tuple[Hamiltonian, ...]:
    """Validate and return a nonempty equal-width generator pool."""
    if isinstance(operator_pool, (str, bytes)) or not isinstance(
        operator_pool, Sequence
    ):
        raise TypeError("operator_pool must be a sequence")
    if not operator_pool:
        raise ValueError("operator_pool must not be empty")
    snapshots = tuple(
        _snapshot_operator(operator, name=f"operator_pool[{index}]")
        for index, operator in enumerate(operator_pool)
    )
    width = snapshots[0].num_qubits
    if any(operator.num_qubits != width for operator in snapshots[1:]):
        raise ValueError("operator_pool entries must have equal width")
    for operator in snapshots:
        if all(pauli.x_mask == 0 and pauli.z_mask == 0 for pauli, _ in operator.terms):
            raise ValueError("operator_pool entries must not be identity-only")
    return snapshots


def _commutator_observable(generator: Hamiltonian, target: Hamiltonian) -> Hamiltonian:
    """Return the Hermitian observable ``i [generator, target]``."""
    result = Hamiltonian(target.num_qubits)
    for left, left_coefficient in generator.terms:
        for right, right_coefficient in target.terms:
            if left.commutes_with(right):
                continue
            result.add_term(
                left * right,
                2j * complex(left_coefficient) * complex(right_coefficient),
            )
    result.simplify()
    return result.normalized_hermitian()


def _has_repeating_suffix(indices: Sequence[int]) -> bool:
    """Return whether the selected-index suffix repeats immediately."""
    for width in range(1, len(indices) // 2 + 1):
        if tuple(indices[-width:]) == tuple(indices[-2 * width : -width]):
            return True
    return False


@final
class _SeededEstimator(Estimator):
    """Assign a reproducible child seed to every delegated request."""

    __slots__ = ("_estimator", "_rng")

    def __init__(self, estimator: Estimator, seed: int) -> None:
        self._estimator = estimator
        self._rng = np.random.default_rng(seed)

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        """Delegate requests after replacing seeds with fresh child seeds."""
        seeded_requests: list[EstimatorRequest] = []
        for request in requests:
            if not isinstance(request, EstimatorRequest):
                raise TypeError("requests must contain EstimatorRequest values")
            request.validate()
            seeded_requests.append(
                EstimatorRequest(
                    request.circuit,
                    request.observable,
                    request.parameter_bindings,
                    request.shots,
                    int(self._rng.integers(0, 2**32, dtype=np.uint64)),
                )
            )
        return self._estimator.run(seeded_requests)


class _AdaptiveAnsatz(Ansatz):
    """Compose a fixed reference circuit with selected generator evolutions."""

    __slots__ = ("_initial_circuit", "_operators", "_strategy")

    def __init__(
        self,
        initial_circuit: Circuit,
        operators: Sequence[Hamiltonian],
        strategy: EvolutionStrategy,
    ) -> None:
        self._initial_circuit = copy.copy(initial_circuit)
        self._operators = tuple(operator.copy() for operator in operators)
        self._strategy = copy.copy(strategy)
        self.validate()

    def validate(self) -> None:
        """Validate the reference circuit and selected evolution sequence."""
        self._initial_circuit.validate()
        if self._initial_circuit.used_symbols:
            raise ValueError("adaptive reference circuit must be fully bound")
        if self._operators:
            EvolvedOperatorAnsatz(
                self._operators,
                strategy=copy.copy(self._strategy),
                remove_identity_operators=False,
            ).validate()

    def build_circuit(self, prefix: str) -> Circuit:
        """Build an independent adaptive circuit with stable parameter names."""
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        self.validate()
        circuit = copy.copy(self._initial_circuit)
        if self._operators:
            evolved = EvolvedOperatorAnsatz(
                self._operators,
                strategy=copy.copy(self._strategy),
                remove_identity_operators=False,
            ).build_circuit(prefix)
            circuit.compose(evolved, circuit.qubits)
        return circuit

    def num_parameters(self) -> int:
        """Return one adaptive parameter per selected generator."""
        return len(self._operators)

    def num_qubits(self) -> int:
        """Return the reference-register width."""
        return self._initial_circuit.num_qubits


@final
class AdaptVQE:
    """Grow and reoptimize a variational ansatz from a generator pool.

    For selected Hermitian generators ``G_j``, the circuit appends
    ``exp(-i theta_j G_j)`` in selection order. At each adaptive step the
    complete pool gradient ``i <psi|[G_j, H]|psi>`` is evaluated, the largest
    absolute component is appended, and every active parameter is
    reoptimized. The primary convergence test is the Euclidean norm of the
    complete pool gradient, as in Ref. [1].

    Args:
        estimator: Expectation estimator used for gradients and VQE.
        optimizer: Optimizer used for every full-parameter reoptimization.
        operator_pool: Nonempty equal-width Hermitian generator sequence.
        initial_state: Optional fixed zero-parameter reference preparation.
        gradient_threshold: Positive full-gradient norm threshold.
        eigenvalue_threshold: Optional non-negative accepted-energy threshold.
        max_iterations: Positive adaptive-step limit, or ``None``.
        evolution_strategy: Generator-evolution strategy; exact by default.
        prefix: Parameter-name prefix for adaptive factors.

    Raises:
        TypeError: If a dependency or option has the wrong type.
        ValueError: If a pool, threshold, width, or evolution is invalid.
    """

    __slots__ = (
        "_eigenvalue_threshold",
        "_estimator",
        "_gradient_threshold",
        "_initial_circuit",
        "_max_iterations",
        "_operator_pool",
        "_optimizer",
        "_prefix",
        "_strategy",
    )

    def __init__(
        self,
        estimator: Estimator,
        optimizer: Optimizer,
        operator_pool: Sequence[Hamiltonian | PauliString],
        *,
        initial_state: StatePreparation | None = None,
        gradient_threshold: float = 1e-5,
        eigenvalue_threshold: float | None = None,
        max_iterations: int | None = 50,
        evolution_strategy: EvolutionStrategy | None = None,
        prefix: str = "adapt",
    ) -> None:
        if not isinstance(estimator, Estimator):
            raise TypeError("estimator must be an Estimator")
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be an Optimizer")
        pool = _snapshot_pool(operator_pool)
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        if max_iterations is not None:
            if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
                raise TypeError("max_iterations must be an integer or None")
            if max_iterations <= 0:
                raise ValueError("max_iterations must be positive")
        strategy = (
            EvolutionStrategy.exact()
            if evolution_strategy is None
            else evolution_strategy
        )
        if not isinstance(strategy, EvolutionStrategy):
            raise TypeError("evolution_strategy must be an EvolutionStrategy or None")
        initial_circuit = snapshot_fixed_initial_state(
            initial_state,
            num_qubits=pool[0].num_qubits,
            prefix="adapt_initial",
        )
        if initial_circuit is None:
            initial_circuit = Circuit(pool[0].num_qubits)
        for operator in pool:
            EvolvedOperatorAnsatz(
                operator,
                strategy=copy.copy(strategy),
                remove_identity_operators=False,
            ).build_circuit("adapt_validation")

        self._estimator = estimator
        self._optimizer = optimizer
        self._operator_pool = pool
        self._initial_circuit = initial_circuit
        self._gradient_threshold = normalize_positive_real(
            gradient_threshold, "gradient_threshold"
        )
        self._eigenvalue_threshold = (
            None
            if eigenvalue_threshold is None
            else normalize_nonnegative_real(
                eigenvalue_threshold, "eigenvalue_threshold"
            )
        )
        self._max_iterations = max_iterations
        self._strategy = copy.copy(strategy)
        self._prefix = prefix

    def _run_vqe(
        self,
        estimator: Estimator,
        target: Hamiltonian,
        selected: Sequence[Hamiltonian],
        initial_point: dict[str, float],
        shots: int | None,
    ) -> VQEResult:
        ansatz = _AdaptiveAnsatz(self._initial_circuit, selected, self._strategy)
        return VQE(
            estimator,
            ansatz,
            self._optimizer,
            prefix=self._prefix,
        ).compute_minimum_eigenvalue(target, initial_point, shots=shots)

    def _gradients(
        self,
        estimator: Estimator,
        target: Hamiltonian,
        circuit: Circuit,
        shots: int | None,
    ) -> tuple[float, ...]:
        values = [0.0 for _ in self._operator_pool]
        pending: list[tuple[int, EstimatorRequest]] = []
        for index, generator in enumerate(self._operator_pool):
            observable = _commutator_observable(generator, target)
            if observable.num_terms == 0:
                continue
            pending.append((index, EstimatorRequest(circuit, observable, {}, shots)))
        if pending:
            requests = [request for _, request in pending]
            results = require_estimator_results(estimator.run(requests), requests)
            for (index, _), result in zip(pending, results, strict=True):
                values[index] = result.expectation
        return tuple(values)

    def compute_minimum_eigenvalue(
        self,
        operator: Hamiltonian | PauliString,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> AdaptVQEResult:
        """Run adaptive ansatz construction and return the final VQE state.

        Finite-shot gradients are statistical estimates; threshold termination
        does not by itself provide a confidence guarantee.

        Args:
            operator: Finite Hermitian target with the pool register width.
            shots: ``None`` for exact estimation or a positive shot count.
            seed: Optional master seed for all finite-shot requests. Equal
                seeds reproduce the request-seed stream when the optimizer
                follows the same evaluation path and the estimator honors
                request seeds.

        Returns:
            Final bound state, optimizer metadata, and adaptive diagnostics.

        Raises:
            TypeError: If inputs or component results have invalid types.
            ValueError: If widths, shots, or component results are invalid.
        """
        target = _snapshot_operator(operator, name="operator")
        if target.num_qubits != self._operator_pool[0].num_qubits:
            raise ValueError("operator and operator_pool must have equal width")
        run_estimator: Estimator = self._estimator
        if shots is not None or seed is not None:
            run_estimator = _SeededEstimator(
                self._estimator,
                spawn_seeds(seed, 1)[0],
            )

        selected_indices: list[int] = []
        selected: list[Hamiltonian] = []
        current = self._run_vqe(run_estimator, target, selected, {}, shots)
        energies = [current.eigenvalue]
        gradient_norms: list[float] = []
        termination = AdaptVQETermination.MAX_ITERATIONS
        final_max_gradient = 0.0

        iteration = 0
        while self._max_iterations is None or iteration < self._max_iterations:
            gradients = self._gradients(
                run_estimator,
                target,
                current.optimal_circuit,
                shots,
            )
            gradient_norm = math.hypot(*gradients)
            gradient_norms.append(gradient_norm)
            best_index = max(
                range(len(gradients)), key=lambda index: abs(gradients[index])
            )
            final_max_gradient = abs(gradients[best_index])
            if gradient_norm <= self._gradient_threshold:
                termination = AdaptVQETermination.GRADIENT_CONVERGED
                break
            candidate_indices = (*selected_indices, best_index)
            if _has_repeating_suffix(candidate_indices):
                termination = AdaptVQETermination.CYCLIC_SEQUENCE
                break

            candidate_selected = (*selected, self._operator_pool[best_index])
            initial_point = dict(current.optimal_parameters)
            initial_point[f"{self._prefix}_{len(selected)}"] = 0.0
            candidate = self._run_vqe(
                run_estimator,
                target,
                candidate_selected,
                initial_point,
                shots,
            )
            if (
                self._eigenvalue_threshold is not None
                and abs(candidate.eigenvalue - current.eigenvalue)
                <= self._eigenvalue_threshold
            ):
                termination = AdaptVQETermination.ENERGY_CONVERGED
                break

            selected_indices.append(best_index)
            selected.append(self._operator_pool[best_index])
            current = candidate
            energies.append(current.eigenvalue)
            iteration += 1
        else:
            gradients = self._gradients(
                run_estimator,
                target,
                current.optimal_circuit,
                shots,
            )
            gradient_norms.append(math.hypot(*gradients))
            final_max_gradient = max(abs(value) for value in gradients)

        return AdaptVQEResult(
            current.optimal_value,
            current.optimal_parameters,
            current.optimal_circuit,
            current.optimizer_result,
            selected_operator_indices=tuple(selected_indices),
            gradient_norms=tuple(gradient_norms),
            eigenvalue_history=tuple(energies),
            final_max_gradient=final_max_gradient,
            termination=termination,
        )


__all__ = ["AdaptVQE", "AdaptVQEResult", "AdaptVQETermination"]
