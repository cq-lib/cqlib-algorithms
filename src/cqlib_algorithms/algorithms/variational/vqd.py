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

"""Variational quantum deflation for low-energy eigenstate estimation.

References
----------
[1] O. Higgott, D. Wang, and S. Brierley, "Variational Quantum Computation of
    Excited States," Quantum 3, 156, 2019. DOI: 10.22331/q-2019-07-01-156.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType

import numpy as np
from cqlib import Circuit
from cqlib.qis import Hamiltonian, PauliString

from ...operators.state_fidelity import CircuitFidelityRequest, CircuitFidelityResult
from ...protocols.ansatz import Ansatz
from ...protocols.estimator import Estimator, EstimatorRequest, EstimatorResult
from ...protocols.optimizer import Optimizer, OptimizerRequest, OptimizerResult
from ...protocols.sampler import spawn_seeds
from ...protocols.state_fidelity import StateFidelity
from .result import _immutable_parameters

VQDCallback = Callable[[int, Mapping[str, float], float, int], None]


def _normalize_finite_values(
    values: tuple[float, ...],
    name: str,
) -> tuple[float, ...]:
    """Validate and return finite real values in normalized form."""
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must contain real numbers")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{name} must contain only finite values")
        normalized.append(numeric)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class VQDResult:
    """Low-energy variational states and their optimization metadata.

    ``eigenvalues`` contains the unpenalized Hamiltonian expectations at the
    final iterates. ``optimal_values`` contains the deflated objectives seen
    by the optimizer. Parameter mappings are immutable defensive copies, but
    the fully bound Core circuits remain mutable despite the frozen fields.

    Args:
        eigenvalues: Unpenalized final Hamiltonian expectations in level order.
        optimal_values: Deflated objectives reported by the optimizers.
        optimal_parameters: Final immutable parameter mappings for each level.
        optimal_circuits: Fully bound circuits corresponding to the levels.
        optimizer_results: Optimizer metadata in level order.
        auxiliary_results: Auxiliary estimates arranged by level and operator.
        betas: Positive penalties associated with preceding reference states.

    Raises:
        TypeError: If a result field has the wrong type.
        ValueError: If lengths, values, circuits, or optimizer metadata are
            inconsistent.
    """

    eigenvalues: tuple[float, ...]
    optimal_values: tuple[float, ...]
    optimal_parameters: tuple[Mapping[str, float], ...]
    optimal_circuits: tuple[Circuit, ...]
    optimizer_results: tuple[OptimizerResult, ...]
    auxiliary_results: tuple[tuple[EstimatorResult, ...], ...]
    betas: tuple[float, ...]

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        eigenvalues = _normalize_finite_values(self.eigenvalues, "eigenvalues")
        optimal_values = _normalize_finite_values(
            self.optimal_values,
            "optimal_values",
        )
        betas = _normalize_finite_values(self.betas, "betas")
        if not eigenvalues:
            raise ValueError("eigenvalues must not be empty")
        if any(beta <= 0.0 for beta in betas):
            raise ValueError("betas must be positive")
        object.__setattr__(self, "eigenvalues", eigenvalues)
        object.__setattr__(self, "optimal_values", optimal_values)
        object.__setattr__(self, "betas", betas)

        state_count = len(eigenvalues)
        fields = (
            (self.optimal_values, "optimal_values"),
            (self.optimal_parameters, "optimal_parameters"),
            (self.optimal_circuits, "optimal_circuits"),
            (self.optimizer_results, "optimizer_results"),
            (self.auxiliary_results, "auxiliary_results"),
        )
        for values, name in fields:
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be a tuple")
            if len(values) != state_count:
                raise ValueError(f"{name} must have one entry per eigenvalue")
        if len(betas) != state_count - 1:
            raise ValueError("betas must contain one value per deflation reference")

        immutable_parameters = []
        for position, parameters in enumerate(self.optimal_parameters):
            immutable = _immutable_parameters(parameters)
            optimizer_result = self.optimizer_results[position]
            if not isinstance(optimizer_result, OptimizerResult):
                raise TypeError("optimizer_results must contain OptimizerResult values")
            if tuple(immutable) != optimizer_result.parameter_names:
                raise ValueError(
                    "optimal parameter names must match the optimizer result"
                )
            if tuple(immutable.values()) != optimizer_result.optimal_parameters:
                raise ValueError(
                    "optimal parameter values must match the optimizer result"
                )
            if optimal_values[position] != optimizer_result.optimal_value:
                raise ValueError("optimal_values must match optimizer results")
            immutable_parameters.append(immutable)
        object.__setattr__(self, "optimal_parameters", tuple(immutable_parameters))

        for circuit in self.optimal_circuits:
            if not isinstance(circuit, Circuit):
                raise TypeError("optimal_circuits must contain cqlib.Circuit values")
            circuit.validate()
            if circuit.used_symbols:
                raise ValueError("optimal circuits must be fully bound")

        auxiliary_width: int | None = None
        for row in self.auxiliary_results:
            if not isinstance(row, tuple):
                raise TypeError("auxiliary result rows must be tuples")
            if any(not isinstance(value, EstimatorResult) for value in row):
                raise TypeError(
                    "auxiliary result rows must contain EstimatorResult values"
                )
            if auxiliary_width is None:
                auxiliary_width = len(row)
            elif len(row) != auxiliary_width:
                raise ValueError("auxiliary result rows must have equal lengths")

    def __repr__(self) -> str:
        """Summarize states and termination statuses without printing circuits."""
        statuses = tuple(result.status.name for result in self.optimizer_results)
        return (
            f"VQDResult(num_states={len(self.eigenvalues)}, "
            f"eigenvalues={self.eigenvalues!r}, num_betas={len(self.betas)}, "
            f"statuses={statuses!r})"
        )


def _snapshot_result_batch(
    results: object,
    expected: int,
    *,
    source: str,
) -> list[EstimatorResult]:
    """Validate and return a defensive Estimator result snapshot."""
    if not isinstance(results, list):
        raise TypeError(f"{source} must return a list of EstimatorResult values")
    if len(results) != expected:
        raise ValueError(f"{source} must return one result per request")
    if any(not isinstance(result, EstimatorResult) for result in results):
        raise TypeError(f"{source} must return EstimatorResult values")
    return list(results)


class _DeflatedEstimator(Estimator):
    """Add overlap penalties to one width-preserving estimator objective.

    Gradient and optimizer components may evaluate shifted parameter points,
    but the resulting requests must retain the ansatz register width.  A
    transformation that appends ancillas cannot be compared with the stored
    reference states and is rejected instead of silently omitting deflation.
    """

    __slots__ = (
        "_betas",
        "_callback",
        "_estimator",
        "_fidelity",
        "_fidelity_rng",
        "_fidelity_shots",
        "_level",
        "_references",
    )

    def __init__(
        self,
        estimator: Estimator,
        fidelity: StateFidelity[CircuitFidelityRequest, CircuitFidelityResult],
        references: tuple[Circuit, ...],
        betas: tuple[float, ...],
        fidelity_shots: int,
        fidelity_seed: int,
        level: int,
        callback: Callable[[Mapping[str, float], float, int], None] | None,
    ) -> None:
        self._estimator = estimator
        self._fidelity = fidelity
        self._references = references
        self._betas = betas
        self._fidelity_shots = fidelity_shots
        self._fidelity_rng = np.random.default_rng(fidelity_seed)
        self._level = level
        self._callback = callback

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        """Evaluate energy and all preceding-state overlap penalties."""
        request_list = list(requests)
        if any(not isinstance(request, EstimatorRequest) for request in request_list):
            raise TypeError("requests must contain EstimatorRequest values")
        for request in request_list:
            request.validate()
        if not request_list:
            return []
        if any(
            request.circuit.num_qubits != reference.num_qubits
            for request in request_list
            for reference in self._references
        ):
            raise ValueError(
                "VQD deflation requires estimator transformations to preserve "
                "the ansatz circuit width"
            )

        energy_results = _snapshot_result_batch(
            self._estimator.run(request_list),
            len(request_list),
            source="estimator",
        )
        fidelity_requests: list[CircuitFidelityRequest] = []
        if self._references:
            count = len(request_list) * len(self._references)
            parent = int(self._fidelity_rng.integers(0, 2**32, dtype=np.uint64))
            child_seeds = iter(spawn_seeds(parent, count))
            for request in request_list:
                for reference in self._references:
                    fidelity_requests.append(
                        CircuitFidelityRequest(
                            request.circuit,
                            reference,
                            self._fidelity_shots,
                            request.parameter_bindings,
                            {},
                            next(child_seeds),
                        )
                    )

        fidelity_results: list[CircuitFidelityResult] = []
        if fidelity_requests:
            raw_fidelities = self._fidelity.run(fidelity_requests)
            if not isinstance(raw_fidelities, list):
                raise TypeError(
                    "state fidelity must return a list of CircuitFidelityResult values"
                )
            if len(raw_fidelities) != len(fidelity_requests):
                raise ValueError("state fidelity must return one result per request")
            if any(
                not isinstance(result, CircuitFidelityResult)
                for result in raw_fidelities
            ):
                raise TypeError(
                    "state fidelity must return CircuitFidelityResult values"
                )
            fidelity_results = raw_fidelities

        results: list[EstimatorResult] = []
        reference_count = len(self._references)
        for position, (request, energy) in enumerate(
            zip(request_list, energy_results, strict=True)
        ):
            start = position * reference_count
            overlaps = fidelity_results[start : start + reference_count]
            objective = energy.expectation + sum(
                beta * overlap.fidelity
                for beta, overlap in zip(self._betas, overlaps, strict=True)
            )
            variances = [energy.variance, *(overlap.variance for overlap in overlaps)]
            if any(variance is None for variance in variances):
                variance = None
            else:
                energy_variance = energy.variance
                if energy_variance is None:
                    raise RuntimeError("variance validation lost a value")
                variance = energy_variance + sum(
                    beta * beta * float(overlap.variance)
                    for beta, overlap in zip(self._betas, overlaps, strict=True)
                    if overlap.variance is not None
                )
            result = EstimatorResult(objective, variance)
            results.append(result)
            if self._callback is not None:
                self._callback(
                    MappingProxyType(dict(request.parameter_bindings)),
                    objective,
                    self._level,
                )
        return results


def _normalize_betas(
    betas: Sequence[float] | None,
    k: int,
) -> tuple[float, ...] | None:
    """Validate and return immutable explicit deflation penalties."""
    if betas is None:
        return None
    if isinstance(betas, (str, bytes)) or not isinstance(betas, Sequence):
        raise TypeError("betas must be a sequence of real numbers or None")
    if len(betas) != k - 1:
        raise ValueError("betas must contain exactly k - 1 values")
    normalized: list[float] = []
    for beta in betas:
        if isinstance(beta, bool) or not isinstance(beta, Real):
            raise TypeError("betas must contain real numbers")
        numeric = float(beta)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError("betas must contain finite positive values")
        normalized.append(numeric)
    return tuple(normalized)


def _operator_bound(operator: Hamiltonian | PauliString) -> float:
    """Return the coefficient one-norm used for automatic penalties."""
    if isinstance(operator, PauliString):
        return 1.0
    return math.fsum(
        abs(complex(coefficient) * pauli.phase.to_complex())
        for pauli, coefficient in operator.terms
    )


def _snapshot_initial_points(
    value: Mapping[str, float] | Sequence[Mapping[str, float]],
    k: int,
) -> tuple[Mapping[str, float], ...]:
    """Validate and return defensive initial-point snapshots."""
    if isinstance(value, Mapping):
        snapshot = MappingProxyType(dict(value))
        return tuple(snapshot for _ in range(k))
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("initial_points must be a mapping or a sequence of mappings")
    if len(value) != k:
        raise ValueError("initial_points must contain exactly k mappings")
    snapshots: list[Mapping[str, float]] = []
    for point in value:
        if not isinstance(point, Mapping):
            raise TypeError("initial_points must contain mappings")
        snapshots.append(MappingProxyType(dict(point)))
    return tuple(snapshots)


class VQD:
    """Estimate low-energy states by sequential variational deflation.

    For level ``s``, the algorithm minimizes the Hamiltonian expectation plus
    positive fidelity penalties against all previously found states [1]. The
    reported eigenvalue is the unpenalized final Hamiltonian expectation;
    convergence and exact-state guarantees remain properties of the chosen
    ansatz, optimizer, shot budget, and penalties.

    Estimator transformations used by an optimizer must preserve the ansatz
    circuit width. Parameter-shift, finite-difference, and SPSA gradients meet
    this requirement; the current ancilla-based linear-combination gradient
    does not and is rejected once deflation references exist.

    Args:
        estimator: Component used for energies and auxiliary observables.
        fidelity: Component used for squared state overlaps.
        ansatz: Parameterized circuit family shared by all levels.
        optimizer: Classical minimizer used independently at every level.
        k: Positive number of low-energy states to estimate.
        betas: Optional ``k - 1`` finite positive penalties. ``None`` derives
            a conservative common value from the operator coefficient norm.
        prefix: Parameter-name prefix passed to the ansatz.
        callback: Optional function receiving evaluation count, immutable
            parameters, deflated objective, and zero-based level.

    Raises:
        TypeError: If a dependency or option has the wrong type.
        ValueError: If an option violates the VQD contract.
    """

    __slots__ = (
        "_ansatz",
        "_betas",
        "_callback",
        "_estimator",
        "_fidelity",
        "_k",
        "_optimizer",
        "_prefix",
    )

    def __init__(
        self,
        estimator: Estimator,
        fidelity: StateFidelity[CircuitFidelityRequest, CircuitFidelityResult],
        ansatz: Ansatz,
        optimizer: Optimizer,
        *,
        k: int = 2,
        betas: Sequence[float] | None = None,
        prefix: str = "theta",
        callback: VQDCallback | None = None,
    ) -> None:
        if not isinstance(estimator, Estimator):
            raise TypeError("estimator must be an Estimator")
        if not isinstance(fidelity, StateFidelity):
            raise TypeError("fidelity must be a StateFidelity")
        if not isinstance(ansatz, Ansatz):
            raise TypeError("ansatz must be an Ansatz")
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be an Optimizer")
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer")
        if k <= 0:
            raise ValueError("k must be positive")
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a str")
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable or None")

        self._estimator = estimator
        self._fidelity = fidelity
        self._ansatz = ansatz
        self._optimizer = optimizer
        self._k = k
        self._betas = _normalize_betas(betas, k)
        self._prefix = prefix
        self._callback = callback

    @property
    def estimator(self) -> Estimator:
        """The injected Hamiltonian expectation estimator."""
        return self._estimator

    @property
    def fidelity(
        self,
    ) -> StateFidelity[CircuitFidelityRequest, CircuitFidelityResult]:
        """The injected squared-state-overlap estimator."""
        return self._fidelity

    @property
    def ansatz(self) -> Ansatz:
        """The variational circuit component shared by all levels."""
        return self._ansatz

    @property
    def optimizer(self) -> Optimizer:
        """The classical minimizer used independently at every level."""
        return self._optimizer

    @property
    def k(self) -> int:
        """The number of requested low-energy states."""
        return self._k

    @property
    def betas(self) -> tuple[float, ...] | None:
        """The explicit penalties, or ``None`` when they are operator-derived."""
        return self._betas

    @property
    def prefix(self) -> str:
        """The prefix passed to the ansatz when constructing symbols."""
        return self._prefix

    def compute_eigenvalues(
        self,
        operator: Hamiltonian | PauliString,
        initial_points: Mapping[str, float] | Sequence[Mapping[str, float]],
        *,
        shots: int | None = None,
        fidelity_shots: int = 1024,
        seed: int | None = None,
        auxiliary_operators: Sequence[Hamiltonian | PauliString] = (),
    ) -> VQDResult:
        """Estimate sequential low-energy states and auxiliary expectations.

        Args:
            operator: Finite Hermitian Hamiltonian whose states are sought.
            initial_points: One exact parameter mapping reused at every level,
                or ``k`` independent mappings in level order.
            shots: Exact energy evaluation when ``None`` or a positive count.
            fidelity_shots: Positive overlap-estimation shot count.
            seed: Non-negative parent seed for reproducible request seeds.
            auxiliary_operators: Ordered Hermitian observables evaluated for
                every final state.

        Returns:
            Eigenvalue approximations, final states, optimizer metadata, and
            auxiliary estimates in level order.

        Raises:
            TypeError: If an input or component response has the wrong type.
            ValueError: If an input, component response, or final state
                violates the VQD contract.
        """
        if isinstance(fidelity_shots, bool) or not isinstance(fidelity_shots, int):
            raise TypeError("fidelity_shots must be an integer")
        if fidelity_shots <= 0:
            raise ValueError("fidelity_shots must be positive")
        if isinstance(auxiliary_operators, (str, bytes)) or not isinstance(
            auxiliary_operators, Sequence
        ):
            raise TypeError("auxiliary_operators must be a sequence")
        auxiliaries = tuple(auxiliary_operators)

        circuit = self._ansatz.build_circuit(self._prefix)
        circuit.validate()
        parameter_names = tuple(circuit.used_symbols)
        if self._k > 2**circuit.num_qubits:
            raise ValueError("k must not exceed the circuit Hilbert-space dimension")
        if self._k > 1 and not parameter_names:
            raise ValueError("k greater than one requires an active ansatz parameter")
        points = _snapshot_initial_points(initial_points, self._k)

        # Construction validates the operator, width, shots, seed, and points.
        request_count = 2 * self._k + self._k * len(auxiliaries)
        seeds = spawn_seeds(seed, request_count)
        energy_seeds = seeds[: self._k]
        fidelity_seeds = seeds[self._k : 2 * self._k]
        auxiliary_seeds = seeds[2 * self._k :]
        base_requests = [
            EstimatorRequest(circuit, operator, point, shots, energy_seeds[level])
            for level, point in enumerate(points)
        ]
        for auxiliary in auxiliaries:
            EstimatorRequest(circuit, auxiliary, points[0], shots, energy_seeds[0])

        if self._betas is None:
            bound = _operator_bound(operator)
            if self._k > 1 and (not math.isfinite(bound) or bound <= 0.0):
                raise ValueError(
                    "a zero operator requires explicit positive betas when k > 1"
                )
            betas = tuple(10.0 * bound for _ in range(self._k - 1))
        else:
            betas = self._betas

        evaluation_count = 0

        def notify(
            parameters: Mapping[str, float],
            value: float,
            level: int,
        ) -> None:
            nonlocal evaluation_count
            evaluation_count += 1
            if self._callback is not None:
                self._callback(evaluation_count, parameters, value, level)

        eigenvalues: list[float] = []
        optimal_values: list[float] = []
        optimal_parameters: list[Mapping[str, float]] = []
        optimal_circuits: list[Circuit] = []
        optimizer_results: list[OptimizerResult] = []

        for level, base_request in enumerate(base_requests):
            deflated = _DeflatedEstimator(
                self._estimator,
                self._fidelity,
                tuple(optimal_circuits),
                betas[:level],
                fidelity_shots,
                fidelity_seeds[level],
                level,
                notify,
            )
            optimizer_request = OptimizerRequest(base_request, parameter_names)
            raw_optimizer_results = self._optimizer.run(
                deflated,
                [optimizer_request],
            )
            if not isinstance(raw_optimizer_results, list):
                raise TypeError(
                    "optimizer must return a list of OptimizerResult values"
                )
            if len(raw_optimizer_results) != 1:
                raise ValueError(
                    "optimizer must return exactly one result per VQD level"
                )
            optimizer_result = raw_optimizer_results[0]
            if not isinstance(optimizer_result, OptimizerResult):
                raise TypeError("optimizer must return OptimizerResult values")
            if optimizer_result.parameter_names != parameter_names:
                raise ValueError(
                    "optimizer result parameter names must match the VQD request"
                )

            parameters = MappingProxyType(
                dict(
                    zip(
                        optimizer_result.parameter_names,
                        optimizer_result.optimal_parameters,
                        strict=True,
                    )
                )
            )
            bound_circuit = circuit.assign_parameters(dict(parameters))
            bound_circuit.validate()
            if bound_circuit.used_symbols:
                raise ValueError(
                    "optimizer result did not fully bind the ansatz circuit"
                )

            energy = _snapshot_result_batch(
                self._estimator.run(
                    [
                        EstimatorRequest(
                            circuit,
                            operator,
                            parameters,
                            shots,
                            energy_seeds[level],
                        )
                    ]
                ),
                1,
                source="estimator",
            )[0]
            eigenvalues.append(energy.expectation)
            optimal_values.append(optimizer_result.optimal_value)
            optimal_parameters.append(parameters)
            optimal_circuits.append(bound_circuit)
            optimizer_results.append(optimizer_result)

        auxiliary_results: tuple[tuple[EstimatorResult, ...], ...]
        if not auxiliaries:
            auxiliary_results = tuple(() for _ in range(self._k))
        else:
            auxiliary_requests: list[EstimatorRequest] = []
            seed_position = 0
            for bound_circuit in optimal_circuits:
                for auxiliary in auxiliaries:
                    auxiliary_requests.append(
                        EstimatorRequest(
                            bound_circuit,
                            auxiliary,
                            {},
                            shots,
                            auxiliary_seeds[seed_position],
                        )
                    )
                    seed_position += 1
            flat_auxiliary = _snapshot_result_batch(
                self._estimator.run(auxiliary_requests),
                len(auxiliary_requests),
                source="estimator",
            )
            width = len(auxiliaries)
            auxiliary_results = tuple(
                tuple(flat_auxiliary[start : start + width])
                for start in range(0, len(flat_auxiliary), width)
            )

        return VQDResult(
            tuple(eigenvalues),
            tuple(optimal_values),
            tuple(optimal_parameters),
            tuple(optimal_circuits),
            tuple(optimizer_results),
            auxiliary_results,
            betas,
        )
