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

"""Projected variational real-time evolution of parameterized circuits.

References
----------
[1] S. Barison, F. Vicentini, and G. Carleo, "An efficient quantum algorithm
    for the time evolution of parameterized circuits," Quantum 5, 512, 2021.
    DOI: 10.22331/q-2021-07-28-512.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from types import MappingProxyType
from typing import final

import numpy as np
from cqlib import Circuit
from cqlib.circuit.ansatz import EvolutionStrategy
from cqlib.device import ExecutionResult
from cqlib.qis import Hamiltonian, PauliString

from ..._utils.hamiltonian import (
    snapshot_nonzero_hermitian_hamiltonian,
)
from ..._utils.numeric import normalize_finite_real, normalize_positive_real
from ...operators.ansatz import EvolvedOperatorAnsatz
from ...operators.state_fidelity import (
    CircuitFidelityRequest,
    CircuitFidelityResult,
    ComputeUncomputeFidelity,
)
from ...protocols import Ansatz, Estimator, EstimatorRequest, EstimatorResult, Optimizer
from ...protocols.estimator import require_estimator_results
from ...protocols.optimizer import OptimizerRequest, OptimizerResult
from ...protocols.sampler import (
    Sampler,
    SamplerRequest,
    require_sampler_results,
    spawn_seeds,
)
from ._common import (
    _snapshot_parameters,
    _validate_positive_integer,
)


def _validate_optional_positive_integer(value: int | None, name: str) -> None:
    """Validate an optional positive integer."""
    if value is None:
        return
    _validate_positive_integer(value, name)


def _snapshot_hamiltonian(
    operator: Hamiltonian | PauliString,
    *,
    name: str,
) -> Hamiltonian:
    """Validate and return a Hermitian Hamiltonian snapshot."""
    if isinstance(operator, PauliString):
        operator = Hamiltonian.from_pauli(operator.copy())
    if not isinstance(operator, Hamiltonian):
        raise TypeError(f"{name} must be a Hamiltonian or PauliString")
    if operator.num_qubits <= 0:
        raise ValueError(f"{name} must act on at least one qubit")
    return snapshot_nonzero_hermitian_hamiltonian(operator, name=name)


def _snapshot_auxiliary_operators(
    operators: Mapping[str, Hamiltonian | PauliString] | None,
    *,
    num_qubits: int,
) -> Mapping[str, Hamiltonian]:
    """Validate and return immutable auxiliary-observable snapshots."""
    if operators is None:
        return MappingProxyType({})
    if not isinstance(operators, Mapping):
        raise TypeError("auxiliary_operators must be a mapping or None")

    snapshots: dict[str, Hamiltonian] = {}
    for name, operator in operators.items():
        if not isinstance(name, str):
            raise TypeError("auxiliary operator names must be strings")
        if not name:
            raise ValueError("auxiliary operator names must not be empty")
        snapshot = _snapshot_hamiltonian(
            operator, name=f"auxiliary_operators[{name!r}]"
        )
        if snapshot.num_qubits != num_qubits:
            raise ValueError("auxiliary operators must match the evolution width")
        snapshots[name] = snapshot
    return MappingProxyType(snapshots)


@dataclass(frozen=True, slots=True, repr=False)
class PVQDResult:
    """Projected variational trajectory and per-step diagnostics.

    ``times`` and ``parameters`` include the initial state. ``fidelities`` and
    ``optimizer_results`` contain one entry per time step. The product of the
    step fidelities is exposed as ``estimated_fidelity``; it is a projection
    diagnostic, not a rigorous lower bound on the final-state fidelity.

    Args:
        times: Strictly increasing integration times beginning at zero.
        parameters: Parameter mappings at the corresponding times.
        fidelities: Estimated projection fidelity of every time step.
        optimizer_results: Optimizer metadata for every projection.
        evolved_circuit: Fully bound ansatz at the final time.
        observables: Optional named expectation histories at all times.

    Raises:
        TypeError: If a field or nested value has the wrong type.
        ValueError: If trajectory lengths, values, or metadata are inconsistent.
    """

    times: tuple[float, ...]
    parameters: tuple[Mapping[str, float], ...]
    fidelities: tuple[float, ...]
    optimizer_results: tuple[OptimizerResult, ...] = field(repr=False)
    evolved_circuit: Circuit = field(repr=False)
    observables: Mapping[str, tuple[EstimatorResult, ...]] = field(default_factory=dict)

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.times, tuple):
            raise TypeError("times must be a tuple")
        times = tuple(normalize_finite_real(value, "time") for value in self.times)
        if len(times) < 2:
            raise ValueError(
                "times must contain the initial and at least one final time"
            )
        if times[0] != 0.0:
            raise ValueError("times must begin at zero")
        if any(right <= left for left, right in pairwise(times)):
            raise ValueError("times must be strictly increasing")
        object.__setattr__(self, "times", times)

        if not isinstance(self.parameters, tuple):
            raise TypeError("parameters must be a tuple")
        parameters = tuple(_snapshot_parameters(entry) for entry in self.parameters)
        if len(parameters) != len(times):
            raise ValueError("parameters must contain one entry per time")
        names = tuple(parameters[0])
        if any(tuple(entry) != names for entry in parameters[1:]):
            raise ValueError("parameter history entries must preserve names and order")
        object.__setattr__(self, "parameters", parameters)

        if not isinstance(self.fidelities, tuple):
            raise TypeError("fidelities must be a tuple")
        fidelities = tuple(
            normalize_finite_real(value, "fidelity") for value in self.fidelities
        )
        if len(fidelities) != len(times) - 1:
            raise ValueError("fidelities must contain one entry per time step")
        if any(value < 0.0 or value > 1.0 for value in fidelities):
            raise ValueError("fidelities must lie between zero and one")
        object.__setattr__(self, "fidelities", fidelities)

        if not isinstance(self.optimizer_results, tuple):
            raise TypeError("optimizer_results must be a tuple")
        if len(self.optimizer_results) != len(fidelities):
            raise ValueError("optimizer_results must contain one entry per time step")
        for index, optimizer_result in enumerate(self.optimizer_results):
            if not isinstance(optimizer_result, OptimizerResult):
                raise TypeError("optimizer_results must contain OptimizerResult values")
            if optimizer_result.parameter_names != names:
                raise ValueError(
                    "optimizer result parameter names must match the trajectory"
                )
            if optimizer_result.optimal_parameters != tuple(
                parameters[index + 1].values()
            ):
                raise ValueError(
                    "optimizer result parameters must match the trajectory"
                )

        if not isinstance(self.evolved_circuit, Circuit):
            raise TypeError("evolved_circuit must be a cqlib.Circuit")
        self.evolved_circuit.validate()
        if self.evolved_circuit.used_symbols:
            raise ValueError("evolved_circuit must be fully bound")

        if not isinstance(self.observables, Mapping):
            raise TypeError("observables must be a mapping")
        observables: dict[str, tuple[EstimatorResult, ...]] = {}
        for name, history in self.observables.items():
            if not isinstance(name, str):
                raise TypeError("observable names must be strings")
            if not name:
                raise ValueError("observable names must not be empty")
            if not isinstance(history, tuple):
                raise TypeError("observable histories must be tuples")
            if len(history) != len(times):
                raise ValueError(
                    "observable histories must contain one result per time"
                )
            if any(not isinstance(result, EstimatorResult) for result in history):
                raise TypeError(
                    "observable histories must contain EstimatorResult values"
                )
            observables[name] = history
        object.__setattr__(self, "observables", MappingProxyType(observables))

    @property
    def final_parameters(self) -> Mapping[str, float]:
        """Return the immutable parameter mapping at the final time."""
        return self.parameters[-1]

    @property
    def estimated_fidelity(self) -> float:
        """Return the product of all estimated step fidelities."""
        return math.prod(self.fidelities)

    @property
    def estimated_error(self) -> float:
        """Return one minus the product of estimated step fidelities."""
        return 1.0 - self.estimated_fidelity

    def __repr__(self) -> str:
        """Summarize the trajectory without printing circuits or parameters."""
        return (
            f"PVQDResult(final_time={self.times[-1]!r}, "
            f"num_timesteps={len(self.fidelities)}, "
            f"num_parameters={len(self.final_parameters)}, "
            f"estimated_fidelity={self.estimated_fidelity!r})"
        )


@final
class _SeededSampler(Sampler):
    """Assign a reproducible child seed to every delegated sample request."""

    __slots__ = ("_rng", "_sampler")

    def __init__(self, sampler: Sampler, seed: int) -> None:
        self._sampler = sampler
        self._rng = np.random.default_rng(seed)

    def run(
        self,
        requests: Sequence[SamplerRequest],
        /,
    ) -> list[ExecutionResult]:
        """Delegate requests after assigning independent child seeds."""
        seeded: list[SamplerRequest] = []
        for request in requests:
            if not isinstance(request, SamplerRequest):
                raise TypeError("requests must contain SamplerRequest values")
            request.validate()
            seeded.append(
                SamplerRequest(
                    request.circuit,
                    request.qubits,
                    request.shots,
                    int(self._rng.integers(0, 2**32, dtype=np.uint64)),
                )
            )
        return require_sampler_results(self._sampler.run(seeded), seeded)


@final
class _ProjectionEstimator(Estimator):
    """Adapt compute-uncompute fidelity to a step-infidelity objective."""

    __slots__ = ("_fidelity", "_shots", "_target")

    def __init__(
        self,
        sampler: Sampler,
        target: Circuit,
        shots: int,
        seed: int,
    ) -> None:
        self._fidelity = ComputeUncomputeFidelity(_SeededSampler(sampler, seed))
        self._target = copy.copy(target)
        self._shots = shots

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        """Return ``1 - fidelity`` for every candidate circuit request."""
        fidelity_requests: list[CircuitFidelityRequest] = []
        for request in requests:
            if not isinstance(request, EstimatorRequest):
                raise TypeError("requests must contain EstimatorRequest values")
            request.validate()
            if request.shots != self._shots:
                raise ValueError("projection objective requests must preserve shots")
            if request.circuit.num_qubits != self._target.num_qubits:
                raise ValueError("candidate and target circuits must have equal width")
            fidelity_requests.append(
                CircuitFidelityRequest(
                    request.circuit,
                    self._target,
                    self._shots,
                    request.parameter_bindings,
                    {},
                )
            )

        results = self._fidelity.run(fidelity_requests)
        if not isinstance(results, list):
            raise TypeError("state fidelity must return a list")
        if len(results) != len(fidelity_requests):
            raise ValueError("state fidelity must return one result per request")
        estimates: list[EstimatorResult] = []
        for result in results:
            if not isinstance(result, CircuitFidelityResult):
                raise TypeError(
                    "state fidelity must return CircuitFidelityResult values"
                )
            estimates.append(EstimatorResult(1.0 - result.fidelity, result.variance))
        return estimates


def _build_evolution_circuit(
    hamiltonian: Hamiltonian,
    time_step: float,
    strategy: EvolutionStrategy,
) -> Circuit:
    """Build and bind one short-time Hamiltonian evolution circuit."""
    template = EvolvedOperatorAnsatz(
        hamiltonian,
        strategy=copy.copy(strategy),
        remove_identity_operators=False,
    ).build_circuit("pvqd_time")
    names = tuple(template.used_symbols)
    if len(names) != 1:
        raise RuntimeError(
            "Hamiltonian evolution must expose exactly one time parameter"
        )
    bound = template.assign_parameters({names[0]: time_step})
    bound.validate()
    if bound.used_symbols:
        raise RuntimeError("Hamiltonian evolution did not bind its time parameter")
    return bound


def _target_circuit(current: Circuit, evolution: Circuit) -> Circuit:
    """Return the short-time evolution applied after the current state."""
    target = Circuit(current.num_qubits)
    target.compose(current, target.qubits)
    target.compose(evolution, target.qubits)
    target.validate()
    return target


def _single_optimizer_result(
    optimizer: Optimizer,
    estimator: Estimator,
    request: OptimizerRequest,
) -> OptimizerResult:
    """Run one projection optimization and validate its outer contract."""
    results = optimizer.run(estimator, [request])
    if not isinstance(results, list):
        raise TypeError("optimizer must return a list of OptimizerResult values")
    if len(results) != 1:
        raise ValueError("optimizer must return exactly one result per time step")
    result = results[0]
    if not isinstance(result, OptimizerResult):
        raise TypeError("optimizer must return OptimizerResult values")
    if result.parameter_names != request.parameter_names:
        raise ValueError("optimizer result parameter names must match the request")
    if result.optimal_value < 0.0 or result.optimal_value > 1.0:
        raise ValueError("optimizer projection loss must lie between zero and one")
    return result


def _evaluate_observables(
    estimator: Estimator,
    circuits: Sequence[Circuit],
    observables: Mapping[str, Hamiltonian],
    *,
    shots: int | None,
    seeds: Sequence[int],
) -> Mapping[str, tuple[EstimatorResult, ...]]:
    """Evaluate flattened time-major observable requests and restore histories."""
    names = tuple(observables)
    requests: list[EstimatorRequest] = []
    seed_index = 0
    for circuit in circuits:
        for observable in observables.values():
            requests.append(
                EstimatorRequest(circuit, observable, {}, shots, seeds[seed_index])
            )
            seed_index += 1
    results = require_estimator_results(
        estimator.run(requests),
        requests,
        count_error="estimator must return one result per observable request",
    )

    histories: dict[str, list[EstimatorResult]] = {name: [] for name in names}
    for index, result in enumerate(results):
        histories[names[index % len(names)]].append(result)
    return MappingProxyType(
        {name: tuple(history) for name, history in histories.items()}
    )


@final
class PVQD:
    """Approximate real-time evolution by iterative global projection.

    At each time step, the exact or product-formula circuit selected by
    ``evolution_strategy`` acts on the current variational state. The optimizer
    then minimizes ``1 - |<psi(theta_next)|target>|**2`` over every active
    ansatz parameter. The previous parameter increment is used as the next
    warm start. Fidelity is measured by compute-uncompute sampling.

    Args:
        sampler: Sampler used for all projection-fidelity measurements.
        ansatz: Parameterized circuit manifold for the evolving state.
        optimizer: Classical optimizer used independently at every time step.
        estimator: Optional estimator for auxiliary-observable histories.
        evolution_strategy: Core strategy for each short-time evolution.
        prefix: Parameter-name prefix passed to the ansatz.

    Raises:
        TypeError: If a dependency or option has the wrong type.
        ValueError: If the prefix is empty.
    """

    __slots__ = (
        "_ansatz",
        "_estimator",
        "_evolution_strategy",
        "_optimizer",
        "_prefix",
        "_sampler",
    )

    def __init__(
        self,
        sampler: Sampler,
        ansatz: Ansatz,
        optimizer: Optimizer,
        *,
        estimator: Estimator | None = None,
        evolution_strategy: EvolutionStrategy | None = None,
        prefix: str = "theta",
    ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError("sampler must be a Sampler")
        if not isinstance(ansatz, Ansatz):
            raise TypeError("ansatz must be an Ansatz")
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be an Optimizer")
        if estimator is not None and not isinstance(estimator, Estimator):
            raise TypeError("estimator must be an Estimator or None")
        if evolution_strategy is not None and not isinstance(
            evolution_strategy, EvolutionStrategy
        ):
            raise TypeError("evolution_strategy must be an EvolutionStrategy or None")
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        if not prefix:
            raise ValueError("prefix must not be empty")

        self._sampler = sampler
        self._ansatz = ansatz
        self._optimizer = optimizer
        self._estimator = estimator
        self._evolution_strategy = copy.copy(
            EvolutionStrategy.auto(1)
            if evolution_strategy is None
            else evolution_strategy
        )
        self._prefix = prefix

    @property
    def sampler(self) -> Sampler:
        """The sampler used for projection fidelities."""
        return self._sampler

    @property
    def ansatz(self) -> Ansatz:
        """The variational state manifold."""
        return self._ansatz

    @property
    def optimizer(self) -> Optimizer:
        """The optimizer used for every projection step."""
        return self._optimizer

    @property
    def estimator(self) -> Estimator | None:
        """The optional auxiliary-observable estimator."""
        return self._estimator

    @property
    def evolution_strategy(self) -> EvolutionStrategy:
        """Return an independent copy of the short-time evolution strategy."""
        return copy.copy(self._evolution_strategy)

    @property
    def prefix(self) -> str:
        """The ansatz parameter-name prefix."""
        return self._prefix

    def run(
        self,
        hamiltonian: Hamiltonian | PauliString,
        initial_parameters: Mapping[str, float],
        time: float,
        *,
        num_timesteps: int,
        fidelity_shots: int,
        auxiliary_operators: Mapping[str, Hamiltonian | PauliString] | None = None,
        observable_shots: int | None = None,
        seed: int | None = None,
    ) -> PVQDResult:
        """Project each real-time evolution step back onto the ansatz.

        Args:
            hamiltonian: Finite nonempty time-independent Hermitian generator.
            initial_parameters: Exact bindings for active ansatz symbols.
            time: Positive total evolution time.
            num_timesteps: Positive number of equal projection steps.
            fidelity_shots: Positive shots per fidelity evaluation.
            auxiliary_operators: Optional named Hermitian observables.
            observable_shots: Shots per auxiliary estimate, or ``None`` for an
                exact estimator request.
            seed: Optional master seed for all finite-shot requests.

        Returns:
            Time and parameter histories, the final bound circuit, per-step
            fidelities and optimizer results, and requested observables.

        Raises:
            TypeError: If an input or component response has the wrong type.
            ValueError: If dimensions, bindings, values, or component responses
                violate the evolution contract.
        """
        total_time = normalize_positive_real(time, "time")
        _validate_positive_integer(num_timesteps, "num_timesteps")
        _validate_positive_integer(fidelity_shots, "fidelity_shots")
        _validate_optional_positive_integer(observable_shots, "observable_shots")

        snapshot = _snapshot_hamiltonian(hamiltonian, name="hamiltonian")
        template = self._ansatz.build_circuit(self._prefix)
        if not isinstance(template, Circuit):
            raise TypeError("ansatz must build a cqlib.Circuit")
        template.validate()
        if template.num_qubits != snapshot.num_qubits:
            raise ValueError("ansatz and Hamiltonian must have the same qubit count")

        parameter_names = tuple(template.used_symbols)
        identity = PauliString.from_str("I" * template.num_qubits)
        initial_request = EstimatorRequest(
            template,
            identity,
            initial_parameters,
            fidelity_shots,
        )
        current_parameters = {
            name: initial_request.parameter_bindings[name] for name in parameter_names
        }
        current_circuit = template.assign_parameters(current_parameters)
        current_circuit.validate()
        if current_circuit.used_symbols:
            raise ValueError("initial parameters did not fully bind the ansatz")

        auxiliaries = _snapshot_auxiliary_operators(
            auxiliary_operators,
            num_qubits=template.num_qubits,
        )
        if auxiliaries and self._estimator is None:
            raise ValueError("estimator is required for auxiliary operators")

        time_step = total_time / num_timesteps
        if not math.isfinite(time_step) or time_step <= 0.0:
            raise ValueError("time step must be finite and representable as positive")
        evolution = _build_evolution_circuit(
            snapshot,
            time_step,
            self._evolution_strategy,
        )
        observable_request_count = (num_timesteps + 1) * len(auxiliaries)
        child_seeds = spawn_seeds(
            seed,
            num_timesteps + observable_request_count,
        )
        projection_seeds = child_seeds[:num_timesteps]
        observable_seeds = child_seeds[num_timesteps:]

        parameter_history: list[Mapping[str, float]] = [dict(current_parameters)]
        circuit_history = [current_circuit]
        fidelities: list[float] = []
        optimizer_results: list[OptimizerResult] = []
        previous_increment = {name: 0.0 for name in parameter_names}

        for step in range(num_timesteps):
            target = _target_circuit(current_circuit, evolution)
            initial_guess = {
                name: current_parameters[name] + previous_increment[name]
                for name in parameter_names
            }
            objective_request = EstimatorRequest(
                template,
                identity,
                initial_guess,
                fidelity_shots,
            )
            optimizer_request = OptimizerRequest(objective_request, parameter_names)
            projection_estimator = _ProjectionEstimator(
                self._sampler,
                target,
                fidelity_shots,
                projection_seeds[step],
            )
            optimizer_result = _single_optimizer_result(
                self._optimizer,
                projection_estimator,
                optimizer_request,
            )
            next_parameters = dict(
                zip(
                    optimizer_result.parameter_names,
                    optimizer_result.optimal_parameters,
                    strict=True,
                )
            )
            next_circuit = template.assign_parameters(next_parameters)
            next_circuit.validate()
            if next_circuit.used_symbols:
                raise ValueError("optimizer result did not fully bind the ansatz")

            previous_increment = {
                name: next_parameters[name] - current_parameters[name]
                for name in parameter_names
            }
            current_parameters = next_parameters
            current_circuit = next_circuit
            parameter_history.append(dict(current_parameters))
            circuit_history.append(current_circuit)
            fidelities.append(1.0 - optimizer_result.optimal_value)
            optimizer_results.append(optimizer_result)

        observables: Mapping[str, tuple[EstimatorResult, ...]] = MappingProxyType({})
        if auxiliaries:
            if self._estimator is None:
                raise RuntimeError("auxiliary estimator disappeared during evolution")
            observables = _evaluate_observables(
                self._estimator,
                circuit_history,
                auxiliaries,
                shots=observable_shots,
                seeds=observable_seeds,
            )

        times = tuple(
            total_time * index / num_timesteps for index in range(num_timesteps + 1)
        )
        return PVQDResult(
            times,
            tuple(parameter_history),
            tuple(fidelities),
            tuple(optimizer_results),
            current_circuit,
            observables,
        )

    def evolve(
        self,
        hamiltonian: Hamiltonian | PauliString,
        initial_parameters: Mapping[str, float],
        time: float,
        *,
        num_timesteps: int,
        fidelity_shots: int,
        auxiliary_operators: Mapping[str, Hamiltonian | PauliString] | None = None,
        observable_shots: int | None = None,
        seed: int | None = None,
    ) -> PVQDResult:
        """Evolve with the same inputs and trajectory semantics as :meth:`run`."""
        return self.run(
            hamiltonian,
            initial_parameters,
            time,
            num_timesteps=num_timesteps,
            fidelity_shots=fidelity_shots,
            auxiliary_operators=auxiliary_operators,
            observable_shots=observable_shots,
            seed=seed,
        )


__all__ = ["PVQD", "PVQDResult"]
