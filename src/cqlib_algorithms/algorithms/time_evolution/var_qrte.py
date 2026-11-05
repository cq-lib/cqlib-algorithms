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

"""Variational real-time evolution by phase-corrected McLachlan projection.

References
----------
[1] X. Yuan, S. Endo, Q. Zhao, Y. Li, and S. C. Benjamin, "Theory of
    variational quantum simulation," Quantum 3, 191, 2019.
    DOI: 10.22331/q-2019-10-07-191.
[2] J. Gacon, J. Nys, R. Rossi, S. Woerner, and G. Carleo, "Variational
    quantum time evolution without the quantum geometric tensor," Physical
    Review Research 6, 013143, 2024. DOI: 10.1103/PhysRevResearch.6.013143.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from numbers import Real
from types import MappingProxyType
from typing import final

import numpy as np
from cqlib import Circuit
from cqlib.qis import Hamiltonian, PauliString

from ..._utils.numeric import normalize_finite_real, normalize_positive_real
from ...operators._statevector_observable import (
    apply_observable,
    hermitian_expectation_from_action,
)
from ...operators.gradient._linear_system import solve_metric_system
from ...operators.gradient._parameter_analysis import validate_circuit
from ...operators.gradient.qgt import _projected_tensor, _statevector_derivatives
from ...protocols import Ansatz, Estimator, EstimatorRequest, EstimatorResult
from ...protocols.gradient import GradientRequest
from ...protocols.sampler import spawn_seeds
from ._common import (
    _evaluate_trajectory,
    _snapshot_auxiliary_operators,
    _snapshot_hamiltonian,
    _snapshot_parameters,
    _validate_positive_integer,
)


def _normalize_regularization(value: float) -> float:
    """Validate and return a finite non-negative ridge strength."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("regularization must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError("regularization must be finite")
    if normalized < 0.0:
        raise ValueError("regularization must be non-negative")
    return normalized


def _mclachlan_velocity(
    request: GradientRequest,
    hamiltonian: Hamiltonian,
    regularization: float,
) -> np.ndarray:
    """Return the phase-corrected real-time McLachlan velocity."""
    names = request.parameter_names
    if not names:
        return np.empty(0, dtype=np.float64)
    state, derivatives = _statevector_derivatives(request)
    tensor = _projected_tensor(state, derivatives)
    metric = np.real(tensor).astype(np.float64, copy=False)
    hamiltonian_state = apply_observable(state, hamiltonian)
    energy = hermitian_expectation_from_action(
        state,
        hamiltonian,
        hamiltonian_state,
    )

    force = np.empty(len(names), dtype=np.float64)
    for index, derivative in enumerate(derivatives):
        projected_overlap = np.vdot(derivative, hamiltonian_state) - (
            np.vdot(derivative, state) * energy
        )
        force[index] = float(projected_overlap.imag)
    if not np.all(np.isfinite(metric)) or not np.all(np.isfinite(force)):
        raise ValueError("real-time metric and force must be finite")
    return solve_metric_system(metric, force, regularization)


@dataclass(frozen=True, slots=True, repr=False)
class VarQRTEResult:
    """A variational real-time trajectory and its observables.

    ``times``, ``parameters``, and ``energies`` include the initial state.
    Every auxiliary-observable history follows the same time grid. The result
    is shallowly frozen; its Core circuit remains mutable.

    Args:
        times: Strictly increasing real times beginning at zero.
        parameters: Immutable parameter mappings at the corresponding times.
        energies: Hamiltonian estimates at every trajectory point.
        evolved_circuit: Fully bound ansatz at the final time.
        observables: Optional named expectation histories at all times.

    Raises:
        TypeError: If a field or nested value has the wrong type.
        ValueError: If trajectory lengths, values, or names are inconsistent.
    """

    times: tuple[float, ...]
    parameters: tuple[Mapping[str, float], ...]
    energies: tuple[EstimatorResult, ...]
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
        parameters = tuple(_snapshot_parameters(item) for item in self.parameters)
        if len(parameters) != len(times):
            raise ValueError("parameters must contain one entry per time")
        names = tuple(parameters[0])
        if any(tuple(item) != names for item in parameters[1:]):
            raise ValueError("parameter history entries must preserve names and order")
        object.__setattr__(self, "parameters", parameters)

        if not isinstance(self.energies, tuple):
            raise TypeError("energies must be a tuple")
        if len(self.energies) != len(times):
            raise ValueError("energies must contain one result per time")
        if any(not isinstance(result, EstimatorResult) for result in self.energies):
            raise TypeError("energies must contain EstimatorResult values")

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
    def final_energy(self) -> EstimatorResult:
        """Return the Hamiltonian estimate at the final time."""
        return self.energies[-1]

    def __repr__(self) -> str:
        """Summarize the trajectory without printing its circuit or arrays."""
        return (
            f"VarQRTEResult(final_time={self.times[-1]!r}, "
            f"num_timesteps={len(self.times) - 1}, "
            f"num_parameters={len(self.final_parameters)}, "
            f"final_energy={self.final_energy.expectation!r})"
        )


@final
class VarQRTE:
    """Evolve a variational state in real time by McLachlan projection.

    For ansatz state ``|psi(theta)>``, each step solves ``g @ theta_dot = b``.
    Here ``g`` is the exact phase-invariant projected QGT metric and
    ``b_i = Im(<d_i psi|H|psi> - <d_i psi|psi> E)``. Parameters advance with
    a fixed Forward Euler step. QGT and force evaluation are exact statevector
    calculations; the injected estimator evaluates only reported trajectories.

    This algorithm differs from PVQD: it integrates the local McLachlan
    equation instead of optimizing fidelity to a short-time target state.
    The ansatz must satisfy the gate and affine-parameter boundary of
    ``StatevectorQGT``. Time-dependent Hamiltonians, adaptive integration and
    integration-error estimates are not supported.

    Args:
        estimator: Estimator for energies and auxiliary values.
        ansatz: Parameterized manifold for the evolving state.
        regularization: Non-negative ridge strength for the metric solve.
        prefix: Non-empty parameter prefix passed to the ansatz.

    Raises:
        TypeError: If a dependency or option has the wrong type.
        ValueError: If an option is non-finite or outside its valid range.
    """

    __slots__ = ("_ansatz", "_estimator", "_prefix", "_regularization")

    def __init__(
        self,
        estimator: Estimator,
        ansatz: Ansatz,
        *,
        regularization: float = 0.0,
        prefix: str = "theta",
    ) -> None:
        if not isinstance(estimator, Estimator):
            raise TypeError("estimator must be an Estimator")
        if not isinstance(ansatz, Ansatz):
            raise TypeError("ansatz must be an Ansatz")
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        if not prefix:
            raise ValueError("prefix must not be empty")
        self._estimator = estimator
        self._ansatz = ansatz
        self._regularization = _normalize_regularization(regularization)
        self._prefix = prefix

    @property
    def estimator(self) -> Estimator:
        """The estimator used for reported trajectory observables."""
        return self._estimator

    @property
    def ansatz(self) -> Ansatz:
        """The variational state manifold."""
        return self._ansatz

    @property
    def regularization(self) -> float:
        """The metric ridge-regularization strength."""
        return self._regularization

    @property
    def prefix(self) -> str:
        """The ansatz parameter-name prefix."""
        return self._prefix

    def evolve(
        self,
        hamiltonian: Hamiltonian | PauliString,
        initial_parameters: Mapping[str, float],
        time: float,
        *,
        num_timesteps: int,
        observable_shots: int | None = None,
        auxiliary_operators: Mapping[str, Hamiltonian | PauliString] | None = None,
        seed: int | None = None,
    ) -> VarQRTEResult:
        """Evolve a time-independent Hamiltonian over equal positive steps.

        Args:
            hamiltonian: Finite Hermitian Hamiltonian, including the zero
                Hamiltonian for a stationary trajectory.
            initial_parameters: Exact finite bindings for active ansatz symbols.
            time: Positive total real time.
            num_timesteps: Positive number of equal Forward Euler steps.
            observable_shots: Shots used only for reported trajectory values;
                ``None`` requests exact observations. It never changes the
                exact statevector QGT, force, or parameter trajectory.
            auxiliary_operators: Optional named finite Hermitian observables.
            seed: Optional master seed for trajectory-observation requests.

        Returns:
            The parameter, energy, circuit, and auxiliary-observable trajectory.

        Raises:
            TypeError: If an input or component response has the wrong type.
            ValueError: If dimensions, bindings, values, circuit structure, or
                component responses violate the evolution contract.
        """
        total_time = normalize_positive_real(time, "time")
        _validate_positive_integer(num_timesteps, "num_timesteps")
        snapshot = _snapshot_hamiltonian(hamiltonian, name="hamiltonian")

        self._ansatz.validate()
        template = self._ansatz.build_circuit(self._prefix)
        if not isinstance(template, Circuit):
            raise TypeError("ansatz must build a cqlib.Circuit")
        template.validate()
        validate_circuit(template)
        if template.num_qubits != snapshot.num_qubits:
            raise ValueError("ansatz and Hamiltonian must have the same qubit count")

        parameter_names = tuple(template.used_symbols)
        initial_request = EstimatorRequest(
            template,
            snapshot,
            initial_parameters,
            observable_shots,
            seed,
        )
        current_parameters = {
            name: initial_request.parameter_bindings[name] for name in parameter_names
        }
        auxiliaries = _snapshot_auxiliary_operators(
            auxiliary_operators,
            num_qubits=template.num_qubits,
        )

        time_step = total_time / num_timesteps
        if not math.isfinite(time_step) or time_step <= 0.0:
            raise ValueError("time step must be finite and representable as positive")
        trajectory_request_count = (num_timesteps + 1) * (1 + len(auxiliaries))
        trajectory_seeds = spawn_seeds(seed, trajectory_request_count)

        parameter_history: list[Mapping[str, float]] = [dict(current_parameters)]
        for _ in range(num_timesteps):
            if parameter_names:
                exact_request = EstimatorRequest(
                    template,
                    snapshot,
                    current_parameters,
                )
                gradient_request = GradientRequest(exact_request, parameter_names)
                velocity = _mclachlan_velocity(
                    gradient_request,
                    snapshot,
                    self._regularization,
                )
                if not np.all(np.isfinite(velocity)):
                    raise ValueError("real-time parameter velocity must be finite")
                current_vector = np.asarray(
                    [current_parameters[name] for name in parameter_names],
                    dtype=np.float64,
                )
                next_vector = current_vector + time_step * velocity
                if not np.all(np.isfinite(next_vector)):
                    raise ValueError("real-time parameter update must be finite")
                current_parameters = dict(
                    zip(
                        parameter_names,
                        (float(value) for value in next_vector),
                        strict=True,
                    )
                )
            parameter_history.append(dict(current_parameters))

        circuit_history: list[Circuit] = []
        for parameters in parameter_history:
            circuit = template.assign_parameters(dict(parameters))
            circuit.validate()
            if circuit.used_symbols:
                raise ValueError("trajectory parameters did not fully bind the ansatz")
            circuit_history.append(circuit)

        energies, observables = _evaluate_trajectory(
            self._estimator,
            circuit_history,
            snapshot,
            auxiliaries,
            shots=observable_shots,
            seeds=trajectory_seeds,
        )
        times = tuple(
            total_time * index / num_timesteps for index in range(num_timesteps + 1)
        )
        return VarQRTEResult(
            times,
            tuple(parameter_history),
            energies,
            circuit_history[-1],
            observables,
        )
