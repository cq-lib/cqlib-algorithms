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

"""Variational imaginary-time evolution by McLachlan projection.

References
----------
[1] S. McArdle, T. Jones, S. Endo, Y. Li, S. C. Benjamin, and X. Yuan,
    "Variational ansatz-based quantum simulation of imaginary time evolution,"
    npj Quantum Information 5, 75, 2019. DOI: 10.1038/s41534-019-0187-2.
[2] X. Yuan, S. Endo, Q. Zhao, Y. Li, and S. C. Benjamin, "Theory of
    variational quantum simulation," Quantum 3, 191, 2019.
    DOI: 10.22331/q-2019-10-07-191.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from types import MappingProxyType
from typing import final

import numpy as np
from cqlib import Circuit
from cqlib.qis import Hamiltonian, PauliString

from ..._utils.numeric import normalize_finite_real, normalize_positive_real
from ...operators.gradient import NaturalGradient
from ...protocols import (
    Ansatz,
    Estimator,
    EstimatorRequest,
    EstimatorResult,
    GradientRequest,
    GradientResult,
)
from ...protocols.sampler import spawn_seeds
from ._common import (
    _evaluate_trajectory,
    _snapshot_auxiliary_operators,
    _snapshot_hamiltonian,
    _snapshot_parameters,
    _validate_positive_integer,
)


@dataclass(frozen=True, slots=True, repr=False)
class VarQITEResult:
    """A variational imaginary-time trajectory and its observables.

    ``times``, ``parameters``, and ``energies`` include the initial state.
    Every auxiliary-observable history follows the same time grid. The result
    is shallowly frozen; its Core circuit remains mutable.

    Args:
        times: Strictly increasing imaginary times beginning at zero.
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
            f"VarQITEResult(final_time={self.times[-1]!r}, "
            f"num_timesteps={len(self.times) - 1}, "
            f"num_parameters={len(self.final_parameters)}, "
            f"final_energy={self.final_energy.expectation!r})"
        )


def _single_natural_direction(
    natural_gradient: NaturalGradient,
    estimator: Estimator,
    request: GradientRequest,
) -> GradientResult:
    """Run and validate one natural-gradient direction."""
    results = natural_gradient.run(estimator, [request])
    if not isinstance(results, list):
        raise TypeError("natural gradient must return a list")
    if len(results) != 1:
        raise ValueError("natural gradient must return exactly one result per step")
    result = results[0]
    if not isinstance(result, GradientResult):
        raise TypeError("natural gradient must return GradientResult values")
    if result.parameter_names != request.parameter_names:
        raise ValueError("natural-gradient parameter names must match the request")
    return result


@final
class VarQITE:
    """Evolve a variational state in imaginary time by McLachlan projection.

    For ansatz state ``|psi(theta)>``, each step solves the projected equation
    ``g(theta) @ theta_dot = -0.5 * grad E(theta)`` and advances it with a
    fixed Forward Euler step. ``g`` is the exact, phase-invariant projected
    quantum-geometric-tensor metric used by :class:`NaturalGradient`; the
    energy gradient uses the injected estimator and may be exact or
    finite-shot. This is the global-phase-corrected form of the McLachlan
    equations described in reference [2].

    The ansatz must satisfy the gate and affine-parameter boundary of
    ``StatevectorQGT`` and ``ParameterShiftGradient``. This implementation
    does not adapt the time step or provide an integration-error estimate.

    Args:
        estimator: Estimator for gradients, energies, and auxiliary values.
        ansatz: Parameterized manifold for the evolving state.
        regularization: Non-negative ridge strength for the metric solve.
        prefix: Non-empty parameter prefix passed to the ansatz.

    Raises:
        TypeError: If a dependency or option has the wrong type.
        ValueError: If an option is non-finite or outside its valid range.
    """

    __slots__ = ("_ansatz", "_estimator", "_natural_gradient", "_prefix")

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
        self._natural_gradient = NaturalGradient(regularization)
        self._prefix = prefix

    @property
    def estimator(self) -> Estimator:
        """The estimator used throughout the evolution."""
        return self._estimator

    @property
    def ansatz(self) -> Ansatz:
        """The variational state manifold."""
        return self._ansatz

    @property
    def regularization(self) -> float:
        """The metric ridge-regularization strength."""
        return self._natural_gradient.regularization

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
        shots: int | None = None,
        auxiliary_operators: Mapping[str, Hamiltonian | PauliString] | None = None,
        seed: int | None = None,
    ) -> VarQITEResult:
        """Evolve the ansatz in imaginary time on a uniform time grid.

        Args:
            hamiltonian: Finite time-independent Hermitian operator, including the zero
                Hamiltonian for a stationary trajectory.
            initial_parameters: Exact finite bindings for active ansatz symbols.
            time: Positive total imaginary time.
            num_timesteps: Positive number of equal Forward Euler steps.
            shots: ``None`` for exact estimator requests or a positive count.
            auxiliary_operators: Optional named finite Hermitian observables.
            seed: Optional master seed for gradient and trajectory requests.

        Returns:
            Time, parameter, energy and observable histories, and the final
            bound ansatz circuit.

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
        if template.num_qubits != snapshot.num_qubits:
            raise ValueError("ansatz and Hamiltonian must have the same qubit count")

        parameter_names = tuple(template.used_symbols)
        initial_request = EstimatorRequest(
            template,
            snapshot,
            initial_parameters,
            shots,
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
        child_seeds = spawn_seeds(
            seed,
            num_timesteps + trajectory_request_count,
        )
        gradient_seeds = child_seeds[:num_timesteps]
        trajectory_seeds = child_seeds[num_timesteps:]

        parameter_history: list[Mapping[str, float]] = [dict(current_parameters)]
        for step in range(num_timesteps):
            estimator_request = EstimatorRequest(
                template,
                snapshot,
                current_parameters,
                shots,
                gradient_seeds[step],
            )
            gradient_request = GradientRequest(estimator_request, parameter_names)
            direction = _single_natural_direction(
                self._natural_gradient,
                self._estimator,
                gradient_request,
            )
            velocity = -0.5 * np.asarray(direction.values, dtype=np.float64)
            if not np.all(np.isfinite(velocity)):
                raise ValueError("imaginary-time parameter velocity must be finite")
            current_vector = np.asarray(
                [current_parameters[name] for name in parameter_names],
                dtype=np.float64,
            )
            next_vector = current_vector + time_step * velocity
            if not np.all(np.isfinite(next_vector)):
                raise ValueError("imaginary-time parameter update must be finite")
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
            shots=shots,
            seeds=trajectory_seeds,
        )
        times = tuple(
            total_time * index / num_timesteps for index in range(num_timesteps + 1)
        )
        return VarQITEResult(
            times,
            tuple(parameter_history),
            energies,
            circuit_history[-1],
            observables,
        )

    def evolve(
        self,
        hamiltonian: Hamiltonian | PauliString,
        initial_parameters: Mapping[str, float],
        time: float,
        *,
        num_timesteps: int,
        shots: int | None = None,
        auxiliary_operators: Mapping[str, Hamiltonian | PauliString] | None = None,
        seed: int | None = None,
    ) -> VarQITEResult:
        """Evolve with the same inputs and trajectory semantics as :meth:`run`."""
        return self.run(
            hamiltonian,
            initial_parameters,
            time,
            num_timesteps=num_timesteps,
            shots=shots,
            auxiliary_operators=auxiliary_operators,
            seed=seed,
        )
