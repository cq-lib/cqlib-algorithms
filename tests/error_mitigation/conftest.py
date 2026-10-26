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

"""Shared fixtures for the error-mitigation test suite.

The helpers are exposed as fixtures so the technique modules can consume
them through pytest's fixture resolution instead of importing across test
directories. Fixtures that carry parameters return a factory callable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pytest
from cqlib import Circuit
from cqlib.circuit import StandardGate
from cqlib.device import NoiseModel, SingleQubitNoise, TwoQubitNoise
from cqlib.qis import Hamiltonian, PauliString
from cqlib.qis.state import DensityMatrixNoise

from cqlib_algorithms.error_mitigation._circuits import is_measurement
from cqlib_algorithms.error_mitigation.core import QemExecutor
from cqlib_algorithms.operators.estimator import StatevectorEstimator
from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.protocols import (
    Estimator,
    EstimatorRequest,
    EstimatorResult,
)

# Gates that the shared bodies use, registered in the noisy backend so a
# depolarizing channel sits after each of them. Decomposition preserves these
# names, so the registration matches the executed circuit.
_SINGLE_QUBIT_GATES = (
    StandardGate.H,
    StandardGate.X,
    StandardGate.Y,
    StandardGate.Z,
    StandardGate.S,
    StandardGate.SDG,
    StandardGate.T,
    StandardGate.TDG,
    StandardGate.RX,
    StandardGate.RY,
    StandardGate.RZ,
)
_TWO_QUBIT_GATES = (StandardGate.CX, StandardGate.CZ, StandardGate.SWAP)

# Width the noisy backend registers noise for; wider circuits simply get no
# extra channels, which never happens in this suite.
_MAX_NOISY_QUBITS = 4


def _noise_model(level: float) -> NoiseModel:
    """Build a depolarizing model covering the shared gate set."""
    model = NoiseModel()
    for gate in _SINGLE_QUBIT_GATES:
        for qubit in range(_MAX_NOISY_QUBITS):
            model.add_single_qubit_error(
                gate, qubit, SingleQubitNoise.depolarizing(level)
            )
    for gate in _TWO_QUBIT_GATES:
        for control in range(_MAX_NOISY_QUBITS):
            for target in range(_MAX_NOISY_QUBITS):
                if control == target:
                    continue
                model.add_two_qubit_error(
                    gate, control, target, TwoQubitNoise.depolarizing(level)
                )
    return model


class _DensityMatrixEstimator(Estimator):
    """Exact expectation values from a depolarizing density-matrix run.

    The simulation is deterministic, so the reported variance is zero and
    ``shots`` must stay ``None``. This backend exists to give the technique
    tests a controlled, reproducible noise response.
    """

    __slots__ = ("_model",)

    def __init__(self, level: float) -> None:
        self._model = _noise_model(level)

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        """Return one exact noisy expectation per request, in input order."""
        results: list[EstimatorResult] = []
        for request in requests:
            if not isinstance(request, EstimatorRequest):
                raise TypeError("requests must contain EstimatorRequest instances")
            request.validate()
            if request.shots is not None:
                raise ValueError("the density-matrix backend is exact")
            bound = request.circuit.assign_parameters(dict(request.parameter_bindings))
            state = DensityMatrixNoise.from_circuit(bound, self._model)
            results.append(EstimatorResult(state.expectation(request.observable), 0.0))
        return results


@pytest.fixture
def bell_circuit() -> Circuit:
    """Return the two-qubit H-plus-CNOT body used across the suite."""
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    return circuit


@pytest.fixture
def rotation_circuit() -> Callable[[float], Circuit]:
    """Return a factory building a one-qubit RY body at a given angle."""

    def build(theta: float) -> Circuit:
        circuit = Circuit(1)
        circuit.ry(0, theta)
        return circuit

    return build


@pytest.fixture
def wide_circuit() -> Circuit:
    """Return a three-operation two-qubit body spanning two layers."""
    circuit = Circuit(2)
    circuit.h(0)
    circuit.x(1)
    circuit.cx(0, 1)
    return circuit


@pytest.fixture
def z_observable() -> PauliString:
    """Return the single-qubit Pauli-Z observable."""
    return PauliString.from_str("Z")


@pytest.fixture
def zz_observable() -> PauliString:
    """Return the two-qubit Pauli-ZZ observable."""
    return PauliString.from_str("ZZ")


@pytest.fixture
def z_hamiltonian(z_observable: PauliString) -> Hamiltonian:
    """Return the single-qubit Pauli-Z observable as a Hamiltonian."""
    return Hamiltonian.from_pauli(z_observable)


@pytest.fixture
def exact_executor() -> QemExecutor:
    """Return a noise-free executor holding both backends."""
    return QemExecutor(
        estimator=StatevectorEstimator(),
        sampler=StatevectorSampler(),
    )


@pytest.fixture
def exact_estimator() -> StatevectorEstimator:
    """Return the noise-free exact expectation backend."""
    return StatevectorEstimator()


@pytest.fixture
def noisy_executor() -> Callable[[float], QemExecutor]:
    """Return a factory building an exact depolarizing executor at a level."""

    def build(level: float) -> QemExecutor:
        return QemExecutor(estimator=_DensityMatrixEstimator(level))

    return build


@pytest.fixture
def phases() -> Callable[[Circuit], complex]:
    """Return the global phase of a circuit as a complex scalar."""

    def read(circuit: Circuit) -> complex:
        return complex(circuit.global_phase.evaluate({}))

    return read


@pytest.fixture
def names() -> Callable[[Circuit], list[str]]:
    """Return the operation names of a circuit in execution order."""

    def read(circuit: Circuit) -> list[str]:
        return [operation.name for operation in circuit.operations]

    return read


@pytest.fixture
def gate_count() -> Callable[[Circuit], int]:
    """Return how many non-measurement operations a circuit carries."""

    def count(circuit: Circuit) -> int:
        return sum(1 for operation in circuit.operations if not is_measurement(operation))

    return count


@pytest.fixture
def sample_from() -> Callable[[Mapping[str, float], int, int], dict[str, int]]:
    """Return a factory drawing counts from a basis distribution."""

    def draw(
        probabilities: Mapping[str, float],
        shots: int,
        seed: int,
    ) -> dict[str, int]:
        keys = list(probabilities)
        weights = np.array([probabilities[key] for key in keys], dtype=float)
        weights /= weights.sum()
        draws = np.random.default_rng(seed).choice(len(keys), size=shots, p=weights)
        tallies = np.bincount(draws, minlength=len(keys))
        return {
            key: int(count)
            for key, count in zip(keys, tallies)
            if count
        }

    return draw
