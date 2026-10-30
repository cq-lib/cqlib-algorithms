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

"""Contract, boundary, and independent-reference tests for VarQITE."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError
from itertools import pairwise

import numpy as np
import pytest
from cqlib import Circuit, Parameter
from cqlib.qis import Hamiltonian, PauliString
from cqlib.qis.state import Statevector
from scipy.linalg import expm  # type: ignore[import-untyped]

from cqlib_algorithms.algorithms.time_evolution import VarQITE, VarQITEResult
from cqlib_algorithms.operators import (
    NaturalGradient,
    SamplingEstimator,
    StatevectorEstimator,
    StatevectorSampler,
)
from cqlib_algorithms.protocols import Ansatz, Estimator
from cqlib_algorithms.protocols.estimator import EstimatorRequest, EstimatorResult
from cqlib_algorithms.protocols.sampler import spawn_seeds


class RYAnsatz(Ansatz):
    """One-angle real state manifold."""

    __slots__ = ()

    def validate(self) -> None:
        """Validate the fixed ansatz configuration."""

    def build_circuit(self, prefix: str) -> Circuit:
        """Build a single RY rotation."""
        circuit = Circuit(1)
        circuit.ry(0, Parameter(f"{prefix}_0"))
        return circuit

    def num_parameters(self) -> int:
        """Return one trainable parameter."""
        return 1

    def num_qubits(self) -> int:
        """Return one ansatz qubit."""
        return 1


class RedundantRYAnsatz(Ansatz):
    """Two redundant RY angles producing a singular projected metric."""

    __slots__ = ()

    def validate(self) -> None:
        """Validate the fixed ansatz configuration."""

    def build_circuit(self, prefix: str) -> Circuit:
        """Build two consecutive RY rotations."""
        circuit = Circuit(1)
        circuit.ry(0, Parameter(f"{prefix}_first"))
        circuit.ry(0, Parameter(f"{prefix}_second"))
        return circuit

    def num_parameters(self) -> int:
        """Return two trainable parameters."""
        return 2

    def num_qubits(self) -> int:
        """Return one ansatz qubit."""
        return 1


class TwoQubitAnsatz(Ansatz):
    """Two-angle entangled manifold with one shared affine occurrence."""

    __slots__ = ()

    def validate(self) -> None:
        """Validate the fixed ansatz configuration."""

    def build_circuit(self, prefix: str) -> Circuit:
        """Build the two-qubit reference ansatz."""
        first = Parameter(f"{prefix}_first")
        second = Parameter(f"{prefix}_second")
        circuit = Circuit(2)
        circuit.ry(0, first)
        circuit.rx(1, second)
        circuit.cx(0, 1)
        circuit.rz(1, 0.4 * first - 0.2)
        return circuit

    def num_parameters(self) -> int:
        """Return two trainable parameters."""
        return 2

    def num_qubits(self) -> int:
        """Return two ansatz qubits."""
        return 2


class BlochSphereAnsatz(Ansatz):
    """One-qubit manifold with independent polar and azimuthal angles."""

    __slots__ = ()

    def validate(self) -> None:
        """Validate the fixed ansatz configuration."""

    def build_circuit(self, prefix: str) -> Circuit:
        """Build a phase-sensitive RY-RZ state manifold."""
        circuit = Circuit(1)
        circuit.ry(0, Parameter(f"{prefix}_polar"))
        circuit.rz(0, Parameter(f"{prefix}_azimuth"))
        return circuit

    def num_parameters(self) -> int:
        """Return two trainable parameters."""
        return 2

    def num_qubits(self) -> int:
        """Return one ansatz qubit."""
        return 1


class StaticAnsatz(Ansatz):
    """Zero-parameter identity manifold of configurable width."""

    __slots__ = ("_num_qubits",)

    def __init__(self, num_qubits: int = 1) -> None:
        self._num_qubits = num_qubits

    def validate(self) -> None:
        """Validate the positive width."""
        if self._num_qubits <= 0:
            raise ValueError("num_qubits must be positive")

    def build_circuit(self, prefix: str) -> Circuit:
        """Build an identity circuit."""
        return Circuit(self._num_qubits)

    def num_parameters(self) -> int:
        """Return zero trainable parameters."""
        return 0

    def num_qubits(self) -> int:
        """Return the configured width."""
        return self._num_qubits


class MeasuredAnsatz(RYAnsatz):
    """RY manifold followed by an unsupported measurement."""

    def build_circuit(self, prefix: str) -> Circuit:
        """Build a measured parameterized circuit."""
        circuit = super().build_circuit(prefix)
        circuit.measure(0)
        return circuit


class RecordingExactEstimator(Estimator):
    """Record requests while evaluating their exact statevector values."""

    __slots__ = ("_delegate", "batches")

    def __init__(self) -> None:
        self.batches: list[tuple[EstimatorRequest, ...]] = []
        self._delegate = StatevectorEstimator()

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        """Record requests and delegate equivalent exact evaluations."""
        batch = tuple(requests)
        self.batches.append(batch)
        exact = [
            EstimatorRequest(
                request.circuit,
                request.observable,
                request.parameter_bindings,
                None,
                request.seed,
            )
            for request in batch
        ]
        return self._delegate.run(exact)


class InvalidEstimator(Estimator):
    """Return an invalid empty response for every batch."""

    __slots__ = ()

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        """Return no estimates."""
        return []


def _z_hamiltonian() -> Hamiltonian:
    """Return the one-qubit Pauli-Z Hamiltonian."""
    return Hamiltonian.from_pauli(PauliString.from_str("Z"))


def _two_qubit_hamiltonian() -> Hamiltonian:
    """Return the two-qubit Hamiltonian used by the dense reference."""
    return Hamiltonian.from_terms(
        2,
        [
            (PauliString.from_str("ZI"), 0.7),
            (PauliString.from_str("XX"), 0.3),
            (PauliString.from_str("IZ"), -0.2),
        ],
    )


def _two_qubit_hamiltonian_matrix() -> np.ndarray:
    """Return an independent NumPy matrix for the reference Hamiltonian."""
    identity = np.eye(2, dtype=np.complex128)
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    pauli_z = np.diag([1.0, -1.0]).astype(np.complex128)
    return (
        0.7 * np.kron(pauli_z, identity)
        + 0.3 * np.kron(pauli_x, pauli_x)
        - 0.2 * np.kron(identity, pauli_z)
    )


def _dense_reference_step(
    template: Circuit,
    parameters: Mapping[str, float],
    names: tuple[str, ...],
    hamiltonian: np.ndarray,
    time_step: float,
    regularization: float,
) -> dict[str, float]:
    """Return one independently differenced McLachlan Euler step."""
    difference = 1e-6

    def state(bindings: Mapping[str, float]) -> np.ndarray:
        bound = template.assign_parameters(dict(bindings))
        return np.asarray(bound.to_matrix([0, 1])[:, 0], dtype=np.complex128)

    psi = state(parameters)
    derivative_states: list[np.ndarray] = []
    energy_gradient: list[float] = []
    for name in names:
        plus = dict(parameters)
        minus = dict(parameters)
        plus[name] += difference
        minus[name] -= difference
        plus_state = state(plus)
        minus_state = state(minus)
        derivative_states.append((plus_state - minus_state) / (2.0 * difference))
        plus_energy = float(np.real(np.vdot(plus_state, hamiltonian @ plus_state)))
        minus_energy = float(np.real(np.vdot(minus_state, hamiltonian @ minus_state)))
        energy_gradient.append((plus_energy - minus_energy) / (2.0 * difference))

    metric = np.empty((len(names), len(names)), dtype=np.float64)
    for row, first in enumerate(derivative_states):
        for column, second in enumerate(derivative_states):
            projected = np.vdot(first, second) - np.vdot(first, psi) * np.vdot(
                psi, second
            )
            metric[row, column] = float(np.real(projected))
    system = metric + regularization * np.eye(len(names))
    direction = np.linalg.solve(system, np.asarray(energy_gradient))
    next_values = np.asarray([parameters[name] for name in names])
    next_values -= 0.5 * time_step * direction
    return dict(zip(names, (float(value) for value in next_values), strict=True))


def test_single_qubit_step_has_mclachlan_half_factor() -> None:
    theta = 0.37
    time_step = 0.08
    result = VarQITE(StatevectorEstimator(), RYAnsatz()).evolve(
        _z_hamiltonian(),
        {"theta_0": theta},
        time_step,
        num_timesteps=1,
    )

    expected = theta + 2.0 * time_step * math.sin(theta)
    assert result.final_parameters["theta_0"] == pytest.approx(expected, abs=1e-12)
    assert tuple(energy.expectation for energy in result.energies) == pytest.approx(
        (math.cos(theta), math.cos(expected)),
        abs=1e-12,
    )


@pytest.mark.reference
def test_single_qubit_trajectory_matches_normalized_imaginary_time() -> None:
    theta = 0.42
    total_time = 0.3
    result = VarQITE(StatevectorEstimator(), RYAnsatz()).evolve(
        _z_hamiltonian(),
        {"theta_0": theta},
        total_time,
        num_timesteps=300,
    )

    initial = np.array(
        [math.cos(theta / 2.0), math.sin(theta / 2.0)],
        dtype=np.complex128,
    )
    exact = expm(-total_time * np.diag([1.0, -1.0])) @ initial
    exact /= np.linalg.norm(exact)
    actual = np.asarray(Statevector.from_circuit(result.evolved_circuit).data)
    assert abs(np.vdot(exact, actual)) ** 2 == pytest.approx(1.0, abs=5e-8)
    energies = [energy.expectation for energy in result.energies]
    assert all(right <= left + 1e-12 for left, right in pairwise(energies))


@pytest.mark.reference
def test_phase_sensitive_trajectory_matches_normalized_imaginary_time() -> None:
    polar = 0.83
    azimuth = -0.61
    total_time = 0.2
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    pauli_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    pauli_z = np.diag([1.0, -1.0]).astype(np.complex128)
    hamiltonian_matrix = 0.6 * pauli_x - 0.35 * pauli_y + 0.2 * pauli_z
    hamiltonian = Hamiltonian.from_terms(
        1,
        [
            (PauliString.from_str("X"), 0.6),
            (PauliString.from_str("Y"), -0.35),
            (PauliString.from_str("Z"), 0.2),
        ],
    )
    result = VarQITE(StatevectorEstimator(), BlochSphereAnsatz()).evolve(
        hamiltonian,
        {"theta_polar": polar, "theta_azimuth": azimuth},
        total_time,
        num_timesteps=400,
    )

    initial = np.array(
        [
            math.cos(polar / 2.0) * np.exp(-0.5j * azimuth),
            math.sin(polar / 2.0) * np.exp(0.5j * azimuth),
        ],
        dtype=np.complex128,
    )
    exact = expm(-total_time * hamiltonian_matrix) @ initial
    exact /= np.linalg.norm(exact)
    actual = np.asarray(Statevector.from_circuit(result.evolved_circuit).data)

    assert abs(np.vdot(exact, actual)) ** 2 == pytest.approx(1.0, abs=2e-8)


@pytest.mark.reference
def test_two_qubit_steps_match_independent_center_difference_reference() -> None:
    ansatz = TwoQubitAnsatz()
    template = ansatz.build_circuit("theta")
    names = tuple(template.used_symbols)
    expected: dict[str, float] = {"theta_first": 0.31, "theta_second": -0.47}
    time_step = 0.02
    regularization = 0.13
    for _ in range(3):
        expected = _dense_reference_step(
            template,
            expected,
            names,
            _two_qubit_hamiltonian_matrix(),
            time_step,
            regularization,
        )

    result = VarQITE(
        StatevectorEstimator(),
        ansatz,
        regularization=regularization,
    ).evolve(
        _two_qubit_hamiltonian(),
        {"theta_first": 0.31, "theta_second": -0.47},
        3.0 * time_step,
        num_timesteps=3,
    )

    assert dict(result.final_parameters) == pytest.approx(expected, abs=4e-7)


@pytest.mark.parametrize("regularization", [0.0, 0.2])
def test_singular_metric_uses_minimum_norm_or_ridge_solution(
    regularization: float,
) -> None:
    initial = {"theta_first": 0.2, "theta_second": 0.35}
    time_step = 0.04
    total = sum(initial.values())
    metric = 0.25 * np.ones((2, 2))
    gradient = -math.sin(total) * np.ones(2)
    if regularization == 0.0:
        direction = np.linalg.lstsq(metric, gradient, rcond=None)[0]
    else:
        direction = np.linalg.solve(
            metric + regularization * np.eye(2),
            gradient,
        )
    expected = np.asarray(list(initial.values())) - 0.5 * time_step * direction

    result = VarQITE(
        StatevectorEstimator(),
        RedundantRYAnsatz(),
        regularization=regularization,
    ).evolve(
        _z_hamiltonian(),
        initial,
        time_step,
        num_timesteps=1,
    )

    assert tuple(result.final_parameters.values()) == pytest.approx(expected, abs=1e-10)


def test_result_records_immutable_trajectory_and_auxiliary_observables() -> None:
    initial = {"motion_0": 0.3}
    auxiliaries = {"x": PauliString.from_str("X")}
    result = VarQITE(
        StatevectorEstimator(),
        RYAnsatz(),
        prefix="motion",
    ).evolve(
        _z_hamiltonian(),
        initial,
        0.1,
        num_timesteps=2,
        auxiliary_operators=auxiliaries,
    )

    assert result.times == pytest.approx((0.0, 0.05, 0.1))
    assert len(result.parameters) == len(result.energies) == 3
    assert len(result.observables["x"]) == 3
    assert result.final_parameters is result.parameters[-1]
    assert result.final_energy is result.energies[-1]
    assert result.evolved_circuit.used_symbols == []
    assert initial == {"motion_0": 0.3}
    assert auxiliaries == {"x": PauliString.from_str("X")}
    with pytest.raises(TypeError):
        result.final_parameters["motion_0"] = 0.0  # type: ignore[index]
    with pytest.raises(TypeError):
        result.observables["new"] = ()  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        result.times = ()  # type: ignore[misc]
    assert repr(result).startswith("VarQITEResult(final_time=0.1")
    assert "Circuit" not in repr(result)


def test_shots_and_child_seeds_are_forwarded_reproducibly() -> None:
    first_estimator = RecordingExactEstimator()
    second_estimator = RecordingExactEstimator()
    arguments = {
        "hamiltonian": _z_hamiltonian(),
        "initial_parameters": {"theta_0": 0.2},
        "time": 0.1,
        "num_timesteps": 2,
        "shots": 64,
        "auxiliary_operators": {"x": PauliString.from_str("X")},
        "seed": 37,
    }
    VarQITE(first_estimator, RYAnsatz()).evolve(**arguments)
    VarQITE(second_estimator, RYAnsatz()).evolve(**arguments)

    expected_seeds = spawn_seeds(37, 8)
    assert len(first_estimator.batches) == 3
    assert [request.seed for request in first_estimator.batches[0]] == [
        expected_seeds[0],
        expected_seeds[0],
    ]
    assert [request.seed for request in first_estimator.batches[1]] == [
        expected_seeds[1],
        expected_seeds[1],
    ]
    assert [request.seed for request in first_estimator.batches[2]] == expected_seeds[
        2:
    ]
    assert all(
        request.shots == 64 for batch in first_estimator.batches for request in batch
    )
    assert [
        [request.seed for request in batch] for batch in first_estimator.batches
    ] == [[request.seed for request in batch] for batch in second_estimator.batches]


@pytest.mark.integration
def test_sampling_estimator_integration_is_seed_reproducible() -> None:
    def run() -> VarQITEResult:
        estimator = SamplingEstimator(StatevectorSampler())
        return VarQITE(estimator, RYAnsatz()).evolve(
            _z_hamiltonian(),
            {"theta_0": 0.4},
            0.04,
            num_timesteps=2,
            shots=4096,
            auxiliary_operators={"x": PauliString.from_str("X")},
            seed=91,
        )

    first = run()
    repeated = run()

    assert first.parameters == repeated.parameters
    assert first.energies == repeated.energies
    assert first.observables == repeated.observables
    assert all(result.variance is not None for result in first.energies)
    assert all(result.variance is not None for result in first.observables["x"])


def test_zero_hamiltonian_with_active_parameters_is_stationary() -> None:
    initial = {"theta_0": 0.37}
    result = VarQITE(StatevectorEstimator(), RYAnsatz()).evolve(
        Hamiltonian(1),
        initial,
        0.2,
        num_timesteps=2,
    )

    assert result.parameters == (initial, initial, initial)
    assert [item.expectation for item in result.energies] == [0.0, 0.0, 0.0]


def test_zero_parameter_ansatz_is_stationary_under_nonzero_hamiltonian() -> None:
    result = VarQITE(StatevectorEstimator(), StaticAnsatz()).evolve(
        _z_hamiltonian(),
        {},
        0.2,
        num_timesteps=2,
    )

    assert result.parameters == ({}, {}, {})
    assert [item.expectation for item in result.energies] == [1.0, 1.0, 1.0]


@pytest.mark.parametrize(
    ("arguments", "exception", "message"),
    [
        ({"estimator": object(), "ansatz": RYAnsatz()}, TypeError, "estimator"),
        (
            {"estimator": StatevectorEstimator(), "ansatz": object()},
            TypeError,
            "ansatz",
        ),
        (
            {
                "estimator": StatevectorEstimator(),
                "ansatz": RYAnsatz(),
                "regularization": True,
            },
            TypeError,
            "regularization",
        ),
        (
            {
                "estimator": StatevectorEstimator(),
                "ansatz": RYAnsatz(),
                "regularization": -0.1,
            },
            ValueError,
            "regularization",
        ),
        (
            {
                "estimator": StatevectorEstimator(),
                "ansatz": RYAnsatz(),
                "prefix": "",
            },
            ValueError,
            "prefix",
        ),
    ],
)
def test_constructor_rejects_invalid_configuration(
    arguments: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        VarQITE(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "exception", "message"),
    [
        ({"time": 0.0}, ValueError, "time"),
        ({"time": float("inf")}, ValueError, "time"),
        ({"num_timesteps": True}, TypeError, "num_timesteps"),
        ({"num_timesteps": 0}, ValueError, "num_timesteps"),
        ({"shots": 0}, ValueError, "shots"),
        ({"seed": -1}, ValueError, "seed"),
        ({"initial_parameters": {}}, ValueError, "parameter bindings"),
        ({"initial_parameters": {"theta_0": float("nan")}}, ValueError, "finite"),
        ({"auxiliary_operators": {"": PauliString.from_str("Z")}}, ValueError, "empty"),
        (
            {"auxiliary_operators": {"wide": PauliString.from_str("ZZ")}},
            ValueError,
            "width",
        ),
    ],
)
def test_evolve_rejects_invalid_input(
    overrides: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "hamiltonian": _z_hamiltonian(),
        "initial_parameters": {"theta_0": 0.2},
        "time": 0.1,
        "num_timesteps": 1,
    }
    arguments.update(overrides)
    with pytest.raises(exception, match=message):
        VarQITE(StatevectorEstimator(), RYAnsatz()).evolve(**arguments)  # type: ignore[arg-type]


def test_evolve_rejects_width_mismatch_and_unsupported_circuit() -> None:
    with pytest.raises(ValueError, match="same qubit count"):
        VarQITE(StatevectorEstimator(), RYAnsatz()).evolve(
            PauliString.from_str("ZZ"),
            {"theta_0": 0.2},
            0.1,
            num_timesteps=1,
        )
    with pytest.raises(ValueError, match="unitary|measure"):
        VarQITE(StatevectorEstimator(), MeasuredAnsatz()).evolve(
            _z_hamiltonian(),
            {"theta_0": 0.2},
            0.1,
            num_timesteps=1,
        )


def test_component_contract_violations_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="trajectory request"):
        VarQITE(InvalidEstimator(), StaticAnsatz()).evolve(
            _z_hamiltonian(),
            {},
            0.1,
            num_timesteps=1,
        )

    monkeypatch.setattr(NaturalGradient, "run", lambda self, estimator, requests: [])
    with pytest.raises(ValueError, match="exactly one result"):
        VarQITE(StatevectorEstimator(), RYAnsatz()).evolve(
            _z_hamiltonian(),
            {"theta_0": 0.2},
            0.1,
            num_timesteps=1,
        )


def test_result_rejects_inconsistent_trajectory() -> None:
    with pytest.raises(ValueError, match="one result per time"):
        VarQITEResult(
            (0.0, 0.1),
            ({}, {}),
            (EstimatorResult(0.0),),
            Circuit(1),
        )
