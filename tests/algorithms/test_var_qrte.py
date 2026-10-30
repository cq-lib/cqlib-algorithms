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

"""Contract, boundary, and independent-reference tests for VarQRTE."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from cqlib import Circuit, Parameter
from cqlib.qis import Hamiltonian, PauliString
from cqlib.qis.state import Statevector
from scipy.linalg import expm  # type: ignore[import-untyped]

from cqlib_algorithms.algorithms.time_evolution.var_qrte import (
    VarQRTE,
    VarQRTEResult,
)
from cqlib_algorithms.operators import (
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


class BlochSphereAnsatz(Ansatz):
    """One-qubit manifold with polar and azimuthal angles."""

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


class TwoQubitAnsatz(Ansatz):
    """Three-angle complex entangled manifold."""

    __slots__ = ()

    def validate(self) -> None:
        """Validate the fixed ansatz configuration."""

    def build_circuit(self, prefix: str) -> Circuit:
        """Build the two-qubit reference ansatz."""
        first = Parameter(f"{prefix}_first")
        second = Parameter(f"{prefix}_second")
        third = Parameter(f"{prefix}_third")
        circuit = Circuit(2)
        circuit.ry(0, first)
        circuit.rx(1, second)
        circuit.cx(0, 1)
        circuit.rz(1, third + 0.3 * first)
        return circuit

    def num_parameters(self) -> int:
        """Return three trainable parameters."""
        return 3

    def num_qubits(self) -> int:
        """Return two ansatz qubits."""
        return 2


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


class StaticAnsatz(Ansatz):
    """Zero-parameter identity manifold."""

    __slots__ = ()

    def validate(self) -> None:
        """Validate the fixed ansatz configuration."""

    def build_circuit(self, prefix: str) -> Circuit:
        """Build a one-qubit identity circuit."""
        return Circuit(1)

    def num_parameters(self) -> int:
        """Return zero trainable parameters."""
        return 0

    def num_qubits(self) -> int:
        """Return one ansatz qubit."""
        return 1


class MeasuredAnsatz(RYAnsatz):
    """RY manifold followed by an unsupported measurement."""

    def build_circuit(self, prefix: str) -> Circuit:
        """Build a measured parameterized circuit."""
        circuit = super().build_circuit(prefix)
        circuit.measure(0)
        return circuit


class RecordingExactEstimator(Estimator):
    """Record requests while evaluating equivalent exact values."""

    __slots__ = ("_delegate", "batches")

    def __init__(self) -> None:
        self._delegate = StatevectorEstimator()
        self.batches: list[tuple[EstimatorRequest, ...]] = []

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        """Record requests and delegate exact evaluations."""
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
    """Return an invalid empty response."""

    __slots__ = ()

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        """Return no estimates."""
        return []


def _one_qubit_hamiltonian() -> Hamiltonian:
    """Return a phase-sensitive one-qubit Hamiltonian."""
    return Hamiltonian.from_terms(
        1,
        [
            (PauliString.from_str("X"), 0.6),
            (PauliString.from_str("Y"), -0.35),
            (PauliString.from_str("Z"), 0.2),
        ],
    )


def _one_qubit_hamiltonian_matrix() -> np.ndarray:
    """Return the independent dense one-qubit Hamiltonian."""
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    pauli_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    pauli_z = np.diag([1.0, -1.0]).astype(np.complex128)
    return 0.6 * pauli_x - 0.35 * pauli_y + 0.2 * pauli_z


def _two_qubit_hamiltonian() -> Hamiltonian:
    """Return the two-qubit Hamiltonian used by the dense reference."""
    return Hamiltonian.from_terms(
        2,
        [
            (PauliString.from_str("ZI"), 0.61),
            (PauliString.from_str("XX"), -0.37),
            (PauliString.from_str("IY"), 0.23),
            (PauliString.from_str("IZ"), 0.19),
        ],
    )


def _two_qubit_hamiltonian_matrix() -> np.ndarray:
    """Return an independent dense two-qubit Hamiltonian."""
    identity = np.eye(2, dtype=np.complex128)
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    pauli_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    pauli_z = np.diag([1.0, -1.0]).astype(np.complex128)
    return (
        0.61 * np.kron(pauli_z, identity)
        - 0.37 * np.kron(pauli_x, pauli_x)
        + 0.23 * np.kron(identity, pauli_y)
        + 0.19 * np.kron(identity, pauli_z)
    )


def _dense_reference_step(
    template: Circuit,
    parameters: Mapping[str, float],
    names: tuple[str, ...],
    hamiltonian: np.ndarray,
    time_step: float,
    regularization: float,
) -> dict[str, float]:
    """Return one independently differenced real-time McLachlan step."""
    difference = 1e-6

    def state(bindings: Mapping[str, float]) -> np.ndarray:
        bound = template.assign_parameters(dict(bindings))
        return np.asarray(bound.to_matrix([0, 1])[:, 0], dtype=np.complex128)

    psi = state(parameters)
    derivatives: list[np.ndarray] = []
    for name in names:
        plus = dict(parameters)
        minus = dict(parameters)
        plus[name] += difference
        minus[name] -= difference
        derivatives.append((state(plus) - state(minus)) / (2.0 * difference))

    metric = np.empty((len(names), len(names)), dtype=np.float64)
    for row, first in enumerate(derivatives):
        for column, second in enumerate(derivatives):
            metric[row, column] = float(
                np.real(
                    np.vdot(first, second) - np.vdot(first, psi) * np.vdot(psi, second)
                )
            )
    hamiltonian_state = hamiltonian @ psi
    energy = float(np.real(np.vdot(psi, hamiltonian_state)))
    force = np.asarray(
        [
            np.imag(
                np.vdot(derivative, hamiltonian_state)
                - np.vdot(derivative, psi) * energy
            )
            for derivative in derivatives
        ]
    )
    if regularization == 0.0:
        velocity = np.linalg.lstsq(metric, force, rcond=None)[0]
    else:
        velocity = np.linalg.solve(
            metric + regularization * np.eye(len(names)),
            force,
        )
    next_values = np.asarray([parameters[name] for name in names])
    next_values += time_step * velocity
    return dict(zip(names, (float(value) for value in next_values), strict=True))


def test_ry_under_y_fixes_sign_imaginary_part_and_qgt_factor() -> None:
    result = VarQRTE(StatevectorEstimator(), RYAnsatz()).evolve(
        PauliString.from_str("Y"),
        {"theta_0": 0.37},
        0.08,
        num_timesteps=1,
    )

    assert result.final_parameters["theta_0"] == pytest.approx(0.53, abs=1e-12)


def test_identity_shift_preserves_phase_corrected_trajectory() -> None:
    initial = {"theta_polar": math.pi / 3.0, "theta_azimuth": -0.4}
    base = Hamiltonian.from_pauli(PauliString.from_str("Z"))
    shifted = Hamiltonian.from_terms(
        1,
        [
            (PauliString.from_str("Z"), 1.0),
            (PauliString.from_str("I"), 7.0),
        ],
    )
    algorithm = VarQRTE(StatevectorEstimator(), BlochSphereAnsatz())

    first = algorithm.evolve(base, initial, 0.12, num_timesteps=3)
    second = algorithm.evolve(shifted, initial, 0.12, num_timesteps=3)

    for first_parameters, second_parameters in zip(
        first.parameters, second.parameters, strict=True
    ):
        assert dict(first_parameters) == pytest.approx(second_parameters, abs=1e-14)
    assert first.final_parameters["theta_polar"] == pytest.approx(
        initial["theta_polar"], abs=1e-12
    )
    assert first.final_parameters["theta_azimuth"] == pytest.approx(-0.16, abs=1e-12)


def test_large_coefficients_with_cancelling_energy_remain_valid() -> None:
    coefficient = 1e16
    hamiltonian = Hamiltonian.from_terms(
        1,
        [
            (PauliString.from_str("X"), coefficient),
            (PauliString.from_str("Y"), coefficient),
        ],
    )

    result = VarQRTE(StatevectorEstimator(), BlochSphereAnsatz()).evolve(
        hamiltonian,
        {"theta_polar": 0.9, "theta_azimuth": -math.pi / 4.0},
        1e-18,
        num_timesteps=1,
    )

    assert result.energies[0].expectation == pytest.approx(0.0, abs=2.0)
    assert all(math.isfinite(value) for value in result.final_parameters.values())


@pytest.mark.reference
def test_phase_sensitive_trajectory_matches_exact_real_time() -> None:
    polar = 0.83
    azimuth = -0.61
    total_time = 0.2
    result = VarQRTE(StatevectorEstimator(), BlochSphereAnsatz()).evolve(
        _one_qubit_hamiltonian(),
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
    exact = expm(-1j * total_time * _one_qubit_hamiltonian_matrix()) @ initial
    actual = np.asarray(Statevector.from_circuit(result.evolved_circuit).data)

    assert abs(np.vdot(exact, actual)) ** 2 == pytest.approx(1.0, abs=2e-8)


@pytest.mark.reference
def test_two_qubit_steps_match_independent_center_difference_reference() -> None:
    ansatz = TwoQubitAnsatz()
    template = ansatz.build_circuit("theta")
    names = tuple(template.used_symbols)
    expected: dict[str, float] = {
        "theta_first": 0.31,
        "theta_second": -0.47,
        "theta_third": 0.22,
    }
    time_step = 0.015
    regularization = 0.11
    for _ in range(3):
        expected = _dense_reference_step(
            template,
            expected,
            names,
            _two_qubit_hamiltonian_matrix(),
            time_step,
            regularization,
        )

    result = VarQRTE(
        StatevectorEstimator(),
        ansatz,
        regularization=regularization,
    ).evolve(
        _two_qubit_hamiltonian(),
        {"theta_first": 0.31, "theta_second": -0.47, "theta_third": 0.22},
        3.0 * time_step,
        num_timesteps=3,
    )

    assert dict(result.final_parameters) == pytest.approx(expected, abs=5e-7)


@pytest.mark.parametrize("regularization", [0.0, 0.2])
def test_singular_metric_uses_minimum_norm_or_ridge_solution(
    regularization: float,
) -> None:
    initial = {"theta_first": 0.2, "theta_second": 0.35}
    metric = 0.25 * np.ones((2, 2))
    force = 0.5 * np.ones(2)
    if regularization == 0.0:
        velocity = np.linalg.lstsq(metric, force, rcond=None)[0]
    else:
        velocity = np.linalg.solve(
            metric + regularization * np.eye(2),
            force,
        )
    expected = np.asarray(list(initial.values())) + 0.04 * velocity

    result = VarQRTE(
        StatevectorEstimator(),
        RedundantRYAnsatz(),
        regularization=regularization,
    ).evolve(
        PauliString.from_str("Y"),
        initial,
        0.04,
        num_timesteps=1,
    )

    assert tuple(result.final_parameters.values()) == pytest.approx(expected, abs=1e-10)


def test_result_records_immutable_trajectory_and_auxiliary_observables() -> None:
    initial = {"motion_0": 0.3}
    auxiliaries = {"x": PauliString.from_str("X")}
    result = VarQRTE(
        StatevectorEstimator(),
        RYAnsatz(),
        prefix="motion",
    ).evolve(
        PauliString.from_str("Y"),
        initial,
        0.1,
        num_timesteps=2,
        auxiliary_operators=auxiliaries,
    )

    assert result.times == pytest.approx((0.0, 0.05, 0.1))
    assert len(result.parameters) == len(result.energies) == 3
    assert len(result.observables["x"]) == 3
    assert result.evolved_circuit.used_symbols == []
    assert initial == {"motion_0": 0.3}
    assert auxiliaries == {"x": PauliString.from_str("X")}
    with pytest.raises(TypeError):
        result.final_parameters["motion_0"] = 0.0  # type: ignore[index]
    with pytest.raises(TypeError):
        result.observables["new"] = ()  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        result.times = ()  # type: ignore[misc]
    assert repr(result).startswith("VarQRTEResult(final_time=0.1")
    assert "Circuit" not in repr(result)


def test_observable_shots_and_seeds_do_not_change_parameters() -> None:
    first_estimator = RecordingExactEstimator()
    second_estimator = RecordingExactEstimator()
    arguments = {
        "hamiltonian": PauliString.from_str("Y"),
        "initial_parameters": {"theta_0": 0.2},
        "time": 0.1,
        "num_timesteps": 2,
        "observable_shots": 64,
        "auxiliary_operators": {"x": PauliString.from_str("X")},
    }
    first = VarQRTE(first_estimator, RYAnsatz()).evolve(seed=37, **arguments)
    second = VarQRTE(second_estimator, RYAnsatz()).evolve(seed=38, **arguments)

    assert first.parameters == second.parameters
    assert len(first_estimator.batches) == len(second_estimator.batches) == 1
    assert [request.seed for request in first_estimator.batches[0]] == spawn_seeds(
        37, 6
    )
    assert [request.seed for request in second_estimator.batches[0]] == spawn_seeds(
        38, 6
    )
    assert all(request.shots == 64 for request in first_estimator.batches[0])


@pytest.mark.integration
def test_sampling_estimator_integration_preserves_exact_parameter_path() -> None:
    sampled = VarQRTE(
        SamplingEstimator(StatevectorSampler()),
        RYAnsatz(),
    ).evolve(
        PauliString.from_str("Y"),
        {"theta_0": 0.2},
        0.1,
        num_timesteps=2,
        observable_shots=4096,
        seed=91,
    )
    exact = VarQRTE(StatevectorEstimator(), RYAnsatz()).evolve(
        PauliString.from_str("Y"),
        {"theta_0": 0.2},
        0.1,
        num_timesteps=2,
    )

    assert sampled.parameters == exact.parameters
    assert all(result.variance is not None for result in sampled.energies)


def test_zero_hamiltonian_and_static_ansatz_are_stationary() -> None:
    active = VarQRTE(StatevectorEstimator(), RYAnsatz()).evolve(
        Hamiltonian(1),
        {"theta_0": 0.37},
        0.2,
        num_timesteps=2,
    )
    static = VarQRTE(StatevectorEstimator(), StaticAnsatz()).evolve(
        PauliString.from_str("Z"),
        {},
        0.2,
        num_timesteps=2,
    )

    assert [dict(item) for item in active.parameters] == [{"theta_0": 0.37}] * 3
    assert static.parameters == ({}, {}, {})
    assert [result.expectation for result in static.energies] == [1.0, 1.0, 1.0]


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
            {"estimator": StatevectorEstimator(), "ansatz": RYAnsatz(), "prefix": ""},
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
        VarQRTE(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "exception", "message"),
    [
        ({"time": 0.0}, ValueError, "time"),
        ({"time": float("inf")}, ValueError, "time"),
        ({"num_timesteps": True}, TypeError, "num_timesteps"),
        ({"num_timesteps": 0}, ValueError, "num_timesteps"),
        ({"observable_shots": 0}, ValueError, "shots"),
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
        "hamiltonian": PauliString.from_str("Y"),
        "initial_parameters": {"theta_0": 0.2},
        "time": 0.1,
        "num_timesteps": 1,
    }
    arguments.update(overrides)
    with pytest.raises(exception, match=message):
        VarQRTE(StatevectorEstimator(), RYAnsatz()).evolve(**arguments)  # type: ignore[arg-type]


def test_evolve_rejects_width_mismatch_and_unsupported_circuit() -> None:
    with pytest.raises(ValueError, match="same qubit count"):
        VarQRTE(StatevectorEstimator(), RYAnsatz()).evolve(
            PauliString.from_str("ZZ"),
            {"theta_0": 0.2},
            0.1,
            num_timesteps=1,
        )
    with pytest.raises(ValueError, match="unitary|measure"):
        VarQRTE(StatevectorEstimator(), MeasuredAnsatz()).evolve(
            PauliString.from_str("Y"),
            {"theta_0": 0.2},
            0.1,
            num_timesteps=1,
        )


def test_invalid_estimator_response_is_rejected() -> None:
    with pytest.raises(ValueError, match="trajectory request"):
        VarQRTE(InvalidEstimator(), StaticAnsatz()).evolve(
            PauliString.from_str("Z"),
            {},
            0.1,
            num_timesteps=1,
        )


def test_result_rejects_inconsistent_trajectory() -> None:
    with pytest.raises(ValueError, match="one result per time"):
        VarQRTEResult(
            (0.0, 0.1),
            ({}, {}),
            (EstimatorResult(0.0),),
            Circuit(1),
        )
