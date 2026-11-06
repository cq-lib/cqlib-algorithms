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

"""Contract, boundary, integration, and reference tests for PVQD."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from cqlib import Circuit, Parameter
from cqlib.circuit.ansatz import EvolutionStrategy
from cqlib.device import ExecutionResult
from cqlib.qis import Hamiltonian, PauliString, TrotterMode
from cqlib.qis.state import Statevector
from scipy.linalg import expm  # type: ignore[import-untyped]

from cqlib_algorithms.algorithms.time_evolution import PVQD, PVQDResult
from cqlib_algorithms.operators import (
    COBYLAOptimizer,
    GradientDescent,
    ParameterShiftGradient,
    StatevectorEstimator,
    StatevectorSampler,
)
from cqlib_algorithms.protocols import Ansatz, Estimator, Optimizer
from cqlib_algorithms.protocols.estimator import (
    EstimatorRequest,
    EstimatorResult,
)
from cqlib_algorithms.protocols.optimizer import (
    OptimizerRequest,
    OptimizerResult,
    OptimizerStatus,
)
from cqlib_algorithms.protocols.sampler import Sampler, SamplerRequest


class RXAnsatz(Ansatz):
    """One-parameter ansatz that exactly follows an X evolution."""

    __slots__ = ()

    def validate(self) -> None:
        """Validate the fixed ansatz configuration."""

    def build_circuit(self, prefix: str) -> Circuit:
        """Build an RX circuit with one active parameter."""
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        circuit = Circuit(1)
        circuit.rx(0, Parameter(f"{prefix}_0"))
        return circuit

    def num_parameters(self) -> int:
        """Return one trainable parameter."""
        return 1

    def num_qubits(self) -> int:
        """Return one ansatz qubit."""
        return 1


class UniversalOneQubitAnsatz(Ansatz):
    """Two-angle ansatz representing every one-qubit pure state."""

    __slots__ = ()

    def validate(self) -> None:
        """Validate the fixed ansatz configuration."""

    def build_circuit(self, prefix: str) -> Circuit:
        """Build RY followed by RZ in circuit time order."""
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


class ZZEntanglingAnsatz(Ansatz):
    """Prepare ``exp(-i theta ZZ / 2)|++>`` on two qubits."""

    __slots__ = ()

    def validate(self) -> None:
        """Validate the fixed ansatz configuration."""

    def build_circuit(self, prefix: str) -> Circuit:
        """Build the parameterized parity-phase preparation."""
        circuit = Circuit(2)
        circuit.h(0)
        circuit.h(1)
        circuit.cx(0, 1)
        circuit.rz(1, Parameter(f"{prefix}_0"))
        circuit.cx(0, 1)
        return circuit

    def num_parameters(self) -> int:
        """Return one trainable parameter."""
        return 1

    def num_qubits(self) -> int:
        """Return two ansatz qubits."""
        return 2


class PaperIsingAnsatz(Ansatz):
    """Three-layer ansatz used for the paper-style Ising benchmark."""

    __slots__ = ()

    _NUM_QUBITS = 3
    _NUM_LAYERS = 3

    def validate(self) -> None:
        """Validate the fixed ansatz configuration."""

    def build_circuit(self, prefix: str) -> Circuit:
        """Build alternating rotation layers followed by nearest-neighbor ZZ."""
        circuit = Circuit(self._NUM_QUBITS)
        for layer in range(self._NUM_LAYERS):
            for qubit in range(self._NUM_QUBITS):
                angle = Parameter(f"{prefix}_{layer}_rotation_{qubit}")
                if layer % 2 == 0:
                    circuit.rx(qubit, angle)
                else:
                    circuit.ry(qubit, angle)
            for edge in range(self._NUM_QUBITS - 1):
                angle = Parameter(f"{prefix}_{layer}_zz_{edge}")
                circuit.cx(edge, edge + 1)
                circuit.rz(edge + 1, angle)
                circuit.cx(edge, edge + 1)
        return circuit

    def num_parameters(self) -> int:
        """Return the number of rotation and entangling parameters."""
        return self._NUM_LAYERS * (2 * self._NUM_QUBITS - 1)

    def num_qubits(self) -> int:
        """Return the three-qubit benchmark width."""
        return self._NUM_QUBITS


class StaticAnsatz(Ansatz):
    """Zero-parameter identity ansatz of configurable width."""

    __slots__ = ("_num_qubits",)

    def __init__(self, num_qubits: int = 1) -> None:
        self._num_qubits = num_qubits

    def validate(self) -> None:
        """Validate the positive width."""
        if self._num_qubits <= 0:
            raise ValueError("num_qubits must be positive")

    def build_circuit(self, prefix: str) -> Circuit:
        """Build the identity preparation."""
        self.validate()
        return Circuit(self._num_qubits)

    def num_parameters(self) -> int:
        """Return zero trainable parameters."""
        return 0

    def num_qubits(self) -> int:
        """Return the configured width."""
        return self._num_qubits


class DeterministicProbabilitySampler(Sampler):
    """Convert exact all-zero probabilities into deterministic finite counts."""

    __slots__ = ("calls",)

    def __init__(self) -> None:
        self.calls: list[tuple[SamplerRequest, ...]] = []

    def run(
        self,
        requests: Sequence[SamplerRequest],
        /,
    ) -> list[ExecutionResult]:
        """Return rounded exact probabilities without allocating random draws."""
        batch = tuple(requests)
        self.calls.append(batch)
        results: list[ExecutionResult] = []
        for index, request in enumerate(batch):
            request.validate()
            state = Statevector.from_circuit(request.circuit)
            probability = abs(complex(state.data[0])) ** 2
            zero_count = min(request.shots, max(0, round(probability * request.shots)))
            zero = "0" * len(request.qubits)
            nonzero = "0" * (len(request.qubits) - 1) + "1"
            counts: dict[str, int] = {}
            if zero_count:
                counts[zero] = zero_count
            if zero_count < request.shots:
                counts[nonzero] = request.shots - zero_count
            results.append(
                ExecutionResult.from_counts(
                    f"deterministic-{index}",
                    list(request.qubits),
                    request.shots,
                    len(request.qubits),
                    counts,
                    "deterministic",
                )
            )
        return results


class RecordingSampler(Sampler):
    """Record delegated requests before exact statevector sampling."""

    __slots__ = ("_delegate", "requests")

    def __init__(self) -> None:
        self.requests: list[SamplerRequest] = []
        self._delegate = StatevectorSampler()

    def run(
        self,
        requests: Sequence[SamplerRequest],
        /,
    ) -> list[ExecutionResult]:
        """Record and delegate the complete request batch."""
        self.requests.extend(requests)
        return self._delegate.run(requests)


class InvalidOptimizer(Optimizer):
    """Return one selected outer-contract violation."""

    __slots__ = ("_mode",)

    def __init__(self, mode: str) -> None:
        self._mode = mode

    def run(
        self,
        estimator: Estimator,
        requests: Sequence[OptimizerRequest],
        /,
    ) -> list[OptimizerResult]:
        """Return malformed optimizer output selected by ``mode``."""
        request = requests[0]
        names = request.parameter_names
        point = tuple(
            request.estimator_request.parameter_bindings[name] for name in names
        )
        if self._mode == "empty":
            return []
        if self._mode == "wrong_names":
            return [
                OptimizerResult(("wrong",), (0.0,), 0.0, 0, OptimizerStatus.CONVERGED)
            ]
        if self._mode == "outside_loss":
            return [OptimizerResult(names, point, 1.1, 0, OptimizerStatus.CONVERGED)]
        raise AssertionError("unknown mode")


class InvalidEstimator(Estimator):
    """Return no results for every nonempty batch."""

    __slots__ = ()

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        """Return an invalid empty result list."""
        return []


def _x_hamiltonian() -> Hamiltonian:
    """Return the one-qubit Pauli-X Hamiltonian."""
    return Hamiltonian.from_pauli(PauliString.from_str("X"))


def _xz_hamiltonian() -> Hamiltonian:
    """Return a noncommuting one-qubit X/Z Hamiltonian."""
    return Hamiltonian.from_terms(
        1,
        [
            (PauliString.from_str("X"), 0.7),
            (PauliString.from_str("Z"), 0.3),
        ],
    )


def _paper_ising_hamiltonian() -> Hamiltonian:
    """Return the open three-site transverse-field Ising benchmark."""
    return Hamiltonian.from_terms(
        3,
        [
            (PauliString.from_str("ZZI"), 0.25),
            (PauliString.from_str("IZZ"), 0.25),
            (PauliString.from_str("XII"), 1.0),
            (PauliString.from_str("IXI"), 1.0),
            (PauliString.from_str("IIX"), 1.0),
        ],
    )


def _paper_ising_reference_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return independent Hamiltonian and total-magnetization matrices."""
    identity = np.eye(2, dtype=np.complex128)
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    def three_site(
        left: np.ndarray, middle: np.ndarray, right: np.ndarray
    ) -> np.ndarray:
        return np.kron(np.kron(left, middle), right)

    x_terms = (
        three_site(pauli_x, identity, identity),
        three_site(identity, pauli_x, identity),
        three_site(identity, identity, pauli_x),
    )
    z_terms = (
        three_site(pauli_z, identity, identity),
        three_site(identity, pauli_z, identity),
        three_site(identity, identity, pauli_z),
    )
    hamiltonian = (
        0.25 * three_site(pauli_z, pauli_z, identity)
        + 0.25 * three_site(identity, pauli_z, pauli_z)
        + sum(x_terms, start=np.zeros((8, 8), dtype=np.complex128))
    )
    magnetization_x = sum(
        x_terms,
        start=np.zeros((8, 8), dtype=np.complex128),
    )
    magnetization_z = sum(
        z_terms,
        start=np.zeros((8, 8), dtype=np.complex128),
    )
    return hamiltonian, magnetization_x, magnetization_z


def _pvqd(
    ansatz: Ansatz,
    optimizer: Optimizer,
    *,
    sampler: Sampler | None = None,
    estimator: Estimator | None = None,
    strategy: EvolutionStrategy | None = None,
) -> PVQD:
    """Return a PVQD instance with a selected short-time evolution strategy."""
    return PVQD(
        DeterministicProbabilitySampler() if sampler is None else sampler,
        ansatz,
        optimizer,
        estimator=estimator,
        evolution_strategy=EvolutionStrategy.exact() if strategy is None else strategy,
    )


def test_constructor_exposes_injected_components_and_copies_strategy() -> None:
    sampler = DeterministicProbabilitySampler()
    ansatz = RXAnsatz()
    optimizer = COBYLAOptimizer(maxiter=0)
    estimator = StatevectorEstimator()
    strategy = EvolutionStrategy.exact()
    algorithm = PVQD(
        sampler,
        ansatz,
        optimizer,
        estimator=estimator,
        evolution_strategy=strategy,
        prefix="motion",
    )

    assert algorithm.sampler is sampler
    assert algorithm.ansatz is ansatz
    assert algorithm.optimizer is optimizer
    assert algorithm.estimator is estimator
    assert algorithm.prefix == "motion"
    assert algorithm.evolution_strategy == strategy
    assert algorithm.evolution_strategy is not strategy


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("sampler", object(), "sampler"),
        ("ansatz", object(), "ansatz"),
        ("optimizer", object(), "optimizer"),
        ("estimator", object(), "estimator"),
        ("evolution_strategy", object(), "evolution_strategy"),
        ("prefix", 1, "prefix"),
    ],
)
def test_constructor_rejects_invalid_dependencies(
    argument: str,
    value: object,
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "sampler": DeterministicProbabilitySampler(),
        "ansatz": RXAnsatz(),
        "optimizer": COBYLAOptimizer(maxiter=0),
    }
    arguments[argument] = value
    with pytest.raises(TypeError, match=message):
        PVQD(**arguments)  # type: ignore[arg-type]


def test_constructor_rejects_empty_prefix() -> None:
    with pytest.raises(ValueError, match="prefix"):
        PVQD(
            DeterministicProbabilitySampler(),
            RXAnsatz(),
            COBYLAOptimizer(maxiter=0),
            prefix="",
        )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"time": True}, TypeError, "time"),
        ({"time": 0.0}, ValueError, "time"),
        ({"time": math.inf}, ValueError, "finite"),
        ({"num_timesteps": True}, TypeError, "num_timesteps"),
        ({"num_timesteps": 0}, ValueError, "num_timesteps"),
        ({"fidelity_shots": 1.5}, TypeError, "fidelity_shots"),
        ({"fidelity_shots": 0}, ValueError, "fidelity_shots"),
        ({"observable_shots": True}, TypeError, "observable_shots"),
        ({"observable_shots": 0}, ValueError, "observable_shots"),
        ({"seed": True}, TypeError, "seed"),
        ({"seed": -1}, ValueError, "seed"),
        ({"time": 5e-324, "num_timesteps": 2}, ValueError, "time step"),
    ],
)
def test_evolve_rejects_invalid_numeric_options(
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "time": 0.2,
        "num_timesteps": 1,
        "fidelity_shots": 10,
    }
    arguments.update(kwargs)
    algorithm = _pvqd(RXAnsatz(), COBYLAOptimizer(maxiter=0))
    with pytest.raises(error, match=message):
        algorithm.evolve(
            _x_hamiltonian(),
            {"theta_0": 0.0},
            **arguments,  # type: ignore[arg-type]
        )


def test_evolve_rejects_operator_width_and_binding_mismatches() -> None:
    algorithm = _pvqd(RXAnsatz(), COBYLAOptimizer(maxiter=0))
    two_qubit = Hamiltonian.from_pauli(PauliString.from_str("ZZ"))

    with pytest.raises(ValueError, match="same qubit count"):
        algorithm.evolve(
            two_qubit,
            {"theta_0": 0.0},
            0.2,
            num_timesteps=1,
            fidelity_shots=10,
        )
    with pytest.raises(ValueError, match="missing"):
        algorithm.evolve(
            _x_hamiltonian(),
            {},
            0.2,
            num_timesteps=1,
            fidelity_shots=10,
        )


def test_evolve_rejects_invalid_auxiliary_operators() -> None:
    algorithm = _pvqd(RXAnsatz(), COBYLAOptimizer(maxiter=0))
    common = {
        "hamiltonian": _x_hamiltonian(),
        "initial_parameters": {"theta_0": 0.0},
        "time": 0.2,
        "num_timesteps": 1,
        "fidelity_shots": 10,
    }

    with pytest.raises(TypeError, match="mapping"):
        algorithm.evolve(**common, auxiliary_operators=[])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="names"):
        algorithm.evolve(**common, auxiliary_operators={1: _x_hamiltonian()})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="must not be empty"):
        algorithm.evolve(**common, auxiliary_operators={"": _x_hamiltonian()})
    with pytest.raises(ValueError, match="match the evolution width"):
        algorithm.evolve(
            **common,
            auxiliary_operators={"wide": PauliString.from_str("ZZ")},
        )
    with pytest.raises(ValueError, match="estimator is required"):
        algorithm.evolve(
            **common,
            auxiliary_operators={"energy": _x_hamiltonian()},
        )


def test_parameter_shift_tracks_exact_x_evolution_and_observables() -> None:
    optimizer = GradientDescent(
        ParameterShiftGradient(),
        learning_rate=0.4,
        maxiter=80,
        tolerance=1e-10,
    )
    algorithm = _pvqd(
        RXAnsatz(),
        optimizer,
        estimator=StatevectorEstimator(),
    )
    result = algorithm.evolve(
        _x_hamiltonian(),
        {"theta_0": 0.0},
        0.6,
        num_timesteps=3,
        fidelity_shots=100_000_000,
        auxiliary_operators={"z": PauliString.from_str("Z")},
        seed=17,
    )

    assert result.times == pytest.approx((0.0, 0.2, 0.4, 0.6))
    assert tuple(point["theta_0"] for point in result.parameters) == pytest.approx(
        (0.0, 0.4, 0.8, 1.2), abs=2e-4
    )
    assert result.fidelities == pytest.approx((1.0, 1.0, 1.0), abs=2e-8)
    assert tuple(
        entry.expectation for entry in result.observables["z"]
    ) == pytest.approx(
        tuple(math.cos(2.0 * time) for time in result.times),
        abs=2e-4,
    )
    assert result.estimated_error < 1e-7


def test_initial_mapping_order_does_not_change_ansatz_parameter_order() -> None:
    algorithm = _pvqd(
        UniversalOneQubitAnsatz(),
        COBYLAOptimizer(maxiter=0),
    )
    result = algorithm.evolve(
        PauliString.from_str("Z"),
        {"theta_azimuth": 0.2, "theta_polar": 0.1},
        0.1,
        num_timesteps=1,
        fidelity_shots=100,
        seed=19,
    )

    assert tuple(result.parameters[0]) == ("theta_polar", "theta_azimuth")
    assert result.optimizer_results[0].parameter_names == (
        "theta_polar",
        "theta_azimuth",
    )


@pytest.mark.reference
def test_noncommuting_hamiltonian_matches_independent_matrix_evolution() -> None:
    algorithm = _pvqd(
        UniversalOneQubitAnsatz(),
        COBYLAOptimizer(maxiter=600, rhobeg=0.4, tolerance=1e-8),
        strategy=EvolutionStrategy.trotter(TrotterMode.second_order(), 16),
    )
    result = algorithm.evolve(
        _xz_hamiltonian(),
        {"theta_polar": 0.0, "theta_azimuth": 0.0},
        0.7,
        num_timesteps=4,
        fidelity_shots=100_000_000,
        seed=123,
    )

    actual = np.asarray(Statevector.from_circuit(result.evolved_circuit).data)
    matrix = np.array([[0.3, 0.7], [0.7, -0.3]], dtype=np.complex128)
    expected = expm(-1j * 0.7 * matrix) @ np.array([1.0, 0.0], dtype=np.complex128)
    fidelity = abs(np.vdot(expected, actual)) ** 2

    assert fidelity > 1.0 - 2e-6
    assert min(result.fidelities) > 1.0 - 2e-7


@pytest.mark.reference
def test_paper_transverse_field_ising_multistep_benchmark() -> None:
    """Reproduce six of the paper benchmark's ``0.05`` evolution steps."""
    ansatz = PaperIsingAnsatz()
    time_step = 0.05
    num_timesteps = 6
    total_time = time_step * num_timesteps
    algorithm = _pvqd(
        ansatz,
        COBYLAOptimizer(maxiter=3_000, rhobeg=0.02, tolerance=1e-8),
        estimator=StatevectorEstimator(),
        strategy=EvolutionStrategy.trotter(TrotterMode.second_order(), 8),
    )
    magnetization_x = Hamiltonian.from_terms(
        3,
        [
            (PauliString.from_str("XII"), 1.0),
            (PauliString.from_str("IXI"), 1.0),
            (PauliString.from_str("IIX"), 1.0),
        ],
    )
    magnetization_z = Hamiltonian.from_terms(
        3,
        [
            (PauliString.from_str("ZII"), 1.0),
            (PauliString.from_str("IZI"), 1.0),
            (PauliString.from_str("IIZ"), 1.0),
        ],
    )
    template = ansatz.build_circuit("theta")
    initial_parameters = {name: 0.0 for name in template.used_symbols}

    result = algorithm.evolve(
        _paper_ising_hamiltonian(),
        initial_parameters,
        total_time,
        num_timesteps=num_timesteps,
        fidelity_shots=1_000_000_000,
        auxiliary_operators={
            "magnetization_x": magnetization_x,
            "magnetization_z": magnetization_z,
        },
        seed=211,
    )

    reference_hamiltonian, reference_x, reference_z = _paper_ising_reference_matrices()
    initial_state = np.zeros(8, dtype=np.complex128)
    initial_state[0] = 1.0
    trajectory_fidelities: list[float] = []
    expected_x: list[float] = []
    expected_z: list[float] = []
    for time, parameters in zip(result.times, result.parameters, strict=True):
        expected_state = expm(-1j * time * reference_hamiltonian) @ initial_state
        actual_circuit = template.assign_parameters(dict(parameters))
        actual_state = np.asarray(Statevector.from_circuit(actual_circuit).data)
        trajectory_fidelities.append(abs(np.vdot(expected_state, actual_state)) ** 2)
        expected_x.append(
            float(np.vdot(expected_state, reference_x @ expected_state).real)
        )
        expected_z.append(
            float(np.vdot(expected_state, reference_z @ expected_state).real)
        )

    integrated_infidelity = time_step * (
        0.5 * (1.0 - trajectory_fidelities[0])
        + sum(1.0 - value for value in trajectory_fidelities[1:-1])
        + 0.5 * (1.0 - trajectory_fidelities[-1])
    )
    actual_x = tuple(
        estimate.expectation for estimate in result.observables["magnetization_x"]
    )
    actual_z = tuple(
        estimate.expectation for estimate in result.observables["magnetization_z"]
    )

    assert result.times == pytest.approx(
        tuple(time_step * step for step in range(num_timesteps + 1))
    )
    assert min(result.fidelities) > 1.0 - 2e-6
    assert min(trajectory_fidelities) > 1.0 - 2e-5
    assert integrated_infidelity < 2e-6
    assert actual_x == pytest.approx(expected_x, abs=2e-4)
    assert actual_z == pytest.approx(expected_z, abs=2e-4)


@pytest.mark.reference
def test_two_qubit_entangling_trajectory_matches_exact_zz_evolution() -> None:
    algorithm = _pvqd(
        ZZEntanglingAnsatz(),
        GradientDescent(
            ParameterShiftGradient(),
            learning_rate=0.4,
            maxiter=80,
            tolerance=1e-10,
        ),
    )
    result = algorithm.evolve(
        PauliString.from_str("ZZ"),
        {"theta_0": 0.0},
        0.5,
        num_timesteps=2,
        fidelity_shots=100_000_000,
        seed=43,
    )

    actual = np.asarray(Statevector.from_circuit(result.evolved_circuit).data)
    zz = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.complex128)
    initial = np.full(4, 0.5, dtype=np.complex128)
    expected = expm(-0.5j * zz) @ initial

    assert abs(np.vdot(expected, actual)) ** 2 > 1.0 - 2e-7
    assert tuple(point["theta_0"] for point in result.parameters) == pytest.approx(
        (0.0, 0.5, 1.0), abs=2e-4
    )


def test_static_ansatz_returns_projection_fidelity_without_parameters() -> None:
    algorithm = _pvqd(StaticAnsatz(), COBYLAOptimizer(maxiter=0))
    result = algorithm.evolve(
        _x_hamiltonian(),
        {},
        0.25,
        num_timesteps=1,
        fidelity_shots=1_000_000,
        seed=5,
    )

    assert result.parameters == ({}, {})
    assert result.optimizer_results[0].status is OptimizerStatus.NO_PARAMETERS
    assert result.fidelities[0] == pytest.approx(math.cos(0.25) ** 2, abs=1e-6)


def test_master_seed_reproduces_and_independently_seeds_sampling_requests() -> None:
    first_sampler = RecordingSampler()
    second_sampler = RecordingSampler()
    first = _pvqd(StaticAnsatz(), COBYLAOptimizer(maxiter=0), sampler=first_sampler)
    second = _pvqd(StaticAnsatz(), COBYLAOptimizer(maxiter=0), sampler=second_sampler)

    first_result = first.evolve(
        _x_hamiltonian(),
        {},
        0.4,
        num_timesteps=2,
        fidelity_shots=257,
        seed=91,
    )
    second_result = second.evolve(
        _x_hamiltonian(),
        {},
        0.4,
        num_timesteps=2,
        fidelity_shots=257,
        seed=91,
    )

    first_seeds = [request.seed for request in first_sampler.requests]
    second_seeds = [request.seed for request in second_sampler.requests]
    assert first_result.fidelities == second_result.fidelities
    assert first_seeds == second_seeds
    assert all(seed is not None for seed in first_seeds)
    assert len(set(first_seeds)) == len(first_seeds)


@pytest.mark.parametrize("mode", ["empty", "wrong_names", "outside_loss"])
def test_evolve_rejects_invalid_optimizer_responses(mode: str) -> None:
    algorithm = _pvqd(RXAnsatz(), InvalidOptimizer(mode))
    with pytest.raises((TypeError, ValueError)):
        algorithm.evolve(
            _x_hamiltonian(),
            {"theta_0": 0.0},
            0.2,
            num_timesteps=1,
            fidelity_shots=10,
        )


def test_evolve_rejects_invalid_auxiliary_estimator_response() -> None:
    algorithm = _pvqd(
        StaticAnsatz(),
        COBYLAOptimizer(maxiter=0),
        estimator=InvalidEstimator(),
    )
    with pytest.raises(ValueError, match="one result per observable"):
        algorithm.evolve(
            _x_hamiltonian(),
            {},
            0.2,
            num_timesteps=1,
            fidelity_shots=10,
            auxiliary_operators={"z": PauliString.from_str("Z")},
        )


def test_inputs_remain_unchanged_and_result_is_immutable() -> None:
    hamiltonian = _x_hamiltonian()
    before = hamiltonian.to_matrix()
    initial = {"theta_0": 0.1}
    algorithm = _pvqd(RXAnsatz(), COBYLAOptimizer(maxiter=0))
    result = algorithm.evolve(
        hamiltonian,
        initial,
        0.2,
        num_timesteps=1,
        fidelity_shots=100,
        seed=3,
    )

    assert np.array_equal(hamiltonian.to_matrix(), before)
    assert initial == {"theta_0": 0.1}
    assert not hasattr(result, "__dict__")
    assert "PVQDResult" in repr(result)
    with pytest.raises(TypeError):
        result.final_parameters["theta_0"] = 2.0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        result.times = (0.0, 1.0)  # type: ignore[misc]


def test_result_rejects_inconsistent_manual_metadata() -> None:
    optimizer_result = OptimizerResult(
        ("theta",),
        (0.2,),
        0.0,
        1,
        OptimizerStatus.CONVERGED,
    )
    circuit = Circuit(1)

    with pytest.raises(ValueError, match="one entry per time"):
        PVQDResult(
            (0.0, 0.1),
            ({"theta": 0.0},),
            (1.0,),
            (optimizer_result,),
            circuit,
        )
    with pytest.raises(ValueError, match="parameters must match"):
        PVQDResult(
            (0.0, 0.1),
            ({"theta": 0.0}, {"theta": 0.3}),
            (1.0,),
            (optimizer_result,),
            circuit,
        )


def test_evolution_strategy_and_source_circuits_are_defensively_isolated() -> None:
    strategy = EvolutionStrategy.auto(1)
    algorithm = PVQD(
        DeterministicProbabilitySampler(),
        RXAnsatz(),
        COBYLAOptimizer(maxiter=0),
        evolution_strategy=strategy,
    )
    first = algorithm.evolve(
        _x_hamiltonian(),
        {"theta_0": 0.0},
        0.2,
        num_timesteps=1,
        fidelity_shots=100,
        seed=8,
    )
    second = algorithm.evolve(
        _x_hamiltonian(),
        {"theta_0": 0.0},
        0.2,
        num_timesteps=1,
        fidelity_shots=100,
        seed=8,
    )

    assert first.fidelities == second.fidelities
    assert first.evolved_circuit is not second.evolved_circuit
    assert copy.copy(algorithm.evolution_strategy) == strategy
