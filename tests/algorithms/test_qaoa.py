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

"""Integration, reference and boundary tests for QAOA."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pytest
from cqlib import Circuit, Parameter
from cqlib.circuit.ansatz import EvolutionStrategy
from cqlib.qis import Hamiltonian, Pauli, PauliString

from cqlib_algorithms.algorithms import QAOA, QAOAResult, VQAResult, VQEResult
from cqlib_algorithms.operators.estimator import StatevectorEstimator
from cqlib_algorithms.operators.gradient import ParameterShiftGradient
from cqlib_algorithms.operators.optimizer import GradientDescent
from cqlib_algorithms.protocols import (
    Estimator,
    EstimatorRequest,
    EstimatorResult,
    Optimizer,
    OptimizerRequest,
    OptimizerResult,
    OptimizerStatus,
    StatePreparation,
)


class RecordingEstimator(Estimator):
    def run(self, requests: Sequence[EstimatorRequest], /) -> list[EstimatorResult]:
        return [EstimatorResult(0.0, 0.0) for _ in requests]


class RecordingOptimizer(Optimizer):
    def __init__(self, value: float = -0.75) -> None:
        self.value = value
        self.estimator: Estimator | None = None
        self.batches: list[list[OptimizerRequest]] = []

    def run(
        self,
        estimator: Estimator,
        requests: Sequence[OptimizerRequest],
        /,
    ) -> list[OptimizerResult]:
        self.estimator = estimator
        batch = list(requests)
        self.batches.append(batch)
        return [
            OptimizerResult(
                request.parameter_names,
                tuple(
                    request.estimator_request.parameter_bindings[name]
                    for name in request.parameter_names
                ),
                self.value,
                0,
                OptimizerStatus.MAX_ITERATIONS,
            )
            for request in batch
        ]


class BasisOnePreparation(StatePreparation):
    def __init__(self, num_qubits: int) -> None:
        self.width = num_qubits

    def validate(self) -> None:
        return None

    def build_circuit(self, prefix: str) -> Circuit:
        circuit = Circuit(self.width)
        circuit.x(0)
        return circuit

    def num_parameters(self) -> int:
        return 0

    def num_qubits(self) -> int:
        return self.width


class ActivePreparation(BasisOnePreparation):
    def build_circuit(self, prefix: str) -> Circuit:
        circuit = Circuit(self.width)
        circuit.ry(0, Parameter(f"{prefix}_input"))
        return circuit

    def num_parameters(self) -> int:
        return 1


def _maxcut_cost(num_qubits: int, edges: Sequence[tuple[int, int]]) -> Hamiltonian:
    cost = Hamiltonian(num_qubits)
    identity = PauliString(num_qubits)
    cost.add_term(identity, len(edges) / 2.0)
    for left, right in edges:
        term = PauliString(num_qubits)
        term.set_pauli(left, Pauli.z())
        term.set_pauli(right, Pauli.z())
        cost.add_term(term, -0.5)
    return cost


def _dense_x_mixer(num_qubits: int) -> np.ndarray:
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    identity = np.eye(2, dtype=np.complex128)
    result = np.zeros((2**num_qubits, 2**num_qubits), dtype=np.complex128)
    for qubit in range(num_qubits):
        factors = [
            x if index == qubit else identity for index in reversed(range(num_qubits))
        ]
        term = factors[0]
        for factor in factors[1:]:
            term = np.kron(term, factor)
        result += term
    return result


def _matrix_exponential(matrix: np.ndarray, angle: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.exp(-1j * angle * values)) @ vectors.conj().T


def _hadamards(num_qubits: int) -> np.ndarray:
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
    result = np.array([[1.0]], dtype=np.complex128)
    for _ in range(num_qubits):
        result = np.kron(result, hadamard / math.sqrt(2.0))
    return result


def _recording_run(
    *,
    reps: int = 2,
    mixer_operator: Hamiltonian | None = None,
    initial_state: StatePreparation | None = None,
    cost_strategy: EvolutionStrategy | None = None,
    mixer_strategy: EvolutionStrategy | None = None,
) -> tuple[QAOAResult, RecordingEstimator, RecordingOptimizer]:
    estimator = RecordingEstimator()
    optimizer = RecordingOptimizer()
    algorithm = QAOA(
        estimator,
        optimizer,
        reps=reps,
        mixer_operator=mixer_operator,
        initial_state=initial_state,
        cost_strategy=cost_strategy,
        mixer_strategy=mixer_strategy,
        prefix="angle",
    )
    point = {
        name: value
        for layer in range(reps)
        for name, value in (
            (f"angle_{layer}_gamma", 0.17 + 0.1 * layer),
            (f"angle_{layer}_beta", -0.23 + 0.05 * layer),
        )
    }
    result = algorithm.compute_maximum(
        _maxcut_cost(2, [(0, 1)]),
        point,
        shots=321,
    )
    return result, estimator, optimizer


def test_qaoa_delegates_negated_objective_and_preserves_parameter_order() -> None:
    result, estimator, optimizer = _recording_run()

    assert optimizer.estimator is estimator
    assert len(optimizer.batches) == 1
    request = optimizer.batches[0][0]
    assert request.parameter_names == (
        "angle_0_gamma",
        "angle_0_beta",
        "angle_1_gamma",
        "angle_1_beta",
    )
    assert request.estimator_request.shots == 321
    expected_cost = _maxcut_cost(2, [(0, 1)]).to_matrix()
    np.testing.assert_allclose(
        request.estimator_request.observable.to_matrix(),
        -expected_cost,
        rtol=0.0,
        atol=0.0,
    )
    assert result.optimal_value == pytest.approx(0.75)
    assert result.optimizer_result.optimal_value == pytest.approx(-0.75)
    assert result.optimal_parameters == request.estimator_request.parameter_bindings
    assert result.optimal_circuit.used_symbols == []


@pytest.mark.reference
def test_two_layer_circuit_matches_independent_dense_matrix_in_exact_order() -> None:
    _, _, optimizer = _recording_run()
    request = optimizer.batches[0][0]
    bindings = request.estimator_request.parameter_bindings
    actual = request.estimator_request.circuit.assign_parameters(
        dict(bindings)
    ).to_matrix()

    cost = _maxcut_cost(2, [(0, 1)]).to_matrix()
    mixer = _dense_x_mixer(2)
    expected = _hadamards(2)
    for layer in range(2):
        expected = (
            _matrix_exponential(mixer, bindings[f"angle_{layer}_beta"])
            @ _matrix_exponential(cost, bindings[f"angle_{layer}_gamma"])
            @ expected
        )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.reference
def test_single_edge_maxcut_reaches_known_optimum_and_optimal_bitstrings() -> None:
    algorithm = QAOA(
        StatevectorEstimator(),
        GradientDescent(
            ParameterShiftGradient(),
            learning_rate=0.2,
            maxiter=10,
            tolerance=1e-12,
        ),
    )

    result = algorithm.run(
        _maxcut_cost(2, [(0, 1)]),
        {
            "qaoa_0_gamma": math.pi / 2.0,
            "qaoa_0_beta": math.pi / 8.0,
        },
        objective_sense="maximize",
    )

    probabilities = np.abs(result.optimal_circuit.to_matrix()[:, 0]) ** 2
    assert result.optimal_value == pytest.approx(1.0, abs=1e-12)
    assert probabilities[1] + probabilities[2] == pytest.approx(1.0, abs=1e-12)
    assert probabilities[0] + probabilities[3] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.reference
def test_three_qubit_path_matches_brute_force_classical_optimum() -> None:
    edges = [(0, 1), (1, 2)]
    cost = _maxcut_cost(3, edges)
    classical_values = [
        sum(((state >> left) & 1) != ((state >> right) & 1) for left, right in edges)
        for state in range(2**3)
    ]
    expected = float(max(classical_values))
    algorithm = QAOA(
        StatevectorEstimator(),
        GradientDescent(
            ParameterShiftGradient(),
            learning_rate=0.1,
            maxiter=1,
            tolerance=1e-10,
        ),
        reps=2,
    )

    result = algorithm.compute_maximum(
        cost,
        {
            "qaoa_0_gamma": 1.0623271687611562,
            "qaoa_0_beta": 0.5198163750795909,
            "qaoa_1_gamma": 1.7576073444416485,
            "qaoa_1_beta": 0.3589872592794749,
        },
    )

    probabilities = np.abs(result.optimal_circuit.to_matrix()[:, 0]) ** 2
    optimal_probability = sum(
        probabilities[index]
        for index, value in enumerate(classical_values)
        if value == expected
    )
    assert expected == 2.0
    assert result.optimal_value == pytest.approx(expected, abs=1e-11)
    assert optimal_probability == pytest.approx(1.0, abs=1e-11)


def test_custom_fixed_initial_state_and_exact_custom_mixer_are_composed() -> None:
    mixer = Hamiltonian(2)
    mixer.add_term(PauliString.from_str("XI"), 0.4)
    mixer.add_term(PauliString.from_str("IX"), -0.2)
    result, _, optimizer = _recording_run(
        reps=1,
        mixer_operator=mixer,
        initial_state=BasisOnePreparation(2),
        cost_strategy=EvolutionStrategy.exact(),
        mixer_strategy=EvolutionStrategy.exact(),
    )
    request = optimizer.batches[0][0]
    bindings = request.estimator_request.parameter_bindings
    actual_state = request.estimator_request.circuit.assign_parameters(
        dict(bindings)
    ).to_matrix()[:, 0]

    initial = np.zeros(4, dtype=np.complex128)
    initial[1] = 1.0
    expected_state = (
        _matrix_exponential(mixer.to_matrix(), bindings["angle_0_beta"])
        @ _matrix_exponential(
            _maxcut_cost(2, [(0, 1)]).to_matrix(),
            bindings["angle_0_gamma"],
        )
        @ initial
    )
    np.testing.assert_allclose(actual_state, expected_state, rtol=0.0, atol=1e-12)
    assert result.optimal_circuit.used_symbols == []


def test_inputs_are_snapshotted_without_mutating_cost_or_mixer() -> None:
    mixer = Hamiltonian(2)
    mixer.add_term(PauliString.from_str("XI"), 1.0)
    mixer_before = mixer.to_matrix().copy()
    algorithm = QAOA(RecordingEstimator(), RecordingOptimizer(), mixer_operator=mixer)
    mixer.add_term(PauliString.from_str("IZ"), 9.0)

    cost = _maxcut_cost(2, [(0, 1)])
    cost_before = cost.to_matrix().copy()
    result = algorithm.compute_maximum(
        cost,
        {"qaoa_0_gamma": 0.2, "qaoa_0_beta": -0.3},
    )

    np.testing.assert_allclose(cost.to_matrix(), cost_before, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(mixer_before, PauliString.from_str("XI").to_matrix())
    expected = (
        _matrix_exponential(mixer_before, -0.3)
        @ _matrix_exponential(cost_before, 0.2)
        @ _hadamards(2)
    )
    np.testing.assert_allclose(
        result.optimal_circuit.to_matrix(), expected, rtol=0.0, atol=1e-12
    )
    assert result.optimal_value == pytest.approx(0.75)


@pytest.mark.parametrize("coefficient", [math.inf, math.nan])
def test_non_finite_mixer_is_rejected_before_optimization(coefficient: float) -> None:
    mixer = Hamiltonian(2)
    mixer.add_term(PauliString.from_str("XI"), coefficient)
    optimizer = RecordingOptimizer()

    with pytest.raises(ValueError, match="mixer_operator coefficients must be finite"):
        QAOA(RecordingEstimator(), optimizer, mixer_operator=mixer)
    assert optimizer.batches == []


def test_non_hermitian_and_empty_mixers_are_rejected_before_optimization() -> None:
    optimizer = RecordingOptimizer()
    non_hermitian = Hamiltonian(2)
    non_hermitian.add_term(PauliString.from_str("XI"), 1j)
    with pytest.raises(ValueError, match="mixer_operator is not Hermitian"):
        QAOA(RecordingEstimator(), optimizer, mixer_operator=non_hermitian)

    empty = Hamiltonian(2)
    with pytest.raises(ValueError, match="empty after simplification"):
        QAOA(RecordingEstimator(), optimizer, mixer_operator=empty)

    cancelling = Hamiltonian(2)
    cancelling.add_term(PauliString.from_str("+iXI"), 1.0)
    cancelling.add_term(PauliString.from_str("XI"), -1j)
    with pytest.raises(ValueError, match="empty after simplification"):
        QAOA(RecordingEstimator(), optimizer, mixer_operator=cancelling)
    assert optimizer.batches == []


def test_mixer_hermiticity_uses_phases_and_aggregated_duplicate_terms() -> None:
    mixer = Hamiltonian(2)
    mixer.add_term(PauliString.from_str("+iXI"), 1.0)
    mixer.add_term(PauliString.from_str("XI"), 1.0 - 1.0j)
    optimizer = RecordingOptimizer()
    algorithm = QAOA(RecordingEstimator(), optimizer, mixer_operator=mixer)

    result = algorithm.compute_maximum(
        _maxcut_cost(2, [(0, 1)]),
        {"qaoa_0_gamma": 0.2, "qaoa_0_beta": -0.3},
    )

    expected_mixer = PauliString.from_str("XI").to_matrix()
    expected = (
        _matrix_exponential(expected_mixer, -0.3)
        @ _matrix_exponential(_maxcut_cost(2, [(0, 1)]).to_matrix(), 0.2)
        @ _hadamards(2)
    )
    np.testing.assert_allclose(
        result.optimal_circuit.to_matrix(), expected, rtol=0.0, atol=1e-12
    )
    assert len(optimizer.batches) == 1


def test_repeated_runs_return_independent_deterministic_circuits() -> None:
    optimizer = RecordingOptimizer()
    algorithm = QAOA(RecordingEstimator(), optimizer)
    point = {"qaoa_0_gamma": 0.2, "qaoa_0_beta": -0.3}

    first = algorithm.compute_maximum(_maxcut_cost(2, [(0, 1)]), point)
    second = algorithm.compute_maximum(_maxcut_cost(2, [(0, 1)]), point)

    assert first.optimal_circuit == second.optimal_circuit
    assert first.optimal_circuit is not second.optimal_circuit
    first.optimal_circuit.x(0)
    assert first.optimal_circuit != second.optimal_circuit


@pytest.mark.parametrize("reps", [True, 1.5, "1", None])
def test_reps_rejects_non_integer_values(reps: object) -> None:
    with pytest.raises(TypeError, match="reps must be an integer"):
        QAOA(RecordingEstimator(), RecordingOptimizer(), reps=reps)  # type: ignore[arg-type]


@pytest.mark.parametrize("reps", [0, -1])
def test_reps_must_be_positive(reps: int) -> None:
    with pytest.raises(ValueError, match="reps must be greater than zero"):
        QAOA(RecordingEstimator(), RecordingOptimizer(), reps=reps)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("estimator", object(), "estimator must be an Estimator"),
        ("optimizer", object(), "optimizer must be an Optimizer"),
        ("mixer_operator", object(), "mixer_operator must be a Hamiltonian or None"),
        (
            "initial_state",
            object(),
            "initial_state must be a StatePreparation or None",
        ),
        (
            "cost_strategy",
            object(),
            "cost_strategy must be an EvolutionStrategy or None",
        ),
        (
            "mixer_strategy",
            object(),
            "mixer_strategy must be an EvolutionStrategy or None",
        ),
        ("prefix", 1, "prefix must be a str"),
    ],
)
def test_constructor_type_validation(keyword: str, value: object, message: str) -> None:
    arguments: dict[str, object] = {
        "estimator": RecordingEstimator(),
        "optimizer": RecordingOptimizer(),
    }
    arguments[keyword] = value
    with pytest.raises(TypeError, match=message):
        QAOA(**arguments)  # type: ignore[arg-type]


def test_cost_type_width_empty_diagonal_and_hermitian_validation() -> None:
    optimizer = RecordingOptimizer()
    algorithm = QAOA(RecordingEstimator(), optimizer)
    point = {"qaoa_0_gamma": 0.2, "qaoa_0_beta": -0.3}

    with pytest.raises(TypeError, match="cost_operator must be a Hamiltonian"):
        algorithm.compute_maximum(PauliString.from_str("Z"), point)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one qubit"):
        algorithm.compute_maximum(Hamiltonian(0), point)
    with pytest.raises(ValueError, match="empty after simplification"):
        algorithm.compute_maximum(Hamiltonian(2), point)
    off_diagonal = Hamiltonian.from_pauli(PauliString.from_str("XI"))
    with pytest.raises(ValueError, match="diagonal in the computational basis"):
        algorithm.compute_maximum(off_diagonal, point)
    non_hermitian = Hamiltonian(2)
    non_hermitian.add_term(PauliString.from_str("ZI"), 1j)
    with pytest.raises(ValueError, match="is not Hermitian"):
        algorithm.compute_maximum(non_hermitian, point)
    non_finite = Hamiltonian(2)
    non_finite.add_term(PauliString.from_str("ZI"), math.inf)
    with pytest.raises(ValueError, match="coefficients must be finite"):
        algorithm.compute_maximum(non_finite, point)
    assert optimizer.batches == []


def test_mixer_and_initial_state_width_must_match_cost() -> None:
    point = {"qaoa_0_gamma": 0.2, "qaoa_0_beta": -0.3}
    one_qubit_mixer = Hamiltonian.from_pauli(PauliString.from_str("X"))
    with pytest.raises(ValueError, match="act on num_qubits qubits"):
        QAOA(
            RecordingEstimator(),
            RecordingOptimizer(),
            mixer_operator=one_qubit_mixer,
        ).compute_maximum(_maxcut_cost(2, [(0, 1)]), point)
    with pytest.raises(ValueError, match="initial_state and cost_operator"):
        QAOA(
            RecordingEstimator(),
            RecordingOptimizer(),
            initial_state=BasisOnePreparation(1),
        ).compute_maximum(_maxcut_cost(2, [(0, 1)]), point)


def test_active_initial_state_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not contain active symbols"):
        QAOA(
            RecordingEstimator(),
            RecordingOptimizer(),
            initial_state=ActivePreparation(2),
        ).compute_maximum(
            _maxcut_cost(2, [(0, 1)]),
            {"qaoa_0_gamma": 0.2, "qaoa_0_beta": -0.3},
        )


def test_initial_point_validation_is_delegated_to_variational_path() -> None:
    algorithm = QAOA(RecordingEstimator(), RecordingOptimizer())
    with pytest.raises(ValueError, match="missing"):
        algorithm.compute_maximum(
            _maxcut_cost(2, [(0, 1)]),
            {"qaoa_0_gamma": 0.2},
        )


def test_algorithms_share_a_flat_variational_result() -> None:
    result, _, _ = _recording_run(reps=1)
    assert QAOAResult is VQEResult is VQAResult
    assert isinstance(result, VQAResult)
    assert result.objective_sense == "maximize"
    assert not hasattr(result, "vqe_result")


@pytest.mark.parametrize("legacy", [False, True], ids=["run", "compute"])
def test_minimum_uses_original_cost_and_same_circuit_as_maximum(legacy: bool) -> None:
    optimizer = RecordingOptimizer()
    algorithm = QAOA(RecordingEstimator(), optimizer)
    cost = _maxcut_cost(2, [(0, 1)])
    point = {"qaoa_0_gamma": 0.2, "qaoa_0_beta": -0.3}
    if legacy:
        minimum = algorithm.compute_minimum_eigenvalue(cost, point, shots=123)
        maximum = algorithm.compute_maximum(cost, point, shots=123)
    else:
        minimum = algorithm.run(cost, point, shots=123)
        maximum = algorithm.run(cost, point, shots=123, objective_sense="maximize")
    assert minimum.optimal_value == -0.75
    assert maximum.optimal_value == 0.75
    assert minimum.objective_sense == "minimize"
    assert maximum.objective_sense == "maximize"
    assert maximum.optimizer_result.optimal_value == -maximum.optimal_value
    assert minimum.eigenvalue == minimum.optimal_value
    positive = optimizer.batches[0][0].estimator_request
    negative = optimizer.batches[1][0].estimator_request
    assert positive.shots == negative.shots == 123
    assert positive.circuit == negative.circuit
    np.testing.assert_allclose(positive.observable.to_matrix(), cost.to_matrix())
    np.testing.assert_allclose(negative.observable.to_matrix(), -cost.to_matrix())


@pytest.mark.parametrize(
    ("sense", "error"), [("minimum", ValueError), (None, TypeError)]
)
def test_invalid_objective_sense_is_rejected_before_optimization(sense, error) -> None:
    optimizer = RecordingOptimizer()
    algorithm = QAOA(RecordingEstimator(), optimizer)
    with pytest.raises(error, match="objective_sense"):
        algorithm.run(
            _maxcut_cost(2, [(0, 1)]),
            {"qaoa_0_gamma": 0.2, "qaoa_0_beta": -0.3},
            objective_sense=sense,
        )
    assert optimizer.batches == []


@pytest.mark.reference
def test_minimum_reaches_known_single_edge_minimum() -> None:
    algorithm = QAOA(
        StatevectorEstimator(),
        GradientDescent(ParameterShiftGradient(), maxiter=1, tolerance=1e-12),
    )
    result = algorithm.run(
        _maxcut_cost(2, [(0, 1)]),
        {"qaoa_0_gamma": math.pi / 2, "qaoa_0_beta": -math.pi / 8},
    )
    probabilities = np.abs(result.optimal_circuit.to_matrix()[:, 0]) ** 2
    assert result.optimal_value == pytest.approx(0.0, abs=1e-12)
    assert probabilities[0] + probabilities[3] == pytest.approx(1.0, abs=1e-12)


def test_subclass_build_hook_is_used_by_both_optimization_paths() -> None:
    class Variant(QAOA):
        def build_circuit(self, cost_operator: Hamiltonian) -> Circuit:
            circuit = Circuit(cost_operator.num_qubits)
            circuit.ry(0, Parameter("custom"))
            return circuit

    optimizer = RecordingOptimizer()
    algorithm = Variant(RecordingEstimator(), optimizer)
    cost = Hamiltonian.from_pauli(PauliString.from_str("Z"))
    for method in (algorithm.compute_minimum_eigenvalue, algorithm.compute_maximum):
        result = method(cost, {"custom": 0.2})
        assert tuple(result.optimal_parameters) == ("custom",)
        assert optimizer.batches[-1][0].parameter_names == ("custom",)


def test_public_build_returns_fresh_circuits_without_optimization() -> None:
    optimizer = RecordingOptimizer()
    algorithm = QAOA(RecordingEstimator(), optimizer)
    cost = _maxcut_cost(2, [(0, 1)])
    first = algorithm.build_circuit(cost)
    second = algorithm.build_circuit(cost)
    assert first == second
    first.x(0)
    assert first != second
    assert optimizer.batches == []


@pytest.mark.parametrize("response", [(), [], [object()]])
def test_invalid_optimizer_responses_are_rejected(response: object) -> None:
    class BrokenOptimizer(RecordingOptimizer):
        def run(
            self, estimator: Estimator, requests: Sequence[OptimizerRequest], /
        ) -> list[OptimizerResult]:
            return response  # type: ignore[return-value]

    algorithm = QAOA(RecordingEstimator(), BrokenOptimizer())
    with pytest.raises((TypeError, ValueError)):
        algorithm.compute_minimum_eigenvalue(
            _maxcut_cost(2, [(0, 1)]),
            {"qaoa_0_gamma": 0.2, "qaoa_0_beta": -0.3},
        )


def test_optimizer_parameter_order_must_match_request() -> None:
    class ReorderedOptimizer(RecordingOptimizer):
        def run(
            self, estimator: Estimator, requests: Sequence[OptimizerRequest], /
        ) -> list[OptimizerResult]:
            return [
                OptimizerResult(
                    ("qaoa_0_beta", "qaoa_0_gamma"),
                    (0.1, 0.2),
                    0.0,
                    0,
                    OptimizerStatus.MAX_ITERATIONS,
                )
            ]

    with pytest.raises(ValueError, match="names must match"):
        QAOA(RecordingEstimator(), ReorderedOptimizer()).compute_maximum(
            _maxcut_cost(2, [(0, 1)]),
            {"qaoa_0_gamma": 0.2, "qaoa_0_beta": -0.3},
        )


def test_minimization_updates_nonoptimal_initial_point_to_z_ground_state() -> None:
    algorithm = QAOA(
        StatevectorEstimator(),
        GradientDescent(
            ParameterShiftGradient(), learning_rate=0.2, maxiter=80, tolerance=1e-10
        ),
    )
    result = algorithm.compute_minimum_eigenvalue(
        Hamiltonian.from_pauli(PauliString.from_str("Z")),
        {"qaoa_0_gamma": 0.31, "qaoa_0_beta": -0.14},
    )
    assert result.optimizer_result.iterations > 0
    assert result.optimal_value == pytest.approx(-1.0, abs=1e-10)
    assert abs(result.optimal_circuit.to_matrix()[1, 0]) ** 2 == pytest.approx(
        1.0, abs=1e-10
    )
