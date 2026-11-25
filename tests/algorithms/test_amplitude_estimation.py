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

"""Tests for phase-estimation-based amplitude estimation."""

from __future__ import annotations

import math

import numpy as np
import pytest
from cqlib import Circuit, Parameter, Qubit
from cqlib.device import ExecutionResult

from cqlib_algorithms import (
    AmplitudeEstimationProblem,
    StandardAmplitudeEstimation,
    StandardAmplitudeEstimationRequest,
    StandardAmplitudeEstimationResult,
)
from cqlib_algorithms.algorithms.amplitude_estimation._common import (
    _build_grover_power_circuit,
)
from cqlib_algorithms.operators import (
    AmplitudeEncoding,
    BooleanFunction,
    GroverOperator,
    PhaseOracle,
    StatevectorSampler,
    UniformSuperposition,
    UserCircuitPreparation,
)
from cqlib_algorithms.protocols import Sampler


def _rotation_problem(theta: float) -> AmplitudeEstimationProblem:
    preparation = AmplitudeEncoding([math.cos(theta), math.sin(theta)])
    grover = GroverOperator(
        PhaseOracle(BooleanFunction.from_marked_states(1, [1])),
        state_preparation=preparation,
    )
    return AmplitudeEstimationProblem(
        preparation,
        grover,
        (0,),
        lambda bitstring: bitstring == "1",
    )


def _qpe_probability(phase: float, point: int, dimension: int) -> float:
    offset = phase - point / dimension
    terms = np.exp(2j * math.pi * offset * np.arange(dimension))
    return float(abs(np.sum(terms) / dimension) ** 2)


def _evaluation_probabilities(
    circuit: Circuit, num_evaluation_qubits: int
) -> np.ndarray:
    dimension = 1 << num_evaluation_qubits
    state = circuit.to_matrix()[:, 0]
    return np.sum(np.abs(state.reshape(-1, dimension)) ** 2, axis=0)


class _ScriptedSampler(Sampler):
    def __init__(self, scripts: list[dict[str, int]]) -> None:
        self.scripts = scripts
        self.calls = 0
        self.requests = []

    def run(self, requests):
        self.calls += 1
        self.requests = list(requests)
        return [
            ExecutionResult.from_counts(
                "scripted",
                list(request.qubits),
                request.shots,
                len(request.qubits),
                counts,
                "test",
            )
            for request, counts in zip(requests, self.scripts)
        ]


@pytest.mark.reference
def test_exact_grid_point_has_only_the_two_expected_phase_peaks() -> None:
    num_evaluation_qubits = 3
    dimension = 1 << num_evaluation_qubits
    point = 1
    problem = _rotation_problem(math.pi * point / dimension)

    circuit = StandardAmplitudeEstimation.build_circuit(problem, num_evaluation_qubits)
    probabilities = _evaluation_probabilities(circuit, num_evaluation_qubits)

    expected = np.zeros(dimension)
    expected[point] = 0.5
    expected[dimension - point] = 0.5
    np.testing.assert_allclose(probabilities, expected, rtol=0.0, atol=1e-12)
    assert math.sin(math.pi * point / dimension) ** 2 == pytest.approx(
        math.sin(math.pi * (dimension - point) / dimension) ** 2,
        abs=1e-15,
    )


@pytest.mark.reference
@pytest.mark.parametrize(
    ("theta", "point"),
    [(0.0, 0), (math.pi / 2.0, 4)],
)
def test_physical_endpoints_map_to_the_exact_phase_grid(
    theta: float, point: int
) -> None:
    num_evaluation_qubits = 3
    dimension = 1 << num_evaluation_qubits

    probabilities = _evaluation_probabilities(
        StandardAmplitudeEstimation.build_circuit(
            _rotation_problem(theta), num_evaluation_qubits
        ),
        num_evaluation_qubits,
    )

    expected = np.zeros(dimension)
    expected[point] = 1.0
    np.testing.assert_allclose(probabilities, expected, rtol=0.0, atol=1e-12)


@pytest.mark.reference
@pytest.mark.parametrize(
    "complex_component", [False, True], ids=["raw-rotation", "complex-components"]
)
def test_non_grid_distribution_matches_independent_phase_estimation_formula(
    complex_component: bool,
) -> None:
    num_evaluation_qubits = 4
    dimension = 1 << num_evaluation_qubits
    theta = 0.271
    phase = theta / math.pi
    if complex_component:
        preparation = AmplitudeEncoding((math.cos(theta), 1j * math.sin(theta)))
        oracle = PhaseOracle(BooleanFunction.from_marked_states(1, [1]))
        iteration = GroverOperator(oracle, state_preparation=preparation)
        problem = AmplitudeEstimationProblem(
            preparation, iteration, (0,), lambda bits: bits == "1"
        )
    else:
        problem = _rotation_problem(theta)
    circuit = StandardAmplitudeEstimation.build_circuit(problem, num_evaluation_qubits)

    expected = np.asarray(
        [
            0.5 * _qpe_probability(phase, point, dimension)
            + 0.5 * _qpe_probability((-phase) % 1.0, point, dimension)
            for point in range(dimension)
        ]
    )
    np.testing.assert_allclose(
        _evaluation_probabilities(circuit, num_evaluation_qubits),
        expected,
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("power", [0, 1, 2, 5])
def test_grover_power_circuit_is_exact_q_power_times_preparation(power: int) -> None:
    preparation = Circuit([8, 3])
    preparation.ry(Qubit(8), 0.21)
    preparation.cx(Qubit(8), Qubit(3))
    preparation.set_global_phase(-0.12)
    preparation_matrix = preparation.to_matrix([8, 3])
    component = UserCircuitPreparation(preparation)
    grover = GroverOperator(
        PhaseOracle(BooleanFunction.from_marked_states(2, [2, 3])),
        state_preparation=component,
    )
    state = preparation_matrix[:, 0]
    reflection = 2 * np.outer(state, state.conj()) - np.eye(4)
    grover_matrix = reflection @ np.diag([1, 1, -1, -1])
    problem = AmplitudeEstimationProblem(
        UserCircuitPreparation(preparation),
        grover,
        (1,),
        lambda bitstring: bitstring == "1",
    )

    actual = _build_grover_power_circuit(problem, power).to_matrix()
    expected = np.linalg.matrix_power(grover_matrix, power) @ preparation_matrix
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_problem_snapshots_sources_and_rejects_dynamic_or_unbound_circuits() -> None:
    preparation = Circuit(1)
    preparation.h(0)
    grover = GroverOperator(
        PhaseOracle(BooleanFunction.from_marked_states(1, [1])),
        state_preparation=UserCircuitPreparation(preparation),
    )
    expected_iteration = grover.build_circuit().to_matrix()
    problem = AmplitudeEstimationProblem(
        UserCircuitPreparation(preparation),
        grover,
        (0,),
        lambda bitstring: bitstring == "1",
    )
    preparation.x(0)
    grover.oracle.x(0)
    assert len(problem.preparation_circuit.operations) == 1
    np.testing.assert_allclose(
        problem.iteration_circuit.to_matrix(), expected_iteration, atol=1e-12
    )

    measured = Circuit(1)
    measured.measure(0)
    with pytest.raises(ValueError, match="classical data|static unitary|invertible"):
        AmplitudeEstimationProblem(
            UserCircuitPreparation(measured),
            GroverOperator(
                PhaseOracle(BooleanFunction.from_marked_states(1, [0])),
                state_preparation=UserCircuitPreparation(Circuit(1)),
            ),
            (0,),
            lambda _: True,
        )
    symbolic = Circuit(1)
    symbolic.ry(0, Parameter("theta"))
    with pytest.raises(ValueError, match="fully bound"):
        AmplitudeEstimationProblem(
            UserCircuitPreparation(symbolic),
            GroverOperator(
                PhaseOracle(BooleanFunction.from_marked_states(1, [0])),
                state_preparation=UserCircuitPreparation(Circuit(1)),
            ),
            (0,),
            lambda _: True,
        )


def test_problem_rejects_width_and_objective_qubit_errors() -> None:
    def predicate(bitstring: str) -> bool:
        return bitstring == "1"

    with pytest.raises(ValueError, match="same qubit count"):
        AmplitudeEstimationProblem(
            UserCircuitPreparation(Circuit(1)),
            GroverOperator(
                PhaseOracle(BooleanFunction.from_marked_states(2, [0])),
                state_preparation=UserCircuitPreparation(Circuit(2)),
            ),
            (0,),
            predicate,
        )
    with pytest.raises(TypeError, match="must be a tuple"):
        AmplitudeEstimationProblem(
            UserCircuitPreparation(Circuit(1)),
            GroverOperator(
                PhaseOracle(BooleanFunction.from_marked_states(1, [0])),
                state_preparation=UserCircuitPreparation(Circuit(1)),
            ),
            [0],
            predicate,
        )
    with pytest.raises(ValueError, match="must not be empty"):
        AmplitudeEstimationProblem(
            UserCircuitPreparation(Circuit(1)),
            GroverOperator(
                PhaseOracle(BooleanFunction.from_marked_states(1, [0])),
                state_preparation=UserCircuitPreparation(Circuit(1)),
            ),
            (),
            predicate,
        )
    with pytest.raises(ValueError, match="duplicates"):
        AmplitudeEstimationProblem(
            UserCircuitPreparation(Circuit(2)),
            GroverOperator(
                PhaseOracle(BooleanFunction.from_marked_states(2, [0])),
                state_preparation=UserCircuitPreparation(Circuit(2)),
            ),
            (0, 0),
            predicate,
        )
    with pytest.raises(ValueError, match="valid system positions"):
        AmplitudeEstimationProblem(
            UserCircuitPreparation(Circuit(1)),
            GroverOperator(
                PhaseOracle(BooleanFunction.from_marked_states(1, [0])),
                state_preparation=UserCircuitPreparation(Circuit(1)),
            ),
            (1,),
            predicate,
        )
    with pytest.raises(TypeError, match="only integers"):
        AmplitudeEstimationProblem(
            UserCircuitPreparation(Circuit(1)),
            GroverOperator(
                PhaseOracle(BooleanFunction.from_marked_states(1, [0])),
                state_preparation=UserCircuitPreparation(Circuit(1)),
            ),
            (True,),
            predicate,
        )
    with pytest.raises(TypeError, match="must be callable"):
        AmplitudeEstimationProblem(
            UserCircuitPreparation(Circuit(1)),
            GroverOperator(
                PhaseOracle(BooleanFunction.from_marked_states(1, [0])),
                state_preparation=UserCircuitPreparation(Circuit(1)),
            ),
            (0,),
            None,
        )


def test_run_preserves_complete_distribution_and_uses_stable_tie_break() -> None:
    sampler = _ScriptedSampler([{"001": 4, "111": 4}])
    result = StandardAmplitudeEstimation(sampler).run(
        [StandardAmplitudeEstimationRequest(_rotation_problem(math.pi / 8), 3, 8)]
    )[0]

    assert sampler.calls == 1
    assert result.grid_point == 1
    assert result.estimation == pytest.approx(math.sin(math.pi / 8) ** 2)
    assert dict(result.evaluation_probabilities) == {
        0: 0.0,
        1: 0.5,
        2: 0.0,
        3: 0.0,
        4: 0.0,
        5: 0.0,
        6: 0.0,
        7: 0.5,
    }
    assert result.num_grover_queries == 7
    assert result.num_oracle_queries == 56
    assert result.confidence_interval is None


def test_finite_shots_select_canonical_phase_pair_before_mode() -> None:
    sampler = _ScriptedSampler([{"001": 3, "111": 3, "010": 4}])
    result = StandardAmplitudeEstimation(sampler).run(
        [StandardAmplitudeEstimationRequest(_rotation_problem(math.pi / 8), 3, 10)]
    )[0]

    assert result.grid_point == 1
    assert result.estimation == pytest.approx(math.sin(math.pi / 8) ** 2)
    assert result.evaluation_probabilities[1] == 0.3
    assert result.evaluation_probabilities[7] == 0.3
    assert result.evaluation_probabilities[2] == 0.4


def test_equal_canonical_bins_select_the_smallest_grid_point() -> None:
    sampler = _ScriptedSampler([{"001": 3, "111": 1, "010": 4}])
    result = StandardAmplitudeEstimation(sampler).run(
        [StandardAmplitudeEstimationRequest(_rotation_problem(math.pi / 8), 3, 8)]
    )[0]

    assert result.grid_point == 1


@pytest.mark.integration
@pytest.mark.parametrize(("marked", "bin_index"), [(1, 1), (0, 3)])
def test_component_search_predicate_selects_the_expected_amplitude(marked, bin_index):
    theta = math.pi / 8
    preparation = AmplitudeEncoding((math.cos(theta), math.sin(theta)))
    oracle = PhaseOracle(BooleanFunction.from_marked_states(1, [marked]))
    iteration = GroverOperator(oracle, state_preparation=preparation)
    problem = AmplitudeEstimationProblem(
        preparation, iteration, (0,), lambda bits: bits == str(marked)
    )
    result = StandardAmplitudeEstimation(StatevectorSampler()).run(
        problem, num_evaluation_qubits=3, shots=64
    )

    expected = np.zeros(8)
    expected[[bin_index, 8 - bin_index]] = 0.5
    circuit = StandardAmplitudeEstimation.build_circuit(problem, 3)
    probabilities = _evaluation_probabilities(circuit, 3)
    np.testing.assert_allclose(probabilities, expected, rtol=0, atol=1e-12)
    reference = StandardAmplitudeEstimation.build_circuit(
        _rotation_problem(bin_index * math.pi / 8), 3
    )
    np.testing.assert_allclose(
        probabilities, _evaluation_probabilities(reference, 3), rtol=0, atol=1e-12
    )
    assert isinstance(result, StandardAmplitudeEstimationResult)
    assert result.grid_point == bin_index
    assert result.estimation == pytest.approx(
        math.sin(theta) ** 2 if marked else math.cos(theta) ** 2
    )
    assert result.num_grover_queries == 7
    assert result.num_oracle_queries == 7 * 64


def test_component_inputs_preserve_the_controlled_global_phase():
    preparation = UniformSuperposition(1)
    oracle = PhaseOracle(BooleanFunction.from_marked_states(1, [0, 1]))
    iteration = GroverOperator(oracle, state_preparation=preparation)
    problem = AmplitudeEstimationProblem(
        preparation,
        iteration,
        (0,),
        lambda bits: True,
    )
    algorithm = StandardAmplitudeEstimation(StatevectorSampler())
    single = algorithm.run(problem=problem, num_evaluation_qubits=3, shots=32)
    batch = algorithm.run([StandardAmplitudeEstimationRequest(problem, 3, 32)])

    assert isinstance(problem.preparation_circuit, Circuit)
    assert isinstance(problem.iteration_circuit, Circuit)
    assert batch == [single]
    assert single.evaluation_probabilities[4] == 1.0
    assert single.estimation == 1.0


def test_component_problem_preserves_objective_positions_and_local_predicate():
    preparation = UniformSuperposition(2)
    oracle = PhaseOracle(BooleanFunction.from_marked_states(2, [2, 3]))
    iteration = GroverOperator(oracle, state_preparation=preparation)

    def upper_bit_is_one(bits):
        assert len(bits) == 1
        return bits == "1"

    problem = AmplitudeEstimationProblem(preparation, iteration, (1,), upper_bit_is_one)
    assert problem.objective_qubits == (1,)
    assert problem.is_good_state is upper_bit_is_one
    assert problem.is_good_state("1") and not problem.is_good_state("0")
    result = StandardAmplitudeEstimation(StatevectorSampler()).run(
        problem, num_evaluation_qubits=2, shots=64
    )
    assert result.estimation == pytest.approx(0.5)


def test_problem_rejects_unbound_preparation_and_incompatible_component_widths():
    symbolic = Circuit(1)
    symbolic.ry(0, Parameter("x"))
    oracle = PhaseOracle(BooleanFunction.from_marked_states(1, [1]))
    iteration = GroverOperator(oracle)
    with pytest.raises(ValueError, match="fully bound"):
        AmplitudeEstimationProblem(
            UserCircuitPreparation(symbolic), iteration, (0,), lambda bits: True
        )
    with pytest.raises(ValueError, match="same qubit count"):
        AmplitudeEstimationProblem(
            UniformSuperposition(2), iteration, (0,), lambda bits: True
        )


def test_problem_requires_components_in_their_declared_roles():
    oracle = PhaseOracle(BooleanFunction.from_marked_states(1, [1]))
    iteration = GroverOperator(oracle)
    with pytest.raises(TypeError, match="state_preparation must be"):
        AmplitudeEstimationProblem(iteration, iteration, (0,), lambda bits: True)
    with pytest.raises(TypeError, match="grover_operator must be"):
        AmplitudeEstimationProblem(
            UniformSuperposition(1), oracle, (0,), lambda bits: True
        )


@pytest.mark.parametrize(
    ("built", "error", "message"),
    [(None, TypeError, "return a Circuit"), (Circuit(2), ValueError, "declared width")],
)
def test_problem_checks_the_circuit_returned_by_the_iteration(
    monkeypatch, built, error, message
):
    iteration = GroverOperator(PhaseOracle(BooleanFunction.from_marked_states(1, [1])))
    monkeypatch.setattr(GroverOperator, "build_circuit", lambda self: built)
    with pytest.raises(error, match=message):
        AmplitudeEstimationProblem(
            UniformSuperposition(1), iteration, (0,), lambda bits: True
        )


def test_batch_still_submits_once_and_returns_results_in_request_order():
    sampler = _ScriptedSampler([{"11": 8}, {"100": 12}])
    problem = _rotation_problem(math.pi / 8)
    results = StandardAmplitudeEstimation(sampler).run(
        [
            StandardAmplitudeEstimationRequest(problem, 2, 8),
            StandardAmplitudeEstimationRequest(problem, 3, 12),
        ]
    )
    assert sampler.calls == 1
    assert [result.grid_point for result in results] == [1, 4]
    assert [result.num_oracle_queries for result in results] == [3 * 8, 7 * 12]


def test_single_and_batch_options_are_validated_before_sampling():
    sampler = _ScriptedSampler([])
    algorithm = StandardAmplitudeEstimation(sampler)
    problem = _rotation_problem(0.2)
    with pytest.raises(TypeError, match="num_evaluation_qubits and shots"):
        algorithm.run(problem, shots=10)
    with pytest.raises(TypeError, match="num_evaluation_qubits and shots"):
        algorithm.run(problem, num_evaluation_qubits=3)
    with pytest.raises(TypeError, match="batch"):
        algorithm.run([], num_evaluation_qubits=3, shots=10)
    with pytest.raises(ValueError, match="shots"):
        algorithm.run(problem, num_evaluation_qubits=3, shots=0)
    assert sampler.calls == 0


def test_dense_control_limit_rejects_before_large_matrix_allocation() -> None:
    problem = AmplitudeEstimationProblem(
        UserCircuitPreparation(Circuit(11)),
        GroverOperator(
            PhaseOracle(BooleanFunction.from_marked_states(11, [0])),
            state_preparation=UserCircuitPreparation(Circuit(11)),
        ),
        (0,),
        lambda bitstring: bitstring == "1",
    )

    with pytest.raises(ValueError, match="dense controlled-matrix limit"):
        StandardAmplitudeEstimation.build_circuit(problem, 1)


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("num_evaluation_qubits", True, TypeError, "integer"),
        ("num_evaluation_qubits", 0, ValueError, "greater than zero"),
        ("shots", True, TypeError, "integer"),
        ("shots", 0, ValueError, "greater than zero"),
    ],
)
def test_standard_request_validation(field, value, error, message) -> None:
    arguments = {
        "problem": _rotation_problem(0.2),
        "num_evaluation_qubits": 3,
        "shots": 100,
    }
    arguments[field] = value
    with pytest.raises(error, match=message):
        StandardAmplitudeEstimationRequest(**arguments)


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("num_grover_queries", True, TypeError, "integer"),
        ("num_oracle_queries", True, TypeError, "integer"),
        ("estimation", 0.25, ValueError, "match grid_point"),
    ],
)
def test_standard_result_rejects_inconsistent_derived_fields(
    field, value, error, message
) -> None:
    arguments = {
        "estimation": 0.0,
        "confidence_interval": None,
        "shots": 1,
        "num_grover_queries": 1,
        "num_oracle_queries": 1,
        "grid_point": 0,
        "evaluation_probabilities": {0: 1.0, 1: 0.0},
        "num_evaluation_qubits": 1,
    }
    arguments[field] = value
    with pytest.raises(error, match=message):
        StandardAmplitudeEstimationResult(**arguments)


def test_empty_batch_does_not_call_sampler() -> None:
    sampler = _ScriptedSampler([])
    assert StandardAmplitudeEstimation(sampler).run([]) == []
    assert sampler.calls == 0
