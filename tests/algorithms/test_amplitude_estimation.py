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
from cqlib_algorithms.operators import GroverOperator, StatevectorSampler
from cqlib_algorithms.protocols import Sampler


def _rotation_problem(theta: float) -> AmplitudeEstimationProblem:
    preparation = Circuit(1)
    preparation.ry(0, 2.0 * theta)
    grover = Circuit(1)
    grover.ry(0, 4.0 * theta)
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
def test_non_grid_distribution_matches_independent_phase_estimation_formula() -> None:
    num_evaluation_qubits = 4
    dimension = 1 << num_evaluation_qubits
    theta = 0.271
    phase = theta / math.pi
    circuit = StandardAmplitudeEstimation.build_circuit(
        _rotation_problem(theta), num_evaluation_qubits
    )

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
    grover = Circuit([4, 9])
    grover.rx(Qubit(4), -0.31)
    grover.cz(Qubit(4), Qubit(9))
    grover.ry(Qubit(9), 0.47)
    grover.set_global_phase(0.19)
    preparation_matrix = preparation.to_matrix([8, 3])
    grover_matrix = grover.to_matrix([4, 9])
    problem = AmplitudeEstimationProblem(
        preparation, grover, (1,), lambda bitstring: bitstring == "1"
    )

    actual = _build_grover_power_circuit(problem, power).to_matrix()
    expected = np.linalg.matrix_power(grover_matrix, power) @ preparation_matrix
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_problem_snapshots_sources_and_rejects_dynamic_or_unbound_circuits() -> None:
    preparation = Circuit(1)
    preparation.h(0)
    grover = Circuit(1)
    grover.z(0)
    problem = AmplitudeEstimationProblem(
        preparation, grover, (0,), lambda bitstring: bitstring == "1"
    )
    preparation.x(0)
    grover.x(0)
    assert len(problem.state_preparation.operations) == 1
    assert len(problem.grover_operator.operations) == 1

    measured = Circuit(1)
    measured.measure(0)
    with pytest.raises(ValueError, match="classical data|static unitary"):
        AmplitudeEstimationProblem(measured, Circuit(1), (0,), lambda _: True)
    symbolic = Circuit(1)
    symbolic.ry(0, Parameter("theta"))
    with pytest.raises(ValueError, match="fully bound"):
        AmplitudeEstimationProblem(symbolic, Circuit(1), (0,), lambda _: True)


def test_problem_rejects_width_and_objective_qubit_errors() -> None:
    def predicate(bitstring: str) -> bool:
        return bitstring == "1"

    with pytest.raises(ValueError, match="same qubit count"):
        AmplitudeEstimationProblem(Circuit(1), Circuit(2), (0,), predicate)
    with pytest.raises(TypeError, match="must be a tuple"):
        AmplitudeEstimationProblem(Circuit(1), Circuit(1), [0], predicate)
    with pytest.raises(ValueError, match="must not be empty"):
        AmplitudeEstimationProblem(Circuit(1), Circuit(1), (), predicate)
    with pytest.raises(ValueError, match="duplicates"):
        AmplitudeEstimationProblem(Circuit(2), Circuit(2), (0, 0), predicate)
    with pytest.raises(ValueError, match="valid system positions"):
        AmplitudeEstimationProblem(Circuit(1), Circuit(1), (1,), predicate)
    with pytest.raises(TypeError, match="only integers"):
        AmplitudeEstimationProblem(Circuit(1), Circuit(1), (True,), predicate)
    with pytest.raises(TypeError, match="must be callable"):
        AmplitudeEstimationProblem(Circuit(1), Circuit(1), (0,), None)


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
def test_dense_unitary_circuit_executes_with_statevector_sampler() -> None:
    theta = math.pi / 8
    preparation = Circuit(1)
    preparation.ry(0, 2.0 * theta)
    oracle = Circuit(1)
    oracle.z(0)
    problem = AmplitudeEstimationProblem(
        preparation,
        GroverOperator(oracle, state_preparation=preparation).build_circuit(),
        (0,),
        lambda bitstring: bitstring == "1",
    )
    result = StandardAmplitudeEstimation(StatevectorSampler()).run(
        [StandardAmplitudeEstimationRequest(problem, 3, 64)]
    )[0]

    observed = {
        point
        for point, probability in result.evaluation_probabilities.items()
        if probability > 0.0
    }
    assert observed <= {1, 7}
    assert result.estimation == pytest.approx(math.sin(theta) ** 2, abs=1e-12)


def test_dense_control_limit_rejects_before_large_matrix_allocation() -> None:
    problem = AmplitudeEstimationProblem(
        Circuit(11), Circuit(11), (0,), lambda bitstring: bitstring == "1"
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
