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

"""Tests for faster amplitude estimation."""

from __future__ import annotations

import copy
import math
from dataclasses import replace

import numpy as np
import pytest
from cqlib.device import ExecutionResult

from cqlib_algorithms import (
    AmplitudeEstimationProblem,
    FasterAmplitudeEstimation,
    FasterAmplitudeEstimationRequest,
    FasterAmplitudeEstimationResult,
)
from cqlib_algorithms.operators import (
    AmplitudeEncoding,
    BooleanFunction,
    GroverOperator,
    PhaseOracle,
    StatevectorSampler,
)
from cqlib_algorithms.protocols import Sampler


def _rotation_problem(theta: float, *, predicate=None) -> AmplitudeEstimationProblem:
    preparation = AmplitudeEncoding([math.cos(theta), math.sin(theta)])
    grover = GroverOperator(
        PhaseOracle(BooleanFunction.from_marked_states(1, [1])),
        state_preparation=preparation,
    )
    return AmplitudeEstimationProblem(
        preparation,
        grover,
        (0,),
        predicate if predicate is not None else lambda bits: bits == "1",
    )


class _AnalyticSampler(Sampler):
    """Return deterministic rounded counts from independent probabilities."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def run(self, requests):
        self.batch_sizes.append(len(requests))
        results = []
        for request in requests:
            state = request.circuit.to_matrix()[:, 0]
            probability = float(abs(state[1]) ** 2)
            assert -1e-12 <= probability <= 1.0 + 1e-12
            probability = min(1.0, max(0.0, probability))
            good = round(request.shots * probability)
            counts = {}
            if request.shots - good:
                counts["0"] = request.shots - good
            if good:
                counts["1"] = good
            results.append(
                ExecutionResult.from_counts(
                    "analytic",
                    list(request.qubits),
                    request.shots,
                    len(request.qubits),
                    counts,
                    "test",
                )
            )
        return results


class _BrokenSampler(Sampler):
    def run(self, requests):
        return []


@pytest.fixture
def sampled_rounds(monkeypatch):
    """Record actual circuits and replay the same shot noise on repeated runs."""
    rounds = []
    sample = StatevectorSampler.run

    def seeded_run(self, requests):
        rounds.append(tuple(requests))
        return sample(
            self,
            [
                replace(request, seed=100 * len(rounds) + index)
                for index, request in enumerate(requests)
            ],
        )

    monkeypatch.setattr(StatevectorSampler, "run", seeded_run)
    return rounds


@pytest.mark.parametrize(("objective", "factor"), [(1, 1.0), (0, 0.25)])
def test_component_powers_and_result_match_independent_rotation_batch(
    objective, factor, sampled_rounds
):
    preparation = AmplitudeEncoding((8, 1, 2j, 0))
    marked = [x for x in range(4) if x & (1 << objective)]
    oracle = PhaseOracle(BooleanFunction.from_marked_states(2, marked))
    iteration = GroverOperator(oracle, state_preparation=preparation)
    problem = AmplitudeEstimationProblem(
        preparation, iteration, (objective,), lambda bits: bits == "1"
    )
    fae = FasterAmplitudeEstimation(StatevectorSampler())
    result = fae.run(
        problem=problem, alpha=0.05, max_iterations=5, amplitude_rescaling_factor=factor
    )

    # The two small marginals of the prepared state are 4/69 and 1/69.
    probability = (4 if objective == 1 else 1) / 69
    theta = math.asin(math.sqrt(probability))
    target = probability / factor**2
    assert isinstance(result, FasterAmplitudeEstimationResult)
    assert result.confidence_interval[0] <= target <= result.confidence_interval[1]
    assert result.num_first_stage_steps < result.num_steps
    requests = [request for batch in sampled_rounds for request in batch]
    for power, request in zip(result.powers, requests, strict=True):
        assert [q.index for q in request.qubits] == [objective]
        state = request.circuit.to_matrix()[:, 0]
        actual = float(np.sum(np.abs(state[marked]) ** 2))
        expected = math.sin((2 * power + 1) * theta) ** 2
        assert actual == pytest.approx(expected, rel=0, abs=1e-12)

    sampled_rounds.clear()
    legacy = FasterAmplitudeEstimationRequest(_rotation_problem(theta), 0.05, 5, factor)
    assert fae.run([legacy]) == [result]


@pytest.mark.reference
def test_two_stage_schedule_contains_the_independent_amplitude() -> None:
    theta = 0.2
    request = FasterAmplitudeEstimationRequest(
        _rotation_problem(theta),
        alpha=0.05,
        max_iterations=5,
    )
    sampler = _AnalyticSampler()
    result = FasterAmplitudeEstimation(sampler).run([request])[0]
    expected = math.sin(theta) ** 2

    assert result.num_first_stage_steps < result.num_steps
    assert result.confidence_interval[0] <= expected <= result.confidence_interval[1]
    assert result.estimation == pytest.approx(expected, abs=0.01)
    assert result.success_probability_lower_bound == 0.95
    assert len(result.theta_intervals) == request.max_iterations + 1
    assert result.powers[: result.num_first_stage_steps] == tuple(
        1 << index for index in range(result.num_first_stage_steps)
    )
    assert sampler.batch_sizes[0] == 1
    assert 2 in sampler.batch_sizes


@pytest.mark.reference
def test_external_quarter_amplitude_rescaling_is_undone() -> None:
    target = 0.36
    factor = 0.25
    scaled_theta = math.asin(factor * math.sqrt(target))
    result = FasterAmplitudeEstimation(_AnalyticSampler()).run(
        [
            FasterAmplitudeEstimationRequest(
                _rotation_problem(scaled_theta),
                alpha=0.05,
                max_iterations=5,
                amplitude_rescaling_factor=factor,
            )
        ]
    )[0]

    assert result.confidence_interval[0] <= target <= result.confidence_interval[1]
    assert result.estimation == pytest.approx(target, abs=0.03)
    assert result.amplitude_rescaling_factor == factor


def test_first_stage_only_path_and_resource_statistics() -> None:
    theta = 0.01
    request = FasterAmplitudeEstimationRequest(
        _rotation_problem(theta), alpha=0.1, max_iterations=2
    )
    result = FasterAmplitudeEstimation(_AnalyticSampler()).run([request])[0]
    delta_c = request.alpha / (2 * request.max_iterations)
    shots = math.ceil(1944 * math.log(2 / delta_c))

    assert result.num_first_stage_steps == result.num_steps == 2
    assert result.powers == (1, 2)
    assert result.circuit_shots == (shots, shots)
    assert result.shots == 2 * shots
    assert result.num_grover_queries == 3
    assert result.num_oracle_queries == 3 * shots


@pytest.mark.parametrize("theta", [0.0, math.asin(0.25)])
def test_promised_domain_boundaries_remain_inside_interval(theta: float) -> None:
    problem = _rotation_problem(theta)
    fae = FasterAmplitudeEstimation(_AnalyticSampler())
    result = fae.run(problem, alpha=0.05, max_iterations=4)
    assert fae.run([FasterAmplitudeEstimationRequest(problem, 0.05, 4)]) == [result]
    expected = math.sin(theta) ** 2

    assert result.confidence_interval[0] <= expected <= result.confidence_interval[1]
    assert result.theta_intervals[0] == (0.0, math.asin(0.25))


def test_same_round_requests_are_batched_and_results_keep_order(sampled_rounds) -> None:
    results = FasterAmplitudeEstimation(StatevectorSampler()).run(
        [
            FasterAmplitudeEstimationRequest(_rotation_problem(0.02), 0.1, 3),
            FasterAmplitudeEstimationRequest(_rotation_problem(0.2), 0.05, 5),
        ]
    )

    first, second = results
    assert first.num_first_stage_steps == first.num_steps == 3
    assert second.num_first_stage_steps == 2 and second.num_steps == 5
    assert (
        first.confidence_interval[0]
        <= math.sin(0.02) ** 2
        <= first.confidence_interval[1]
    )
    assert (
        second.confidence_interval[0]
        <= math.sin(0.2) ** 2
        <= second.confidence_interval[1]
    )
    assert [len(batch) for batch in sampled_rounds] == [2, 2, 3, 2, 2]
    assert [request.shots for request in sampled_rounds[2]] == [
        first.circuit_shots[2],
        *second.circuit_shots[2:4],
    ]


@pytest.mark.integration
def test_statevector_sampler_executes_fae_end_to_end() -> None:
    theta = 0.15
    result = FasterAmplitudeEstimation(StatevectorSampler()).run(
        [FasterAmplitudeEstimationRequest(_rotation_problem(theta), 0.1, 4)]
    )[0]

    expected = math.sin(theta) ** 2
    assert result.confidence_interval[0] <= expected <= result.confidence_interval[1]
    assert result.estimation == pytest.approx(expected, abs=0.03)


def test_empty_batch_does_not_call_sampler() -> None:
    sampler = _AnalyticSampler()
    assert FasterAmplitudeEstimation(sampler).run([]) == []
    assert sampler.batch_sizes == []


@pytest.mark.parametrize("options", [{}, {"alpha": 0.1}, {"max_iterations": 3}])
def test_single_problem_requires_both_options_before_sampling(options):
    sampler = _AnalyticSampler()
    with pytest.raises(TypeError, match="alpha and max_iterations"):
        FasterAmplitudeEstimation(sampler).run(_rotation_problem(0.1), **options)
    assert sampler.batch_sizes == []


@pytest.mark.parametrize(
    "options",
    [{"alpha": 0.05}, {"max_iterations": 3}, {"amplitude_rescaling_factor": 1.0}],
)
def test_batch_rejects_single_problem_options(options):
    request = FasterAmplitudeEstimationRequest(_rotation_problem(0.1), 0.05, 3)
    sampler = _AnalyticSampler()
    with pytest.raises(TypeError, match="batch"):
        FasterAmplitudeEstimation(sampler).run([request], **options)
    assert sampler.batch_sizes == []


def test_stale_later_request_is_rejected_before_the_first_round():
    first = FasterAmplitudeEstimationRequest(_rotation_problem(0.1), 0.05, 3)
    second = FasterAmplitudeEstimationRequest(_rotation_problem(0.2), 0.05, 5)
    second.problem.iteration_circuit.measure(0)
    sampler = _AnalyticSampler()
    with pytest.raises(ValueError, match="static unitary|classical data"):
        FasterAmplitudeEstimation(sampler).run([first, second])
    assert sampler.batch_sizes == []


def test_algorithm_rejects_invalid_sampler_and_request_values() -> None:
    with pytest.raises(TypeError, match="Sampler instance"):
        FasterAmplitudeEstimation(object())
    algorithm = FasterAmplitudeEstimation(_AnalyticSampler())
    with pytest.raises(TypeError, match="FasterAmplitudeEstimationRequest"):
        algorithm.run([object()])


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("alpha", True, TypeError, "real number"),
        ("alpha", 0.0, ValueError, r"\(0, 1\)"),
        ("alpha", 1.0, ValueError, r"\(0, 1\)"),
        ("alpha", math.nan, ValueError, "finite"),
        ("alpha", math.ulp(1.0) / 4.0, ValueError, "binary64"),
        ("max_iterations", True, TypeError, "integer"),
        ("max_iterations", 0, ValueError, "greater than zero"),
        ("max_iterations", 21, ValueError, "1,000,000"),
        ("max_iterations", 10**9, ValueError, "safe explicit-power"),
        ("amplitude_rescaling_factor", True, TypeError, "real number"),
        ("amplitude_rescaling_factor", 0.5, ValueError, "1.0 or 0.25"),
        ("amplitude_rescaling_factor", math.inf, ValueError, "finite"),
    ],
)
def test_request_validation(field, value, error, message) -> None:
    arguments = {
        "problem": _rotation_problem(0.1),
        "alpha": 0.05,
        "max_iterations": 4,
        "amplitude_rescaling_factor": 1.0,
    }
    arguments[field] = value
    with pytest.raises(error, match=message):
        FasterAmplitudeEstimationRequest(**arguments)
    sampler = _AnalyticSampler()
    with pytest.raises(error, match=message):
        FasterAmplitudeEstimation(sampler).run(**arguments)
    assert sampler.batch_sizes == []


def test_sampler_contract_failure_is_rejected() -> None:
    request = FasterAmplitudeEstimationRequest(_rotation_problem(0.1), 0.05, 2)
    with pytest.raises(ValueError, match="one result per request"):
        FasterAmplitudeEstimation(_BrokenSampler()).run([request])


def test_twenty_iterations_fit_the_explicit_power_limit() -> None:
    request = FasterAmplitudeEstimationRequest(_rotation_problem(0.1), 0.05, 20)

    assert request.max_iterations == 20


def test_good_state_predicate_must_return_bool() -> None:
    problem = _rotation_problem(0.1, predicate=lambda _: 1)
    request = FasterAmplitudeEstimationRequest(problem, 0.05, 2)
    with pytest.raises(TypeError, match="must return a bool"):
        FasterAmplitudeEstimation(_AnalyticSampler()).run([request])


def test_execution_does_not_mutate_problem_circuits() -> None:
    problem = _rotation_problem(0.1)
    preparation_before = copy.deepcopy(problem.preparation_circuit).to_matrix([0])
    grover_before = copy.deepcopy(problem.iteration_circuit).to_matrix([0])

    FasterAmplitudeEstimation(_AnalyticSampler()).run(
        [FasterAmplitudeEstimationRequest(problem, 0.05, 3)]
    )

    assert np.allclose(problem.preparation_circuit.to_matrix([0]), preparation_before)
    assert np.allclose(problem.iteration_circuit.to_matrix([0]), grover_before)


def test_deterministic_sampler_gives_repeatable_execution_record() -> None:
    theta = 0.1
    request = FasterAmplitudeEstimationRequest(_rotation_problem(theta), 0.05, 4)

    first = FasterAmplitudeEstimation(_AnalyticSampler()).run([request])[0]
    second = FasterAmplitudeEstimation(_AnalyticSampler()).run([request])[0]

    assert first == second
