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

"""Tests for maximum-likelihood amplitude estimation."""

from __future__ import annotations

import math

import numpy as np
import pytest
from cqlib import Circuit
from cqlib.device import ExecutionResult

from cqlib_algorithms import (
    AmplitudeEstimationProblem,
    MaximumLikelihoodAmplitudeEstimation,
    MaximumLikelihoodAmplitudeEstimationRequest,
    MaximumLikelihoodAmplitudeEstimationResult,
)
from cqlib_algorithms.algorithms.amplitude_estimation.maximum_likelihood import (
    _global_maximum_likelihood,
)
from cqlib_algorithms.operators import (
    AmplitudeEncoding,
    BooleanFunction,
    GroverOperator,
    PhaseOracle,
    StatevectorSampler,
)
from cqlib_algorithms.operators.state_preparation import UserCircuitPreparation
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


class _ScriptedSampler(Sampler):
    def __init__(self, scripts: list[dict[str, int]]) -> None:
        self.scripts = scripts
        self.calls = 0
        self.batch_sizes: list[int] = []
        self.requests = []

    def run(self, requests):
        self.calls += 1
        self.requests = list(requests)
        self.batch_sizes.append(len(requests))
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


def _independent_log_likelihood(
    theta: float,
    powers: tuple[int, ...],
    good_counts: tuple[int, ...],
    shots: int,
) -> float:
    value = 0.0
    for power, good in zip(powers, good_counts):
        angle = (2 * power + 1) * theta
        sine = abs(math.sin(angle))
        cosine = abs(math.cos(angle))
        if good:
            if sine == 0.0:
                return -math.inf
            value += 2.0 * good * math.log(sine)
        failures = shots - good
        if failures:
            if cosine == 0.0:
                return -math.inf
            value += 2.0 * failures * math.log(cosine)
    return value


@pytest.mark.reference
def test_multimodal_likelihood_matches_independent_complete_grid_search() -> None:
    powers = (0, 1, 2, 4)
    shots = 200
    good_counts = (31, 168, 173, 13)
    scripts = [{"0": shots - count, "1": count} for count in good_counts]
    sampler = _ScriptedSampler(scripts)
    result = MaximumLikelihoodAmplitudeEstimation(sampler).run(
        [
            MaximumLikelihoodAmplitudeEstimationRequest(
                _rotation_problem(0.4), powers, shots, 1e-10
            )
        ]
    )[0]

    grid = np.linspace(0.0, math.pi / 2.0, 200_001)
    values = np.asarray(
        [
            _independent_log_likelihood(float(theta), powers, good_counts, shots)
            for theta in grid
        ]
    )
    expected_theta = float(grid[int(np.argmax(values))])
    actual_theta = math.asin(math.sqrt(result.estimation))

    assert sampler.calls == 1
    assert sampler.batch_sizes == [len(powers)]
    assert actual_theta == pytest.approx(expected_theta, abs=1e-5)
    assert result.log_likelihood == pytest.approx(
        _independent_log_likelihood(actual_theta, powers, good_counts, shots),
        abs=1e-9,
    )
    assert result.good_counts == good_counts
    assert result.num_grover_queries == sum(powers)
    assert result.num_oracle_queries == shots * sum(powers)
    assert result.shots == shots * len(powers)
    assert result.confidence_interval == result.asymptotic_interval
    fisher_information = 4.0 * shots * sum((2 * power + 1) ** 2 for power in powers)
    radius = 1.959963984540054 / math.sqrt(fisher_information)
    assert result.asymptotic_interval == pytest.approx(
        (
            math.sin(max(0.0, actual_theta - radius)) ** 2,
            math.sin(min(math.pi / 2.0, actual_theta + radius)) ** 2,
        ),
        abs=1e-12,
    )


@pytest.mark.parametrize("marked", [0, 1])
def test_component_powers_match_analytic_probabilities_and_legacy_batch(marked):
    preparation = AmplitudeEncoding((3, 0, 1j, 0))
    oracle = PhaseOracle(
        BooleanFunction.from_marked_states(2, (2 * marked, 2 * marked + 1))
    )
    iteration = GroverOperator(oracle, state_preparation=preparation)

    def good(bits):
        assert len(bits) == 1
        return bits == str(marked)

    problem = AmplitudeEstimationProblem(preparation, iteration, (1,), good)
    expected = 0.1 if marked else 0.9
    theta = math.asin(math.sqrt(expected))
    powers = (0, 1, 2, 4)
    shots = 1000
    probabilities = [math.sin((2 * k + 1) * theta) ** 2 for k in powers]
    hits = tuple(round(shots * p) for p in probabilities)
    scripts = [{str(marked): h, str(1 - marked): shots - h} for h in hits]
    sampler = _ScriptedSampler(scripts)
    result = MaximumLikelihoodAmplitudeEstimation(sampler).run(
        problem=problem, powers=powers, shots=shots
    )

    assert isinstance(result, MaximumLikelihoodAmplitudeEstimationResult)
    assert sampler.batch_sizes == [len(powers)]
    for experiment, expected_p in zip(sampler.requests, probabilities):
        assert [qubit.index for qubit in experiment.qubits] == [1]
        state = experiment.circuit.to_matrix()[:, 0]
        observed_p = np.sum(np.abs(state[2 * marked : 2 * marked + 2]) ** 2)
        assert observed_p == pytest.approx(expected_p, abs=1e-12, rel=0)
    assert result.good_counts == hits
    assert result.estimation == pytest.approx(expected, abs=1e-3, rel=0)
    legacy = AmplitudeEstimationProblem(preparation, iteration, (1,), good)
    assert MaximumLikelihoodAmplitudeEstimation(_ScriptedSampler(scripts)).run(
        [MaximumLikelihoodAmplitudeEstimationRequest(legacy, powers, shots)]
    ) == [result]


@pytest.mark.parametrize(
    ("bitstring", "expected"),
    [("0", 0.0), ("1", 1.0)],
)
def test_all_failure_and_all_success_data_select_physical_endpoints(
    bitstring: str,
    expected: float,
) -> None:
    powers = (0, 1, 3)
    shots = 50
    sampler = _ScriptedSampler([{bitstring: shots} for _ in powers])
    result = MaximumLikelihoodAmplitudeEstimation(sampler).run(
        [
            MaximumLikelihoodAmplitudeEstimationRequest(
                _rotation_problem(0.2), powers, shots
            )
        ]
    )[0]

    assert result.estimation == pytest.approx(expected, abs=1e-14)
    assert math.isfinite(result.log_likelihood)
    assert result.asymptotic_interval[0] <= result.estimation
    assert result.estimation <= result.asymptotic_interval[1]

    preparation = AmplitudeEncoding((1, 0) if bitstring == "0" else (0, 1))
    oracle = PhaseOracle(BooleanFunction.from_marked_states(1, [1]))
    iteration = GroverOperator(oracle, state_preparation=preparation)
    problem = AmplitudeEstimationProblem(
        preparation, iteration, (0,), lambda bits: bits == "1"
    )
    assert (
        MaximumLikelihoodAmplitudeEstimation(StatevectorSampler()).run(
            problem, powers=powers, shots=shots
        )
        == result
    )


def test_multiple_problem_schedules_are_flattened_into_one_sampler_call() -> None:
    shots = 100
    sampler = _ScriptedSampler(
        [
            {"0": 75, "1": 25},
            {"0": 25, "1": 75},
            {"0": 100, "1": 100},
            {"0": 20, "1": 180},
            {"0": 180, "1": 20},
        ]
    )
    results = MaximumLikelihoodAmplitudeEstimation(sampler).run(
        [
            MaximumLikelihoodAmplitudeEstimationRequest(
                _rotation_problem(math.pi / 6.0), (0, 1), shots
            ),
            MaximumLikelihoodAmplitudeEstimationRequest(
                _rotation_problem(math.pi / 4.0), (0, 2, 5), 2 * shots
            ),
        ]
    )

    assert sampler.calls == 1
    assert sampler.batch_sizes == [5]
    assert [request.shots for request in sampler.requests] == [100, 100, 200, 200, 200]
    assert [result.powers for result in results] == [(0, 1), (0, 2, 5)]
    assert results[0].good_counts == (25, 75)
    assert results[1].good_counts == (100, 180, 20)
    assert [result.shots for result in results] == [200, 600]


def test_noncontiguous_objective_order_reaches_multibit_predicate() -> None:
    problem = AmplitudeEstimationProblem(
        UserCircuitPreparation(Circuit(3)),
        GroverOperator(
            PhaseOracle(BooleanFunction.from_marked_states(3, [0])),
            state_preparation=UserCircuitPreparation(Circuit(3)),
        ),
        (2, 0),
        lambda bitstring: bitstring == "10",
    )
    sampler = _ScriptedSampler([{"00": 3, "01": 4, "10": 7, "11": 6}])

    result = MaximumLikelihoodAmplitudeEstimation(sampler).run(
        [MaximumLikelihoodAmplitudeEstimationRequest(problem, (0,), 20)]
    )[0]

    assert [qubit.index for qubit in sampler.requests[0].qubits] == [2, 0]
    assert result.good_counts == (7,)
    assert result.estimation == pytest.approx(7 / 20, abs=1e-10)


def test_global_search_does_not_merge_distinct_large_log_likelihood_peaks() -> None:
    powers = (0, 4)
    shots = 14_481_265
    good_counts = (1_684_335, 2_896_253)
    sampler = _ScriptedSampler([{"0": shots - good, "1": good} for good in good_counts])
    result = MaximumLikelihoodAmplitudeEstimation(sampler).run(
        [
            MaximumLikelihoodAmplitudeEstimationRequest(
                _rotation_problem(0.4), powers, shots, 1e-10
            )
        ]
    )[0]

    theta = math.asin(math.sqrt(result.estimation))
    false_peak = 0.298205658582271
    assert theta == pytest.approx(0.3999769072569902, abs=1e-9)
    assert result.log_likelihood > _independent_log_likelihood(
        false_peak, powers, good_counts, shots
    )


def test_search_precision_below_float_resolution_is_rejected() -> None:
    with pytest.raises(ValueError, match="floating-point resolution"):
        MaximumLikelihoodAmplitudeEstimationRequest(
            _rotation_problem(0.2), (0, 1), 100, 1e-300
        )


@pytest.mark.parametrize(
    ("good_counts", "expected_theta"),
    [
        ((25, 100), math.pi / 6.0),
        ((75, 0), math.pi / 3.0),
    ],
)
def test_stationary_singularity_is_selected_within_search_precision(
    good_counts: tuple[int, int],
    expected_theta: float,
) -> None:
    shots = 100
    sampler = _ScriptedSampler([{"0": shots - good, "1": good} for good in good_counts])
    result = MaximumLikelihoodAmplitudeEstimation(sampler).run(
        [
            MaximumLikelihoodAmplitudeEstimationRequest(
                _rotation_problem(expected_theta),
                (0, 1),
                shots,
                search_precision=1e-12,
            )
        ]
    )[0]

    assert result.estimation == math.sin(expected_theta) ** 2
    assert result.log_likelihood == 25 * math.log(0.25) + 75 * math.log(0.75)


def test_near_singularity_score_root_does_not_snap_to_boundary() -> None:
    shots = 100_000_000
    good_counts = (25_000_001, 100_000_000)
    sampler = _ScriptedSampler([{"0": shots - good, "1": good} for good in good_counts])
    precision = 1e-12
    problem = _rotation_problem(math.pi / 6.0)
    algorithm = MaximumLikelihoodAmplitudeEstimation(sampler)
    result = algorithm.run(
        problem, powers=(0, 1), shots=shots, search_precision=precision
    )
    assert algorithm.run(
        [MaximumLikelihoodAmplitudeEstimationRequest(problem, (0, 1), shots, precision)]
    ) == [result]

    theta = math.asin(math.sqrt(result.estimation))
    assert theta != math.pi / 6.0
    assert theta == pytest.approx(0.5235987776977544, abs=precision)


def test_high_precision_global_score_prevents_binary64_peak_reversal() -> None:
    powers = (0, 4)
    shots = 10**18
    good_counts = (108_867_391_320_086_708, shots)
    precision = 1e-12
    theta, _ = _global_maximum_likelihood(powers, good_counts, shots, precision)
    assert theta > 0.5
    assert theta == pytest.approx(0.5196556374964838, abs=precision)


def test_high_precision_global_score_refines_roots_before_ranking() -> None:
    powers = (0, 4)
    shots = 10**30
    good_counts = (108_867_391_320_086_683_245_963_796_036, shots)
    precision = 1e-12
    theta, _ = _global_maximum_likelihood(powers, good_counts, shots, precision)
    assert theta < 0.2
    assert theta == pytest.approx(0.179917238665130259, abs=precision)


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("powers", [], TypeError, "tuple"),
        ("powers", (), ValueError, "must not be empty"),
        ("powers", (1, 2), ValueError, "start with zero"),
        ("powers", (0, 2, 2), ValueError, "strictly increasing"),
        ("powers", (0, True), TypeError, "integers"),
        ("shots", 0, ValueError, "greater than zero"),
        ("search_precision", True, TypeError, "real number"),
        ("search_precision", 0.0, ValueError, "greater than zero"),
        ("search_precision", math.inf, ValueError, "finite"),
    ],
)
def test_request_validation(field, value, error, message) -> None:
    arguments = {
        "problem": _rotation_problem(0.2),
        "powers": (0, 1, 2),
        "shots": 100,
        "search_precision": 1e-8,
    }
    arguments[field] = value
    with pytest.raises(error, match=message):
        MaximumLikelihoodAmplitudeEstimationRequest(**arguments)
    sampler = _ScriptedSampler([])
    with pytest.raises(error, match=message):
        MaximumLikelihoodAmplitudeEstimation(sampler).run(**arguments)
    assert sampler.calls == 0


def test_public_request_enforces_binary64_candidate_geometry_shot_limit() -> None:
    maximum = (1 << 53) - 1
    sampler = _ScriptedSampler([])
    request = MaximumLikelihoodAmplitudeEstimationRequest(
        _rotation_problem(0.2),
        (0, 1),
        maximum,
    )
    assert request.shots == maximum

    with pytest.raises(ValueError, match="candidate geometry numeric limit"):
        MaximumLikelihoodAmplitudeEstimationRequest(
            _rotation_problem(0.2),
            (0, 1),
            maximum + 1,
        )
    assert sampler.calls == 0


def test_empty_batch_does_not_call_sampler() -> None:
    sampler = _ScriptedSampler([])
    assert MaximumLikelihoodAmplitudeEstimation(sampler).run([]) == []
    assert sampler.calls == 0


@pytest.mark.parametrize("options", [{"shots": 100}, {"powers": (0,)}])
def test_single_requires_both_experiment_options_before_sampling(options):
    sampler = _ScriptedSampler([])
    with pytest.raises(TypeError, match="powers and shots"):
        MaximumLikelihoodAmplitudeEstimation(sampler).run(
            _rotation_problem(0.2), **options
        )
    assert sampler.calls == 0


@pytest.mark.parametrize(
    "options", [{"powers": (0,)}, {"shots": 100}, {"search_precision": 1e-8}]
)
def test_batch_rejects_options_belonging_to_individual_requests(options):
    request = MaximumLikelihoodAmplitudeEstimationRequest(
        _rotation_problem(0.2), (0,), 100
    )
    sampler = _ScriptedSampler([])
    with pytest.raises(TypeError, match="batch"):
        MaximumLikelihoodAmplitudeEstimation(sampler).run([request], **options)
    assert sampler.calls == 0


def test_invalid_later_request_prevents_submission_of_the_whole_batch():
    first = MaximumLikelihoodAmplitudeEstimationRequest(
        _rotation_problem(0.2), (0,), 100
    )
    second = MaximumLikelihoodAmplitudeEstimationRequest(
        _rotation_problem(0.3), (0, 2), 200
    )
    second.problem.iteration_circuit.measure(0)
    sampler = _ScriptedSampler([])
    with pytest.raises(ValueError, match="static unitary|classical data"):
        MaximumLikelihoodAmplitudeEstimation(sampler).run([first, second])
    assert sampler.calls == 0
