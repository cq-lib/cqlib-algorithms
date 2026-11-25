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

"""Tests for iterative amplitude estimation."""

from __future__ import annotations

import math
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
from cqlib.device import ExecutionResult

from cqlib_algorithms import (
    AmplitudeEstimationProblem,
    IterativeAmplitudeEstimation,
    IterativeAmplitudeEstimationRequest,
    IterativeAmplitudeEstimationResult,
    StandardAmplitudeEstimation,
)
from cqlib_algorithms.algorithms.amplitude_estimation.iterative import (
    _find_next_power,
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
    """Return rounded Bernoulli counts from the known rotation problem."""

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


class _SeededBernoulliSampler(Sampler):
    def __init__(self, seed: int) -> None:
        self.generator = np.random.default_rng(seed)

    def run(self, requests):
        results = []
        for request in requests:
            state = request.circuit.to_matrix()[:, 0]
            probability = float(abs(state[1]) ** 2)
            assert -1e-12 <= probability <= 1.0 + 1e-12
            probability = min(1.0, max(0.0, probability))
            good = int(self.generator.binomial(request.shots, probability))
            results.append(
                ExecutionResult.from_counts(
                    "seeded",
                    list(request.qubits),
                    request.shots,
                    len(request.qubits),
                    {"0": request.shots - good, "1": good},
                    "test",
                )
            )
        return results


class _SeededStatevectorSampler(Sampler):
    """Replay finite-shot sampling of the actual component circuits."""

    def __init__(self):
        self.generator = np.random.default_rng(2026)
        self.sampler = StatevectorSampler()

    def run(self, requests):
        return self.sampler.run(
            [
                replace(request, seed=int(self.generator.integers(2**32)))
                for request in requests
            ]
        )


@pytest.mark.parametrize("marked", [0, 1])
def test_components_share_standard_problem_and_match_batch(marked):
    theta = math.pi / 8
    preparation = AmplitudeEncoding((math.cos(theta), 0, 1j * math.sin(theta), 0))
    oracle = PhaseOracle(
        BooleanFunction.from_marked_states(2, [2 * marked, 2 * marked + 1])
    )
    iteration = GroverOperator(oracle, state_preparation=preparation)

    def good(bits):
        assert len(bits) == 1
        return bits == str(marked)

    problem = AmplitudeEstimationProblem(preparation, iteration, (1,), good)
    standard = StandardAmplitudeEstimation(_SeededStatevectorSampler()).run(
        problem, num_evaluation_qubits=3, shots=128
    )
    result = IterativeAmplitudeEstimation(_SeededStatevectorSampler()).run(
        problem=problem, epsilon=0.01, alpha=0.05, shots=500
    )
    batch = IterativeAmplitudeEstimation(_SeededStatevectorSampler()).run(
        [IterativeAmplitudeEstimationRequest(problem, 0.01, 0.05, 500)]
    )

    expected = math.sin(theta) ** 2 if marked else math.cos(theta) ** 2
    assert standard.estimation == pytest.approx(expected)
    assert isinstance(result, IterativeAmplitudeEstimationResult)
    assert batch == [result]
    low, high = result.confidence_interval
    assert low <= expected <= high
    assert high - low <= 0.02
    assert result.estimation == (low + high) / 2


@pytest.mark.reference
def test_adaptive_schedule_and_intervals_follow_chernoff_half_circle_rule() -> None:
    theta = math.pi / 8.0
    sampler = _AnalyticSampler()
    result = IterativeAmplitudeEstimation(sampler).run(
        [
            IterativeAmplitudeEstimationRequest(
                _rotation_problem(theta),
                epsilon=0.005,
                alpha=0.05,
                shots=500,
                max_rounds=20,
            )
        ]
    )[0]

    assert result.powers == (0, 1, 4, 17)
    assert result.round_shots == (500, 500, 500, 500)
    assert sampler.batch_sizes == [1, 1, 1, 1]
    rounds = math.ceil(math.log2(math.pi / (8 * 0.005)))
    radius = math.sqrt(math.log(2 * rounds / 0.05) / (2 * 500))
    observed = round(500 * math.sin(theta) ** 2) / 500
    assert result.round_intervals[0] == pytest.approx(
        (observed - radius, observed + radius)
    )
    assert result.confidence_interval[0] <= math.sin(theta) ** 2
    assert math.sin(theta) ** 2 <= result.confidence_interval[1]
    assert result.confidence_interval[1] - result.confidence_interval[0] <= 0.01
    assert result.round_intervals[-1] == result.confidence_interval
    assert result.shots == 2000
    assert result.num_grover_queries == 22
    assert result.num_oracle_queries == 11_000


@pytest.mark.parametrize("theta", [0.0, math.pi / 4.0, math.pi / 2.0])
def test_boundary_and_midpoint_amplitudes_remain_inside_interval(theta: float) -> None:
    result = IterativeAmplitudeEstimation(_AnalyticSampler()).run(
        [
            IterativeAmplitudeEstimationRequest(
                _rotation_problem(theta),
                epsilon=0.05,
                alpha=0.05,
                shots=2000,
                max_rounds=10,
            )
        ]
    )[0]
    amplitude = math.sin(theta) ** 2

    assert result.confidence_interval[0] <= amplitude
    assert amplitude <= result.confidence_interval[1]
    assert result.confidence_interval[1] - result.confidence_interval[0] <= 0.1


@pytest.mark.reference
def test_seeded_monte_carlo_has_conservative_coverage_regression() -> None:
    theta = 0.37
    amplitude = math.sin(theta) ** 2
    covered = 0
    for seed in range(20):
        result = IterativeAmplitudeEstimation(_SeededBernoulliSampler(seed)).run(
            [
                IterativeAmplitudeEstimationRequest(
                    _rotation_problem(theta), 0.02, 0.1, 300, 20
                )
            ]
        )[0]
        covered += (
            result.confidence_interval[0] <= amplitude <= result.confidence_interval[1]
        )

    assert covered >= 18


def test_active_requests_are_batched_and_completed_requests_drop_out() -> None:
    theta = math.pi / 8.0
    sampler = _AnalyticSampler()
    results = IterativeAmplitudeEstimation(sampler).run(
        [
            IterativeAmplitudeEstimationRequest(
                _rotation_problem(theta), 0.05, 0.05, 500, 20
            ),
            IterativeAmplitudeEstimationRequest(
                _rotation_problem(theta), 0.005, 0.05, 500, 20
            ),
        ]
    )

    assert sampler.batch_sizes[0] == 2
    assert sampler.batch_sizes[1:] == [2, 1, 1]
    assert len(results[0].powers) == 2
    assert results[1].powers == (0, 1, 4, 17)
    for result, epsilon in zip(results, (0.05, 0.005)):
        single = IterativeAmplitudeEstimation(_AnalyticSampler()).run(
            _rotation_problem(theta),
            epsilon=epsilon,
            alpha=0.05,
            shots=500,
            max_rounds=20,
        )
        assert single == result


@pytest.mark.reference
def test_no_overshooting_reduces_shots_and_repeated_powers_are_combined() -> None:
    theta = math.pi / 8.0
    request = IterativeAmplitudeEstimationRequest(
        _rotation_problem(theta), 0.002, 0.05, 500, 20
    )
    result = IterativeAmplitudeEstimation(_AnalyticSampler()).run([request])[0]

    rounds = math.ceil(math.log2(math.pi / (8.0 * request.epsilon)))
    lmax = math.asin(
        (2.0 / request.shots * math.log(2.0 * rounds / request.alpha)) ** 0.25
    )
    expected_reduced_shots = math.ceil(
        request.shots * lmax / request.epsilon / (4 * 68 + 2) / 10.0
    )
    assert result.powers == (0, 1, 4, 17, 68, 68, 68, 145)
    assert result.round_shots == (
        500,
        500,
        500,
        500,
        expected_reduced_shots,
        expected_reduced_shots,
        expected_reduced_shots,
        18,
    )
    assert result.round_intervals[6][1] - result.round_intervals[6][0] < (
        result.round_intervals[4][1] - result.round_intervals[4][0]
    )
    assert result.num_oracle_queries == sum(
        power * shots for power, shots in zip(result.powers, result.round_shots)
    )


def test_max_rounds_is_a_hard_safety_limit() -> None:
    theta = 0.1
    with pytest.raises(RuntimeError, match="did not converge"):
        IterativeAmplitudeEstimation(_AnalyticSampler()).run(
            [
                IterativeAmplitudeEstimationRequest(
                    _rotation_problem(theta), 0.05, 0.05, 200, 3
                )
            ]
        )
    with pytest.raises(RuntimeError, match="did not converge"):
        IterativeAmplitudeEstimation(_AnalyticSampler()).run(
            _rotation_problem(theta), epsilon=0.05, alpha=0.05, shots=200, max_rounds=3
        )


def test_theoretical_round_budget_is_independent_of_safety_limit() -> None:
    theta = math.pi / 8.0
    problem = _rotation_problem(theta)
    short_limit = IterativeAmplitudeEstimation(_AnalyticSampler()).run(
        [IterativeAmplitudeEstimationRequest(problem, 0.005, 0.05, 500, 20)]
    )[0]
    long_limit = IterativeAmplitudeEstimation(_AnalyticSampler()).run(
        [IterativeAmplitudeEstimationRequest(problem, 0.005, 0.05, 500, 50)]
    )[0]

    assert short_limit == long_limit


def test_find_next_power_rejects_float_tolerance_half_circle_crossing() -> None:
    power, _ = _find_next_power(
        0,
        (0.04999999999999995, 0.09),
        True,
    )

    assert power != 2


def test_find_next_power_rejects_high_scale_float_product_alias() -> None:
    power, _ = _find_next_power(
        0,
        (0.21114872684741617, 0.21114872695018716),
        True,
    )

    assert power != 1_216_267_786


@pytest.mark.parametrize(
    ("previous_power", "interval", "previous_upper"),
    [
        (0, (0.05, 0.09), True),
        (0, (0.1, 0.14), True),
        (10, (0.101, 0.105), True),
        (20, (0.211, 0.213), False),
        (0, (0.21114872684741617, 0.21114872695018716), True),
    ],
)
def test_every_new_power_is_exactly_contained_in_one_half_circle(
    previous_power: int,
    interval: tuple[float, float],
    previous_upper: bool,
) -> None:
    power, _ = _find_next_power(previous_power, interval, previous_upper)
    assert power > previous_power

    scale = 4 * power + 2
    scaled_low = scale * Fraction.from_float(interval[0])
    scaled_high = scale * Fraction.from_float(interval[1])
    low_floor = math.floor(scaled_low)
    high_floor = math.floor(scaled_high)
    fractional_low = scaled_low - low_floor
    fractional_high = scaled_high - high_floor
    assert low_floor == high_floor
    assert fractional_high <= Fraction(1, 2) or fractional_low >= Fraction(1, 2)


def test_good_state_predicate_must_return_bool() -> None:
    problem = _rotation_problem(0.2, predicate=lambda _: 1)
    with pytest.raises(TypeError, match="must return a bool"):
        IterativeAmplitudeEstimation(_AnalyticSampler()).run(
            [IterativeAmplitudeEstimationRequest(problem, 0.1, 0.05, 100, 5)]
        )


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("epsilon", True, TypeError, "real number"),
        ("epsilon", 0.0, ValueError, "must lie"),
        ("epsilon", 0.6, ValueError, "must lie"),
        ("alpha", 1.0, ValueError, "must lie"),
        ("alpha", math.nan, ValueError, "finite"),
        ("shots", True, TypeError, "integer"),
        ("shots", 0, ValueError, "greater than zero"),
        ("max_rounds", 0, ValueError, "greater than zero"),
    ],
)
def test_request_validation(field, value, error, message) -> None:
    arguments = {
        "problem": _rotation_problem(0.2),
        "epsilon": 0.1,
        "alpha": 0.05,
        "shots": 100,
        "max_rounds": 5,
    }
    arguments[field] = value
    with pytest.raises(error, match=message):
        IterativeAmplitudeEstimationRequest(**arguments)


def test_request_rejects_invalid_paper_round_and_lmax_parameters() -> None:
    problem = _rotation_problem(0.2)
    with pytest.raises(ValueError, match="at least ceil"):
        IterativeAmplitudeEstimationRequest(problem, 0.005, 0.05, 500, 6)
    with pytest.raises(ValueError, match="too small.*Lmax"):
        IterativeAmplitudeEstimationRequest(problem, 0.1, 0.05, 1, 5)


def test_request_enforces_binary64_chernoff_shot_limit() -> None:
    problem = _rotation_problem(0.2)
    maximum = (1 << 53) - 1

    request = IterativeAmplitudeEstimationRequest(problem, 0.1, 0.05, maximum)
    assert request.shots == maximum
    with pytest.raises(ValueError, match=r"numeric limit of 2\*\*53 - 1"):
        IterativeAmplitudeEstimationRequest(problem, 0.1, 0.05, maximum + 1)


def test_epsilon_respects_binary64_geometry_resolution() -> None:
    minimum = math.pi * math.ulp(0.25)
    problem = _rotation_problem(0.2)
    with pytest.raises(ValueError, match=r"pi \* ulp\(0\.25\)"):
        IterativeAmplitudeEstimationRequest(
            problem,
            math.nextafter(minimum, 0.0),
            0.05,
            1000,
            60,
        )

    request = IterativeAmplitudeEstimationRequest(
        problem,
        minimum,
        0.05,
        1000,
        60,
    )
    assert request.epsilon / math.pi >= math.ulp(0.25)


def test_empty_batch_does_not_call_sampler() -> None:
    sampler = _AnalyticSampler()
    assert IterativeAmplitudeEstimation(sampler).run([]) == []
    assert sampler.batch_sizes == []


def test_entire_batch_is_revalidated_before_the_first_round():
    first = IterativeAmplitudeEstimationRequest(_rotation_problem(0.2), 0.05, 0.05, 500)
    second = IterativeAmplitudeEstimationRequest(
        _rotation_problem(0.3), 0.05, 0.05, 500
    )
    second.problem.iteration_circuit.measure(0)
    sampler = _AnalyticSampler()
    with pytest.raises(ValueError, match="static unitary|classical data"):
        IterativeAmplitudeEstimation(sampler).run((first, second))
    assert sampler.batch_sizes == []


@pytest.mark.parametrize("missing", ["epsilon", "alpha", "shots"])
def test_single_requires_its_estimation_options_before_sampling(missing):
    options = {"epsilon": 0.05, "alpha": 0.05, "shots": 500}
    del options[missing]
    sampler = _AnalyticSampler()
    with pytest.raises(TypeError, match="epsilon, alpha, and shots"):
        IterativeAmplitudeEstimation(sampler).run(_rotation_problem(0.2), **options)
    assert sampler.batch_sizes == []


@pytest.mark.parametrize(
    "option", [{"epsilon": 0.05}, {"alpha": 0.1}, {"shots": 256}, {"max_rounds": 50}]
)
def test_batch_rejects_single_problem_options_before_sampling(option):
    sampler = _AnalyticSampler()
    request = IterativeAmplitudeEstimationRequest(
        _rotation_problem(0.2), 0.05, 0.05, 500
    )
    with pytest.raises(TypeError, match="batch"):
        IterativeAmplitudeEstimation(sampler).run([request], **option)
    assert sampler.batch_sizes == []
