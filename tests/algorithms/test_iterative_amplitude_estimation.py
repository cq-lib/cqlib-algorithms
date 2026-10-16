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
from fractions import Fraction

import numpy as np
import pytest
from cqlib import Circuit
from cqlib.device import ExecutionResult

from cqlib_algorithms import (
    AmplitudeEstimationProblem,
    IterativeAmplitudeEstimation,
    IterativeAmplitudeEstimationRequest,
)
from cqlib_algorithms.algorithms.amplitude_estimation.iterative import (
    _find_next_power,
)
from cqlib_algorithms.protocols import Sampler


def _rotation_problem(theta: float, *, predicate=None) -> AmplitudeEstimationProblem:
    preparation = Circuit(1)
    preparation.ry(0, 2.0 * theta)
    grover = Circuit(1)
    grover.ry(0, 4.0 * theta)
    return AmplitudeEstimationProblem(
        preparation,
        grover,
        (0,),
        predicate if predicate is not None else lambda bits: bits == "1",
    )


class _AnalyticSampler(Sampler):
    """Return rounded Bernoulli counts from the known rotation problem."""

    def __init__(self, theta: float) -> None:
        self.theta = theta
        self.batch_sizes: list[int] = []

    def run(self, requests):
        self.batch_sizes.append(len(requests))
        results = []
        for request in requests:
            # The test problem has one RY in A and one in every Q copy.
            power = (
                sum(operation.name == "RY" for operation in request.circuit.operations)
                - 1
            )
            probability = math.sin((2 * power + 1) * self.theta) ** 2
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
    def __init__(self, theta: float, seed: int) -> None:
        self.theta = theta
        self.generator = np.random.default_rng(seed)

    def run(self, requests):
        results = []
        for request in requests:
            power = (
                sum(operation.name == "RY" for operation in request.circuit.operations)
                - 1
            )
            probability = math.sin((2 * power + 1) * self.theta) ** 2
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


@pytest.mark.reference
def test_adaptive_schedule_and_intervals_follow_chernoff_half_circle_rule() -> None:
    theta = math.pi / 8.0
    sampler = _AnalyticSampler(theta)
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
    assert result.confidence_interval[0] <= math.sin(theta) ** 2
    assert math.sin(theta) ** 2 <= result.confidence_interval[1]
    assert result.confidence_interval[1] - result.confidence_interval[0] <= 0.01
    assert result.round_intervals[-1] == result.confidence_interval
    assert result.shots == 2000
    assert result.num_grover_queries == 22
    assert result.num_oracle_queries == 11_000


@pytest.mark.parametrize("theta", [0.0, math.pi / 4.0, math.pi / 2.0])
def test_boundary_and_midpoint_amplitudes_remain_inside_interval(theta: float) -> None:
    result = IterativeAmplitudeEstimation(_AnalyticSampler(theta)).run(
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
        result = IterativeAmplitudeEstimation(_SeededBernoulliSampler(theta, seed)).run(
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
    sampler = _AnalyticSampler(theta)
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


@pytest.mark.reference
def test_no_overshooting_reduces_shots_and_repeated_powers_are_combined() -> None:
    theta = math.pi / 8.0
    request = IterativeAmplitudeEstimationRequest(
        _rotation_problem(theta), 0.002, 0.05, 500, 20
    )
    result = IterativeAmplitudeEstimation(_AnalyticSampler(theta)).run([request])[0]

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
        IterativeAmplitudeEstimation(_AnalyticSampler(theta)).run(
            [
                IterativeAmplitudeEstimationRequest(
                    _rotation_problem(theta), 0.05, 0.05, 200, 3
                )
            ]
        )


def test_theoretical_round_budget_is_independent_of_safety_limit() -> None:
    theta = math.pi / 8.0
    problem = _rotation_problem(theta)
    short_limit = IterativeAmplitudeEstimation(_AnalyticSampler(theta)).run(
        [IterativeAmplitudeEstimationRequest(problem, 0.005, 0.05, 500, 20)]
    )[0]
    long_limit = IterativeAmplitudeEstimation(_AnalyticSampler(theta)).run(
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
        IterativeAmplitudeEstimation(_AnalyticSampler(0.2)).run(
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
    sampler = _AnalyticSampler(0.2)
    assert IterativeAmplitudeEstimation(sampler).run([]) == []
    assert sampler.batch_sizes == []
