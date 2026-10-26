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

"""SPSA optimizer tests."""

import math

import numpy as np
import pytest
from cqlib import Circuit, Parameter
from cqlib.qis import PauliString

from cqlib_algorithms.operators.optimizer import SPSAOptimizer
from cqlib_algorithms.protocols import (
    EstimatorRequest,
    OptimizerRequest,
    OptimizerStatus,
)


def _request(bindings, names=None):
    circuit = Circuit(1)
    for name in bindings:
        circuit.ry(0, Parameter(name))
    return OptimizerRequest(
        EstimatorRequest(circuit, PauliString.from_str("Z"), bindings),
        tuple(bindings) if names is None else names,
    )


def _one_step_quadratic(values, delta, learning_rate, perturbation):
    values = np.asarray(values, dtype=np.float64)
    plus = np.sum(np.square(values + perturbation * delta))
    minus = np.sum(np.square(values - perturbation * delta))
    direction = (plus - minus) / (2.0 * perturbation)
    return values - learning_rate * direction * delta


@pytest.mark.parametrize("seed", [1, 8, 21])
def test_one_dimensional_rademacher_sign_cancels(
    quadratic_estimator_factory, seed
) -> None:
    result = SPSAOptimizer(0.25, 0.1, maxiter=1, seed=seed).run(
        quadratic_estimator_factory(), [_request({"x": 1.0})]
    )[0]
    assert result.optimal_parameters == pytest.approx((0.5,))
    assert result.optimal_value == pytest.approx(0.25)
    assert result.iterations == 1
    assert result.status is OptimizerStatus.MAX_ITERATIONS


def test_seeded_two_dimensional_step_and_evaluation_count(
    quadratic_estimator_factory,
) -> None:
    seed = 7
    expected_rng = np.random.default_rng(seed)
    delta = expected_rng.integers(0, 2, size=2, dtype=np.int64) * 2 - 1
    expected = _one_step_quadratic([1.0, -0.5], delta, 0.1, 0.2)
    estimator = quadratic_estimator_factory()
    result = SPSAOptimizer(0.1, 0.2, maxiter=1, seed=seed).run(
        estimator, [_request({"x": 1.0, "y": -0.5})]
    )[0]
    assert result.optimal_parameters == pytest.approx(tuple(expected), abs=1e-14)
    assert result.optimal_value == pytest.approx(float(expected @ expected))
    assert estimator.batch_sizes == [2, 1]


def test_random_stream_advances_across_runs(quadratic_estimator_factory) -> None:
    seed = 11
    expected_rng = np.random.default_rng(seed)
    first_delta = expected_rng.integers(0, 2, size=2, dtype=np.int64) * 2 - 1
    second_delta = expected_rng.integers(0, 2, size=2, dtype=np.int64) * 2 - 1
    first_expected = _one_step_quadratic([1.0, 0.2], first_delta, 0.1, 0.2)
    second_expected = _one_step_quadratic([1.0, 0.2], second_delta, 0.1, 0.2)
    optimizer = SPSAOptimizer(0.1, 0.2, maxiter=1, seed=seed)
    first = optimizer.run(
        quadratic_estimator_factory(), [_request({"x": 1.0, "y": 0.2})]
    )[0]
    second = optimizer.run(
        quadratic_estimator_factory(), [_request({"x": 1.0, "y": 0.2})]
    )[0]
    assert first.optimal_parameters == pytest.approx(tuple(first_expected))
    assert second.optimal_parameters == pytest.approx(tuple(second_expected))


def test_schedule_uses_request_local_iteration_index(
    quadratic_estimator_factory,
) -> None:
    learning_rate = 0.3
    perturbation = 0.2
    alpha = 0.7
    gamma = 0.2
    stability = 2.0
    value = 1.0
    for iteration in range(2):
        rate = learning_rate / (stability + iteration + 1) ** alpha
        value -= rate * 2.0 * value
    estimator = quadratic_estimator_factory()
    result = SPSAOptimizer(
        learning_rate,
        perturbation,
        alpha=alpha,
        gamma=gamma,
        stability_constant=stability,
        maxiter=2,
        seed=4,
    ).run(estimator, [_request({"x": 1.0})])[0]
    assert result.optimal_parameters == pytest.approx((value,), abs=1e-12)
    first_radius = abs(estimator.bindings[0]["x"] - estimator.bindings[1]["x"]) / 2
    second_radius = abs(estimator.bindings[2]["x"] - estimator.bindings[3]["x"]) / 2
    assert first_radius == pytest.approx(perturbation)
    assert second_radius == pytest.approx(perturbation / 2**gamma)


def test_equal_tolerance_stops_after_update(quadratic_estimator_factory) -> None:
    result = SPSAOptimizer(0.25, 0.5, maxiter=3, tolerance=0.5, seed=1).run(
        quadratic_estimator_factory(), [_request({"x": 1.0})]
    )[0]
    assert result.optimal_parameters == pytest.approx((0.5,))
    assert result.iterations == 1
    assert result.status is OptimizerStatus.STEP_TOLERANCE


@pytest.mark.parametrize(
    ("maxiter", "names", "status"),
    [
        (0, ("x",), OptimizerStatus.MAX_ITERATIONS),
        (3, (), OptimizerStatus.NO_PARAMETERS),
    ],
)
def test_non_iterating_cases_have_one_final_evaluation(
    quadratic_estimator_factory, maxiter, names, status
) -> None:
    estimator = quadratic_estimator_factory()
    result = SPSAOptimizer(0.1, 0.2, maxiter=maxiter, seed=2).run(
        estimator, [_request({"x": 1.0}, names)]
    )[0]
    assert result.iterations == 0
    assert result.status is status
    assert estimator.batch_sizes == [1]


def test_empty_batch_skips_estimator(quadratic_estimator_factory) -> None:
    estimator = quadratic_estimator_factory()
    assert SPSAOptimizer(0.1, 0.2, seed=2).run(estimator, []) == []
    assert estimator.batch_sizes == []


def test_requests_stop_independently_and_keep_order(
    quadratic_estimator_factory,
) -> None:
    estimator = quadratic_estimator_factory()
    results = SPSAOptimizer(0.1, 0.5, maxiter=3, tolerance=0.0, seed=3).run(
        estimator, [_request({"first": 0.0}), _request({"second": 1.0})]
    )
    assert [result.parameter_names for result in results] == [
        ("first",),
        ("second",),
    ]
    assert [result.iterations for result in results] == [1, 3]
    assert [result.status for result in results] == [
        OptimizerStatus.STEP_TOLERANCE,
        OptimizerStatus.MAX_ITERATIONS,
    ]
    assert estimator.batch_sizes == [4, 2, 2, 2]


def test_integrates_with_real_ry_pauli_objective(matrix_estimator) -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    request = OptimizerRequest(
        EstimatorRequest(circuit, PauliString.from_str("Z"), {"theta": 0.4}),
        ("theta",),
    )
    result = SPSAOptimizer(0.2, 0.05, maxiter=2, seed=5).run(
        matrix_estimator, [request]
    )[0]
    assert result.optimal_value < math.cos(0.4)
    assert matrix_estimator.batch_sizes == [2, 2, 1]


def test_perturbation_result_count_is_checked(quadratic_estimator_factory) -> None:
    class BrokenEstimator(quadratic_estimator_factory):
        def run(self, requests, /):
            return []

    with pytest.raises(ValueError, match="one result per perturbation"):
        SPSAOptimizer(0.1, 0.2, maxiter=1, seed=1).run(
            BrokenEstimator(), [_request({"x": 1.0})]
        )


@pytest.mark.parametrize(
    ("alpha", "stability_constant"),
    [(2.0, 1e308), (1e308, 1.0)],
    ids=("power-overflow", "schedule-underflow"),
)
def test_invalid_runtime_schedule_raises_before_estimator(
    quadratic_estimator_factory, alpha, stability_constant
) -> None:
    estimator = quadratic_estimator_factory()
    optimizer = SPSAOptimizer(
        0.1,
        0.2,
        alpha=alpha,
        stability_constant=stability_constant,
        maxiter=1,
        seed=1,
    )
    with pytest.raises(ValueError, match="learning_rate schedule"):
        optimizer.run(estimator, [_request({"x": 1.0})])
    assert estimator.batch_sizes == []


def test_nonzero_subnormal_step_does_not_satisfy_zero_tolerance(
    quadratic_estimator_factory,
) -> None:
    smallest = float.fromhex("0x0.0000000000001p-1022")
    result = SPSAOptimizer(
        smallest,
        0.5,
        maxiter=1,
        tolerance=0.0,
        seed=1,
    ).run(quadratic_estimator_factory(), [_request({"x": 0.5})])[0]
    assert result.iterations == 1
    assert result.status is OptimizerStatus.MAX_ITERATIONS


@pytest.mark.parametrize(
    ("args", "kwargs", "error"),
    [
        ((True, 0.1), {}, TypeError),
        ((0.0, 0.1), {}, ValueError),
        ((0.1, True), {}, TypeError),
        ((0.1, 0.0), {}, ValueError),
        ((0.1, 0.1), {"alpha": 0.0}, ValueError),
        ((0.1, 0.1), {"gamma": float("nan")}, ValueError),
        ((0.1, 0.1), {"stability_constant": True}, TypeError),
        ((0.1, 0.1), {"stability_constant": -1.0}, ValueError),
        ((0.1, 0.1), {"maxiter": True}, TypeError),
        ((0.1, 0.1), {"maxiter": -1}, ValueError),
        ((0.1, 0.1), {"tolerance": True}, TypeError),
        ((0.1, 0.1), {"tolerance": -1.0}, ValueError),
        ((0.1, 0.1), {"seed": True}, TypeError),
        ((0.1, 0.1), {"seed": 1.5}, TypeError),
    ],
)
def test_constructor_validation(args, kwargs, error) -> None:
    with pytest.raises(error):
        SPSAOptimizer(*args, **kwargs)
