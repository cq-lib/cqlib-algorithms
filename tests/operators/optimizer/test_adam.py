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

"""Adam optimizer tests."""

import math

import pytest
from cqlib.qis import PauliString

from cqlib import Circuit, Parameter
from cqlib_algorithms.operators.gradient import (
    FiniteDifferenceGradient,
    ParameterShiftGradient,
)
from cqlib_algorithms.operators.optimizer import Adam
from cqlib_algorithms.protocols import (
    EstimatorRequest,
    Gradient,
    GradientResult,
    OptimizerRequest,
    OptimizerStatus,
)


def _request(value=1.0, names=("x",)):
    parameter = Parameter("x")
    circuit = Circuit(1)
    circuit.ry(0, parameter)
    return OptimizerRequest(
        EstimatorRequest(circuit, PauliString.from_str("Z"), {"x": value}),
        names,
    )


def _adam_two_step_reference() -> float:
    value = 1.0
    first_moment = 0.0
    second_moment = 0.0
    for iteration in (1, 2):
        gradient = 2.0 * value
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient**2
        corrected_first = first_moment / (1.0 - 0.9**iteration)
        corrected_second = second_moment / (1.0 - 0.999**iteration)
        value -= 0.1 * corrected_first / (math.sqrt(corrected_second) + 1e-8)
    return value


def test_first_two_steps_match_independent_bias_correction(
    quadratic_estimator_factory, quadratic_gradient_factory
) -> None:
    result = Adam(quadratic_gradient_factory(), learning_rate=0.1, maxiter=2).run(
        quadratic_estimator_factory(), [_request()]
    )[0]
    expected = _adam_two_step_reference()
    assert result.optimal_parameters == pytest.approx((expected,), abs=1e-14)
    assert result.optimal_value == pytest.approx(expected**2, abs=1e-14)
    assert result.iterations == 2
    assert result.status is OptimizerStatus.MAX_ITERATIONS


def test_moments_reset_for_each_run(
    quadratic_estimator_factory, quadratic_gradient_factory
) -> None:
    optimizer = Adam(quadratic_gradient_factory(), learning_rate=0.1, maxiter=2)
    first = optimizer.run(quadratic_estimator_factory(), [_request()])[0]
    second = optimizer.run(quadratic_estimator_factory(), [_request()])[0]
    assert second.optimal_parameters == first.optimal_parameters


def test_equal_tolerance_stops_after_counted_update(
    quadratic_estimator_factory, quadratic_gradient_factory
) -> None:
    epsilon = 1e-8
    first_step = 0.1 * 2.0 / (2.0 + epsilon)
    result = Adam(
        quadratic_gradient_factory(),
        learning_rate=0.1,
        epsilon=epsilon,
        maxiter=5,
        tolerance=first_step,
    ).run(quadratic_estimator_factory(), [_request()])[0]
    assert result.iterations == 1
    assert result.status is OptimizerStatus.STEP_TOLERANCE


@pytest.mark.parametrize(
    ("maxiter", "names", "status"),
    [
        (0, ("x",), OptimizerStatus.MAX_ITERATIONS),
        (3, (), OptimizerStatus.NO_PARAMETERS),
    ],
)
def test_non_iterating_cases_still_evaluate_final_point(
    quadratic_estimator_factory,
    quadratic_gradient_factory,
    maxiter,
    names,
    status,
) -> None:
    estimator = quadratic_estimator_factory()
    gradient = quadratic_gradient_factory()
    result = Adam(gradient, maxiter=maxiter).run(estimator, [_request(names=names)])[0]
    assert result.iterations == 0
    assert result.status is status
    assert estimator.batch_sizes == [1]
    assert gradient.batch_sizes == []


def test_empty_batch_skips_dependencies(
    quadratic_estimator_factory, quadratic_gradient_factory
) -> None:
    estimator = quadratic_estimator_factory()
    gradient = quadratic_gradient_factory()
    assert Adam(gradient).run(estimator, []) == []
    assert estimator.batch_sizes == []
    assert gradient.batch_sizes == []


def test_requests_stop_independently_and_keep_order(
    quadratic_estimator_factory, quadratic_gradient_factory
) -> None:
    gradient = quadratic_gradient_factory()
    results = Adam(gradient, learning_rate=0.1, maxiter=3, tolerance=0.0).run(
        quadratic_estimator_factory(),
        [_request(0.0), _request(1.0)],
    )
    assert results[0].optimal_parameters == (0.0,)
    assert [result.iterations for result in results] == [1, 3]
    assert [result.status for result in results] == [
        OptimizerStatus.STEP_TOLERANCE,
        OptimizerStatus.MAX_ITERATIONS,
    ]
    assert gradient.batch_sizes == [2, 1, 1]


@pytest.mark.parametrize(
    "gradient", [ParameterShiftGradient(), FiniteDifferenceGradient()]
)
def test_integrates_with_concrete_gradients(gradient, matrix_estimator) -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    request = OptimizerRequest(
        EstimatorRequest(circuit, PauliString.from_str("Z"), {"theta": 0.4}),
        ("theta",),
    )
    result = Adam(gradient, learning_rate=0.1, maxiter=3).run(
        matrix_estimator, [request]
    )[0]
    assert result.optimal_value < math.cos(0.4)
    assert result.status is OptimizerStatus.MAX_ITERATIONS


def test_gradient_result_count_is_checked(
    quadratic_estimator_factory, quadratic_gradient_factory
) -> None:
    class BrokenGradient(Gradient):
        def run(self, estimator, requests, /):
            return []

    with pytest.raises(ValueError, match="one result per active"):
        Adam(BrokenGradient(), maxiter=1).run(
            quadratic_estimator_factory(), [_request()]
        )


def test_unrepresentable_second_moment_raises_before_final_evaluation(
    quadratic_estimator_factory,
) -> None:
    class HugeGradient(Gradient):
        def run(self, estimator, requests, /):
            return [
                GradientResult(request.parameter_names, (1e308,))
                for request in requests
            ]

    estimator = quadratic_estimator_factory()
    with pytest.raises(ValueError, match="second moment became non-finite"):
        Adam(HugeGradient(), maxiter=1).run(estimator, [_request()])
    assert estimator.batch_sizes == []


def test_nonzero_subnormal_step_does_not_satisfy_zero_tolerance(
    quadratic_estimator_factory,
) -> None:
    class UnitGradient(Gradient):
        def run(self, estimator, requests, /):
            return [
                GradientResult(request.parameter_names, (1.0,)) for request in requests
            ]

    smallest = float.fromhex("0x0.0000000000001p-1022")
    result = Adam(
        UnitGradient(),
        learning_rate=smallest,
        epsilon=smallest,
        maxiter=1,
        tolerance=0.0,
    ).run(quadratic_estimator_factory(), [_request(0.0)])[0]
    assert result.iterations == 1
    assert result.status is OptimizerStatus.MAX_ITERATIONS


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"learning_rate": True}, TypeError),
        ({"learning_rate": 0.0}, ValueError),
        ({"beta1": True}, TypeError),
        ({"beta1": -0.1}, ValueError),
        ({"beta1": 1.0}, ValueError),
        ({"beta2": float("nan")}, ValueError),
        ({"epsilon": 0.0}, ValueError),
        ({"maxiter": True}, TypeError),
        ({"maxiter": -1}, ValueError),
        ({"tolerance": True}, TypeError),
        ({"tolerance": -1.0}, ValueError),
    ],
)
def test_constructor_validation(quadratic_gradient_factory, kwargs, error) -> None:
    with pytest.raises(error):
        Adam(quadratic_gradient_factory(), **kwargs)
