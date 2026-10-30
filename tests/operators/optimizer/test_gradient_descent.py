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

"""Gradient-descent optimizer tests."""

import math

import pytest
from cqlib import Circuit, Parameter
from cqlib.qis import PauliString

from cqlib_algorithms.operators.gradient import (
    FiniteDifferenceGradient,
    ParameterShiftGradient,
)
from cqlib_algorithms.operators.optimizer import GradientDescent
from cqlib_algorithms.protocols import (
    EstimatorRequest,
    Gradient,
    GradientResult,
    OptimizerRequest,
    OptimizerStatus,
)


def _request(bindings, names):
    circuit = Circuit(1)
    for name in bindings:
        circuit.ry(0, Parameter(name))
    return OptimizerRequest(
        EstimatorRequest(circuit, PauliString.from_str("Z"), bindings),
        names,
    )


def test_quadratic_updates_have_exact_final_value(
    quadratic_estimator_factory, quadratic_gradient_factory
) -> None:
    estimator = quadratic_estimator_factory()
    gradient = quadratic_gradient_factory()
    result = GradientDescent(gradient, 0.25, maxiter=3).run(
        estimator, [_request({"x": 1.0}, ("x",))]
    )[0]
    assert result.optimal_parameters == pytest.approx((0.125,))
    assert result.optimal_value == pytest.approx(0.125**2)
    assert result.iterations == 3
    assert result.status is OptimizerStatus.MAX_ITERATIONS
    assert not result.converged
    assert gradient.batch_sizes == [1, 1, 1]
    assert estimator.batch_sizes == [1]


def test_result_reports_final_iterate_not_best_seen(
    quadratic_estimator_factory, quadratic_gradient_factory
) -> None:
    result = GradientDescent(
        quadratic_gradient_factory(), learning_rate=1.5, maxiter=1
    ).run(quadratic_estimator_factory(), [_request({"x": 1.0}, ("x",))])[0]
    assert result.optimal_parameters == (-2.0,)
    assert result.optimal_value == 4.0


def test_equal_step_tolerance_converges_after_update(
    quadratic_estimator_factory, quadratic_gradient_factory
) -> None:
    result = GradientDescent(
        quadratic_gradient_factory(),
        0.25,
        maxiter=5,
        tolerance=0.5,
    ).run(quadratic_estimator_factory(), [_request({"x": 1.0}, ("x",))])[0]
    assert result.optimal_parameters == pytest.approx((0.5,))
    assert result.iterations == 1
    assert result.status is OptimizerStatus.STEP_TOLERANCE
    assert result.converged


def test_tolerance_none_disables_zero_step_stopping(
    quadratic_estimator_factory, quadratic_gradient_factory
) -> None:
    result = GradientDescent(
        quadratic_gradient_factory(), 0.2, maxiter=2, tolerance=None
    ).run(quadratic_estimator_factory(), [_request({"x": 0.0}, ("x",))])[0]
    assert result.iterations == 2
    assert result.status is OptimizerStatus.MAX_ITERATIONS


def test_maxiter_zero_evaluates_initial_point_without_gradient(
    quadratic_estimator_factory, quadratic_gradient_factory
) -> None:
    gradient = quadratic_gradient_factory()
    estimator = quadratic_estimator_factory()
    result = GradientDescent(gradient, maxiter=0).run(
        estimator, [_request({"x": 2.0}, ("x",))]
    )[0]
    assert result.optimal_parameters == (2.0,)
    assert result.optimal_value == 4.0
    assert result.iterations == 0
    assert result.status is OptimizerStatus.MAX_ITERATIONS
    assert gradient.batch_sizes == []
    assert estimator.batch_sizes == [1]


def test_no_parameters_evaluates_once_and_converges(
    quadratic_estimator_factory, quadratic_gradient_factory
) -> None:
    gradient = quadratic_gradient_factory()
    estimator = quadratic_estimator_factory()
    result = GradientDescent(gradient).run(estimator, [_request({"fixed": 2.0}, ())])[0]
    assert result.parameter_names == ()
    assert result.optimal_parameters == ()
    assert result.optimal_value == 4.0
    assert result.iterations == 0
    assert result.status is OptimizerStatus.NO_PARAMETERS
    assert result.converged
    assert gradient.batch_sizes == []
    assert estimator.batch_sizes == [1]


def test_empty_batch_does_not_call_dependencies(
    quadratic_estimator_factory, quadratic_gradient_factory
) -> None:
    gradient = quadratic_gradient_factory()
    estimator = quadratic_estimator_factory()
    assert GradientDescent(gradient).run(estimator, []) == []
    assert gradient.batch_sizes == []
    assert estimator.batch_sizes == []


def test_requests_stop_independently_and_keep_order(
    quadratic_estimator_factory, quadratic_gradient_factory
) -> None:
    gradient = quadratic_gradient_factory()
    estimator = quadratic_estimator_factory()
    results = GradientDescent(gradient, 0.25, maxiter=3, tolerance=0.0).run(
        estimator,
        [_request({"first": 0.0}, ("first",)), _request({"second": 1.0}, ("second",))],
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
    assert gradient.batch_sizes == [2, 1, 1]
    assert estimator.batch_sizes == [2]


def test_non_trainable_bindings_remain_fixed(
    quadratic_estimator_factory, quadratic_gradient_factory
) -> None:
    estimator = quadratic_estimator_factory()
    result = GradientDescent(quadratic_gradient_factory(), 0.25, maxiter=1).run(
        estimator,
        [_request({"trainable": 1.0, "fixed": 0.7}, ("trainable",))],
    )[0]
    assert result.optimal_parameters == pytest.approx((0.5,))
    assert estimator.bindings[-1]["fixed"] == 0.7


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
    result = GradientDescent(gradient, 0.2, maxiter=3).run(matrix_estimator, [request])[
        0
    ]
    assert result.optimal_value < math.cos(0.4)
    assert result.status is OptimizerStatus.MAX_ITERATIONS
    assert matrix_estimator.batch_sizes[-1] == 1


def test_gradient_result_count_is_checked(
    quadratic_estimator_factory, quadratic_gradient_factory
) -> None:
    class BrokenGradient(Gradient):
        def run(self, estimator, requests, /):
            return []

    with pytest.raises(ValueError, match="one result per active"):
        GradientDescent(BrokenGradient(), maxiter=1).run(
            quadratic_estimator_factory(), [_request({"x": 1.0}, ("x",))]
        )


def test_nonzero_subnormal_step_does_not_satisfy_zero_tolerance(
    quadratic_estimator_factory,
) -> None:
    class SmallestGradient(Gradient):
        def run(self, estimator, requests, /):
            smallest = float.fromhex("0x0.0000000000001p-1022")
            return [
                GradientResult(request.parameter_names, (smallest,))
                for request in requests
            ]

    result = GradientDescent(
        SmallestGradient(),
        learning_rate=1.0,
        maxiter=1,
        tolerance=0.0,
    ).run(quadratic_estimator_factory(), [_request({"x": 0.0}, ("x",))])[0]
    assert result.iterations == 1
    assert result.status is OptimizerStatus.MAX_ITERATIONS


def test_final_estimator_result_count_is_checked(
    quadratic_estimator_factory, quadratic_gradient_factory
) -> None:
    class BrokenEstimator(quadratic_estimator_factory):
        def run(self, requests, /):
            return []

    with pytest.raises(ValueError, match="one result per final"):
        GradientDescent(quadratic_gradient_factory(), maxiter=0).run(
            BrokenEstimator(), [_request({"x": 1.0}, ("x",))]
        )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"learning_rate": True}, TypeError),
        ({"learning_rate": 0.0}, ValueError),
        ({"learning_rate": float("inf")}, ValueError),
        ({"maxiter": True}, TypeError),
        ({"maxiter": -1}, ValueError),
        ({"tolerance": True}, TypeError),
        ({"tolerance": -1.0}, ValueError),
        ({"tolerance": float("nan")}, ValueError),
    ],
)
def test_constructor_validation(quadratic_gradient_factory, kwargs, error) -> None:
    with pytest.raises(error):
        GradientDescent(quadratic_gradient_factory(), **kwargs)


def test_gradient_and_runtime_types_are_checked(
    quadratic_estimator_factory, quadratic_gradient_factory
) -> None:
    with pytest.raises(TypeError, match="Gradient"):
        GradientDescent(object())  # type: ignore[arg-type]
    optimizer = GradientDescent(quadratic_gradient_factory())
    with pytest.raises(TypeError, match="Estimator"):
        optimizer.run(object(), [])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="OptimizerRequest"):
        optimizer.run(quadratic_estimator_factory(), [object()])  # type: ignore[list-item]
