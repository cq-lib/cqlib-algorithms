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

"""Tests for SciPy-backed L-BFGS-B optimization."""

import math

import pytest
from cqlib import Circuit, Parameter
from cqlib.qis import PauliString

from cqlib_algorithms.operators.gradient import ParameterShiftGradient
from cqlib_algorithms.operators.optimizer import LBFGSBOptimizer
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


@pytest.mark.reference
def test_analytic_gradient_minimizes_quadratic_and_preserves_order(
    quadratic_estimator_factory, quadratic_gradient_factory
) -> None:
    targets = {"x": 1.25, "y": -0.75}
    estimator = quadratic_estimator_factory(targets)
    gradient = quadratic_gradient_factory(targets)
    result = LBFGSBOptimizer(gradient, maxiter=100, gtol=1e-10).run(
        estimator,
        [_request({"x": -2.0, "y": 2.0}, ("y", "x"))],
    )[0]

    assert result.parameter_names == ("y", "x")
    assert result.optimal_parameters == pytest.approx((-0.75, 1.25), abs=1e-8)
    assert result.optimal_value == pytest.approx(0.0, abs=1e-14)
    assert result.iterations > 0
    assert result.status is OptimizerStatus.CONVERGED
    assert gradient.batch_sizes


@pytest.mark.reference
def test_numerical_gradient_minimizes_quadratic(
    quadratic_estimator_factory,
) -> None:
    estimator = quadratic_estimator_factory({"x": -0.4})
    result = LBFGSBOptimizer(maxiter=100, gtol=1e-9).run(
        estimator,
        [_request({"x": 2.0}, ("x",))],
    )[0]

    assert result.optimal_parameters == pytest.approx((-0.4,), abs=2e-6)
    assert result.optimal_value == pytest.approx(0.0, abs=1e-11)
    assert result.status is OptimizerStatus.CONVERGED


def test_box_bound_constrains_solution_and_is_copied(
    quadratic_estimator_factory, quadratic_gradient_factory
) -> None:
    mutable_bounds = {"x": (-1.0, 1.0)}
    optimizer = LBFGSBOptimizer(
        quadratic_gradient_factory({"x": 3.0}), bounds=mutable_bounds
    )
    mutable_bounds["x"] = (-10.0, 10.0)
    result = optimizer.run(
        quadratic_estimator_factory({"x": 3.0}),
        [_request({"x": 0.0}, ("x",))],
    )[0]

    assert optimizer.bounds["x"] == (-1.0, 1.0)
    assert result.optimal_parameters == pytest.approx((1.0,))
    assert result.status is OptimizerStatus.CONVERGED
    with pytest.raises(TypeError):
        optimizer.bounds["x"] = (0.0, 2.0)  # type: ignore[index]


@pytest.mark.integration
def test_integrates_with_parameter_shift_gradient(matrix_estimator) -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    request = OptimizerRequest(
        EstimatorRequest(circuit, PauliString.from_str("Z"), {"theta": 0.4}),
        ("theta",),
    )
    result = LBFGSBOptimizer(
        ParameterShiftGradient(),
        bounds={"theta": (0.0, 2.0 * math.pi)},
        maxiter=100,
        gtol=1e-9,
    ).run(matrix_estimator, [request])[0]

    assert result.optimal_parameters == pytest.approx((math.pi,), abs=2e-6)
    assert result.optimal_value == pytest.approx(-1.0, abs=1e-12)
    assert result.status is OptimizerStatus.CONVERGED


def test_no_parameters_and_zero_budget_avoid_gradient(
    quadratic_estimator_factory, quadratic_gradient_factory
) -> None:
    gradient = quadratic_gradient_factory()
    estimator = quadratic_estimator_factory()
    results = LBFGSBOptimizer(gradient, maxiter=0).run(
        estimator,
        [_request({"fixed": 0.25}, ()), _request({"x": 1.0}, ("x",))],
    )

    assert [result.status for result in results] == [
        OptimizerStatus.NO_PARAMETERS,
        OptimizerStatus.MAX_ITERATIONS,
    ]
    assert gradient.batch_sizes == []
    assert estimator.batch_sizes == [2]


def test_fixed_binding_is_preserved(
    quadratic_estimator_factory, quadratic_gradient_factory
) -> None:
    targets = {"x": 0.5, "fixed": 0.25}
    estimator = quadratic_estimator_factory(targets)
    LBFGSBOptimizer(quadratic_gradient_factory(targets)).run(
        estimator,
        [_request({"x": -1.0, "fixed": 0.25}, ("x",))],
    )
    assert estimator.bindings[-1]["fixed"] == 0.25


def test_gradient_result_count_and_names_are_checked(
    quadratic_estimator_factory,
) -> None:
    class BrokenGradient(Gradient):
        def __init__(self, mode):
            self.mode = mode

        def run(self, estimator, requests, /):
            if self.mode == "count":
                return []
            return [GradientResult(("wrong",), (0.0,))]

    request = _request({"x": 1.0}, ("x",))
    with pytest.raises(ValueError, match="derivative evaluation"):
        LBFGSBOptimizer(BrokenGradient("count")).run(
            quadratic_estimator_factory(), [request]
        )
    with pytest.raises(ValueError, match="names must match"):
        LBFGSBOptimizer(BrokenGradient("names")).run(
            quadratic_estimator_factory(), [request]
        )


@pytest.mark.parametrize(
    ("bounds", "error"),
    [
        ([], TypeError),
        ({1: (0.0, 1.0)}, TypeError),
        ({"x": [0.0, 1.0]}, TypeError),
        ({"x": (0.0,)}, TypeError),
        ({"x": (True, 1.0)}, TypeError),
        ({"x": (float("-inf"), 1.0)}, ValueError),
        ({"x": (2.0, 1.0)}, ValueError),
    ],
)
def test_bound_validation(bounds, error) -> None:
    with pytest.raises(error):
        LBFGSBOptimizer(bounds=bounds)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"maxiter": True}, TypeError),
        ({"maxiter": -1}, ValueError),
        ({"maxfun": True}, TypeError),
        ({"maxfun": 0}, ValueError),
        ({"maxcor": 0}, ValueError),
        ({"ftol": True}, TypeError),
        ({"ftol": -1.0}, ValueError),
        ({"gtol": float("nan")}, ValueError),
        ({"finite_difference_step": 0.0}, ValueError),
        ({"maxls": 0}, ValueError),
    ],
)
def test_constructor_validation(kwargs, error) -> None:
    with pytest.raises(error):
        LBFGSBOptimizer(**kwargs)


def test_gradient_type_and_empty_batch_are_checked(
    quadratic_estimator_factory,
) -> None:
    with pytest.raises(TypeError, match="Gradient"):
        LBFGSBOptimizer(object())  # type: ignore[arg-type]
    estimator = quadratic_estimator_factory()
    assert LBFGSBOptimizer().run(estimator, []) == []
    assert estimator.batch_sizes == []
