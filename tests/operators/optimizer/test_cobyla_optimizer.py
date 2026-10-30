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

"""Tests for SciPy-backed COBYLA optimization."""

from types import SimpleNamespace

import numpy as np
import pytest
from cqlib import Circuit, Parameter
from cqlib.qis import PauliString

from cqlib_algorithms.operators.optimizer import COBYLAOptimizer
from cqlib_algorithms.protocols import (
    Estimator,
    EstimatorRequest,
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
def test_minimizes_multivariable_quadratic_against_analytic_solution(
    quadratic_estimator_factory,
) -> None:
    estimator = quadratic_estimator_factory({"x": 1.25, "y": -0.75})
    result = COBYLAOptimizer(maxiter=500, tolerance=1e-9).run(
        estimator,
        [_request({"x": -2.0, "y": 2.0}, ("x", "y"))],
    )[0]

    assert result.optimal_parameters == pytest.approx((1.25, -0.75), abs=2e-6)
    assert result.optimal_value == pytest.approx(0.0, abs=1e-11)
    assert result.iterations > 0
    assert result.status is OptimizerStatus.CONVERGED
    assert result.converged
    assert estimator.batch_sizes[-1] == 1


def test_preserves_fixed_bindings(quadratic_estimator_factory) -> None:
    estimator = quadratic_estimator_factory({"trainable": 0.5, "fixed": 0.25})
    result = COBYLAOptimizer(maxiter=300, tolerance=1e-8).run(
        estimator,
        [_request({"trainable": -1.0, "fixed": 0.25}, ("trainable",))],
    )[0]

    assert result.optimal_parameters == pytest.approx((0.5,), abs=2e-5)
    assert estimator.bindings[-1]["fixed"] == 0.25


def test_multiple_requests_keep_order_and_share_final_batch(
    quadratic_estimator_factory,
) -> None:
    targets = {"first": 0.25, "second": -0.5}
    estimator = quadratic_estimator_factory(targets)
    results = COBYLAOptimizer(maxiter=300, tolerance=1e-8).run(
        estimator,
        [
            _request({"first": 1.5}, ("first",)),
            _request({"second": 1.5}, ("second",)),
        ],
    )

    assert [result.parameter_names for result in results] == [
        ("first",),
        ("second",),
    ]
    assert results[0].optimal_parameters == pytest.approx((0.25,), abs=2e-5)
    assert results[1].optimal_parameters == pytest.approx((-0.5,), abs=2e-5)
    assert estimator.batch_sizes[-1] == 2


def test_batch_order_no_parameters_and_zero_budget(
    quadratic_estimator_factory,
) -> None:
    estimator = quadratic_estimator_factory({"x": 1.0})
    requests = [
        _request({"fixed": 0.25}, ()),
        _request({"x": -1.0}, ("x",)),
    ]
    results = COBYLAOptimizer(maxiter=0).run(estimator, requests)

    assert [result.parameter_names for result in results] == [(), ("x",)]
    assert [result.status for result in results] == [
        OptimizerStatus.NO_PARAMETERS,
        OptimizerStatus.MAX_EVALUATIONS,
    ]
    assert [result.iterations for result in results] == [0, 0]
    assert results[1].optimal_parameters == (-1.0,)
    assert estimator.batch_sizes == [2]


def test_empty_batch_does_not_call_estimator(quadratic_estimator_factory) -> None:
    estimator = quadratic_estimator_factory()
    assert COBYLAOptimizer().run(estimator, []) == []
    assert estimator.batch_sizes == []


def test_evaluation_budget_exhaustion_has_distinct_status(
    quadratic_estimator_factory,
) -> None:
    result = COBYLAOptimizer(maxiter=3).run(
        quadratic_estimator_factory({"x": 10.0}),
        [_request({"x": -10.0}, ("x",))],
    )[0]
    assert result.status is OptimizerStatus.MAX_EVALUATIONS
    assert not result.converged


def test_positive_budget_must_meet_solver_dimension_minimum(
    quadratic_estimator_factory,
) -> None:
    with pytest.raises(ValueError, match="parameters plus two"):
        COBYLAOptimizer(maxiter=2).run(
            quadratic_estimator_factory(),
            [_request({"x": 1.0}, ("x",))],
        )


def test_objective_result_count_is_checked(quadratic_estimator_factory) -> None:
    class BrokenEstimator(quadratic_estimator_factory):
        def run(self, requests, /):
            return []

    with pytest.raises(ValueError, match="objective evaluation"):
        COBYLAOptimizer(maxiter=10).run(
            BrokenEstimator(), [_request({"x": 1.0}, ("x",))]
        )


def test_unexpected_solver_failure_is_not_reported_as_convergence(
    monkeypatch, quadratic_estimator_factory
) -> None:
    import cqlib_algorithms.operators.optimizer.cobyla_optimizer as module

    monkeypatch.setattr(
        module,
        "minimize",
        lambda *args, **kwargs: SimpleNamespace(
            success=False,
            x=np.array([1.0]),
            nfev=1,
            message="invalid trust-region state",
        ),
    )
    with pytest.raises(RuntimeError, match="invalid trust-region state"):
        COBYLAOptimizer(maxiter=10).run(
            quadratic_estimator_factory(), [_request({"x": 1.0}, ("x",))]
        )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"maxiter": True}, TypeError),
        ({"maxiter": -1}, ValueError),
        ({"rhobeg": True}, TypeError),
        ({"rhobeg": 0.0}, ValueError),
        ({"rhobeg": float("inf")}, ValueError),
        ({"tolerance": True}, TypeError),
        ({"tolerance": -1.0}, ValueError),
        ({"tolerance": float("nan")}, ValueError),
    ],
)
def test_constructor_validation(kwargs, error) -> None:
    with pytest.raises(error):
        COBYLAOptimizer(**kwargs)


def test_runtime_types_are_checked(quadratic_estimator_factory) -> None:
    optimizer = COBYLAOptimizer()
    with pytest.raises(TypeError, match="Estimator"):
        optimizer.run(object(), [])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="OptimizerRequest"):
        optimizer.run(quadratic_estimator_factory(), [object()])  # type: ignore[list-item]


def test_estimator_result_type_is_checked() -> None:
    class BrokenEstimator(Estimator):
        def run(self, requests, /):
            return [object() for _ in requests]

    with pytest.raises(TypeError, match="EstimatorResult"):
        COBYLAOptimizer(maxiter=0).run(
            BrokenEstimator(), [_request({"x": 1.0}, ("x",))]
        )
