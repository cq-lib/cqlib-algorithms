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

"""Natural-gradient tests against analytic and algebraic references."""

from __future__ import annotations

import math

import numpy as np
import pytest
from cqlib import Circuit, Parameter
from cqlib.qis import PauliString

from cqlib_algorithms.operators.gradient import NaturalGradient
from cqlib_algorithms.protocols import EstimatorRequest, GradientRequest


@pytest.fixture
def natural() -> NaturalGradient:
    return NaturalGradient()


def _request(circuit, observable, bindings, names):
    return GradientRequest(EstimatorRequest(circuit, observable, bindings), names)


def test_single_rotation_direction_is_minus_four_sine(
    natural, matrix_estimator
) -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    request = _request(circuit, PauliString.from_str("Z"), {"theta": 0.4}, ("theta",))
    result = natural.run(matrix_estimator, [request])[0]
    assert result.parameter_names == ("theta",)
    assert result.values == pytest.approx((-4.0 * math.sin(0.4),), abs=1e-12)


def test_ridge_regularization_scales_the_direction(natural, matrix_estimator) -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    request = _request(circuit, PauliString.from_str("Z"), {"theta": 0.4}, ("theta",))
    ridge = NaturalGradient(regularization=1.0)
    result = ridge.run(matrix_estimator, [request])[0]
    expected = -math.sin(0.4) / (0.25 + 1.0)
    assert result.values == pytest.approx((expected,), abs=1e-12)


def test_singular_same_axis_metric_returns_minimum_norm_direction(
    matrix_estimator,
) -> None:
    first = Parameter("first")
    second = Parameter("second")
    circuit = Circuit(1)
    circuit.ry(0, first)
    circuit.ry(0, second)
    bindings = {"first": 0.2, "second": 0.35}
    request = _request(
        circuit, PauliString.from_str("Z"), bindings, ("first", "second")
    )
    result = NaturalGradient().run(matrix_estimator, [request])[0]
    total = bindings["first"] + bindings["second"]
    # g = 0.25 * ones(2); min-norm solution of g @ eta = grad.
    expected = -2.0 * math.sin(total)
    assert result.values == pytest.approx((expected, expected), abs=1e-9)


def test_direction_solves_the_metric_system(natural, matrix_estimator) -> None:
    first = Parameter("first")
    second = Parameter("second")
    circuit = Circuit(2)
    circuit.ry(0, first)
    circuit.cx(0, 1)
    circuit.ry(1, second)
    bindings = {"first": 0.3, "second": -0.7}
    names = ("first", "second")
    request = _request(circuit, PauliString.from_str("ZI"), bindings, names)

    from cqlib_algorithms.operators.gradient import (
        ParameterShiftGradient,
        StatevectorQGT,
    )

    gradient = ParameterShiftGradient().run(matrix_estimator, [request])[0]
    metric = StatevectorQGT().compute(request).metric
    direction = natural.run(matrix_estimator, [request])[0]
    residual = metric @ np.asarray(direction.values) - np.asarray(gradient.values)
    assert np.max(np.abs(residual)) < 1e-9


def test_multi_parameter_direction_matches_independent_finite_differences(
    matrix_estimator,
) -> None:
    first = Parameter("first")
    second = Parameter("second")
    circuit = Circuit(2)
    circuit.h(0)
    circuit.ry(0, first)
    circuit.rx(1, second)
    circuit.cx(0, 1)
    circuit.rz(1, 0.4 * first - 0.2)
    observable = PauliString.from_str("YY")
    bindings = {"first": 0.37, "second": -0.41}
    names = ("first", "second")
    request = _request(circuit, observable, bindings, names)
    step = 1e-6

    def state(values: dict[str, float]) -> np.ndarray:
        bound = circuit.assign_parameters(values)
        qubit_order = [qubit.index for qubit in bound.qubits]
        return bound.to_matrix(qubit_order)[:, 0]

    psi = state(bindings)
    derivative_states: list[np.ndarray] = []
    objective_gradient: list[float] = []
    observable_matrix = observable.to_matrix()
    for name in names:
        plus_bindings = dict(bindings)
        minus_bindings = dict(bindings)
        plus_bindings[name] += step
        minus_bindings[name] -= step
        plus = state(plus_bindings)
        minus = state(minus_bindings)
        derivative_states.append((plus - minus) / (2.0 * step))
        plus_value = float(np.real(np.vdot(plus, observable_matrix @ plus)))
        minus_value = float(np.real(np.vdot(minus, observable_matrix @ minus)))
        objective_gradient.append((plus_value - minus_value) / (2.0 * step))

    metric = np.empty((len(names), len(names)), dtype=np.float64)
    for row, d_first in enumerate(derivative_states):
        for column, d_second in enumerate(derivative_states):
            projected = np.vdot(d_first, d_second) - np.vdot(d_first, psi) * np.vdot(
                psi, d_second
            )
            metric[row, column] = float(np.real(projected))

    regularization = 0.17
    assert all(abs(value) > 1e-5 for value in objective_gradient)
    expected = np.linalg.solve(
        metric + regularization * np.eye(len(names)),
        np.asarray(objective_gradient),
    )
    result = NaturalGradient(regularization=regularization).run(
        matrix_estimator, [request]
    )[0]

    assert np.asarray(result.values) == pytest.approx(expected, abs=2e-6)


def test_multiple_requests_preserve_order(natural, matrix_estimator) -> None:
    theta = Parameter("theta")
    first = Circuit(1)
    first.ry(0, theta)
    second = Circuit(1)
    second.h(0)
    second.rz(0, theta)
    requests = [
        _request(first, PauliString.from_str("Z"), {"theta": 0.2}, ("theta",)),
        _request(second, PauliString.from_str("X"), {"theta": 0.5}, ("theta",)),
    ]
    results = natural.run(matrix_estimator, requests)
    assert len(results) == 2
    assert results[0].values == pytest.approx((-4.0 * math.sin(0.2),), abs=1e-12)
    assert results[1].values == pytest.approx((-4.0 * math.sin(0.5),), abs=1e-12)


def test_empty_names_return_empty_direction(natural, matrix_estimator) -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    request = _request(circuit, PauliString.from_str("Z"), {"theta": 0.3}, ())
    result = natural.run(matrix_estimator, [request])[0]
    assert result.parameter_names == ()
    assert result.values == ()


def test_constructor_validates_regularization() -> None:
    with pytest.raises(TypeError):
        NaturalGradient(regularization=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        NaturalGradient(regularization=-1.0)
    with pytest.raises(ValueError):
        NaturalGradient(regularization=float("inf"))
    assert NaturalGradient().regularization == 0.0
    assert NaturalGradient(regularization=0.5).regularization == 0.5


def test_rejects_non_estimator_and_non_gradient_requests(
    natural, matrix_estimator
) -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    request = _request(circuit, PauliString.from_str("Z"), {"theta": 0.2}, ("theta",))
    with pytest.raises(TypeError, match="estimator"):
        natural.run(object(), [request])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="GradientRequest"):
        natural.run(matrix_estimator, [object()])  # type: ignore[list-item]
