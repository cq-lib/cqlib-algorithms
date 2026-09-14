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
"""Tests for the public gradient contract."""

from __future__ import annotations

import inspect
from collections.abc import Sequence

import pytest
from cqlib import Circuit, Parameter
from cqlib.qis import PauliString

from cqlib_algorithms.protocols.estimator import (
    Estimator,
    EstimatorRequest,
    EstimatorResult,
)
from cqlib_algorithms.protocols.gradient import (
    Gradient,
    GradientRequest,
    GradientResult,
)


def _estimator_request(x: float = 2.0, y: float = 3.0) -> EstimatorRequest:
    circuit = Circuit(1)
    circuit.rx(0, Parameter("x"))
    circuit.ry(0, Parameter("y"))
    return EstimatorRequest(
        circuit,
        PauliString.from_str("Z"),
        {"x": x, "y": y},
    )


class _QuadraticEstimator:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        self.batch_sizes.append(len(requests))
        results: list[EstimatorResult] = []
        for request in requests:
            request.validate()
            results.append(
                EstimatorResult(
                    sum(value * value for value in request.parameter_bindings.values()),
                    0.0,
                )
            )
        return results


class _CentralDifferenceGradient:
    def __init__(self, step: float = 1e-5) -> None:
        self.step = step

    def run(
        self,
        estimator: Estimator,
        requests: Sequence[GradientRequest],
        /,
    ) -> list[GradientResult]:
        shifted: list[EstimatorRequest] = []
        spans: list[tuple[int, int]] = []
        for request in requests:
            request.validate()
            start = len(shifted)
            base = request.estimator_request
            for name in request.parameter_names:
                plus = dict(base.parameter_bindings)
                minus = dict(base.parameter_bindings)
                plus[name] += self.step
                minus[name] -= self.step
                shifted.extend(
                    [
                        EstimatorRequest(
                            base.circuit,
                            base.observable,
                            plus,
                            base.shots,
                        ),
                        EstimatorRequest(
                            base.circuit,
                            base.observable,
                            minus,
                            base.shots,
                        ),
                    ]
                )
            spans.append((start, len(shifted)))

        estimates = estimator.run(shifted) if shifted else []
        results: list[GradientResult] = []
        for request, (start, stop) in zip(requests, spans, strict=True):
            values = tuple(
                (estimates[index].expectation - estimates[index + 1].expectation)
                / (2.0 * self.step)
                for index in range(start, stop, 2)
            )
            results.append(GradientResult(request.parameter_names, values))
        return results


def _run_gradient(
    gradient: Gradient,
    estimator: Estimator,
    requests: Sequence[GradientRequest],
) -> list[GradientResult]:
    return gradient.run(estimator, requests)


def test_gradient_request_requires_explicit_parameter_names():
    parameter = inspect.signature(GradientRequest).parameters["parameter_names"]

    assert parameter.default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="parameter_names"):
        GradientRequest(_estimator_request())  # type: ignore[call-arg]


def test_gradient_request_accepts_an_ordered_binding_subset():
    request = GradientRequest(_estimator_request(), ("y",))

    assert request.parameter_names == ("y",)
    with pytest.raises(TypeError):
        hash(request)


def test_gradient_request_allows_an_explicit_empty_parameter_subset():
    request = GradientRequest(_estimator_request(), ())

    assert request.parameter_names == ()


def test_gradient_request_requires_tuple_parameter_names():
    with pytest.raises(TypeError, match="tuple"):
        GradientRequest(_estimator_request(), ["x"])  # type: ignore[arg-type]


def test_gradient_request_rejects_duplicate_parameter_names():
    with pytest.raises(ValueError, match="unique"):
        GradientRequest(_estimator_request(), ("x", "x"))


def test_gradient_request_rejects_unknown_parameter_names():
    with pytest.raises(ValueError, match="must be bound"):
        GradientRequest(_estimator_request(), ("z",))


def test_gradient_request_rejects_non_string_parameter_names():
    with pytest.raises(TypeError, match="strings"):
        GradientRequest(_estimator_request(), (1,))  # type: ignore[arg-type]


def test_gradient_request_revalidates_nested_estimator_request():
    request = GradientRequest(_estimator_request(), ("x",))
    request.estimator_request.circuit.rz(0, Parameter("late"))

    with pytest.raises(ValueError, match="missing"):
        request.validate()
    with pytest.raises(ValueError, match="missing"):
        _CentralDifferenceGradient().run(_QuadraticEstimator(), [request])


def test_gradient_result_normalizes_finite_real_values():
    result = GradientResult(("x", "y"), (1, 2.5))

    assert result.values == (1.0, 2.5)


def test_gradient_result_requires_matching_lengths():
    with pytest.raises(ValueError, match="equal length"):
        GradientResult(("x",), ())


def test_gradient_result_rejects_duplicate_names():
    with pytest.raises(ValueError, match="unique"):
        GradientResult(("x", "x"), (1.0, 2.0))


@pytest.mark.parametrize("value", [True, float("nan"), float("inf")])
def test_gradient_result_rejects_invalid_values(value):
    expected = TypeError if value is True else ValueError
    with pytest.raises(expected):
        GradientResult(("x",), (value,))


def test_structural_gradient_batches_all_perturbed_estimator_requests():
    estimator = _QuadraticEstimator()
    gradient = _CentralDifferenceGradient()
    requests = [
        GradientRequest(_estimator_request(2.0, 3.0), ("y", "x")),
        GradientRequest(_estimator_request(4.0, 5.0), ("x",)),
    ]

    results = _run_gradient(gradient, estimator, requests)

    assert estimator.batch_sizes == [6]
    assert results[0].parameter_names == ("y", "x")
    assert results[0].values == pytest.approx((6.0, 4.0))
    assert results[1].parameter_names == ("x",)
    assert results[1].values == pytest.approx((8.0,))


def test_structural_gradient_supports_empty_batches_without_evaluation():
    estimator = _QuadraticEstimator()
    gradient = _CentralDifferenceGradient()

    assert _run_gradient(gradient, estimator, []) == []
    assert estimator.batch_sizes == []


def test_gradient_protocol_is_not_runtime_checkable():
    with pytest.raises(TypeError, match="runtime_checkable"):
        isinstance(_CentralDifferenceGradient(), Gradient)  # type: ignore[misc]

