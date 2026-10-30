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

"""Regression tests for Estimator seed forwarding across derived requests."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest
from cqlib import Circuit
from cqlib.circuit import Parameter
from cqlib.qis import PauliString

from cqlib_algorithms.operators.gradient import (
    FiniteDifferenceGradient,
    LinCombEstimatorGradient,
    ParameterShiftGradient,
    SPSAGradient,
)
from cqlib_algorithms.operators.optimizer import (
    Adam,
    COBYLAOptimizer,
    GradientDescent,
    SPSAOptimizer,
)
from cqlib_algorithms.protocols import (
    Estimator,
    EstimatorRequest,
    EstimatorResult,
    Gradient,
    GradientRequest,
    Optimizer,
    OptimizerRequest,
)

_SEED = 91


class _RecordingEstimator(Estimator):
    """Record request seeds and return a deterministic quadratic objective."""

    def __init__(self) -> None:
        self.seeds: list[int | None] = []

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        results: list[EstimatorResult] = []
        for request in requests:
            request.validate()
            self.seeds.append(request.seed)
            value = sum(
                number * number for number in request.parameter_bindings.values()
            )
            results.append(EstimatorResult(value, 0.0))
        return results


def _estimator_request() -> EstimatorRequest:
    circuit = Circuit(1)
    circuit.ry(0, Parameter("theta"))
    return EstimatorRequest(
        circuit,
        PauliString.from_str("Z"),
        {"theta": 0.25},
        None,
        _SEED,
    )


@pytest.mark.parametrize(
    "factory",
    [
        FiniteDifferenceGradient,
        ParameterShiftGradient,
        lambda: SPSAGradient(0.01, seed=3),
        LinCombEstimatorGradient,
    ],
    ids=("finite-difference", "parameter-shift", "spsa", "lin-comb"),
)
def test_gradients_forward_parent_seed_to_derived_requests(
    factory: Callable[[], Gradient],
) -> None:
    estimator = _RecordingEstimator()
    request = GradientRequest(_estimator_request(), ("theta",))

    factory().run(estimator, [request])

    assert estimator.seeds
    assert set(estimator.seeds) == {_SEED}


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Adam(ParameterShiftGradient(), maxiter=1),
        lambda: GradientDescent(ParameterShiftGradient(), maxiter=1),
        lambda: SPSAOptimizer(0.1, 0.01, maxiter=1, seed=3),
        lambda: COBYLAOptimizer(maxiter=4),
    ],
    ids=("adam", "gradient-descent", "spsa", "cobyla"),
)
def test_optimizers_forward_parent_seed_to_derived_requests(
    factory: Callable[[], Optimizer],
) -> None:
    estimator = _RecordingEstimator()
    request = OptimizerRequest(_estimator_request(), ("theta",))

    factory().run(estimator, [request])

    assert estimator.seeds
    assert set(estimator.seeds) == {_SEED}
