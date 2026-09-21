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

"""Independent estimator and gradient fixtures for optimizer tests."""

import numpy as np
import pytest

from cqlib_algorithms.protocols import (
    Estimator,
    EstimatorResult,
    Gradient,
    GradientResult,
)


class MatrixEstimator(Estimator):
    """Exact dense-matrix estimator independent of optimizer code."""

    def __init__(self) -> None:
        self.batch_sizes = []

    def run(self, requests, /):
        requests = list(requests)
        self.batch_sizes.append(len(requests))
        results = []
        for request in requests:
            request.validate()
            bound = request.circuit.assign_parameters(dict(request.parameter_bindings))
            order = [qubit.index for qubit in bound.qubits]
            state = bound.to_matrix(order)[:, 0]
            matrix = request.observable.to_matrix()
            value = float(np.real(np.vdot(state, matrix @ state)))
            results.append(EstimatorResult(value, 0.0))
        return results


class QuadraticEstimator(Estimator):
    """Classical separable quadratic objective over request bindings."""

    def __init__(self, targets=None) -> None:
        self.targets = dict(targets or {})
        self.batch_sizes = []
        self.bindings = []

    def run(self, requests, /):
        requests = list(requests)
        self.batch_sizes.append(len(requests))
        results = []
        for request in requests:
            request.validate()
            bindings = dict(request.parameter_bindings)
            self.bindings.append(bindings)
            value = sum(
                (number - self.targets.get(name, 0.0)) ** 2
                for name, number in bindings.items()
            )
            results.append(EstimatorResult(value, 0.0))
        return results


class QuadraticGradient(Gradient):
    """Analytic gradient matching :class:`QuadraticEstimator`."""

    def __init__(self, targets=None) -> None:
        self.targets = dict(targets or {})
        self.batch_sizes = []

    def run(self, estimator, requests, /):
        requests = list(requests)
        self.batch_sizes.append(len(requests))
        return [
            GradientResult(
                request.parameter_names,
                tuple(
                    2.0
                    * (
                        request.estimator_request.parameter_bindings[name]
                        - self.targets.get(name, 0.0)
                    )
                    for name in request.parameter_names
                ),
            )
            for request in requests
        ]


@pytest.fixture
def matrix_estimator():
    return MatrixEstimator()


@pytest.fixture
def quadratic_estimator_factory():
    return QuadraticEstimator


@pytest.fixture
def quadratic_gradient_factory():
    return QuadraticGradient
