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

"""Shared test fixtures for the gradient components.

The exact reference estimator evaluates expectations through dense
circuit and observable matrices with ``numpy``.  This path shares no
code with the components under test and does not touch Core's
expectation APIs, which have a known sign defect for Pauli strings
containing an odd number of Y factors.

The helpers are exposed through fixtures because the test tree uses
``--import-mode=importlib`` without package structure, so sibling
module imports are unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from cqlib_algorithms.protocols import Estimator


class MatrixEstimator(Estimator):
    """Exact estimator using ``Circuit.to_matrix`` and observable matrices."""

    __slots__ = ()

    def run(self, requests, /):
        results = []
        for request in requests:
            request.validate()
            if request.shots is not None:
                raise ValueError("the reference estimator is exact only")
            bound = request.circuit.assign_parameters(dict(request.parameter_bindings))
            qubit_order = [qubit.index for qubit in bound.qubits]
            state = bound.to_matrix(qubit_order)[:, 0]
            observable = request.observable.to_matrix()
            expectation = float(np.real(np.vdot(state, observable @ state)))
            results.append(ExactResult(expectation))
        return results


class ExactResult:
    """A minimal exact estimator result."""

    __slots__ = ("expectation", "variance")

    def __init__(self, expectation: float) -> None:
        self.expectation = expectation
        self.variance = 0.0


class RecordingEstimator(MatrixEstimator):
    """Reference estimator that records every batch it executes."""

    __slots__ = ("batch_sizes",)

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def run(self, requests, /):
        self.batch_sizes.append(len(list(requests)))
        return super().run(requests)


class ConstantEstimator(Estimator):
    """Estimator returning a fixed expectation for counting evaluations."""

    __slots__ = ("calls", "sizes", "value")

    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = 0
        self.sizes: list[int] = []

    def run(self, requests, /):
        requests = list(requests)
        self.calls += 1
        self.sizes.append(len(requests))
        return [ExactResult(self.value) for _ in requests]


@pytest.fixture
def matrix_estimator() -> MatrixEstimator:
    return MatrixEstimator()


@pytest.fixture
def matrix_estimator_factory():
    """Return the reference estimator class for subclassing in tests."""
    return MatrixEstimator


@pytest.fixture
def recording_estimator_factory():
    return RecordingEstimator


@pytest.fixture
def constant_estimator_factory():
    return ConstantEstimator


@pytest.fixture
def exact_result_factory():
    return ExactResult


@pytest.fixture
def finite_difference_step() -> float:
    return 1e-6
