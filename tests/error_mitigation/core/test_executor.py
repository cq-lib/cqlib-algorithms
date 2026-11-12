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

"""Tests for the unified execution backend and its request cache."""

from __future__ import annotations

from collections.abc import Sequence
from math import cos

import pytest
from cqlib import Circuit, Parameter
from cqlib.device import ExecutionResult
from cqlib.qis import PauliString

from cqlib_algorithms.error_mitigation.core import QemExecutor
from cqlib_algorithms.operators.estimator import StatevectorEstimator
from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.protocols import (
    Estimator,
    EstimatorRequest,
    EstimatorResult,
    Sampler,
    SamplerRequest,
)

_ATOL = 1e-12


class _CountingEstimator(Estimator):
    """Record every request it processes and answer a distinct estimate.

    Counting requests rather than calls keeps the observable independent of
    whether the executor hands the backend an empty batch: a fully cached
    batch delivers no request, so the recorded count stays flat.
    """

    def __init__(self) -> None:
        self.requests: list[EstimatorRequest] = []

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        """Store each request and return one estimate per request."""
        results: list[EstimatorResult] = []
        for request in requests:
            self.requests.append(request)
            results.append(EstimatorResult(float(len(self.requests)), 0.0))
        return results

    @property
    def processed(self) -> int:
        """Return how many requests have reached the backend."""
        return len(self.requests)


class _CountingSampler(Sampler):
    """Record every request it processes and answer an all-zero draw."""

    def __init__(self) -> None:
        self.requests: list[SamplerRequest] = []

    def run(
        self,
        requests: Sequence[SamplerRequest],
        /,
    ) -> list[ExecutionResult]:
        """Store each request and return one terminal result per request."""
        results: list[ExecutionResult] = []
        for request in requests:
            self.requests.append(request)
            width = len(request.qubits)
            results.append(
                ExecutionResult.from_counts(
                    "counting",
                    list(request.qubits),
                    request.shots,
                    width,
                    {"0" * width: request.shots},
                    "counting",
                )
            )
        return results

    @property
    def processed(self) -> int:
        """Return how many requests have reached the backend."""
        return len(self.requests)


def _angled_circuit(theta: float) -> Circuit:
    """Return a one-qubit RY body, distinguishable by its angle."""
    circuit = Circuit(1)
    circuit.ry(0, theta)
    return circuit


def _estimator_request(
    circuit: Circuit,
    *,
    shots: int | None = None,
    seed: int | None = None,
    bindings: dict[str, float] | None = None,
) -> EstimatorRequest:
    """Return a Pauli-Z request over ``circuit`` with optional overrides."""
    return EstimatorRequest(
        circuit=circuit,
        observable=PauliString.from_str("Z"),
        parameter_bindings=bindings or {},
        shots=shots,
        seed=seed,
    )


def _sampler_request(circuit: Circuit, *, shots: int = 8) -> SamplerRequest:
    """Return a full-width sampling request over ``circuit``."""
    return SamplerRequest(circuit, tuple(circuit.qubits), shots)


def test_estimate_returns_results_in_request_order() -> None:
    executor = QemExecutor(estimator=StatevectorEstimator())
    requests = [_estimator_request(_angled_circuit(theta)) for theta in (0.0, 1.3, 2.7)]

    results = executor.estimate(requests)

    expected = [1.0, cos(1.3), cos(2.7)]
    assert [result.expectation for result in results] == pytest.approx(expected)
    assert results[0].expectation == pytest.approx(1.0, abs=_ATOL)


def test_estimate_without_a_backend_raises_value_error() -> None:
    executor = QemExecutor(sampler=StatevectorSampler())

    with pytest.raises(ValueError, match="no estimator backend"):
        executor.estimate([_estimator_request(_angled_circuit(0.0))])


def test_sample_without_a_backend_raises_value_error() -> None:
    executor = QemExecutor(estimator=StatevectorEstimator())

    with pytest.raises(ValueError, match="no sampler backend"):
        executor.sample([_sampler_request(_angled_circuit(0.0))])


def test_use_cache_defaults_to_false() -> None:
    executor = QemExecutor(estimator=StatevectorEstimator())

    assert executor.use_cache is False


def test_cache_skips_a_repeated_request() -> None:
    backend = _CountingEstimator()
    executor = QemExecutor(estimator=backend, use_cache=True)
    request = _estimator_request(_angled_circuit(0.4))

    first = executor.estimate([request])
    second = executor.estimate([request])

    assert backend.processed == 1
    assert first == second
    assert first[0].expectation == pytest.approx(1.0)


def test_cache_key_includes_the_seed() -> None:
    backend = _CountingEstimator()
    executor = QemExecutor(estimator=backend, use_cache=True)
    circuit = _angled_circuit(0.4)

    executor.estimate([_estimator_request(circuit, seed=1)])
    executor.estimate([_estimator_request(circuit, seed=2)])

    assert backend.processed == 2


def test_cache_key_includes_the_shots() -> None:
    backend = _CountingEstimator()
    executor = QemExecutor(estimator=backend, use_cache=True)
    circuit = _angled_circuit(0.4)

    executor.estimate([_estimator_request(circuit, shots=10)])
    executor.estimate([_estimator_request(circuit, shots=20)])

    assert backend.processed == 2


def test_cache_key_includes_parameter_bindings() -> None:
    backend = _CountingEstimator()
    executor = QemExecutor(estimator=backend, use_cache=True)
    circuit = Circuit(1)
    circuit.ry(0, Parameter("theta"))

    executor.estimate([_estimator_request(circuit, bindings={"theta": 0.1})])
    executor.estimate([_estimator_request(circuit, bindings={"theta": 0.2})])

    assert backend.processed == 2


def test_duplicate_requests_in_one_batch_share_one_backend_call() -> None:
    backend = _CountingEstimator()
    executor = QemExecutor(estimator=backend, use_cache=True)
    request = _estimator_request(_angled_circuit(0.4))

    results = executor.estimate([request, request])

    assert backend.processed == 1
    assert len(results) == 2
    assert results[0] == results[1]


def test_sample_returns_results_in_request_order() -> None:
    executor = QemExecutor(sampler=StatevectorSampler())
    requests = [
        _sampler_request(_angled_circuit(theta), shots=10) for theta in (0.1, 1.1)
    ]

    results = executor.sample(requests)

    assert [result.shots for result in results] == [10, 10]
    assert all(sum(result.counts.values()) == 10 for result in results)


def test_sampler_cache_key_includes_the_seed() -> None:
    backend = _CountingSampler()
    executor = QemExecutor(sampler=backend, use_cache=True)
    circuit = _angled_circuit(0.4)

    executor.sample([SamplerRequest(circuit, tuple(circuit.qubits), 8, 1)])
    executor.sample([SamplerRequest(circuit, tuple(circuit.qubits), 8, 2)])

    assert backend.processed == 2
