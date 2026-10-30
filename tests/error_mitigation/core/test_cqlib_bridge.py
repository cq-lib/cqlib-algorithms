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

"""Tests for the cqlib mitigation bridges."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from cqlib import Circuit
from cqlib.qis import Hamiltonian, PauliString

from cqlib_algorithms.error_mitigation.core import (
    QemExecutor,
    make_cqlib_estimator,
    pair_to_result,
)
from cqlib_algorithms.protocols import (
    Estimator,
    EstimatorRequest,
    EstimatorResult,
)


class _RecordingEstimator(Estimator):
    """Record every request it receives and answer a constant estimate."""

    def __init__(self) -> None:
        self.requests: list[EstimatorRequest] = []

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        """Store the batch and return one fixed result per request."""
        batch = list(requests)
        self.requests.extend(batch)
        return [EstimatorResult(0.5, 0.0) for _ in batch]


def test_pair_to_result_maps_mean_and_variance() -> None:
    result = pair_to_result((0.7, 0.02))

    assert result.expectation == pytest.approx(0.7)
    assert result.variance == pytest.approx(0.02)


def test_pair_to_result_rejects_a_non_pair() -> None:
    with pytest.raises(TypeError):
        pair_to_result(0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        pair_to_result((0.5, 0.1, 0.2))  # type: ignore[arg-type]


def test_bridge_binds_the_observable_and_ignores_the_one_passed_in(
    z_hamiltonian: Hamiltonian,
) -> None:
    executor = QemExecutor(estimator=_RecordingEstimator())
    callback = make_cqlib_estimator(executor, z_hamiltonian)
    circuit = Circuit(1)

    mean, variance = callback(
        circuit,
        Hamiltonian.from_list([(PauliString.from_str("X"), 1.0)]),
        64,
    )

    assert mean == pytest.approx(0.5)
    assert variance == pytest.approx(0.0)


def test_bridge_forwards_shots_and_seed_to_the_request(
    z_hamiltonian: Hamiltonian,
) -> None:
    backend = _RecordingEstimator()
    executor = QemExecutor(estimator=backend)
    callback = make_cqlib_estimator(executor, z_hamiltonian, shots=32, seed=7)

    callback(Circuit(1), None, None)

    assert len(backend.requests) == 1
    assert backend.requests[0].shots == 32
    assert backend.requests[0].seed == 7
    assert backend.requests[0].observable is z_hamiltonian


def test_bridge_raises_without_an_estimator_backend(
    z_hamiltonian: Hamiltonian,
) -> None:
    callback = make_cqlib_estimator(QemExecutor(), z_hamiltonian)

    with pytest.raises(ValueError, match="no estimator backend"):
        callback(Circuit(1), None, None)


def test_bridge_forwards_the_observable_when_none_is_bound(
    z_hamiltonian: Hamiltonian,
) -> None:
    backend = _RecordingEstimator()
    executor = QemExecutor(estimator=backend)
    callback = make_cqlib_estimator(executor)

    callback(Circuit(1), z_hamiltonian, None)

    assert len(backend.requests) == 1
    assert backend.requests[0].observable is z_hamiltonian


def test_bridge_requires_an_observable_from_either_side() -> None:
    callback = make_cqlib_estimator(QemExecutor(estimator=_RecordingEstimator()))

    with pytest.raises(ValueError, match="observable"):
        callback(Circuit(1), None, None)
