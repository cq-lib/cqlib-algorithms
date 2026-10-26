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

"""Tests for virtual distillation."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from cqlib import Circuit
from cqlib.qis import PauliString

from cqlib_algorithms.error_mitigation.core import QemExecutor
from cqlib_algorithms.error_mitigation.techniques.vd import (
    combine_results,
    construct_circuits,
    vd_executor,
)
from cqlib_algorithms.protocols import (
    Estimator,
    EstimatorRequest,
    EstimatorResult,
    spawn_seeds,
)


class _RecordingEstimator(Estimator):
    """Record the seed of every request and return a fixed expectation."""

    __slots__ = ("seeds",)

    def __init__(self) -> None:
        self.seeds: list[int | None] = []

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        """Return one fixed result per request, recording its seed."""
        results: list[EstimatorResult] = []
        for request in requests:
            self.seeds.append(request.seed)
            results.append(EstimatorResult(expectation=0.5, variance=0.0))
        return results


def test_combine_results_is_the_ratio_of_the_moments() -> None:
    # The moments are plain real numbers, so the ratio is decided by
    # arithmetic alone and needs no copy-swap state.
    distilled = combine_results([EstimatorResult(0.75), EstimatorResult(0.5)])

    assert distilled == pytest.approx(1.5)

    with pytest.raises(ValueError, match="exactly two"):
        combine_results([EstimatorResult(1.0)])
    with pytest.raises(ValueError, match="undefined"):
        combine_results([EstimatorResult(1.0), EstimatorResult(0.0)])


def test_the_two_copies_use_independent_draws() -> None:
    recorder = _RecordingEstimator()
    executor = QemExecutor(estimator=recorder)
    circuit = Circuit(1)
    circuit.h(0)
    observable = PauliString.from_str("Z")
    request = EstimatorRequest(circuit=circuit, observable=observable)
    settings = {"copies": 2, "shots_numerator": 4096, "shots_denominator": 4097}

    def run_seeds(seed: int) -> list[int | None]:
        recorder.seeds.clear()
        vd_executor(executor, seed=seed, **settings).estimate([request])
        return list(recorder.seeds)

    first = run_seeds(11)
    repeat = run_seeds(11)
    other = run_seeds(12)

    # The numerator and denominator each take their own child seed, and one run
    # seed reproduces that pair while a different seed resamples it. Asserting
    # on the seed set keeps the check independent of how many times cqlib
    # replays each callback.
    expected = set(spawn_seeds(11, 2))
    assert set(first) == expected
    assert len(expected) == 2
    assert set(repeat) == expected
    assert set(other) == set(spawn_seeds(12, 2))


def test_copy_swap_circuit_is_built_by_cqlib(rotation_circuit) -> None:
    circuit = rotation_circuit(0.9)

    circuits = construct_circuits(circuit, copies=2)

    assert len(circuits) == 1
    assert circuits[0].width == 2 * circuit.num_qubits + 1
