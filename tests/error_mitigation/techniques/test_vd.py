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

import math
from collections.abc import Sequence

import numpy as np
import pytest
from cqlib import Circuit
from cqlib.qis import Hamiltonian, PauliString

from cqlib_algorithms.error_mitigation.core import QemExecutor
from cqlib_algorithms.error_mitigation.techniques.vd import (
    combine_results,
    construct_circuits,
    vd_executor,
)
from cqlib_algorithms.operators.estimator import StatevectorEstimator
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


class _ExactRecordingEstimator(Estimator):
    """Evaluate callback observables exactly while retaining each request."""

    __slots__ = ("requests",)

    def __init__(self) -> None:
        self.requests: list[EstimatorRequest] = []

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        """Drop finite-shot metadata and delegate the same observables exactly."""
        self.requests.extend(requests)
        exact = [
            EstimatorRequest(
                circuit=request.circuit,
                observable=request.observable,
                parameter_bindings=request.parameter_bindings,
            )
            for request in requests
        ]
        return StatevectorEstimator().run(exact)


class _BitFlipDensityEstimator(Estimator):
    """Evaluate two-copy callback observables on a known mixed state."""

    __slots__ = ("_density",)

    def __init__(self, marginal: float) -> None:
        single = np.diag([1.0 - marginal, marginal]).astype(complex)
        ancilla_zero = np.diag([1.0, 0.0]).astype(complex)
        # Core qubit zero is least significant, so the ancilla is the leading
        # Kronecker factor for the three-qubit copy-swap circuit.
        self._density = np.kron(ancilla_zero, np.kron(single, single))

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        """Return exact dense-matrix expectations for callback observables."""
        results: list[EstimatorResult] = []
        for request in requests:
            qubits = list(range(request.circuit.num_qubits))
            unitary = request.circuit.to_matrix(qubits)
            final_density = unitary @ self._density @ unitary.conj().T
            value = np.trace(request.observable.to_matrix() @ final_density)
            assert abs(value.imag) < 1e-12
            results.append(EstimatorResult(float(value.real)))
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


def test_virtual_distillation_preserves_exact_pure_state_expectation() -> None:
    circuit = Circuit(1)
    circuit.ry(0, 0.7)
    observable = PauliString.from_str("Z")
    expected = math.cos(0.7)

    for copies in (2, 3):
        estimator = _ExactRecordingEstimator()
        executor = vd_executor(
            QemExecutor(estimator=estimator),
            copies=copies,
            shots_numerator=101,
            shots_denominator=103,
            seed=17,
        )
        result = executor.estimate([EstimatorRequest(circuit, observable)])[0]

        assert result.expectation == pytest.approx(expected, abs=1e-12)


def test_virtual_distillation_matches_analytic_mixed_bit_flip_state() -> None:
    marginal = 0.1
    circuit = Circuit(1)
    raw_expectation = 1.0 - 2.0 * marginal
    expected = raw_expectation / ((1.0 - marginal) ** 2 + marginal**2)
    executor = vd_executor(
        QemExecutor(estimator=_BitFlipDensityEstimator(marginal)),
        copies=2,
        shots_numerator=101,
        shots_denominator=103,
        seed=19,
    )

    result = executor.estimate([EstimatorRequest(circuit, PauliString.from_str("Z"))])[
        0
    ]

    assert result.expectation == pytest.approx(expected, abs=1e-12)
    assert abs(result.expectation - 1.0) < abs(raw_expectation - 1.0)


def test_virtual_distillation_forwards_cqlib_moment_observables() -> None:
    circuit = Circuit(1)
    circuit.x(0)
    estimator = _ExactRecordingEstimator()
    executor = vd_executor(
        QemExecutor(estimator=estimator),
        copies=2,
        shots_numerator=101,
        shots_denominator=103,
        seed=23,
    )

    result = executor.estimate([EstimatorRequest(circuit, PauliString.from_str("Z"))])[
        0
    ]

    assert result.expectation == pytest.approx(-1.0, abs=1e-12)
    assert len(estimator.requests) == 2
    assert [request.observable.num_qubits for request in estimator.requests] == [3, 3]
    assert all(
        isinstance(request.observable, Hamiltonian) for request in estimator.requests
    )
    masks = []
    for request in estimator.requests:
        assert isinstance(request.observable, Hamiltonian)
        masks.append(
            {(pauli.x_mask, pauli.z_mask) for pauli, _ in request.observable.terms}
        )
    ancilla_x = 1 << 2
    assert (ancilla_x, 1) in masks[0]
    assert (ancilla_x, 0) in masks[1]


def test_vd_executor_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="seed must be non-negative"):
        vd_executor(
            QemExecutor(estimator=_RecordingEstimator()),
            copies=2,
            shots_numerator=101,
            shots_denominator=103,
            seed=-1,
        )
