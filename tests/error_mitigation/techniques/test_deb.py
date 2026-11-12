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

"""Tests for debiasing by qubit-label symmetrization."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from cqlib import Circuit
from cqlib.device import ExecutionResult
from cqlib.qis import PauliString

from cqlib_algorithms.error_mitigation._circuits import rebuild
from cqlib_algorithms.error_mitigation.core import QemExecutor
from cqlib_algorithms.error_mitigation.techniques.deb import (
    _draw_permutations,
    combine_results,
    construct_circuits,
    deb_executor,
    execute_with_deb,
)
from cqlib_algorithms.operators.estimator import StatevectorEstimator
from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.protocols import (
    EstimatorRequest,
    Sampler,
    SamplerRequest,
    spawn_seeds,
)

_ATOL = 1e-9
_ZZ = PauliString.from_str("ZZ")


def _result(
    counts: dict[str, int],
    shots: int,
    num_qubits: int,
    /,
) -> ExecutionResult:
    """Wrap a bitstring-to-count mapping as one variant's execution result."""
    return ExecutionResult.from_counts(
        "deb-variant", list(range(num_qubits)), shots, num_qubits, counts
    )


def _permutation_variants(
    num_variants: int,
    num_qubits: int,
    seed: int,
    /,
) -> list[tuple[int, ...]]:
    """Return the qubit-label permutations DEB draws for a seed.

    The tuples mirror the draw :func:`construct_circuits` performs, so a test
    can identify a variant and tell whether its relabelling does anything.
    """
    return [
        tuple(int(value) for value in row)
        for row in _draw_permutations(num_variants, num_qubits, seed)
    ]


class _BiasedSampler(Sampler):
    """Sampler appending a fixed coherent rotation to physical qubit zero.

    The over-rotation always acts on physical qubit 0, so from the logical
    frame it lands on a different logical qubit for each permutation. That is
    exactly the physical-qubit-attached, direction-dependent error that
    qubit-label symmetrization is meant to average away.
    """

    __slots__ = ("_base", "_theta")

    def __init__(self, base: Sampler, theta: float) -> None:
        self._base = base
        self._theta = theta

    def run(
        self,
        requests: Sequence[SamplerRequest],
        /,
    ) -> list[ExecutionResult]:
        """Return one biased count record per request, in input order."""
        biased: list[SamplerRequest] = []
        for request in requests:
            template = Circuit(request.circuit.num_qubits)
            template.ry(0, self._theta)
            tilt = template.operations[0]
            circuit = rebuild(request.circuit, [*request.circuit.operations, tilt])
            biased.append(
                SamplerRequest(
                    circuit=circuit,
                    qubits=tuple(circuit.qubits),
                    shots=request.shots,
                    seed=request.seed,
                )
            )
        return self._base.run(biased)


def _biased_executor(theta: float, /) -> QemExecutor:
    """Return an executor whose sampler tilts physical qubit zero by theta."""
    return QemExecutor(sampler=_BiasedSampler(StatevectorSampler(), theta))


class _RecordingSampler(Sampler):
    """Sampler returning an all-zero record while logging every request."""

    __slots__ = ("requests",)

    def __init__(self) -> None:
        self.requests: list[SamplerRequest] = []

    def run(
        self,
        requests: Sequence[SamplerRequest],
        /,
    ) -> list[ExecutionResult]:
        """Log the requests and return one deterministic record each."""
        self.requests.extend(requests)
        results: list[ExecutionResult] = []
        for request in requests:
            width = request.circuit.num_qubits
            results.append(
                ExecutionResult.from_counts(
                    "deb-record",
                    list(request.qubits),
                    request.shots,
                    width,
                    {"0" * width: request.shots},
                )
            )
        return results


def test_the_aggregate_equals_the_expectation_without_coherent_bias(
    exact_executor: QemExecutor,
    exact_estimator: StatevectorEstimator,
) -> None:
    circuit = Circuit(3)
    circuit.x(0)
    circuit.x(1)
    observable = PauliString.from_str("ZIZ")
    ideal = exact_estimator.run(
        [EstimatorRequest(circuit=circuit, observable=observable)]
    )[0].expectation

    assert abs(ideal) == pytest.approx(1.0, abs=_ATOL)
    for mode in ("averaging", "sharpening"):
        value = execute_with_deb(
            circuit,
            exact_executor,
            observable,
            mode=mode,
            shots=4096,
            num_variants=8,
            seed=5,
        )
        # Every relabelling estimates the same ideal value on a noise-free
        # sampler, so aggregating the variants cannot move the estimate.
        assert value == pytest.approx(ideal, abs=_ATOL)


def test_averaging_reduces_a_fixed_directional_bias(
    exact_estimator: StatevectorEstimator,
) -> None:
    shots, num_variants, theta = 20000, 64, 0.6
    circuit = Circuit(2)
    circuit.x(0)
    # The label order puts Z on qubit 0, the qubit the bias acts on.
    observable = PauliString.from_str("IZ")
    ideal = exact_estimator.run(
        [EstimatorRequest(circuit=circuit, observable=observable)]
    )[0].expectation
    executor = _biased_executor(theta)

    raw = executor.sample(
        [
            SamplerRequest(
                circuit=circuit,
                qubits=tuple(circuit.qubits),
                shots=shots,
            )
        ]
    )
    unmitigated = combine_results(
        raw, mode="averaging", observable=observable, shots=shots
    )
    mitigated = execute_with_deb(
        circuit,
        executor,
        observable,
        mode="averaging",
        shots=shots,
        num_variants=num_variants,
        seed=11,
    )

    # The physical-qubit rotation biases the direct estimate; relabelling
    # spreads that single direction over the logical qubits, so the mean over
    # variants lands strictly closer to the ideal value.
    assert abs(unmitigated - ideal) > 0.05
    assert abs(mitigated - ideal) < abs(unmitigated - ideal)


def test_sharpening_output_lies_in_the_sample_value_set() -> None:
    shots = 3
    variants = [
        _result({"00": 3}, shots, 2),
        _result({"00": 1, "01": 2}, shots, 2),
        _result({"00": 1, "01": 2}, shots, 2),
    ]

    value = combine_results(variants, mode="sharpening", observable=_ZZ, shots=shots)

    # Majority voting keeps one "00" and two "01" records, so the sharpened
    # estimate is the convex combination 1/3 - 2/3 = -1/3, a realisable value.
    assert -1.0 <= value <= 1.0
    assert value == pytest.approx(-1.0 / 3.0, abs=_ATOL)


def test_sharpening_requires_equal_shot_counts() -> None:
    variants = [
        _result({"00": 4}, 4, 2),
        _result({"00": 5}, 5, 2),
    ]

    with pytest.raises(ValueError, match="same shot count"):
        combine_results(variants, mode="sharpening", observable=_ZZ, shots=4)


def test_sharpening_discards_a_tie() -> None:
    shots = 2
    variants = [
        _result({"00": 2}, shots, 2),
        _result({"00": 1, "01": 1}, shots, 2),
        _result({"00": 1, "10": 1}, shots, 2),
    ]

    value = combine_results(variants, mode="sharpening", observable=_ZZ, shots=shots)

    # The second shot draws three different records, so its vote ties and the
    # shot is dropped; the estimate is built from the surviving "00" alone.
    assert value == pytest.approx(1.0, abs=_ATOL)

    tied = [
        _result({"00": 2}, shots, 2),
        _result({"11": 2}, shots, 2),
    ]
    with pytest.raises(ValueError, match="dropped every shot"):
        combine_results(tied, mode="sharpening", observable=_ZZ, shots=shots)


def test_combine_results_uses_the_original_observable(
    exact_executor: QemExecutor,
    exact_estimator: StatevectorEstimator,
) -> None:
    shots, num_variants, seed = 4096, 8, 2
    circuit = Circuit(2)
    circuit.x(0)
    observable = PauliString.from_str("ZI")
    ideal = exact_estimator.run(
        [EstimatorRequest(circuit=circuit, observable=observable)]
    )[0].expectation
    circuits, observables = construct_circuits(
        circuit, observable, num_variants=num_variants, seed=seed
    )
    permutations = _permutation_variants(num_variants, 2, seed)
    index = next(
        position
        for position, permutation in enumerate(permutations)
        if permutation != (0, 1)
    )
    results = exact_executor.sample(
        [
            SamplerRequest(circuit=variant, qubits=tuple(variant.qubits), shots=shots)
            for variant in circuits
        ]
    )

    # The recorded bitstring is already back in logical order, so the original
    # observable is the one that evaluates a relabelled variant correctly,
    # while pairing a variant with its own permuted observable is a mismatch.
    relabelled = combine_results(
        [results[index]], mode="averaging", observable=observable, shots=shots
    )
    mismatched = combine_results(
        [results[index]],
        mode="averaging",
        observable=observables[index],
        shots=shots,
    )
    aggregate = combine_results(
        results, mode="averaging", observable=observable, shots=shots
    )

    assert relabelled == pytest.approx(ideal, abs=_ATOL)
    assert abs(mismatched - ideal) > 0.5
    assert aggregate == pytest.approx(ideal, abs=_ATOL)


def test_the_permutations_are_statistically_independent() -> None:
    shots, num_variants, seed = 512, 6, 11
    recorder = _RecordingSampler()
    executor = QemExecutor(sampler=recorder)
    circuit = Circuit(2)
    circuit.x(0)
    observable = PauliString.from_str("ZI")
    request = EstimatorRequest(circuit=circuit, observable=observable, shots=shots)

    def run_seeds(run_seed: int) -> list[int]:
        recorder.requests.clear()
        deb_executor(
            executor,
            mode="averaging",
            shots=shots,
            num_variants=num_variants,
            seed=run_seed,
        ).estimate([request])
        return [entry.seed for entry in recorder.requests]

    expected = list(spawn_seeds(seed, num_variants))
    assert run_seeds(seed) == expected
    assert len(set(expected)) == num_variants
    assert run_seeds(seed) == expected
    assert run_seeds(seed + 1) == list(spawn_seeds(seed + 1, num_variants))

    # The circuit permutations themselves vary across variants, so the runs
    # are not silently sharing one relabelling.
    assert len(set(_permutation_variants(num_variants, 2, seed))) > 1
