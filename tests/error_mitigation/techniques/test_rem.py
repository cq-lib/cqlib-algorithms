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

"""Tests for readout-error mitigation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import reduce

import numpy as np
import pytest
from cqlib import Circuit
from cqlib.device import ExecutionResult, ReadoutError
from cqlib.qis import PauliString

from cqlib_algorithms.error_mitigation.core import QemExecutor
from cqlib_algorithms.error_mitigation.inference import ConvergenceWarning
from cqlib_algorithms.error_mitigation.techniques.rem import (
    ConfusionMatrix,
    _local_matrix,
    calibrate_readout,
    execute_with_rem,
    mitigate_counts,
)
from cqlib_algorithms.operators.estimator import StatevectorEstimator
from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.protocols import (
    EstimatorRequest,
    Sampler,
    SamplerRequest,
)

_ATOL = 1e-9


def _confusion(p0: float, p1: float, num_qubits: int, /) -> ConfusionMatrix:
    """Return the same single-qubit readout error on every qubit."""
    error = ReadoutError(p_0_given_1=p1, p_1_given_0=p0)
    return ConfusionMatrix(per_qubit=tuple(error for _ in range(num_qubits)))


def _probs_to_counts(
    probabilities: Mapping[str, float],
    shots: int,
    /,
) -> dict[str, int]:
    """Turn a basis distribution into integer counts over ``shots`` draws."""
    return {
        key: int(round(value * shots)) for key, value in probabilities.items()
    }


class _ReadoutNoiseSampler(Sampler):
    """Sampler adding independent per-qubit readout flips to a clean run.

    The ideal distribution is produced by a noiseless statevector sampler and
    then convolved with the Kronecker product of the single-qubit confusion
    matrices, so the reported counts carry exactly the readout error the
    technique is meant to invert.
    """

    __slots__ = ("_inner", "_p0", "_p1")

    def __init__(self, p0: float, p1: float) -> None:
        self._inner = StatevectorSampler()
        self._p0 = p0
        self._p1 = p1

    def run(
        self,
        requests: Sequence[SamplerRequest],
        /,
    ) -> list[ExecutionResult]:
        """Return one flipped count record per request, in input order."""
        results: list[ExecutionResult] = []
        for request, exact in zip(
            requests, self._inner.run(requests), strict=True
        ):
            width = exact.num_qubits
            ideal = np.zeros(1 << width, dtype=float)
            for key, count in exact.counts.items():
                ideal[int(key, 2)] += count
            ideal /= ideal.sum()

            # The confusion index puts bit i on qubit i, the same convention
            # the little-endian count keys use.
            local = np.array(
                [[1.0 - self._p0, self._p1], [self._p0, 1.0 - self._p1]],
                dtype=float,
            )
            full = reduce(np.kron, [local] * width)
            observed = full @ ideal
            observed /= observed.sum()

            draws = np.random.default_rng(request.seed).multinomial(
                request.shots, observed
            )
            counts = {
                format(index, f"0{width}b"): int(draws[index])
                for index in range(1 << width)
                if draws[index]
            }
            results.append(
                ExecutionResult.from_counts(
                    "readout-noise",
                    list(request.qubits),
                    request.shots,
                    width,
                    counts,
                )
            )
        return results


def test_mitigate_counts_recovers_a_known_distribution() -> None:
    confusion = _confusion(0.1, 0.2, 2)
    truth = np.array([0.4, 0.3, 0.2, 0.1])

    # Independent construction from the definition A_{yx} = P(y | x): the
    # two-qubit channel is the Kronecker product of the single-qubit blocks.
    local = np.array([[0.9, 0.2], [0.1, 0.8]])
    observed = reduce(np.kron, [local, local]) @ truth
    counts = {
        format(index, "02b"): int(round(1.0e6 * observed[index]))
        for index in range(4)
    }

    recovered = mitigate_counts(counts, confusion)

    assert np.allclose(recovered, truth, atol=1e-4)


def test_mitigate_counts_is_the_identity_for_a_unit_confusion() -> None:
    confusion = _confusion(0.0, 0.0, 2)
    assert confusion.p0 == (0.0, 0.0)

    probabilities = {"00": 0.5, "01": 0.25, "10": 0.15, "11": 0.1}
    counts = _probs_to_counts(probabilities, 4000)

    recovered = mitigate_counts(counts, confusion)

    # A = I, so the corrected vector is the observed distribution unchanged.
    expected = np.array(
        [probabilities[key] for key in ("00", "01", "10", "11")]
    )
    assert np.allclose(recovered, expected, atol=_ATOL)


def test_confusion_matrix_direction_p0_is_measured_one_given_prepared_zero() -> None:
    confusion = _confusion(0.1, 0.2, 1)

    assert confusion.p0 == (0.1,)
    assert confusion.p1 == (0.2,)

    # Column 0 is the prepared-zero case and column 1 the prepared-one case,
    # so the false-positive rate sits at [1, 0] and the false-negative at
    # [0, 1]; swapping them would invert the correction.
    matrix = _local_matrix(confusion.per_qubit[0])
    assert matrix[1, 0] == pytest.approx(0.1, abs=_ATOL)
    assert matrix[0, 1] == pytest.approx(0.2, abs=_ATOL)


def test_mitigate_counts_uses_a_pseudoinverse_for_an_ill_conditioned_matrix() -> None:
    # p0 = p1 = 0.5 is fully depolarized readout: the confusion matrix is
    # singular, so the correction falls back to the pseudo-inverse instead of
    # inverting a zero eigenvalue.
    confusion = _confusion(0.5, 0.5, 2)
    counts = _probs_to_counts({"00": 0.4, "01": 0.3, "10": 0.2, "11": 0.1}, 10000)

    with pytest.warns(ConvergenceWarning):
        recovered = mitigate_counts(counts, confusion)

    assert np.all(np.isfinite(recovered))
    assert np.all(recovered >= 0.0)
    assert recovered.sum() == pytest.approx(1.0, abs=_ATOL)


def test_post_select_drops_the_negative_entries() -> None:
    confusion = _confusion(0.3, 0.3, 1)
    counts = _probs_to_counts({"0": 0.9, "1": 0.1}, 1000)

    recovered = mitigate_counts(counts, confusion, post_select=True)

    assert np.all(recovered >= 0.0)
    assert recovered.sum() == pytest.approx(1.0, abs=_ATOL)

    # Both repair strategies must return a physical vector.
    projected = mitigate_counts(counts, confusion)
    assert np.all(projected >= 0.0)
    assert projected.sum() == pytest.approx(1.0, abs=_ATOL)


def test_calibrate_readout_returns_one_readout_error_per_qubit(
    exact_executor: QemExecutor,
) -> None:
    confusion = calibrate_readout(exact_executor, 3, shots=1024, seed=0)

    assert len(confusion.per_qubit) == 3
    assert all(
        isinstance(error, ReadoutError) for error in confusion.per_qubit
    )
    # A noiseless backend reports no flips.
    assert confusion.p0 == (0.0, 0.0, 0.0)
    assert confusion.p1 == (0.0, 0.0, 0.0)


def test_calibrate_readout_builds_a_uniform_matrix_from_p0_and_p1() -> None:
    p0, p1 = 0.1, 0.2
    executor = QemExecutor(sampler=_ReadoutNoiseSampler(p0, p1))

    confusion = calibrate_readout(executor, 2, shots=200000, seed=5)

    # The two calibration circuits expose opposite flips, and the recovered
    # rates land in the matching slots for every qubit.
    for value in confusion.p0:
        assert value == pytest.approx(p0, abs=0.01)
    for value in confusion.p1:
        assert value == pytest.approx(p1, abs=0.01)


def test_execute_with_rem_reproduces_the_analytic_bit_flip_example(
    rotation_circuit: Callable[[float], Circuit],
    z_observable: PauliString,
    exact_estimator: StatevectorEstimator,
) -> None:
    shots = 200000
    target = rotation_circuit(1.0)
    executor = QemExecutor(sampler=_ReadoutNoiseSampler(0.05, 0.05))

    ideal = exact_estimator.run(
        [EstimatorRequest(circuit=target, observable=z_observable)]
    )[0].expectation
    sampled = executor.sample(
        [
            SamplerRequest(
                circuit=target,
                qubits=tuple(target.qubits),
                shots=shots,
                seed=7,
            )
        ]
    )[0]
    noisy = sum(
        (1.0 if key == "0" else -1.0) * count
        for key, count in sampled.counts.items()
    ) / shots

    mitigated = execute_with_rem(
        target,
        executor,
        z_observable,
        shots=shots,
        p0=0.05,
        p1=0.05,
        seed=7,
    )

    # Symmetric bit-flip readout scales <Z> by (1 - 2 p), so inverting it must
    # recover the ideal value and beat the raw one.
    assert mitigated == pytest.approx(ideal, abs=0.02)
    assert abs(mitigated - ideal) < abs(noisy - ideal)
