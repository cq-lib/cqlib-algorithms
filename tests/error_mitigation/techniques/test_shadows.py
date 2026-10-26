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

"""Tests for classical shadows with robust readout calibration."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest
from cqlib import Circuit
from cqlib.device import ExecutionResult
from cqlib.qis import Hamiltonian, PauliString

from cqlib_algorithms.error_mitigation.core import QemExecutor
from cqlib_algorithms.error_mitigation.techniques.shadows import (
    execute_with_shadows,
    snapshot_expectation,
)
from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.protocols import Sampler, SamplerRequest

_ATOL = 1e-9

# Shot counts are large enough that the snapshot estimator, whose per-shot
# variance is a few times the observable's, stays well inside the tolerance
# below in the sample mean.
_SHOTS = 40000

# Snapshot estimates of a two-qubit Z-diagonal observable have a per-term
# standard error of about 0.01 at ``_SHOTS`` shots, so this bound covers a
# few sigma without being loose enough to hide a systematic error.
_SAMPLE_TOL = 0.08


class _RecordingSampler(Sampler):
    """Delegate to a real sampler while recording every request."""

    __slots__ = ("_inner", "requests")

    def __init__(self, inner: Sampler) -> None:
        self._inner = inner
        self.requests: list[SamplerRequest] = []

    def run(
        self,
        requests: Sequence[SamplerRequest],
        /,
    ) -> list[ExecutionResult]:
        """Record ``requests`` in order and forward them to the sampler."""
        self.requests.extend(requests)
        return self._inner.run(requests)


def _z_diagonal_observable() -> Hamiltonian:
    """Return a two-qubit Z-diagonal observable with one- and two-site terms.

    The two single-site coefficients are equal, so the ideal value below is
    independent of how the Pauli string labels are mapped onto the qubits.
    """
    observable = Hamiltonian(2)
    observable.add_term(PauliString.from_str("ZI"), 0.5)
    observable.add_term(PauliString.from_str("IZ"), 0.5)
    observable.add_term(PauliString.from_str("ZZ"), 0.25)
    observable.simplify()
    return observable


def _direct_z_mean(
    executor: QemExecutor,
    circuit: Circuit,
    observable: Hamiltonian,
    shots: int,
    /,
) -> float:
    """Return the Z-basis expectation of a diagonal observable from counts.

    Parameters
    ----------
    executor
        Backend supplying the sampling surface.
    circuit
        Circuit whose Z-basis distribution is measured.
    observable
        Z-diagonal observable whose expectation is read off the distribution.
    shots
        Number of shots sampled for the direct estimate.

    Returns
    -------
    float
        The expectation value of ``observable`` under the measured
        computational-basis distribution.
    """
    sampled = executor.sample(
        [
            SamplerRequest(
                circuit=circuit,
                qubits=tuple(circuit.qubits),
                shots=shots,
            )
        ]
    )[0]
    distribution = {
        bitstring: count / sampled.shots
        for bitstring, count in sampled.counts.items()
    }
    measurement = PauliString.from_str("Z" * circuit.num_qubits)
    expectation = observable.expectation_probs([(measurement, distribution)])
    return float(expectation.real)


def test_the_estimate_matches_a_direct_z_measurement_in_the_large_sample_limit(
    exact_executor,
) -> None:
    observable = _z_diagonal_observable()
    circuit = Circuit(2)
    circuit.ry(0, math.pi / 3.0)
    circuit.ry(1, -math.pi / 5.0)

    estimate = execute_with_shadows(
        circuit,
        exact_executor,
        observable,
        num_total_measurements_shadow=_SHOTS,
        seed=3,
    )
    direct = _direct_z_mean(exact_executor, circuit, observable, _SHOTS)

    # On a product state the single-site terms are cos(theta_0) and
    # cos(theta_1) and the two-site term their product, so the ideal value is
    # known in closed form.
    cosines = math.cos(math.pi / 3.0), math.cos(math.pi / 5.0)
    target = 0.5 * sum(cosines) + 0.25 * cosines[0] * cosines[1]
    assert estimate == pytest.approx(target, abs=_SAMPLE_TOL)
    assert estimate == pytest.approx(direct, abs=_SAMPLE_TOL)


def test_the_reconstructed_state_is_trace_normalised() -> None:
    observable = PauliString.from_str("II")
    outcomes = [[0, 1], [1, 1], [0, 0], [1, 0], [1, 1]]
    bases = [["Z", "X"], ["X", "Y"], ["Z", "Z"], ["Y", "Z"], ["X", "X"]]

    # The identity is unsupported, so every snapshot product is one and the
    # average returns the trace of the reconstruction for any shot set.
    assert snapshot_expectation(outcomes, bases, observable) == pytest.approx(
        1.0, abs=_ATOL
    )


def test_rse_reduces_to_the_standard_shadow_without_readout_error() -> None:
    outcomes = [[0, 1], [0, 1], [1, 0], [0, 0]]
    bases = [["Z", "Z"], ["Z", "X"], ["Z", "Z"], ["Y", "Z"]]
    observable = PauliString.from_str("ZI")

    standard = snapshot_expectation(outcomes, bases, observable)
    unit_fidelity = snapshot_expectation(
        outcomes, bases, observable, readout_fidelities=(1.0, 1.0)
    )

    # Dividing every matching factor by a unit readout fidelity leaves the
    # standard estimator bit for bit unchanged.
    assert standard == pytest.approx(0.75, abs=_ATOL)
    assert unit_fidelity == standard
    # Halving both fidelities doubles the calibrated estimate, because only
    # the support qubit's single factor enters the snapshot product.
    halved = snapshot_expectation(
        outcomes, bases, observable, readout_fidelities=(0.5, 0.5)
    )
    assert halved == pytest.approx(2.0 * standard, abs=_ATOL)


def test_the_random_bases_are_reproducible_for_a_seed() -> None:
    recorder = _RecordingSampler(StatevectorSampler())
    executor = QemExecutor(sampler=recorder)
    circuit = Circuit(1)
    circuit.h(0)
    observable = PauliString.from_str("Z")
    shots = 512

    def signature(seed: int) -> list[tuple[int, int | None]]:
        recorder.requests.clear()
        execute_with_shadows(
            circuit,
            executor,
            observable,
            num_total_measurements_shadow=shots,
            seed=seed,
        )
        return [(request.shots, request.seed) for request in recorder.requests]

    first = signature(5)
    repeat = signature(5)
    other = signature(6)

    # Shots that share an axis assignment are grouped into one request, so the
    # request sequence is a direct fingerprint of the drawn axes: one run seed
    # reproduces it, another seed resamples it.
    assert first == repeat
    assert first != other
