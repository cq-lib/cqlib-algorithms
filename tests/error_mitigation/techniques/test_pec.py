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

"""Tests for probabilistic error cancellation."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import pytest
from cqlib import Circuit
from cqlib.circuit import Instruction, StandardGate
from cqlib.device import SingleQubitNoise

from cqlib_algorithms.error_mitigation._circuits import strip_measurements
from cqlib_algorithms.error_mitigation.core import QemExecutor
from cqlib_algorithms.error_mitigation.quasiprobability import (
    QuasiprobabilityRepresentation,
    quasiprobability_of_operation,
)
from cqlib_algorithms.error_mitigation.techniques.pec import (
    combine_results,
    construct_circuits,
    execute_with_pec,
)
from cqlib_algorithms.protocols import (
    Estimator,
    EstimatorRequest,
    EstimatorResult,
)

_ATOL = 1e-9


def _bit_flip_representation(p: float, /) -> QuasiprobabilityRepresentation:
    """Return the PEC expansion of a Hadamard against a bit-flip channel."""
    return quasiprobability_of_operation(
        Instruction.from_standard_gate(StandardGate.H),
        SingleQubitNoise.bit_flip(p),
    )


class _BitFlipDevice(Estimator):
    """Fake device damping ``<Z>`` by ``1 - 2p`` and flipping it on a flip.

    A variant with no correction realises the bare Hadamard and damps the
    ideal value; a variant carrying the ``X`` correction flips the sign of
    ``<Z>``, which is exactly the response the signed average must undo.
    """

    __slots__ = ("_base", "_p", "_truth")

    def __init__(self, base: int, truth: float, p: float) -> None:
        self._base = base
        self._truth = truth
        self._p = p

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        """Return the damped response of each variant, in input order."""
        results: list[EstimatorResult] = []
        for request in requests:
            body = len(strip_measurements(request.circuit)[0].operations)
            response = (1.0 - 2.0 * self._p) * self._truth
            if body > self._base:
                response = -response
            results.append(EstimatorResult(expectation=response))
        return results


def test_gamma_is_at_least_one_and_equals_one_without_noise(
    exact_executor, rotation_circuit, z_observable
) -> None:
    circuit = rotation_circuit(0.9)
    representation = _bit_flip_representation(0.0)

    base = exact_executor.estimate(
        [EstimatorRequest(circuit=circuit, observable=z_observable)]
    )[0].expectation
    mitigated = execute_with_pec(
        circuit,
        exact_executor,
        z_observable,
        quasiprobabilities=[representation],
        seed=0,
    )

    # A vanishing noise level leaves the identity as the only correction, so
    # the one-norm collapses to one and PEC reproduces the bare value.
    assert representation.norm >= 1.0
    assert representation.norm == pytest.approx(1.0, abs=_ATOL)
    assert mitigated == pytest.approx(base, abs=_ATOL)


def test_the_bit_flip_inverse_channel_has_the_closed_form_coefficients() -> None:
    p = 0.1
    denominator = 1.0 - 2.0 * p

    representation = _bit_flip_representation(p)
    coefficients = representation.coefficients

    # The inverse of N(rho) = (1 - p) rho + p X rho X is a * I + b * X with
    # a = (1 - p)/(1 - 2p) and b = -p/(1 - 2p), so only the identity and X
    # terms survive and the one-norm is 1/(1 - 2p).
    assert coefficients[0] == pytest.approx((1.0 - p) / denominator, abs=_ATOL)
    assert coefficients[1] == pytest.approx(-p / denominator, abs=_ATOL)
    assert all(
        value == pytest.approx(0.0, abs=_ATOL) for value in coefficients[2:]
    )
    assert representation.norm == pytest.approx(1.0 / denominator, abs=_ATOL)


def test_combine_results_applies_the_gamma_compensation() -> None:
    norm = 3.0
    ideal = 0.6

    # One sample's sign-weighted mean is only 1 / Gamma of the ideal value;
    # the multiplication by the one-norm inside combine_results restores it.
    assert combine_results(
        [EstimatorResult(expectation=ideal / norm)], norm, [1]
    ) == pytest.approx(ideal, abs=_ATOL)
    assert combine_results(
        [EstimatorResult(expectation=-ideal / norm)], norm, [-1]
    ) == pytest.approx(ideal, abs=_ATOL)

    with pytest.raises(ValueError, match="paired one to one"):
        combine_results([EstimatorResult(expectation=0.0)], norm, [1, -1])


def test_signed_sampling_converges_to_the_true_expectation(
    z_observable,
) -> None:
    truth, p = 0.6, 0.1
    circuit = Circuit(1)
    circuit.h(0)
    representation = _bit_flip_representation(p)
    base = len(strip_measurements(circuit)[0].operations)
    executor = QemExecutor(estimator=_BitFlipDevice(base, truth, p))

    # A precision of Gamma / 316 lands the sample count near 1e5, so the
    # sampling error is small enough for the true value to fall inside a few
    # standard errors.
    precision = representation.norm / 316.23
    # The sample count deliberately crosses the reporting threshold, so the
    # expected large-run warning is silenced here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        circuits, signs, norm = construct_circuits(
            circuit,
            quasiprobabilities=[representation],
            precision=precision,
            seed=0,
        )
    results = executor.estimate(
        [
            EstimatorRequest(circuit=variant, observable=z_observable)
            for variant in circuits
        ]
    )
    estimate = combine_results(results, norm, signs)

    assert len(circuits) > 100_000
    assert abs(estimate - truth) <= 5.0 * norm / math.sqrt(len(circuits))


def test_rejects_a_gate_beyond_two_qubits() -> None:
    circuit = Circuit(3)
    circuit.ccx(0, 1, 2)

    with pytest.raises(ValueError, match="two-qubit"):
        construct_circuits(
            circuit,
            quasiprobabilities=[_bit_flip_representation(0.1)],
        )


def test_rejects_a_gamma_above_the_overflow_limit() -> None:
    circuit = Circuit(1)
    for _ in range(13):
        circuit.h(0)
    representation = _bit_flip_representation(0.45)

    # Thirteen operations at one-norm ten put Gamma at 1e13, past the 1e12
    # ceiling above which the sample count is unattainable.
    with pytest.raises(ValueError, match="one-norm"):
        construct_circuits(
            circuit,
            quasiprobabilities=[representation] * 13,
        )


def test_warns_when_the_sample_count_is_large() -> None:
    circuit = Circuit(1)
    circuit.h(0)
    representation = _bit_flip_representation(0.1)

    with pytest.warns(UserWarning, match="circuit executions"):
        construct_circuits(
            circuit,
            quasiprobabilities=[representation],
            precision=representation.norm / 316.24,
        )
