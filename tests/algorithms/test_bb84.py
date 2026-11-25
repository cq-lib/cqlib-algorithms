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

"""Tests for the BB84 prepare-and-measure protocol experiment."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from cqlib.device import ExecutionResult

from cqlib_algorithms.algorithms import BB84, BB84Basis, BB84Request, BB84Result
from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.protocols import Sampler


def _result(request, bit):
    return ExecutionResult.from_counts(
        "scripted",
        list(request.qubits),
        request.shots,
        len(request.qubits),
        {str(bit): request.shots},
        "fake-backend",
    )


class _ScriptedSampler(Sampler):
    def __init__(self, scripts):
        self.scripts = [tuple(script) for script in scripts]
        self.calls = []

    def run(self, requests):
        call_index = len(self.calls)
        self.calls.append(list(requests))
        bits = self.scripts[call_index]
        assert len(bits) == len(requests)
        return [_result(request, bit) for request, bit in zip(requests, bits)]


def _request(
    bits=(0, 1, 1, 0),
    alice=(BB84Basis.Z, BB84Basis.X, BB84Basis.Z, BB84Basis.X),
    bob=(BB84Basis.Z, BB84Basis.X, BB84Basis.X, BB84Basis.X),
    sample=(0, 1),
    max_qber=0.11,
    **kwargs,
):
    return BB84Request(
        bits,
        alice,
        bob,
        max_qber=max_qber,
        parameter_estimation_indices=sample,
        **kwargs,
    )


def _state(bit, basis):
    if basis is BB84Basis.Z:
        return np.array([1.0, 0.0]) if bit == 0 else np.array([0.0, 1.0])
    sign = 1.0 if bit == 0 else -1.0
    return np.array([1.0, sign]) / math.sqrt(2.0)


def _measurement_probability(bit, prepared_bit, prepared_basis, measured_basis):
    return float(
        abs(np.vdot(_state(bit, measured_basis), _state(prepared_bit, prepared_basis)))
        ** 2
    )


@pytest.mark.reference
def test_all_four_states_and_eight_measurement_combinations_match_numpy() -> None:
    bits = tuple(bit for basis in BB84Basis for bit in (0, 1) for _ in BB84Basis)
    alice_bases = tuple(
        basis for basis in BB84Basis for _bit in (0, 1) for _ in BB84Basis
    )
    bob_bases = tuple(
        measured for _basis in BB84Basis for _bit in (0, 1) for measured in BB84Basis
    )
    sampler = _ScriptedSampler([tuple(0 for _ in bits)])
    BB84(sampler).run([BB84Request(bits, alice_bases, bob_bases, max_qber=0.11)])

    circuits = [request.circuit for request in sampler.calls[0]]
    assert len(circuits) == 8
    for circuit, prepared_bit, prepared_basis, measured_basis in zip(
        circuits, bits, alice_bases, bob_bases
    ):
        amplitudes = circuit.to_matrix()[:, 0]
        probabilities = np.abs(amplitudes) ** 2
        reference = np.array(
            [
                _measurement_probability(
                    measured_bit,
                    prepared_bit,
                    prepared_basis,
                    measured_basis,
                )
                for measured_bit in (0, 1)
            ]
        )
        np.testing.assert_allclose(probabilities, reference, rtol=0.0, atol=1e-12)


@pytest.mark.integration
def test_statevector_sampler_recovers_matching_basis_bits_end_to_end() -> None:
    request = BB84Request(
        (0, 1, 0, 1, 1, 0),
        (
            BB84Basis.Z,
            BB84Basis.Z,
            BB84Basis.X,
            BB84Basis.X,
            BB84Basis.Z,
            BB84Basis.X,
        ),
        (
            BB84Basis.Z,
            BB84Basis.Z,
            BB84Basis.X,
            BB84Basis.X,
            BB84Basis.Z,
            BB84Basis.X,
        ),
        max_qber=0.11,
        parameter_estimation_indices=(0, 2, 5),
    )

    result = BB84(StatevectorSampler()).run([request])[0]

    assert result.alice_sifted_bits == request.alice_bits
    assert result.bob_sifted_bits == request.alice_bits
    assert result.qber == 0.0
    assert result.parameter_estimation_passed is True
    assert result.candidate_indices == (1, 3, 4)
    assert result.candidate_keys_match


def test_intercept_resend_uses_actual_eve_outcome_for_second_stage() -> None:
    sampler = _ScriptedSampler([(1,), (1,)])
    request = BB84Request(
        (0,),
        (BB84Basis.Z,),
        (BB84Basis.X,),
        max_qber=0.11,
        intercept_resend_bases=(BB84Basis.X,),
    )

    BB84(sampler).run([request])

    assert len(sampler.calls) == 2
    eve_probabilities = np.abs(sampler.calls[0][0].circuit.to_matrix()[:, 0]) ** 2
    resend_probabilities = np.abs(sampler.calls[1][0].circuit.to_matrix()[:, 0]) ** 2
    np.testing.assert_allclose(eve_probabilities, (0.5, 0.5), atol=1e-12)
    np.testing.assert_allclose(resend_probabilities, (0.0, 1.0), atol=1e-12)


def test_partial_intercept_resend_only_routes_selected_signals_through_eve() -> None:
    sampler = _ScriptedSampler([(1,), (0, 1, 1)])
    request = BB84Request(
        (0, 0, 1),
        (BB84Basis.Z, BB84Basis.Z, BB84Basis.Z),
        (BB84Basis.Z, BB84Basis.Z, BB84Basis.Z),
        max_qber=0.11,
        intercept_resend_bases=(None, BB84Basis.X, None),
    )

    result = BB84(sampler).run([request])[0]

    assert [len(call) for call in sampler.calls] == [1, 3]
    np.testing.assert_allclose(
        np.abs(sampler.calls[0][0].circuit.to_matrix()[:, 0]) ** 2,
        (0.5, 0.5),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.abs(sampler.calls[1][0].circuit.to_matrix()[:, 0]) ** 2,
        (1.0, 0.0),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.abs(sampler.calls[1][2].circuit.to_matrix()[:, 0]) ** 2,
        (0.0, 1.0),
        atol=1e-12,
    )
    assert result.bob_sifted_bits == (0, 1, 1)


def test_all_none_intercept_plan_is_equivalent_to_no_interception() -> None:
    request = BB84Request(
        (0, 1),
        (BB84Basis.Z, BB84Basis.X),
        (BB84Basis.Z, BB84Basis.X),
        max_qber=0.11,
        intercept_resend_bases=(None, None),
    )
    sampler = _ScriptedSampler([(0, 1)])

    result = BB84(sampler).run([request])[0]

    assert len(sampler.calls) == 1
    assert len(sampler.calls[0]) == 2
    assert result.alice_sifted_bits == result.bob_sifted_bits == (0, 1)


def test_partial_intercepts_preserve_flattened_request_and_signal_order() -> None:
    sampler = _ScriptedSampler([(0, 1), (0, 0, 1, 1)])
    requests = [
        BB84Request(
            (0, 1),
            (BB84Basis.Z, BB84Basis.X),
            (BB84Basis.Z, BB84Basis.X),
            max_qber=0.11,
            intercept_resend_bases=(None, BB84Basis.X),
        ),
        BB84Request(
            (1, 0),
            (BB84Basis.Z, BB84Basis.X),
            (BB84Basis.Z, BB84Basis.X),
            max_qber=0.11,
            intercept_resend_bases=(BB84Basis.Z, None),
        ),
    ]

    results = BB84(sampler).run(requests)

    assert [len(call) for call in sampler.calls] == [2, 4]
    first_eve_state = sampler.calls[0][0].circuit.to_matrix()[:, 0]
    second_eve_state = sampler.calls[0][1].circuit.to_matrix()[:, 0]
    np.testing.assert_allclose(np.abs(first_eve_state) ** 2, (0.0, 1.0), atol=1e-12)
    np.testing.assert_allclose(np.abs(second_eve_state) ** 2, (0.0, 1.0), atol=1e-12)
    assert [result.bob_sifted_bits for result in results] == [(0, 0), (1, 1)]


@pytest.mark.reference
def test_uniform_intercept_resend_has_analytic_25_percent_sifted_qber() -> None:
    expected_error = 0.0
    cases = 0
    for alice_basis in BB84Basis:
        bob_basis = alice_basis
        for alice_bit in (0, 1):
            for eve_basis in BB84Basis:
                conditional_error = 0.0
                for eve_bit in (0, 1):
                    sampler = _ScriptedSampler([(eve_bit,), (alice_bit,)])
                    BB84(sampler).run(
                        [
                            BB84Request(
                                (alice_bit,),
                                (alice_basis,),
                                (bob_basis,),
                                max_qber=0.11,
                                intercept_resend_bases=(eve_basis,),
                            )
                        ]
                    )
                    eve_probabilities = (
                        np.abs(sampler.calls[0][0].circuit.to_matrix()[:, 0]) ** 2
                    )
                    bob_probabilities = (
                        np.abs(sampler.calls[1][0].circuit.to_matrix()[:, 0]) ** 2
                    )
                    p_eve = float(eve_probabilities[eve_bit])
                    p_error = float(bob_probabilities[1 - alice_bit])
                    conditional_error += p_eve * p_error
                expected_error += conditional_error
                cases += 1

    assert expected_error / cases == pytest.approx(0.25, abs=1e-12)


@pytest.mark.parametrize(
    ("bob_bits", "max_qber", "expected_qber", "passed"),
    [
        ((0, 1), 0.0, 0.0, True),
        ((1, 0), 1.0, 1.0, True),
        ((0, 0), 0.5, 0.5, True),
        ((0, 0), 0.49, 0.5, False),
    ],
)
def test_qber_and_threshold_boundaries(
    bob_bits, max_qber, expected_qber, passed
) -> None:
    sampler = _ScriptedSampler([bob_bits])
    request = BB84Request(
        (0, 1),
        (BB84Basis.Z, BB84Basis.Z),
        (BB84Basis.Z, BB84Basis.Z),
        max_qber=max_qber,
        parameter_estimation_indices=(0, 1),
    )

    result = BB84(sampler).run([request])[0]

    assert result.qber == expected_qber
    assert result.parameter_estimation_passed is passed


def test_empty_sifting_and_empty_sample_are_supported() -> None:
    sampler = _ScriptedSampler([(0, 1)])
    result = BB84(sampler).run(
        [
            BB84Request(
                (0, 1),
                (BB84Basis.Z, BB84Basis.X),
                (BB84Basis.X, BB84Basis.Z),
                max_qber=0.11,
            )
        ]
    )[0]

    assert result.sifted_indices == ()
    assert result.sifted_count == 0
    assert result.qber is None
    assert result.parameter_estimation_passed is None
    assert result.candidate_indices == ()
    assert result.candidate_keys_match


def test_all_sifted_bits_may_be_disclosed() -> None:
    sampler = _ScriptedSampler([(0, 1)])
    result = BB84(sampler).run(
        [
            BB84Request(
                (0, 1),
                (BB84Basis.Z, BB84Basis.X),
                (BB84Basis.Z, BB84Basis.X),
                max_qber=0.11,
                parameter_estimation_indices=(0, 1),
            )
        ]
    )[0]

    assert result.parameter_estimation_count == 2
    assert result.candidate_count == 0
    assert result.alice_candidate_bits == ()
    assert result.bob_candidate_bits == ()


def test_passing_sample_does_not_claim_candidate_bits_match() -> None:
    sampler = _ScriptedSampler([(0, 0)])
    result = BB84(sampler).run(
        [
            BB84Request(
                (0, 1),
                (BB84Basis.Z, BB84Basis.Z),
                (BB84Basis.Z, BB84Basis.Z),
                max_qber=0.0,
                parameter_estimation_indices=(0,),
            )
        ]
    )[0]

    assert result.parameter_estimation_passed is True
    assert result.candidate_indices == (1,)
    assert not result.candidate_keys_match


def test_no_eve_batch_uses_one_flattened_sampler_call() -> None:
    sampler = _ScriptedSampler([(0, 1, 1)])
    results = BB84(sampler).run(
        [
            BB84Request((0,), (BB84Basis.Z,), (BB84Basis.Z,), max_qber=0.11),
            BB84Request(
                (1, 1),
                (BB84Basis.X, BB84Basis.Z),
                (BB84Basis.X, BB84Basis.Z),
                max_qber=0.11,
            ),
        ]
    )

    assert len(sampler.calls) == 1
    assert len(sampler.calls[0]) == 3
    assert [result.bob_sifted_bits for result in results] == [(0,), (1, 1)]


def test_mixed_batch_uses_one_eve_call_then_one_bob_call() -> None:
    sampler = _ScriptedSampler([(1, 0), (0, 1, 0)])
    results = BB84(sampler).run(
        [
            BB84Request((0,), (BB84Basis.Z,), (BB84Basis.Z,), max_qber=0.11),
            BB84Request(
                (1, 0),
                (BB84Basis.Z, BB84Basis.X),
                (BB84Basis.Z, BB84Basis.X),
                max_qber=0.11,
                intercept_resend_bases=(BB84Basis.X, BB84Basis.Z),
            ),
        ]
    )

    assert [len(call) for call in sampler.calls] == [2, 3]
    assert results[0].bob_sifted_bits == (0,)
    assert results[1].bob_sifted_bits == (1, 0)


def test_empty_batch_does_not_call_sampler() -> None:
    sampler = _ScriptedSampler([])
    assert BB84(sampler).run([]) == []
    assert sampler.calls == []


def test_complete_batch_is_validated_before_sampler_execution() -> None:
    sampler = _ScriptedSampler([])
    with pytest.raises(TypeError, match="BB84Request"):
        BB84(sampler).run([_request(), object()])  # type: ignore[list-item]
    assert sampler.calls == []


def test_request_is_revalidated_before_sampler_execution() -> None:
    sampler = _ScriptedSampler([])
    request = BB84Request((0,), (BB84Basis.Z,), (BB84Basis.Z,), max_qber=0.11)
    object.__setattr__(request, "alice_bits", ())

    with pytest.raises(ValueError, match="must not be empty"):
        BB84(sampler).run([request])

    assert sampler.calls == []


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("alice_bits", "01", TypeError, "sequence of bits"),
        ("alice_bits", (False,), TypeError, "integer bits"),
        ("alice_bits", (2,), ValueError, "only 0 or 1"),
        ("alice_bases", ("Z",), TypeError, "BB84Basis"),
        ("bob_bases", (), ValueError, "one entry per Alice bit"),
        ("parameter_estimation_indices", (1, 0), ValueError, "sorted"),
        ("parameter_estimation_indices", (0, 0), ValueError, "unique"),
        ("parameter_estimation_indices", (4,), ValueError, "in range"),
        ("max_qber", True, TypeError, "real number"),
        ("max_qber", math.inf, ValueError, "finite"),
        ("max_qber", -0.1, ValueError, "between zero and one"),
        ("max_qber", 1.1, ValueError, "between zero and one"),
    ],
)
def test_request_rejects_invalid_values(field, value, error, message) -> None:
    kwargs = {
        "alice_bits": (0,),
        "alice_bases": (BB84Basis.Z,),
        "bob_bases": (BB84Basis.Z,),
        "max_qber": 0.11,
    }
    kwargs[field] = value
    with pytest.raises(error, match=message):
        BB84Request(**kwargs)


def test_request_requires_explicit_max_qber_keyword() -> None:
    with pytest.raises(TypeError, match="max_qber"):
        BB84Request(  # type: ignore[call-arg]
            (0,),
            (BB84Basis.Z,),
            (BB84Basis.Z,),
        )


def test_request_rejects_nonmatching_parameter_estimation_position() -> None:
    with pytest.raises(ValueError, match="matching bases"):
        BB84Request(
            (0,),
            (BB84Basis.Z,),
            (BB84Basis.X,),
            max_qber=0.11,
            parameter_estimation_indices=(0,),
        )


def test_request_rejects_invalid_intercept_resend_bases() -> None:
    with pytest.raises(ValueError, match="one entry per Alice bit"):
        BB84Request(
            (0, 1),
            (BB84Basis.Z, BB84Basis.X),
            (BB84Basis.Z, BB84Basis.X),
            max_qber=0.11,
            intercept_resend_bases=(BB84Basis.Z,),
        )
    with pytest.raises(TypeError, match="BB84Basis"):
        BB84Request(
            (0,),
            (BB84Basis.Z,),
            (BB84Basis.Z,),
            max_qber=0.11,
            intercept_resend_bases=("Z",),  # type: ignore[arg-type]
        )


def test_request_copies_sequences_and_supports_dataclasses_replace() -> None:
    bits = [0, 1]
    bases = [BB84Basis.Z, BB84Basis.X]
    request = BB84Request(  # type: ignore[arg-type]
        bits,
        bases,
        bases,
        max_qber=0.11,
        parameter_estimation_indices=(0,),
    )
    bits[0] = 1
    bases[0] = BB84Basis.X

    assert request.alice_bits == (0, 1)
    assert request.alice_bases == (BB84Basis.Z, BB84Basis.X)
    assert replace(request, max_qber=0.2).max_qber == 0.2
    with pytest.raises(FrozenInstanceError):
        request.max_qber = 0.3  # type: ignore[misc]


def test_result_validation_and_properties() -> None:
    result = BB84Result((1, 3), (1,), (0, 1), (1, 1), 0.5)
    assert result.parameter_estimation_error_count == 1
    assert result.qber == 1.0
    assert result.candidate_indices == (3,)
    assert result.alice_candidate_bits == (1,)
    assert result.bob_candidate_bits == (1,)
    assert result.candidate_keys_match
    with pytest.raises(FrozenInstanceError):
        result.max_qber = 0.2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (((0,), (), (0, 1), (0,), 0.1), "align"),
        (((0,), (), (), (), 0.1), "align"),
        (((0,), (1,), (0,), (0,), 0.1), "sifted indices"),
    ],
)
def test_result_rejects_inconsistent_fields(args, message) -> None:
    with pytest.raises(ValueError, match=message):
        BB84Result(*args)


def test_algorithm_rejects_non_sampler() -> None:
    with pytest.raises(TypeError, match="Sampler"):
        BB84(object())  # type: ignore[arg-type]


class _InvalidSampler(Sampler):
    def __init__(self, mode):
        self.mode = mode

    def run(self, requests):
        if self.mode == "tuple":
            return tuple(_result(request, 0) for request in requests)
        if self.mode == "count":
            return []
        if self.mode == "type":
            return [object() for _ in requests]
        if self.mode == "shots":
            return [
                ExecutionResult.from_counts(
                    "bad-shots",
                    list(request.qubits),
                    2,
                    len(request.qubits),
                    {"0": 2},
                    "fake-backend",
                )
                for request in requests
            ]
        raise AssertionError("unknown mode")


@pytest.mark.parametrize(
    ("mode", "error", "message"),
    [
        ("tuple", TypeError, "return a list"),
        ("count", ValueError, "one result per request"),
        ("type", TypeError, "ExecutionResult"),
        ("shots", ValueError, "shots must match"),
    ],
)
def test_invalid_sampler_contract_is_rejected(mode, error, message) -> None:
    with pytest.raises(error, match=message):
        BB84(_InvalidSampler(mode)).run(
            [
                BB84Request(
                    (0,),
                    (BB84Basis.Z,),
                    (BB84Basis.Z,),
                    max_qber=0.11,
                )
            ]
        )


def test_single_run_and_legacy_batch_use_same_sifted_bits():
    bases = (BB84Basis.Z, BB84Basis.X, BB84Basis.Z, BB84Basis.X)
    bits = (0, 1, 1, 0)
    algorithm = BB84(StatevectorSampler())
    actual = algorithm.run(
        bits, bases, bases, max_qber=0.11, parameter_estimation_indices=(0, 1)
    )
    expected = algorithm.run(
        [
            BB84Request(
                bits, bases, bases, max_qber=0.11, parameter_estimation_indices=(0, 1)
            )
        ]
    )[0]
    assert actual == expected


def test_single_intercept_resend_matches_batch_and_two_stage_submission():
    sampler = _ScriptedSampler([(0, 1), (1, 0)])
    actual = BB84(sampler).run(
        (0, 1),
        (BB84Basis.Z,) * 2,
        (BB84Basis.Z,) * 2,
        max_qber=0.11,
        parameter_estimation_indices=(0, 1),
        intercept_resend_bases=(BB84Basis.X,) * 2,
    )
    legacy_sampler = _ScriptedSampler([(0, 1), (1, 0)])
    expected = BB84(legacy_sampler).run(
        [
            BB84Request(
                (0, 1),
                (BB84Basis.Z,) * 2,
                (BB84Basis.Z,) * 2,
                max_qber=0.11,
                parameter_estimation_indices=(0, 1),
                intercept_resend_bases=(BB84Basis.X,) * 2,
            )
        ]
    )[0]
    assert actual == expected
    assert len(sampler.calls) == 2
    for new_batch, old_batch in zip(sampler.calls, legacy_sampler.calls):
        assert len(new_batch) == len(old_batch) == 2
        for new, old in zip(new_batch, old_batch):
            np.testing.assert_allclose(
                new.circuit.to_matrix(), old.circuit.to_matrix(), atol=1e-12
            )


@pytest.mark.parametrize(
    "case", ["missing_bob", "missing_qber", "mismatch", "bad_bit", "bad_index"]
)
def test_single_bb84_rejects_invalid_data_before_sampling(case):
    sampler = _ScriptedSampler([])
    algorithm = BB84(sampler)
    bases = (BB84Basis.Z,) * 2
    with pytest.raises((TypeError, ValueError)):
        if case == "missing_bob":
            algorithm.run((0, 1), bases, max_qber=0.1)
        elif case == "missing_qber":
            algorithm.run((0, 1), bases, bases)
        elif case == "mismatch":
            algorithm.run((0,), bases, bases, max_qber=0.1)
        elif case == "bad_bit":
            algorithm.run((0, 2), bases, bases, max_qber=0.1)
        else:
            algorithm.run(
                (0, 1), bases, bases, max_qber=0.1, parameter_estimation_indices=(2,)
            )
    assert sampler.calls == []
