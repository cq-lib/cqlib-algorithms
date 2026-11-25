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

"""Independent references and contract tests for superdense coding."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from cqlib.device import ExecutionResult

from cqlib_algorithms.algorithms import (
    SuperdenseCoding,
    SuperdenseCodingRequest,
    SuperdenseCodingResult,
)
from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.protocols import Sampler

_MESSAGES = ((0, 0), (0, 1), (1, 0), (1, 1))
_I = np.eye(2, dtype=complex)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / math.sqrt(2.0)


def _single(matrix: np.ndarray, qubit: int) -> np.ndarray:
    return np.kron(_I, matrix) if qubit == 0 else np.kron(matrix, _I)


def _cx() -> np.ndarray:
    matrix = np.zeros((4, 4), dtype=complex)
    for source in range(4):
        destination = source ^ (2 if source & 1 else 0)
        matrix[destination, source] = 1.0
    return matrix


def _reference_matrix(message: tuple[int, int]) -> np.ndarray:
    x, z = message
    matrix = _single(_H, 0)
    matrix = _cx() @ matrix
    if z:
        matrix = _single(_Z, 0) @ matrix
    if x:
        matrix = _single(_X, 0) @ matrix
    matrix = _cx() @ matrix
    return _single(_H, 0) @ matrix


@pytest.mark.parametrize("message", _MESSAGES)
def test_circuit_matches_independent_matrix(message):
    circuit = SuperdenseCoding.build_circuit(message)

    np.testing.assert_allclose(
        circuit.to_matrix(),
        _reference_matrix(message),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("message", _MESSAGES)
def test_each_message_decodes_to_its_core_bitstring(message):
    circuit = SuperdenseCoding.build_circuit(message)
    output = circuit.to_matrix()[:, 0]
    expected = np.zeros(4, dtype=complex)
    # The decoded computational bits are q0=z and q1=x, hence little-endian
    # basis index z + 2*x. Core formats the requested (q0, q1) measurement as
    # the string "xz", which is the public message order.
    expected[message[1] + 2 * message[0]] = 1.0

    # Encoding 11 uses XZ and may contribute only a physically irrelevant
    # global phase; compare projectors to retain that distinction explicitly.
    np.testing.assert_allclose(
        np.outer(output, output.conj()),
        np.outer(expected, expected.conj()),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("message", _MESSAGES)
def test_ideal_sampler_decodes_every_shot(message):
    result = SuperdenseCoding(StatevectorSampler()).run(
        [SuperdenseCodingRequest(message, 17)]
    )[0]

    assert result.sent_message == message
    assert result.correct_count == 17
    assert result.success_rate == 1.0
    assert result.all_succeeded
    assert result.decoded_message == message
    assert result.modal_messages == (message,)
    assert result.decoded_probabilities[_MESSAGES.index(message)] == 1.0


def test_batch_preserves_order_and_individual_shots():
    requests = [
        SuperdenseCodingRequest(message, shots)
        for message, shots in zip(_MESSAGES, (1, 3, 7, 11))
    ]

    results = SuperdenseCoding(StatevectorSampler()).run(requests)

    assert [result.sent_message for result in results] == list(_MESSAGES)
    assert [result.shots for result in results] == [1, 3, 7, 11]
    assert all(result.all_succeeded for result in results)


def test_seeded_randomized_batches_decode_all_messages():
    generator = np.random.default_rng(20260929)
    requests = [
        SuperdenseCodingRequest(
            _MESSAGES[int(generator.integers(0, len(_MESSAGES)))],
            int(generator.integers(1, 13)),
        )
        for _ in range(32)
    ]

    results = SuperdenseCoding(StatevectorSampler()).run(requests)

    assert [result.sent_message for result in results] == [
        request.message for request in requests
    ]
    assert [result.shots for result in results] == [
        request.shots for request in requests
    ]
    assert all(result.all_succeeded for result in results)


def test_build_circuit_returns_fresh_independent_circuits():
    first = SuperdenseCoding.build_circuit((1, 0))
    second = SuperdenseCoding.build_circuit((1, 0))

    assert first is not second
    assert first == second
    second_size = len(second.operations)
    first.z(0)
    assert len(second.operations) == second_size


class _ScriptedSampler(Sampler):
    def __init__(self, scripts):
        self.scripts = scripts
        self.calls = 0
        self.requests = []

    def run(self, requests):
        self.calls += 1
        self.requests = list(requests)
        return [
            ExecutionResult.from_counts(
                "scripted",
                list(request.qubits),
                request.shots,
                len(request.qubits),
                counts,
                "fake-backend",
            )
            for request, counts in zip(requests, self.scripts)
        ]


def test_scripted_counts_are_preserved_without_fake_shot_memory():
    sampler = _ScriptedSampler([{"00": 2, "01": 3, "10": 4, "11": 1}])

    result = SuperdenseCoding(sampler).run([SuperdenseCodingRequest((1, 0), 10)])[0]

    assert result.decoded_counts == (2, 3, 4, 1)
    assert result.decoded_probabilities == (0.2, 0.3, 0.4, 0.1)
    assert result.correct_count == 4
    assert result.success_rate == 0.4
    assert result.modal_messages == ((1, 0),)
    assert result.decoded_message == (1, 0)
    assert not hasattr(result, "measurements")


def test_tied_modes_are_reported_without_inventing_a_decision():
    sampler = _ScriptedSampler([{"00": 4, "11": 4}])

    result = SuperdenseCoding(sampler).run([SuperdenseCodingRequest((0, 0), 8)])[0]

    assert result.modal_messages == ((0, 0), (1, 1))
    assert result.decoded_message is None
    assert result.correct_count == 4
    assert result.success_rate == 0.5
    assert not result.all_succeeded


def test_complete_batch_uses_one_sampler_call_and_requested_qubit_order():
    sampler = _ScriptedSampler([{"00": 2}, {"11": 5}])

    SuperdenseCoding(sampler).run(
        [
            SuperdenseCodingRequest((0, 0), 2),
            SuperdenseCodingRequest((1, 1), 5),
        ]
    )

    assert sampler.calls == 1
    assert [request.shots for request in sampler.requests] == [2, 5]
    assert all(
        request.qubits == tuple(request.circuit.qubits) for request in sampler.requests
    )


def test_empty_batch_does_not_call_sampler():
    sampler = _ScriptedSampler([])

    assert SuperdenseCoding(sampler).run([]) == []
    assert sampler.calls == 0


def test_complete_batch_is_validated_before_sampler_call():
    sampler = _ScriptedSampler([{"00": 2}])

    with pytest.raises(TypeError, match="SuperdenseCodingRequest instances"):
        SuperdenseCoding(sampler).run(
            [SuperdenseCodingRequest((0, 0), 2), "invalid"]  # type: ignore[list-item]
        )

    assert sampler.calls == 0


def test_sampler_counts_determine_result_instead_of_sent_message():
    sampler = _ScriptedSampler([{"11": 6}])

    result = SuperdenseCoding(sampler).run([SuperdenseCodingRequest((0, 0), 6)])[0]

    assert result.sent_message == (0, 0)
    assert result.decoded_message == (1, 1)
    assert result.correct_count == 0


def test_algorithm_is_slotted_and_exposes_sampler():
    sampler = StatevectorSampler()
    algorithm = SuperdenseCoding(sampler)

    assert algorithm.sampler is sampler
    assert not hasattr(algorithm, "__dict__")
    with pytest.raises(AttributeError):
        algorithm.sampler = sampler  # type: ignore[misc]


def test_request_and_result_support_standard_frozen_dataclass_behavior():
    request = SuperdenseCodingRequest((0, 1), 4)
    result = SuperdenseCodingResult((0, 1), (0, 4, 0, 0), 4)

    assert replace(request, message=(1, 0)).message == (1, 0)
    assert replace(result, sent_message=(1, 0)).sent_message == (1, 0)
    with pytest.raises(FrozenInstanceError):
        request.shots = 8  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.shots = 8  # type: ignore[misc]


@pytest.mark.parametrize("message", [[0, 1], "01", (0,), (0, 1, 0)])
def test_message_rejects_non_two_bit_tuples(message):
    with pytest.raises(TypeError, match="two-bit tuple"):
        SuperdenseCodingRequest(message, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize("message", [(False, 0), (0, True), (0.0, 1), (0, "1")])
def test_message_rejects_non_integer_bits(message):
    with pytest.raises(TypeError, match="message bits must be integers"):
        SuperdenseCodingRequest(message, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize("message", [(-1, 0), (0, 2), (3, 1)])
def test_message_rejects_non_binary_bits(message):
    with pytest.raises(ValueError, match="message bits must be zero or one"):
        SuperdenseCodingRequest(message, 1)


@pytest.mark.parametrize("shots", [True, 1.0, "1"])
def test_shots_reject_non_integer_values(shots):
    with pytest.raises(TypeError, match="shots must be an integer"):
        SuperdenseCodingRequest((0, 0), shots)  # type: ignore[arg-type]


@pytest.mark.parametrize("shots", [0, -1])
def test_shots_reject_non_positive_values(shots):
    with pytest.raises(ValueError, match="shots must be greater than zero"):
        SuperdenseCodingRequest((0, 0), shots)


def test_result_validates_counts_and_consistency():
    with pytest.raises(TypeError, match="four-count tuple"):
        SuperdenseCodingResult((0, 0), [1, 0, 0, 0], 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="four-count tuple"):
        SuperdenseCodingResult((0, 0), (1, 0, 0), 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="contain integers"):
        SuperdenseCodingResult((0, 0), (True, 0, 0, 0), 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        SuperdenseCodingResult((0, 0), (-1, 2, 0, 0), 1)
    with pytest.raises(ValueError, match="sum to shots"):
        SuperdenseCodingResult((0, 0), (1, 0, 0, 0), 2)


def test_constructor_rejects_non_sampler():
    with pytest.raises(TypeError, match="sampler must be a Sampler instance"):
        SuperdenseCoding(object())  # type: ignore[arg-type]


def test_run_rejects_non_list_sampler_container():
    class _TupleSampler(Sampler):
        def run(self, requests):
            request = requests[0]
            return (
                ExecutionResult.from_counts(
                    "tuple",
                    list(request.qubits),
                    request.shots,
                    len(request.qubits),
                    {"00": request.shots},
                ),
            )

    with pytest.raises(TypeError, match="sampler must return a list"):
        SuperdenseCoding(_TupleSampler()).run([SuperdenseCodingRequest((0, 0), 2)])


def test_run_rejects_wrong_number_of_results():
    sampler = _ScriptedSampler([{"00": 2}])

    with pytest.raises(ValueError, match="one result per request"):
        SuperdenseCoding(sampler).run(
            [
                SuperdenseCodingRequest((0, 0), 2),
                SuperdenseCodingRequest((1, 1), 2),
            ]
        )


def test_run_rejects_filtered_counts():
    sampler = _ScriptedSampler([{"00": 1}])

    with pytest.raises(ValueError, match="sum to the requested shots"):
        SuperdenseCoding(sampler).run([SuperdenseCodingRequest((0, 0), 2)])


def test_run_rejects_wrong_result_qubit_order():
    class _ReversedSampler(Sampler):
        def run(self, requests):
            request = requests[0]
            return [
                ExecutionResult.from_counts(
                    "reversed",
                    list(reversed(request.qubits)),
                    request.shots,
                    len(request.qubits),
                    {"00": request.shots},
                )
            ]

    with pytest.raises(ValueError, match="qubits"):
        SuperdenseCoding(_ReversedSampler()).run([SuperdenseCodingRequest((0, 0), 2)])


@pytest.mark.parametrize("message", _MESSAGES)
def test_single_message_matches_legacy_batch(message):
    algorithm = SuperdenseCoding(StatevectorSampler())
    actual = algorithm.run(message, shots=16)
    assert actual == algorithm.run([SuperdenseCodingRequest(message, 16)])[0]


@pytest.mark.parametrize(
    "message,shots", [((0, 2), 16), ((0,), 16), ((0, 1), 0), ([0, 1], 16)]
)
def test_single_message_rejects_invalid_inputs(message, shots):
    with pytest.raises((TypeError, ValueError)):
        SuperdenseCoding(StatevectorSampler()).run(message, shots=shots)
