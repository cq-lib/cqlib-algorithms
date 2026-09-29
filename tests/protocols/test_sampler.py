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

"""Tests for the public sampler contract."""

from __future__ import annotations

from collections.abc import Sequence
from copy import copy
from types import SimpleNamespace

import pytest
from cqlib import Circuit, Parameter, Qubit
from cqlib.device import ExecutionResult

from cqlib_algorithms import protocols
from cqlib_algorithms.protocols.sampler import (
    Sampler,
    SamplerRequest,
    validate_sampler_result,
    validate_sampler_results,
)


def _sample_request(shots: int = 4) -> SamplerRequest:
    circuit = Circuit(2)
    circuit.h(0)
    return SamplerRequest(circuit, tuple(circuit.qubits), shots)


class _FakeSampler(Sampler):
    def run(
        self,
        requests: Sequence[SamplerRequest],
        /,
    ) -> list[ExecutionResult]:
        results: list[ExecutionResult] = []
        for index, request in enumerate(requests):
            request.validate()
            execution_circuit = copy(request.circuit)
            measurement = execution_circuit.measure_bits(list(request.qubits))
            bitstring = "0" * measurement.width
            results.append(
                ExecutionResult.from_counts(
                    f"fake-{index}",
                    measurement.qubits,
                    request.shots,
                    measurement.width,
                    {bitstring: request.shots},
                    "fake",
                )
            )
        return results


def _run_sampler(
    sampler: Sampler, requests: Sequence[SamplerRequest]
) -> list[ExecutionResult]:
    return sampler.run(requests)


def test_sampler_request_accepts_explicit_ordered_qubits():
    request = _sample_request()

    assert request.qubits == (Qubit(0), Qubit(1))
    assert request.shots == 4
    with pytest.raises(TypeError):
        hash(request)


def test_sampler_request_requires_a_tuple_of_qubits():
    circuit = Circuit(1)

    with pytest.raises(TypeError, match="qubits must be a tuple"):
        SamplerRequest(circuit, circuit.qubits, 10)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="only cqlib.Qubit"):
        SamplerRequest(circuit, (0,), 10)  # type: ignore[arg-type]


@pytest.mark.parametrize("shots", [0, -1])
def test_sampler_request_rejects_non_positive_shots(shots):
    circuit = Circuit(1)

    with pytest.raises(ValueError, match="greater than zero"):
        SamplerRequest(circuit, tuple(circuit.qubits), shots)


@pytest.mark.parametrize("shots", [True, 1.5, "1"])
def test_sampler_request_rejects_non_integer_shots(shots):
    circuit = Circuit(1)

    with pytest.raises(TypeError, match="shots must be an integer"):
        SamplerRequest(circuit, tuple(circuit.qubits), shots)


def test_sampler_request_rejects_unbound_circuit():
    circuit = Circuit(1)
    circuit.rx(0, Parameter("theta"))

    with pytest.raises(ValueError, match="symbolic"):
        SamplerRequest(circuit, tuple(circuit.qubits), 10)


def test_sampler_request_rejects_empty_duplicate_or_foreign_qubits():
    circuit = Circuit(2)

    with pytest.raises(ValueError, match="must not be empty"):
        SamplerRequest(circuit, (), 10)
    with pytest.raises(ValueError, match="unique"):
        SamplerRequest(circuit, (Qubit(0), Qubit(0)), 10)
    with pytest.raises(ValueError, match="belong to circuit"):
        SamplerRequest(circuit, (Qubit(0), Qubit(7)), 10)


def test_sampler_preserves_reversed_non_contiguous_qubit_order():
    circuit = Circuit([7, 2])
    circuit.h(Qubit(7))
    request = SamplerRequest(circuit, (Qubit(2), Qubit(7)), 5)
    original_operation_count = len(circuit.operations)

    result = _FakeSampler().run([request])[0]

    assert result.qubits == [Qubit(2), Qubit(7)]
    assert result.num_qubits == 2
    assert result.shots == 5
    assert len(circuit.operations) == original_operation_count


def test_sampler_revalidates_mutable_circuit_before_execution():
    request = _sample_request()
    request.circuit.rx(0, Parameter("late"))

    with pytest.raises(ValueError, match="symbolic"):
        _FakeSampler().run([request])


def test_sampler_preserves_batch_order_and_supports_empty_batches():
    sampler = _FakeSampler()
    requests = [_sample_request(2), _sample_request(3)]

    assert _run_sampler(sampler, []) == []
    results = _run_sampler(sampler, requests)

    assert [result.task_id for result in results] == ["fake-0", "fake-1"]
    assert [result.shots for result in results] == [2, 3]
    assert all(result.status.is_terminal() for result in results)
    assert all(result.status.is_success() for result in results)
    assert [sum(result.counts.values()) for result in results] == [2, 3]


def test_sampler_uses_nominal_runtime_membership():
    assert isinstance(_FakeSampler(), Sampler)


def test_result_validators_are_exported_from_protocol_namespace():
    assert protocols.validate_sampler_result is validate_sampler_result
    assert protocols.validate_sampler_results is validate_sampler_results


def test_public_result_validators_accept_and_preserve_a_valid_batch():
    requests = [_sample_request(2), _sample_request(3)]
    results = _FakeSampler().run(requests)

    validate_sampler_result(results[0], requests[0])
    assert validate_sampler_results(results, requests) is results


def test_batch_validator_rejects_invalid_container_size_and_entry_type():
    request = _sample_request()

    with pytest.raises(TypeError, match="sampler must return a list"):
        validate_sampler_results(tuple(_FakeSampler().run([request])), [request])
    with pytest.raises(ValueError, match="one result per request"):
        validate_sampler_results([], [request])
    with pytest.raises(TypeError, match="cqlib.device.ExecutionResult"):
        validate_sampler_results([{"00": 4}], [request])


def test_result_validator_rejects_wrong_request_type():
    request = _sample_request()
    result = _FakeSampler().run([request])[0]

    with pytest.raises(TypeError, match="request must be a SamplerRequest"):
        validate_sampler_result(result, object())  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", ["queued", "failed"])
def test_result_validator_rejects_nonterminal_or_failed_results(mode):
    request = _sample_request()
    result = ExecutionResult(
        mode,
        list(request.qubits),
        request.shots,
        len(request.qubits),
    )
    if mode == "failed":
        result.fail("deliberate failure", 1)

    with pytest.raises(ValueError, match="successful terminal"):
        validate_sampler_result(result, request)


def test_result_validator_rejects_metadata_mismatches():
    request = _sample_request()
    wrong_shots = ExecutionResult.from_counts(
        "shots", list(request.qubits), 5, 2, {"00": 5}, "fake"
    )
    wrong_width = ExecutionResult.from_counts(
        "width", [request.qubits[0]], 4, 1, {"0": 4}, "fake"
    )
    wrong_order = ExecutionResult.from_counts(
        "order", list(reversed(request.qubits)), 4, 2, {"00": 4}, "fake"
    )

    with pytest.raises(ValueError, match="shots must match"):
        validate_sampler_result(wrong_shots, request)
    with pytest.raises(ValueError, match="width"):
        validate_sampler_result(wrong_width, request)
    with pytest.raises(ValueError, match="preserve request order"):
        validate_sampler_result(wrong_order, request)


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        ({"0": 4}, "width"),
        ({"0x": 4}, "binary"),
        ({"00": True, "11": 3}, "non-negative integers"),
        ({"00": 1.5, "11": 2.5}, "non-negative integers"),
        ({"00": -1, "11": 5}, "non-negative integers"),
        ({"00": 3}, "sum to the requested shots"),
    ],
)
def test_result_validator_rejects_malformed_or_incomplete_counts(counts, message):
    request = _sample_request()
    valid = _FakeSampler().run([request])[0]
    result = SimpleNamespace(
        status=valid.status,
        shots=valid.shots,
        num_qubits=valid.num_qubits,
        qubits=valid.qubits,
        counts=counts,
    )

    with pytest.raises(ValueError, match=message):
        validate_sampler_result(result, request)  # type: ignore[arg-type]
