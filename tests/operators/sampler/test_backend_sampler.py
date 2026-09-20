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

"""Contract, isolation and boundary checks for backend-adapted sampling."""

from __future__ import annotations

import pytest
from cqlib import Circuit, Parameter, Qubit
from cqlib.circuit import Measurement
from cqlib.device import ExecutionResult

from cqlib_algorithms.operators.sampler import BackendSampler
from cqlib_algorithms.operators.sampler.backend_sampler import (
    BackendSampler as ModuleBackendSampler,
)
from cqlib_algorithms.protocols.sampler import Sampler, SamplerRequest


def _successful_result(
    request: SamplerRequest,
    counts: dict[str, int],
    *,
    backend: str = "test-backend",
) -> ExecutionResult:
    return ExecutionResult.from_counts(
        "test-task",
        list(request.qubits),
        request.shots,
        len(request.qubits),
        counts,
        backend,
    )


def test_local_export_abc_and_empty_batch_do_not_call_executor() -> None:
    calls = 0

    def executor(
        circuit: Circuit,
        measurement: Measurement,
        shots: int,
    ) -> ExecutionResult:
        nonlocal calls
        calls += 1
        raise AssertionError("empty batches must not invoke the executor")

    sampler = BackendSampler(executor)

    assert BackendSampler is ModuleBackendSampler
    assert isinstance(sampler, Sampler)
    assert sampler.run([]) == []
    assert calls == 0


def test_batch_preserves_order_and_does_not_mutate_sources() -> None:
    captured: list[tuple[Circuit, Measurement, int]] = []
    expected_qubits = [(Qubit(2), Qubit(7)), (Qubit(9),)]
    counts = [{"10": 3}, {"1": 5}]

    def executor(
        circuit: Circuit,
        measurement: Measurement,
        shots: int,
    ) -> ExecutionResult:
        index = len(captured)
        captured.append((circuit, measurement, shots))
        request = SamplerRequest(circuit, expected_qubits[index], shots)
        return _successful_result(request, counts[index])

    first = Circuit([7, 2])
    first.x(Qubit(7))
    second = Circuit([9])
    first_operations = len(first.operations)
    second_operations = len(second.operations)
    requests = [
        SamplerRequest(first, (Qubit(2), Qubit(7)), 3),
        SamplerRequest(second, (Qubit(9),), 5),
    ]

    results = BackendSampler(executor).run(requests)

    assert len(captured) == 2
    assert [result.counts for result in results] == [{"10": 3}, {"1": 5}]
    assert [item[2] for item in captured] == [3, 5]
    assert [tuple(item[1].qubits) for item in captured] == expected_qubits
    assert [len(item[0].operations) for item in captured] == [2, 1]
    assert [len(item[0].classical_values) for item in captured] == [1, 1]
    assert len(first.operations) == first_operations
    assert len(second.operations) == second_operations
    assert not first.classical_values
    assert not second.classical_values


def test_executor_mutation_is_confined_to_private_circuit() -> None:
    source = Circuit(1)

    def executor(
        circuit: Circuit,
        measurement: Measurement,
        shots: int,
    ) -> ExecutionResult:
        circuit.x(0)
        request = SamplerRequest(circuit, tuple(measurement.qubits), shots)
        return _successful_result(request, {"0": 2})

    BackendSampler(executor).run([SamplerRequest(source, (Qubit(0),), 2)])

    assert not source.operations
    assert not source.classical_values


def test_failed_terminal_result_is_preserved() -> None:
    circuit = Circuit(1)

    def executor(
        circuit: Circuit,
        measurement: Measurement,
        shots: int,
    ) -> ExecutionResult:
        result = ExecutionResult("failed-task", [Qubit(0)], 4, 1, "backend")
        result.fail("device unavailable", 17)
        return result

    result = BackendSampler(executor).run([SamplerRequest(circuit, (Qubit(0),), 4)])[0]

    assert result.status.is_terminal()
    assert not result.status.is_success()
    assert result.status.error_msg == "device unavailable"


def test_cancelled_terminal_result_is_preserved() -> None:
    circuit = Circuit(1)

    def executor(
        private_circuit: Circuit,
        measurement: Measurement,
        shots: int,
    ) -> ExecutionResult:
        result = ExecutionResult("cancelled-task", [Qubit(0)], shots, 1)
        result.cancel()
        return result

    result = BackendSampler(executor).run([SamplerRequest(circuit, (Qubit(0),), 4)])[0]

    assert result.status.is_terminal()
    assert result.status.kind == "cancelled"


def test_late_symbolic_mutation_is_revalidated_before_executor_call() -> None:
    calls = 0

    def executor(
        circuit: Circuit,
        measurement: Measurement,
        shots: int,
    ) -> ExecutionResult:
        nonlocal calls
        calls += 1
        raise AssertionError("invalid requests must not invoke the executor")

    circuit = Circuit(1)
    request = SamplerRequest(circuit, (Qubit(0),), 3)
    circuit.rx(0, Parameter("late"))

    with pytest.raises(ValueError, match="symbolic"):
        BackendSampler(executor).run([request])
    assert calls == 0


def test_rejects_non_callable_executor_and_wrong_batch_member() -> None:
    with pytest.raises(TypeError, match="callable"):
        BackendSampler(object())  # type: ignore[arg-type]

    sampler = BackendSampler(lambda circuit, measurement, shots: object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="SamplerRequest"):
        sampler.run([object()])  # type: ignore[list-item]


def test_rejects_wrong_result_type() -> None:
    request = SamplerRequest(Circuit(1), (Qubit(0),), 2)
    with pytest.raises(TypeError, match="ExecutionResult"):
        BackendSampler(
            lambda circuit, measurement, shots: object()  # type: ignore[arg-type]
        ).run([request])


def test_executor_exception_propagates_unchanged() -> None:
    request = SamplerRequest(Circuit(1), (Qubit(0),), 2)
    failure = RuntimeError("adapter failure")

    def executor(
        circuit: Circuit,
        measurement: Measurement,
        shots: int,
    ) -> ExecutionResult:
        raise failure

    with pytest.raises(RuntimeError, match="adapter failure") as caught:
        BackendSampler(executor).run([request])
    assert caught.value is failure


def test_rejects_nonterminal_and_inconsistent_successful_metadata() -> None:
    request = SamplerRequest(Circuit(1), (Qubit(0),), 2)
    queued = ExecutionResult("queued", [Qubit(0)], 2, 1, "backend")
    with pytest.raises(ValueError, match="terminal"):
        BackendSampler(lambda circuit, measurement, shots: queued).run([request])

    wrong_order = ExecutionResult.from_counts(
        "wrong-order",
        [Qubit(1)],
        2,
        1,
        {"0": 2},
        "backend",
    )
    with pytest.raises(ValueError, match="qubits"):
        BackendSampler(lambda circuit, measurement, shots: wrong_order).run([request])

    wrong_shots = ExecutionResult.from_counts(
        "wrong-shots",
        [Qubit(0)],
        3,
        1,
        {"0": 3},
        "backend",
    )
    with pytest.raises(ValueError, match="shots"):
        BackendSampler(lambda circuit, measurement, shots: wrong_shots).run([request])

    wrong_width = ExecutionResult.from_counts(
        "wrong-width",
        [Qubit(0), Qubit(1)],
        2,
        2,
        {"00": 2},
        "backend",
    )
    with pytest.raises(ValueError, match="width"):
        BackendSampler(lambda circuit, measurement, shots: wrong_width).run([request])

    filtered = ExecutionResult.from_counts(
        "filtered",
        [Qubit(0)],
        2,
        1,
        {"0": 1},
        "backend",
    )
    returned = BackendSampler(lambda circuit, measurement, shots: filtered).run(
        [request]
    )
    assert returned == [filtered]
