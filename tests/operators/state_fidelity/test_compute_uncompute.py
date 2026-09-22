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

"""Contract, reference, integration, and boundary tests for circuit fidelity."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
from cqlib import Circuit, Parameter, Qubit
from cqlib.circuit import ClassicalExpr, ClassicalType
from cqlib.circuit.gates import UnitaryGate
from cqlib.device import ExecutionResult

from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.operators.state_fidelity import (
    CircuitFidelityRequest,
    CircuitFidelityResult,
    ComputeUncomputeFidelity,
)
from cqlib_algorithms.operators.state_fidelity.compute_uncompute import (
    ComputeUncomputeFidelity as ModuleComputeUncomputeFidelity,
)
from cqlib_algorithms.operators.state_fidelity.compute_uncompute import (
    _validate_sample_result,
)
from cqlib_algorithms.protocols.sampler import Sampler, SamplerRequest


class ScriptedSampler(Sampler):
    """Return fixed counts and record each complete submitted batch."""

    def __init__(self, scripts: Sequence[dict[str, int]]) -> None:
        self.scripts = list(scripts)
        self.calls: list[tuple[SamplerRequest, ...]] = []

    def run(
        self,
        requests: Sequence[SamplerRequest],
        /,
    ) -> list[ExecutionResult]:
        batch = tuple(requests)
        self.calls.append(batch)
        if len(batch) != len(self.scripts):
            raise AssertionError("script count must match the submitted batch")
        return [
            ExecutionResult.from_counts(
                f"script-{index}",
                list(request.qubits),
                request.shots,
                len(request.qubits),
                counts,
                "scripted",
            )
            for index, (request, counts) in enumerate(zip(batch, self.scripts))
        ]


def test_request_is_shallow_frozen_slotted_and_copies_bindings() -> None:
    theta = Parameter("theta")
    circuit_a = Circuit(1)
    circuit_a.ry(0, theta)
    source_bindings = {"theta": 0.25}

    request = CircuitFidelityRequest(
        circuit_a,
        Circuit(1),
        10,
        source_bindings,
    )
    source_bindings["theta"] = 0.75

    assert request.parameter_bindings_a == {"theta": 0.25}
    assert isinstance(request.parameter_bindings_a["theta"], float)
    assert request.validate() is None
    assert not hasattr(request, "__dict__")
    with pytest.raises(TypeError):
        request.parameter_bindings_a["theta"] = 0.5  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        request.shots = 20  # type: ignore[misc]
    with pytest.raises(TypeError):
        hash(request)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"circuit_a": object()}, TypeError, "circuit_a"),
        ({"circuit_b": object()}, TypeError, "circuit_b"),
        ({"shots": True}, TypeError, "shots"),
        ({"shots": 1.5}, TypeError, "shots"),
        ({"shots": 0}, ValueError, "greater than zero"),
    ],
)
def test_request_rejects_invalid_nominal_fields(
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "circuit_a": Circuit(1),
        "circuit_b": Circuit(1),
        "shots": 2,
    }
    arguments.update(kwargs)
    with pytest.raises(error, match=message):
        CircuitFidelityRequest(**arguments)  # type: ignore[arg-type]


def test_request_rejects_zero_or_different_circuit_widths() -> None:
    with pytest.raises(ValueError, match="at least one"):
        CircuitFidelityRequest(Circuit(0), Circuit(0), 2)
    with pytest.raises(ValueError, match="same qubit count"):
        CircuitFidelityRequest(Circuit(1), Circuit(2), 2)


@pytest.mark.parametrize(
    ("bindings", "error", "message"),
    [
        ([], TypeError, "mapping"),
        ({1: 0.2}, TypeError, "names"),
        ({"theta": True}, TypeError, "real numbers"),
        ({"theta": 0.2j}, TypeError, "real numbers"),
        ({"theta": math.nan}, ValueError, "finite"),
        ({"theta": math.inf}, ValueError, "finite"),
    ],
)
def test_request_rejects_invalid_binding_values(
    bindings: object,
    error: type[Exception],
    message: str,
) -> None:
    circuit = Circuit(1)
    circuit.ry(0, Parameter("theta"))
    with pytest.raises(error, match=message):
        CircuitFidelityRequest(
            circuit,
            Circuit(1),
            2,
            bindings,  # type: ignore[arg-type]
        )


def test_request_requires_exact_bindings_for_each_circuit() -> None:
    circuit_a = Circuit(1)
    circuit_a.rx(0, Parameter("alpha"))
    circuit_b = Circuit(1)
    circuit_b.ry(0, Parameter("beta"))

    with pytest.raises(ValueError, match=r"circuit_a.*missing"):
        CircuitFidelityRequest(circuit_a, circuit_b, 2, {}, {"beta": 0.1})
    with pytest.raises(ValueError, match=r"circuit_b.*unexpected"):
        CircuitFidelityRequest(
            circuit_a,
            circuit_b,
            2,
            {"alpha": 0.1},
            {"beta": 0.2, "extra": 0.3},
        )


@pytest.mark.parametrize(
    ("fidelity", "error"),
    [
        (True, TypeError),
        (1j, TypeError),
        (math.nan, ValueError),
        (math.inf, ValueError),
        (-0.1, ValueError),
        (1.1, ValueError),
    ],
)
def test_result_rejects_invalid_fidelity(
    fidelity: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        CircuitFidelityResult(fidelity)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("variance", "error"),
    [
        (True, TypeError),
        (1j, TypeError),
        (math.nan, ValueError),
        (math.inf, ValueError),
        (-0.1, ValueError),
    ],
)
def test_result_rejects_invalid_variance(
    variance: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        CircuitFidelityResult(0.5, variance)  # type: ignore[arg-type]


def test_result_normalizes_real_values_and_is_frozen() -> None:
    result = CircuitFidelityResult(1, 0)

    assert result.fidelity == 1.0
    assert isinstance(result.fidelity, float)
    assert result.variance == 0.0
    assert isinstance(result.variance, float)
    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.fidelity = 0.0  # type: ignore[misc]


def test_local_export_non_sampler_and_empty_batch() -> None:
    assert ComputeUncomputeFidelity is ModuleComputeUncomputeFidelity
    with pytest.raises(TypeError, match="Sampler"):
        ComputeUncomputeFidelity(object())  # type: ignore[arg-type]

    sampler = ScriptedSampler([])
    assert ComputeUncomputeFidelity(sampler).run([]) == []
    assert sampler.calls == []


@pytest.mark.integration
def test_identical_orthogonal_global_phase_and_entangled_states() -> None:
    identical = Circuit(1)
    identical.h(0)
    orthogonal = Circuit(1)
    orthogonal.x(0)
    phase_shifted = Circuit(1)
    phase_shifted.h(0)
    phase_shifted.set_global_phase(0.731)
    bell = Circuit(2)
    bell.h(0)
    bell.cx(0, 1)

    requests = [
        CircuitFidelityRequest(identical, identical, 16),
        CircuitFidelityRequest(Circuit(1), orthogonal, 16),
        CircuitFidelityRequest(identical, phase_shifted, 16),
        CircuitFidelityRequest(bell, bell, 16),
    ]
    results = ComputeUncomputeFidelity(StatevectorSampler()).run(requests)

    assert [result.fidelity for result in results] == [1.0, 0.0, 1.0, 1.0]
    assert [result.variance for result in results] == [0.0, 0.0, 0.0, 0.0]


@pytest.mark.integration
def test_barrier_is_accepted_as_a_nonsemantic_directive() -> None:
    circuit_a = Circuit(1)
    circuit_a.h(0)
    circuit_a.barrier([0])
    circuit_b = Circuit(1)
    circuit_b.h(0)

    result = ComputeUncomputeFidelity(StatevectorSampler()).run(
        [CircuitFidelityRequest(circuit_a, circuit_b, 8)]
    )[0]

    assert result == CircuitFidelityResult(1.0, 0.0)


def test_analytic_ry_overlap_bindings_and_variance() -> None:
    angle_a = 0.2
    angle_b = 1.4
    expected = math.cos((angle_a - angle_b) / 2.0) ** 2
    shots = 1000
    zero_counts = round(expected * shots)
    sampler = ScriptedSampler([{"0": zero_counts, "1": shots - zero_counts}])
    circuit_a = Circuit(1)
    circuit_a.ry(0, Parameter("alpha"))
    circuit_b = Circuit(1)
    circuit_b.ry(0, Parameter("beta"))

    result = ComputeUncomputeFidelity(sampler).run(
        [
            CircuitFidelityRequest(
                circuit_a,
                circuit_b,
                shots,
                {"alpha": angle_a},
                {"beta": angle_b},
            )
        ]
    )[0]

    sampled = zero_counts / shots
    assert result.fidelity == pytest.approx(sampled)
    assert sampled == pytest.approx(expected, abs=1.0 / shots)
    assert result.variance == pytest.approx(sampled * (1.0 - sampled) / 999)
    submitted = sampler.calls[0][0]
    assert submitted.circuit.used_symbols == []
    assert circuit_a.used_symbols == ["alpha"]
    assert circuit_b.used_symbols == ["beta"]

    def analytic_ry(angle: float) -> np.ndarray:
        half_angle = angle / 2.0
        return np.array(
            [
                [math.cos(half_angle), -math.sin(half_angle)],
                [math.sin(half_angle), math.cos(half_angle)],
            ],
            dtype=complex,
        )

    expected_matrix = analytic_ry(-angle_a) @ analytic_ry(angle_b)
    np.testing.assert_allclose(
        submitted.circuit.to_matrix(),
        expected_matrix,
        rtol=0.0,
        atol=1e-14,
    )


@pytest.mark.reference
def test_submitted_matrix_matches_independent_u_a_dagger_u_b_reference() -> None:
    circuit_a = Circuit(2)
    circuit_a.h(0)
    circuit_a.ry(1, 0.43)
    circuit_a.cx(0, 1)
    circuit_a.set_global_phase(0.17)
    circuit_b = Circuit(2)
    circuit_b.rx(0, -0.29)
    circuit_b.cz(0, 1)
    circuit_b.rz(1, 0.61)
    circuit_b.set_global_phase(-0.31)
    sampler = ScriptedSampler([{"00": 5}])

    ComputeUncomputeFidelity(sampler).run(
        [CircuitFidelityRequest(circuit_a, circuit_b, 5)]
    )

    submitted = sampler.calls[0][0].circuit.to_matrix()
    expected = circuit_a.to_matrix().conj().T @ circuit_b.to_matrix()
    np.testing.assert_allclose(submitted, expected, rtol=0.0, atol=1e-14)


def test_qubits_are_associated_by_insertion_position_and_compacted() -> None:
    circuit_a = Circuit([7, 2])
    circuit_a.x(Qubit(2))
    circuit_b = Circuit([11, 5])
    circuit_b.x(Qubit(5))
    sampler = ScriptedSampler([{"00": 4}])

    result = ComputeUncomputeFidelity(sampler).run(
        [CircuitFidelityRequest(circuit_a, circuit_b, 4)]
    )[0]

    submitted = sampler.calls[0][0]
    assert submitted.qubits == (Qubit(0), Qubit(1))
    assert result.fidelity == 1.0
    np.testing.assert_allclose(
        submitted.circuit.to_matrix(),
        np.eye(4, dtype=complex),
        rtol=0.0,
        atol=1e-14,
    )


def test_batch_is_flattened_once_preserves_order_and_inputs() -> None:
    first_a = Circuit(1)
    first_a.h(0)
    first_b = Circuit(1)
    first_b.x(0)
    second = Circuit(2)
    second.h(0)
    second.cx(0, 1)
    snapshots = [
        (len(first_a.operations), first_a.to_matrix().copy()),
        (len(first_b.operations), first_b.to_matrix().copy()),
        (len(second.operations), second.to_matrix().copy()),
    ]
    sampler = ScriptedSampler([{"0": 1}, {"00": 3}])

    results = ComputeUncomputeFidelity(sampler).run(
        [
            CircuitFidelityRequest(first_a, first_b, 1),
            CircuitFidelityRequest(second, second, 3),
        ]
    )

    assert len(sampler.calls) == 1
    assert [request.shots for request in sampler.calls[0]] == [1, 3]
    assert [(result.fidelity, result.variance) for result in results] == [
        (1.0, None),
        (1.0, 0.0),
    ]
    for circuit, (operation_count, matrix) in zip(
        (first_a, first_b, second), snapshots
    ):
        assert len(circuit.operations) == operation_count
        np.testing.assert_allclose(circuit.to_matrix(), matrix, rtol=0.0, atol=0.0)


def test_entire_batch_is_validated_before_sampler_call() -> None:
    first = CircuitFidelityRequest(Circuit(1), Circuit(1), 2)
    late_mutated = Circuit(1)
    second = CircuitFidelityRequest(late_mutated, Circuit(1), 2)
    late_mutated.ry(0, Parameter("late"))
    sampler = ScriptedSampler([])

    with pytest.raises(ValueError, match="missing"):
        ComputeUncomputeFidelity(sampler).run([first, second])
    assert sampler.calls == []


@pytest.mark.parametrize("operation", ["measure", "reset", "delay"])
def test_rejects_nonunitary_and_delayed_circuits(operation: str) -> None:
    circuit = Circuit(1)
    if operation == "measure":
        circuit.measure(0)
    elif operation == "reset":
        circuit.reset(0)
    else:
        circuit.delay(0, 1.0)

    with pytest.raises(ValueError, match="static unitary"):
        CircuitFidelityRequest(circuit, Circuit(1), 2)


def test_rejects_nested_nonunitary_and_core_noninvertible_unitary() -> None:
    inner = Circuit(1)
    inner.measure(0)
    nested = Circuit(1)
    nested.append_circuit_gate(inner.to_gate("hidden_measurement"), [0])
    with pytest.raises(ValueError, match="decomposable static unitary"):
        CircuitFidelityRequest(nested, Circuit(1), 2)

    unsupported = Circuit(3)
    unsupported.append_unitary_gate(
        UnitaryGate("identity-3q", 3).with_matrix(np.eye(8, dtype=complex)),
        [0, 1, 2],
    )
    with pytest.raises(ValueError, match="statically invertible"):
        CircuitFidelityRequest(unsupported, Circuit(3), 2)


@pytest.mark.parametrize("operation", ["classical_data", "classical_control"])
def test_rejects_real_core_classical_data_and_control(operation: str) -> None:
    circuit = Circuit(1)
    variable = circuit.var(ClassicalType.bool())
    if operation == "classical_data":
        circuit.store(variable, ClassicalExpr.bool_literal(True))
    else:
        circuit.if_(variable.expr(), lambda body: body.x(0))

    circuit.validate()
    assert circuit.operations[0].instruction_type == operation
    with pytest.raises(ValueError, match="static unitary"):
        CircuitFidelityRequest(circuit, Circuit(1), 2)


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        ({0: 2}, "width"),
        ({"00": 2}, "width"),
        ({"x": 2}, "binary"),
        ({"0": True, "1": 1}, "non-negative integers"),
        ({"0": 1.5, "1": 0.5}, "non-negative integers"),
        ({"0": -1, "1": 3}, "non-negative integers"),
    ],
)
def test_sample_result_validator_rejects_malformed_counts(
    counts: dict[object, object],
    message: str,
) -> None:
    request = SamplerRequest(Circuit(1), (Qubit(0),), 2)
    valid = ExecutionResult.from_counts(
        "valid",
        list(request.qubits),
        request.shots,
        len(request.qubits),
        {"0": request.shots},
        "test",
    )
    malformed = SimpleNamespace(
        status=valid.status,
        shots=valid.shots,
        num_qubits=valid.num_qubits,
        qubits=valid.qubits,
        counts=counts,
    )

    with pytest.raises(ValueError, match=message):
        _validate_sample_result(malformed, request)  # type: ignore[arg-type]


class InvalidSampler(Sampler):
    """Produce one deliberately contract-violating batch."""

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def run(
        self,
        requests: Sequence[SamplerRequest],
        /,
    ) -> list[ExecutionResult]:
        if self.mode == "missing":
            return []
        if self.mode == "wrong_type":
            return [object()]  # type: ignore[list-item]

        request = requests[0]
        if self.mode == "queued":
            return [
                ExecutionResult(
                    "queued",
                    list(request.qubits),
                    request.shots,
                    len(request.qubits),
                )
            ]
        if self.mode == "failed":
            result = ExecutionResult(
                "failed",
                list(request.qubits),
                request.shots,
                len(request.qubits),
            )
            result.fail("failure", 1)
            return [result]
        if self.mode == "shots":
            return [
                ExecutionResult.from_counts(
                    "shots",
                    list(request.qubits),
                    request.shots + 1,
                    len(request.qubits),
                    {"0": request.shots + 1},
                )
            ]
        if self.mode == "width":
            return [
                ExecutionResult.from_counts(
                    "width",
                    [Qubit(0), Qubit(1)],
                    request.shots,
                    2,
                    {"00": request.shots},
                )
            ]
        if self.mode == "qubits":
            return [
                ExecutionResult.from_counts(
                    "qubits",
                    [Qubit(9)],
                    request.shots,
                    1,
                    {"0": request.shots},
                )
            ]
        if self.mode == "total":
            return [
                ExecutionResult.from_counts(
                    "total",
                    list(request.qubits),
                    request.shots,
                    len(request.qubits),
                    {"0": request.shots - 1},
                )
            ]
        raise AssertionError("unsupported invalid-sampler mode")


@pytest.mark.parametrize(
    ("mode", "error", "message"),
    [
        ("missing", ValueError, "one result"),
        ("wrong_type", TypeError, "ExecutionResult"),
        ("queued", ValueError, "successful terminal"),
        ("failed", ValueError, "successful terminal"),
        ("shots", ValueError, "shots"),
        ("width", ValueError, "width"),
        ("qubits", ValueError, "qubits"),
        ("total", ValueError, "sum"),
    ],
)
def test_rejects_invalid_sampler_results(
    mode: str,
    error: type[Exception],
    message: str,
) -> None:
    request = CircuitFidelityRequest(Circuit(1), Circuit(1), 2)
    with pytest.raises(error, match=message):
        ComputeUncomputeFidelity(InvalidSampler(mode)).run([request])


def test_wrong_batch_member_is_rejected_before_sampling() -> None:
    sampler = ScriptedSampler([])
    with pytest.raises(TypeError, match="CircuitFidelityRequest"):
        ComputeUncomputeFidelity(sampler).run([object()])  # type: ignore[list-item]
    assert sampler.calls == []
