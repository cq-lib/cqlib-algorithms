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

"""Golden, statistical and boundary checks for finite-shot estimation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from copy import copy

import pytest
from cqlib import Circuit, Parameter, Qubit
from cqlib.device import ExecutionResult
from cqlib.qis import Hamiltonian, PauliString, Phase
from cqlib.qis.state import Statevector

from cqlib_algorithms.operators.estimator import SamplingEstimator
from cqlib_algorithms.operators.estimator.sampling_estimator import (
    SamplingEstimator as ModuleSamplingEstimator,
)
from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.protocols.estimator import Estimator, EstimatorRequest
from cqlib_algorithms.protocols.sampler import Sampler, SamplerRequest, spawn_seeds


class ScriptedSampler(Sampler):
    """Return predefined raw counts while recording every flattened batch."""

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
            raise AssertionError("script count must match the flattened batch")
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


class CoreStatevectorSampler(Sampler):
    """Small independent test adapter for deterministic eigenstate checks."""

    def run(
        self,
        requests: Sequence[SamplerRequest],
        /,
    ) -> list[ExecutionResult]:
        results: list[ExecutionResult] = []
        for request in requests:
            request.validate()
            execution = copy(request.circuit)
            state = Statevector.from_circuit(execution)
            receipt = execution.measure_bits(list(request.qubits))
            sampled = state.sample(receipt, request.shots)
            results.append(
                ExecutionResult.from_counts(
                    sampled.task_id,
                    list(request.qubits),
                    request.shots,
                    len(request.qubits),
                    sampled.counts,
                    "test-statevector",
                )
            )
        return results


def test_local_export_abc_and_empty_batch_do_not_call_sampler() -> None:
    sampler = ScriptedSampler([])
    estimator = SamplingEstimator(sampler)

    assert SamplingEstimator is ModuleSamplingEstimator
    assert isinstance(estimator, Estimator)
    assert estimator.run([]) == []
    assert sampler.calls == []


def test_rejects_non_sampler_and_exact_request() -> None:
    with pytest.raises(TypeError, match="Sampler"):
        SamplingEstimator(object())  # type: ignore[arg-type]

    sampler = ScriptedSampler([])
    request = EstimatorRequest(Circuit(1), PauliString.from_str("Z"))
    with pytest.raises(ValueError, match="positive shot"):
        SamplingEstimator(sampler).run([request])
    assert sampler.calls == []


def test_x_y_z_basis_rotations_and_total_shot_distribution() -> None:
    observable = Hamiltonian(1)
    observable.add_term(PauliString.from_str("X"), 1.0)
    observable.add_term(PauliString.from_str("Y"), 2.0)
    observable.add_term(PauliString.from_str("Z"), -0.5)
    sampler = ScriptedSampler([{"0": 3}, {"0": 3}, {"0": 2}])

    result = SamplingEstimator(sampler).run(
        [EstimatorRequest(Circuit(1), observable, shots=8)]
    )[0]

    assert result.expectation == pytest.approx(2.5)
    assert result.variance == 0.0
    assert len(sampler.calls) == 1
    batch = sampler.calls[0]
    assert [request.shots for request in batch] == [3, 3, 2]
    assert [
        [operation.name for operation in request.circuit.operations]
        for request in batch
    ] == [
        ["H"],
        ["SDG", "H"],
        [],
    ]
    assert all(not request.circuit.classical_values for request in batch)


def test_estimator_assigns_reproducible_independent_term_seeds() -> None:
    observable = Hamiltonian(1)
    observable.add_term(PauliString.from_str("X"), 1.0)
    observable.add_term(PauliString.from_str("Z"), 1.0)
    sampler = ScriptedSampler([{"0": 2}, {"0": 2}])

    SamplingEstimator(sampler).run(
        [EstimatorRequest(Circuit(1), observable, shots=4, seed=2026)]
    )

    actual = [request.seed for request in sampler.calls[0]]
    expected = spawn_seeds(2026, 2)
    assert actual == expected
    assert len(set(actual)) == 2


def test_deterministic_pauli_eigenstates_match_quantum_reference() -> None:
    x_state = Circuit(1)
    x_state.h(0)
    y_state = Circuit(1)
    y_state.h(0)
    y_state.rz(0, math.pi / 2)
    minus_z = Circuit(1)
    minus_z.x(0)
    requests = [
        EstimatorRequest(x_state, PauliString.from_str("X"), shots=20),
        EstimatorRequest(y_state, PauliString.from_str("Y"), shots=20),
        EstimatorRequest(minus_z, PauliString.from_str("Z"), shots=20),
    ]

    results = SamplingEstimator(CoreStatevectorSampler()).run(requests)

    assert [result.expectation for result in results] == pytest.approx([1.0, 1.0, -1.0])
    assert [result.variance for result in results] == [0.0, 0.0, 0.0]


@pytest.mark.integration
def test_statevector_sampler_integration_preserves_noncontiguous_qubit_order() -> None:
    circuit = Circuit([7, 2])
    circuit.x(Qubit(2))
    request = EstimatorRequest(
        circuit,
        PauliString.from_str("ZI"),
        shots=20,
    )

    result = SamplingEstimator(StatevectorSampler()).run([request])[0]

    assert result.expectation == -1.0
    assert result.variance == 0.0


def test_fixed_counts_produce_unbiased_sample_mean_variance() -> None:
    sampler = ScriptedSampler([{"0": 3, "1": 1}])
    request = EstimatorRequest(
        Circuit(1),
        PauliString.from_str("Z"),
        shots=4,
    )

    result = SamplingEstimator(sampler).run([request])[0]

    assert result.expectation == pytest.approx(0.5)
    assert result.variance == pytest.approx((1.0 - 0.5**2) / 3.0)


def test_one_shot_variance_is_unknown_but_identity_variance_is_zero() -> None:
    sampled = SamplingEstimator(ScriptedSampler([{"1": 1}])).run(
        [EstimatorRequest(Circuit(1), PauliString.from_str("Z"), shots=1)]
    )[0]
    identity = SamplingEstimator(ScriptedSampler([])).run(
        [EstimatorRequest(Circuit(1), PauliString.from_str("I"), shots=1)]
    )[0]

    assert sampled.expectation == -1.0
    assert sampled.variance is None
    assert identity.expectation == 1.0
    assert identity.variance == 0.0


def test_pauli_phase_identity_and_duplicate_hamiltonian_terms() -> None:
    plus_z = PauliString.from_str("Z")
    minus_z = PauliString.from_str("Z")
    minus_z.phase = Phase.minus()
    imaginary_z = PauliString.from_str("Z")
    imaginary_z.phase = Phase.i()
    identity = PauliString.from_str("I")
    identity.phase = Phase.minus()

    observable = Hamiltonian(1)
    observable.add_term(plus_z, 0.75)
    observable.add_term(minus_z, -0.25)
    observable.add_term(imaginary_z, 1j)
    observable.add_term(plus_z, 1.0)
    observable.add_term(identity, 2.0)
    # Effective Z coefficient: 0.75 + 0.25 - 1 + 1 = 1.
    # Effective identity coefficient: -2.
    sampler = ScriptedSampler([{"0": 5}])

    result = SamplingEstimator(sampler).run(
        [EstimatorRequest(Circuit(1), observable, shots=5)]
    )[0]

    assert result.expectation == pytest.approx(-1.0)
    assert result.variance == 0.0
    assert len(sampler.calls[0]) == 1


def test_standalone_negative_pauli_phase_is_respected() -> None:
    observable = PauliString.from_str("X")
    observable.phase = Phase.minus()
    result = SamplingEstimator(ScriptedSampler([{"0": 3}])).run(
        [EstimatorRequest(Circuit(1), observable, shots=3)]
    )[0]

    assert result.expectation == -1.0


def test_empty_hamiltonian_and_zero_width_identity_need_no_samples() -> None:
    sampler = ScriptedSampler([])
    requests = [
        EstimatorRequest(Circuit(1), Hamiltonian(1), shots=2),
        EstimatorRequest(Circuit(0), PauliString(0), shots=2),
    ]

    results = SamplingEstimator(sampler).run(requests)

    assert [(result.expectation, result.variance) for result in results] == [
        (0.0, 0.0),
        (1.0, 0.0),
    ]
    assert sampler.calls == []


def test_insufficient_total_shots_is_rejected_before_sampling() -> None:
    observable = Hamiltonian(1)
    observable.add_term(PauliString.from_str("X"), 1.0)
    observable.add_term(PauliString.from_str("Z"), 1.0)
    sampler = ScriptedSampler([])

    with pytest.raises(ValueError, match="at least"):
        SamplingEstimator(sampler).run(
            [EstimatorRequest(Circuit(1), observable, shots=1)]
        )
    assert sampler.calls == []


def test_parameter_binding_is_private_and_noncontiguous_order_is_little_endian() -> (
    None
):
    circuit = Circuit([7, 2])
    circuit.ry(Qubit(7), Parameter("theta"))
    sampler = ScriptedSampler([{"10": 4}])
    request = EstimatorRequest(
        circuit,
        PauliString.from_str("ZI"),
        {"theta": 0.2},
        shots=4,
    )

    result = SamplingEstimator(sampler).run([request])[0]

    assert result.expectation == -1.0
    assert circuit.used_symbols == ["theta"]
    submitted = sampler.calls[0][0]
    assert submitted.circuit.used_symbols == []
    assert submitted.qubits == (Qubit(7), Qubit(2))


def test_multiple_estimator_requests_use_one_flattened_sampler_batch() -> None:
    first = EstimatorRequest(
        Circuit(1),
        PauliString.from_str("Z"),
        shots=2,
    )
    second_observable = Hamiltonian(1)
    second_observable.add_term(PauliString.from_str("X"), 2.0)
    second_observable.add_term(PauliString.from_str("Z"), -1.0)
    second = EstimatorRequest(Circuit(1), second_observable, shots=4)
    sampler = ScriptedSampler([{"0": 2}, {"0": 2}, {"1": 2}])

    results = SamplingEstimator(sampler).run([first, second])

    assert [result.expectation for result in results] == pytest.approx([1.0, 3.0])
    assert len(sampler.calls) == 1
    assert [request.shots for request in sampler.calls[0]] == [2, 2, 2]


@pytest.mark.parametrize("directive", ["measure", "reset", "delay"])
def test_nonunitary_preparation_is_rejected_before_sampling(directive: str) -> None:
    circuit = Circuit(1)
    if directive == "measure":
        circuit.measure(0)
    elif directive == "reset":
        circuit.reset(0)
    else:
        circuit.delay(0, 1.0)
    sampler = ScriptedSampler([])

    with pytest.raises(ValueError, match="unitary circuit"):
        SamplingEstimator(sampler).run(
            [EstimatorRequest(circuit, PauliString.from_str("Z"), shots=2)]
        )
    assert sampler.calls == []


def test_nested_measurement_is_normalized_to_component_value_error() -> None:
    inner = Circuit(1)
    inner.measure(0)
    outer = Circuit(1)
    outer.append_circuit_gate(inner.to_gate("hidden_measurement"), [0])
    request = EstimatorRequest(
        outer,
        PauliString.from_str("Z"),
        shots=2,
    )

    with pytest.raises(ValueError, match="decomposable unitary"):
        SamplingEstimator(ScriptedSampler([])).run([request])


def test_late_mutation_and_wrong_batch_member_are_revalidated() -> None:
    sampler = ScriptedSampler([])
    circuit = Circuit(1)
    request = EstimatorRequest(
        circuit,
        PauliString.from_str("Z"),
        shots=2,
    )
    circuit.ry(0, Parameter("late"))

    with pytest.raises(ValueError, match="missing"):
        SamplingEstimator(sampler).run([request])
    with pytest.raises(TypeError, match="EstimatorRequest"):
        SamplingEstimator(sampler).run([object()])  # type: ignore[list-item]


class InvalidSampler(Sampler):
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def run(
        self,
        requests: Sequence[SamplerRequest],
        /,
    ) -> list[ExecutionResult]:
        if self.mode == "missing":
            return []
        request = requests[0]
        if self.mode == "failed":
            result = ExecutionResult("failed", list(request.qubits), request.shots, 1)
            result.fail("failure", 1)
            return [result]
        if self.mode == "filtered":
            return [
                ExecutionResult.from_counts(
                    "filtered",
                    list(request.qubits),
                    request.shots,
                    len(request.qubits),
                    {"0": request.shots - 1},
                )
            ]
        raise AssertionError("unsupported test mode")


@pytest.mark.parametrize(
    ("mode", "message"),
    [("missing", "one result"), ("failed", "successful"), ("filtered", "sum")],
)
def test_invalid_sampler_results_are_rejected(mode: str, message: str) -> None:
    request = EstimatorRequest(
        Circuit(1),
        PauliString.from_str("Z"),
        shots=2,
    )

    with pytest.raises(ValueError, match=message):
        SamplingEstimator(InvalidSampler(mode)).run([request])
