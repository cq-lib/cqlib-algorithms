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

"""Independent and boundary checks for finite-shot local sampling."""

from __future__ import annotations

import math

import pytest
from cqlib import Circuit, Parameter, Qubit

from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.protocols.sampler import Sampler, SamplerRequest


def test_nominal_sampler_and_empty_batch() -> None:
    sampler = StatevectorSampler()

    assert isinstance(sampler, Sampler)
    assert sampler.run([]) == []


def test_basis_outcome_retains_original_noncontiguous_qubit_order() -> None:
    circuit = Circuit([7, 2])
    circuit.x(Qubit(7))
    requested = (Qubit(2), Qubit(7))
    initial_operations = len(circuit.operations)
    initial_classical_values = len(circuit.classical_values)

    result = StatevectorSampler().run([SamplerRequest(circuit, requested, 17)])[0]

    assert result.qubits == list(requested)
    assert result.num_qubits == 2
    assert result.shots == 17
    assert result.counts == {"10": 17}
    assert result.probabilities == {"10": 1.0}
    assert result.status.is_terminal()
    assert result.status.is_success()
    assert result.backend == "statevector"
    assert len(circuit.operations) == initial_operations
    assert len(circuit.classical_values) == initial_classical_values


def test_partial_measurement_projects_bell_state() -> None:
    circuit = Circuit([9, 4, 7])
    circuit.h(Qubit(9))
    circuit.cx(Qubit(9), Qubit(4))
    circuit.x(Qubit(7))
    shots = 4000
    request = SamplerRequest(circuit, (Qubit(7), Qubit(4)), shots)

    result = StatevectorSampler().run([request])[0]

    # The first measured qubit is the least-significant result bit.
    # Qubit 7 is fixed at 1; Qubit 4 is 0 or 1 with analytic probability 1/2.
    assert set(result.counts) <= {"01", "11"}
    assert sum(result.counts.values()) == shots
    assert abs(result.counts.get("11", 0) / shots - 0.5) < 0.05
    assert result.probabilities is not None
    assert math.isclose(sum(result.probabilities.values()), 1.0, abs_tol=1e-12)
    assert len(circuit.operations) == 3


def test_explicit_seed_reproduces_statevector_counts() -> None:
    circuit = Circuit(1)
    circuit.h(0)
    request = SamplerRequest(circuit, (Qubit(0),), 128, seed=2026)
    sampler = StatevectorSampler()

    first = sampler.run([request])[0]
    repeat = sampler.run([request])[0]

    assert first.counts == repeat.counts
    assert sum(first.counts.values()) == request.shots


def test_single_qubit_marginal_ignores_unmeasured_register() -> None:
    circuit = Circuit(2)
    circuit.x(1)
    result = StatevectorSampler().run([SamplerRequest(circuit, (Qubit(0),), 12)])[0]

    assert result.qubits == [Qubit(0)]
    assert result.num_qubits == 1
    assert result.counts == {"0": 12}


def test_batch_preserves_result_and_shot_order() -> None:
    zero = Circuit(1)
    one = Circuit([8])
    one.x(Qubit(8))
    requests = [
        SamplerRequest(zero, (Qubit(0),), 2),
        SamplerRequest(one, (Qubit(8),), 5),
    ]

    results = StatevectorSampler().run(requests)

    assert [result.counts for result in results] == [{"0": 2}, {"1": 5}]
    assert [result.shots for result in results] == [2, 5]
    assert [result.qubits for result in results] == [[Qubit(0)], [Qubit(8)]]


@pytest.mark.parametrize("operation", ["measure", "reset"])
def test_rejects_nonunitary_operations(operation: str) -> None:
    circuit = Circuit(1)
    if operation == "measure":
        circuit.measure(0)
    else:
        circuit.reset(0)

    request = SamplerRequest(circuit, (Qubit(0),), 5)
    with pytest.raises(ValueError, match="static unitary"):
        StatevectorSampler().run([request])


def test_rejects_late_nonunitary_or_symbolic_mutation() -> None:
    circuit = Circuit(1)
    request = SamplerRequest(circuit, (Qubit(0),), 3)
    circuit.rx(0, Parameter("late"))
    with pytest.raises(ValueError, match="symbolic"):
        StatevectorSampler().run([request])

    other = Circuit(1)
    other_request = SamplerRequest(other, (Qubit(0),), 3)
    other.measure(0)
    with pytest.raises(ValueError, match="static unitary"):
        StatevectorSampler().run([other_request])


def test_rejects_dynamic_control() -> None:
    circuit = Circuit(1)
    measurement = circuit.measure(0)
    circuit.if_(measurement.expr().bit_to_bool(), lambda body: body.x(0))
    request = SamplerRequest(circuit, (Qubit(0),), 4)

    with pytest.raises(ValueError, match="static unitary"):
        StatevectorSampler().run([request])


def test_rejects_wrong_batch_member() -> None:
    with pytest.raises(TypeError, match="SamplerRequest"):
        StatevectorSampler().run([object()])  # type: ignore[list-item]


def test_barrier_is_a_harmless_static_directive() -> None:
    circuit = Circuit(1)
    circuit.x(0)
    circuit.barrier([0])

    result = StatevectorSampler().run([SamplerRequest(circuit, (Qubit(0),), 7)])[0]

    assert result.counts == {"1": 7}


def test_static_circuit_gate_is_supported() -> None:
    inner = Circuit(1)
    inner.x(0)
    outer = Circuit([6])
    outer.append_circuit_gate(inner.to_gate("xsub"), [Qubit(6)])

    result = StatevectorSampler().run([SamplerRequest(outer, (Qubit(6),), 9)])[0]

    assert result.counts == {"1": 9}
    assert len(outer.operations) == 1


def test_rejects_reset_inside_circuit_gate() -> None:
    inner = Circuit(1)
    inner.reset(0)
    outer = Circuit([6])
    outer.append_circuit_gate(inner.to_gate("reset-sub"), [Qubit(6)])
    request = SamplerRequest(outer, (Qubit(6),), 9)

    with pytest.raises(ValueError, match="static unitary"):
        StatevectorSampler().run([request])
