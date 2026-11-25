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

"""Component-based measurement planning and grouped sampler execution."""

import math

import numpy as np
import pytest
from cqlib import Circuit, Parameter, Qubit
from cqlib.qis import PauliString

from cqlib_algorithms.operators import (
    QFT,
    BasisStateInitialization,
    PauliMeasurementPlanner,
    RealAmplitudesAnsatz,
    StatevectorSampler,
    UniformSuperposition,
    UnitaryDilation,
    UnitaryOverlap,
    UserCircuitPreparation,
)
from cqlib_algorithms.protocols import Sampler, SamplerRequest, StatePreparation


def test_preparation_component_can_be_planned_and_run_without_external_circuits():
    plan = PauliMeasurementPlanner(
        UniformSuperposition(1), [PauliString.from_str("X")]
    ).plan()
    result = plan.run(StatevectorSampler(), shots=64, seed=7)
    assert result[0].counts == {"0": 64}
    assert plan.groups[0].term_indices == (0,)


def test_parameterized_ansatz_can_be_bound_at_grouped_execution():
    plan = PauliMeasurementPlanner(
        RealAmplitudesAnsatz(1, reps=0),
        [PauliString.from_str("Z")],
        prefix="trial",
    ).plan()
    for angle, expected in [(0.0, "0"), (math.pi, "1")]:
        results = plan.run(
            StatevectorSampler(),
            shots=64,
            parameter_bindings={"trial_0": angle},
        )
        assert results[0].counts == {expected: 64}
    with pytest.raises(ValueError):
        plan.run(StatevectorSampler(), shots=64)


@pytest.mark.parametrize("grouping", ["none", "qwc"])
@pytest.mark.parametrize(
    "factory",
    [
        lambda: UniformSuperposition(2),
        lambda: UserCircuitPreparation(QFT(2).build_circuit()),
        lambda: UserCircuitPreparation(
            UnitaryOverlap(
                QFT(2).build_circuit(), BasisStateInitialization(2, 0)
            ).build_circuit()
        ),
        lambda: UserCircuitPreparation(
            UnitaryDilation(
                np.diag([0.4, -0.2]).astype(complex), alpha=1.0
            ).build_circuit()
        ),
        lambda: UserCircuitPreparation.from_component(
            RealAmplitudesAnsatz(2, reps=0),
            {"theta_0": math.pi / 2, "theta_1": 0.0},
        ),
    ],
)
def test_component_plans_match_legacy_basis_matrices_and_seeded_counts(
    factory, grouping
):
    component = factory()
    source = (
        component.build_circuit("theta")
        if isinstance(component, StatePreparation)
        else component.build_circuit()
    )
    terms = [PauliString.from_str(term) for term in ("IX", "YI", "IZ", "II")]
    direct = PauliMeasurementPlanner(component, terms, grouping=grouping).plan()
    legacy = PauliMeasurementPlanner(source, terms, grouping=grouping).plan()
    assert direct.groups == legacy.groups
    assert direct.identity_indices == legacy.identity_indices == (3,)
    for index in range(len(direct.groups)):
        np.testing.assert_allclose(
            direct.build_circuit(index).to_matrix(),
            legacy.build_circuit(index).to_matrix(),
            atol=1e-14,
        )
    sampler = StatevectorSampler()
    expected = sampler.run(
        [
            SamplerRequest(legacy.build_circuit(i), legacy.qubits, 256, 13)
            for i in range(len(legacy.groups))
        ]
    )
    actual = direct.run(sampler, shots=256, seed=13)
    assert [result.counts for result in actual] == [
        result.counts for result in expected
    ]


class _RecordingSampler(Sampler):
    def __init__(self):
        self.batches = []

    def run(self, requests):
        self.batches.append(list(requests))
        return StatevectorSampler().run(requests)


class _Preparation(StatePreparation):
    def __init__(self, circuit, *, width=None, slots=None):
        self.circuit = circuit
        self.width = circuit.num_qubits if width is None else width
        self.slots = len(circuit.used_symbols) if slots is None else slots
        self.builds = 0

    def validate(self):
        pass

    def num_parameters(self):
        return self.slots

    def num_qubits(self):
        return self.width

    def build_circuit(self, prefix):
        self.builds += 1
        return self.circuit


def test_group_order_identity_bookkeeping_and_single_batch_submission():
    terms = [PauliString.from_str(term) for term in ("X", "Z", "Y", "I", "-X")]
    plan = PauliMeasurementPlanner(
        UniformSuperposition(1), terms, grouping="none"
    ).plan()
    sampler = _RecordingSampler()
    results = plan.run(sampler, shots=128, seed=7)
    assert [group.term_indices for group in plan.groups] == [(0,), (1,), (2,), (4,)]
    assert plan.identity_indices == (3,)
    assert plan.term(4).phase == terms[4].phase
    assert len(sampler.batches) == 1
    assert len(results) == len(sampler.batches[0]) == 4
    assert all(
        request.seed == 7 and request.shots == 128 for request in sampler.batches[0]
    )
    assert results[0].counts == results[3].counts == {"0": 128}
    assert [r.counts for r in results] == [
        r.counts for r in plan.run(StatevectorSampler(), shots=128, seed=7)
    ]


def test_noncontiguous_component_qubits_keep_insertion_order_and_independent_snapshots():
    source = Circuit([Qubit(9), Qubit(2)])
    source.h(Qubit(9))
    source.set_global_phase(0.4)
    expected = source.to_matrix([9, 2]).copy()
    component = _Preparation(source)
    terms = [PauliString.from_str("IX"), PauliString.from_str("ZI")]
    planner = PauliMeasurementPlanner(component, terms)
    source.x(Qubit(2))
    terms[0].set_pauli(0, PauliString.from_str("Z").get_pauli(0))
    first, second = planner.plan(), planner.plan()
    assert first.qubits == (Qubit(9), Qubit(2))
    sampler = _RecordingSampler()
    assert first.run(sampler, shots=32)[0].counts == {"00": 32}
    assert sampler.batches[0][0].qubits == first.qubits
    assert component.builds == 1
    first.build_circuit(0).x(Qubit(9))
    assert second.run(StatevectorSampler(), shots=32)[0].counts == {"00": 32}
    h_matrix = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
    np.testing.assert_allclose(
        second.build_circuit(0).to_matrix([9, 2]),
        np.kron(np.eye(2), h_matrix) @ expected,
        atol=1e-14,
    )


def test_static_delay_keeps_measurement_specific_semantics_for_component_inputs():
    circuit = Circuit(1)
    circuit.delay(0, 1.0)
    plan = PauliMeasurementPlanner(
        _Preparation(circuit), [PauliString.from_str("Z")]
    ).plan()
    assert plan.run(StatevectorSampler(), shots=16)[0].counts == {"0": 16}
    assert any(operation.is_delay for operation in plan.build_circuit(0).operations)


@pytest.mark.parametrize("terms", [[], [PauliString.from_str("I")]])
def test_empty_or_identity_only_plan_avoids_backend_calls_but_validates_bindings(terms):
    plan = PauliMeasurementPlanner(RealAmplitudesAnsatz(1, reps=0), terms).plan()
    sampler = _RecordingSampler()
    assert plan.run(sampler, shots=16, parameter_bindings={"theta_0": 0.2}) == []
    assert sampler.batches == []
    with pytest.raises(ValueError):
        plan.run(sampler, shots=16)
    with pytest.raises(ValueError):
        plan.run(sampler, shots=0, parameter_bindings={"theta_0": 0.2})
    assert sampler.batches == []


@pytest.mark.parametrize(
    "options, error",
    [
        ({"parameter_bindings": {}}, ValueError),
        ({"parameter_bindings": {"other": 0.2}}, ValueError),
        ({"parameter_bindings": {"theta_0": math.nan}}, ValueError),
        ({"parameter_bindings": {"theta_0": math.inf}}, ValueError),
        ({"parameter_bindings": {"theta_0": True}}, TypeError),
        ({"parameter_bindings": {"theta_0": "bad"}}, TypeError),
        ({"parameter_bindings": []}, TypeError),
        ({"shots": 0}, ValueError),
        ({"shots": True}, TypeError),
        ({"seed": -1}, ValueError),
        ({"seed": True}, TypeError),
    ],
)
def test_bad_bindings_or_execution_options_fail_before_submission(options, error):
    plan = PauliMeasurementPlanner(
        RealAmplitudesAnsatz(1, reps=0), [PauliString.from_str("Z")]
    ).plan()
    sampler = _RecordingSampler()
    arguments = {"shots": 16, "parameter_bindings": {"theta_0": 0.2}, **options}
    with pytest.raises(error):
        plan.run(sampler, **arguments)
    assert sampler.batches == []
    with pytest.raises(TypeError, match="sampler"):
        plan.run(object(), shots=16, parameter_bindings={"theta_0": 0.2})


def test_binding_once_keeps_source_symbols_and_mapping_independent():
    circuit = Circuit(1)
    circuit.ry(0, Parameter("angle"))
    plan = PauliMeasurementPlanner(
        _Preparation(circuit), [PauliString.from_str("Z")]
    ).plan()
    sampler = _RecordingSampler()
    bindings = {"angle": math.pi}
    assert plan.run(sampler, shots=32, parameter_bindings=bindings)[0].counts == {
        "1": 32
    }
    bindings["angle"] = 0.0
    assert sampler.batches[0][0].circuit.used_symbols == []
    assert circuit.used_symbols == ["angle"]
    assert plan.build_circuit(0).used_symbols == ["angle"]
    assert plan.run(sampler, shots=32, parameter_bindings=bindings)[0].counts == {
        "0": 32
    }


@pytest.mark.parametrize(
    "kind", ["width", "slots", "return", "measurement", "reset", "nested"]
)
def test_invalid_preparation_metadata_and_operations_are_rejected(kind):
    circuit = Circuit(1)
    if kind == "slots":
        circuit.ry(0, Parameter("angle"))
    if kind == "measurement":
        circuit.measure(0)
    if kind == "reset":
        circuit.reset(0)
    if kind == "nested":
        inner = Circuit(1)
        inner.measure(0)
        circuit.append_circuit_gate(inner.to_gate("nested"), [0])
    component = _Preparation(circuit, width=2 if kind == "width" else 1, slots=0)
    if kind == "return":
        component.circuit = object()
    with pytest.raises((TypeError, ValueError)):
        PauliMeasurementPlanner(component, [PauliString.from_str("Z")])


@pytest.mark.parametrize("bad", [None, object(), "circuit"])
def test_invalid_input_types_are_rejected(bad):
    with pytest.raises(TypeError, match="StatePreparation"):
        PauliMeasurementPlanner(bad, [PauliString.from_str("Z")])
    with pytest.raises(TypeError, match="prefix"):
        PauliMeasurementPlanner(UniformSuperposition(1), [], prefix=1)


@pytest.mark.parametrize("response", [None, (), [], [object()]])
def test_sampler_result_contract_is_checked(response):
    class BrokenSampler(Sampler):
        def run(self, requests):
            return response

    plan = PauliMeasurementPlanner(
        UniformSuperposition(1), [PauliString.from_str("X")]
    ).plan()
    with pytest.raises((TypeError, ValueError)):
        plan.run(BrokenSampler(), shots=16)
