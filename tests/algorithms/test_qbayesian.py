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

"""Independent matrix, posterior, budget and execution-contract references."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from cqlib import Circuit, Parameter, Qubit
from cqlib.device import ExecutionResult

from cqlib_algorithms.algorithms.qbayesian import (
    QBayesian,
    QBayesianResult,
    QBayesianRound,
    QBayesianStatus,
)
from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.operators.state_preparation import ArbitraryStatePreparation
from cqlib_algorithms.protocols import Sampler


class ScriptedSampler(Sampler):
    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.calls = []

    def run(self, requests):
        self.calls.append(tuple(requests))
        assert len(requests) == 1
        request = requests[0]
        return [
            ExecutionResult.from_counts(
                "script",
                list(request.qubits),
                request.shots,
                len(request.qubits),
                self.scripts.pop(0),
            )
        ]


def model(sampler=None, circuit=None):
    return QBayesian(
        Circuit(2) if circuit is None else circuit,
        {"q": 0, "e": 1},
        ScriptedSampler([]) if sampler is None else sampler,
    )


@pytest.mark.reference
@pytest.mark.parametrize("power", [0, 1, 2, 4])
def test_one_qubit_amplification_matches_independent_full_matrix(power):
    angle = math.pi / 3
    preparation = Circuit(1)
    preparation.ry(0, angle)
    algorithm = QBayesian(preparation, {"x": 0}, ScriptedSampler([]))
    a = np.array(
        [
            [math.cos(angle / 2), -math.sin(angle / 2)],
            [math.sin(angle / 2), math.cos(angle / 2)],
        ]
    )
    psi = a[:, 0]
    reflection = 2 * np.outer(psi, psi) - np.eye(2)
    oracle = np.diag([1, -1])
    expected = np.linalg.matrix_power(reflection @ oracle, power) @ a
    actual = algorithm.build_circuit({"x": 1}, grover_power=power).to_matrix()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    if power in (0, 1):
        assert abs(actual[1, 0]) ** 2 == pytest.approx([0.25, 1][power], abs=1e-12)


@pytest.mark.reference
@pytest.mark.parametrize("power", [0, 1, 2])
def test_complex_joint_conjunction_hidden_marginal_and_noncontiguous_positions(power):
    weights = np.array([1, 3, 5, 2, 7, 1, 2, 4, 3, 8, 2, 6, 1, 9, 5, 4], dtype=float)
    probabilities = weights / weights.sum()
    psi = np.sqrt(probabilities) * np.exp(1j * np.arange(16) * 0.37)
    synthesized = ArbitraryStatePreparation(psi.tolist()).build_circuit("")
    original = Circuit([7, 2, 11, 4])
    original.compose(synthesized, original.qubits)
    algorithm = QBayesian(original, {"a": 0, "b": 1, "c": 3}, ScriptedSampler([]))
    good = np.array([bool(index & 2) and not bool(index & 8) for index in range(16)])
    oracle = np.diag(np.where(good, -1, 1))
    reflection = 2 * np.outer(psi, psi.conj()) - np.eye(16)
    expected_state = np.linalg.matrix_power(reflection @ oracle, power) @ psi
    actual = algorithm.build_circuit({"c": 0, "b": 1}, grover_power=power).to_matrix()[
        :, 0
    ]
    np.testing.assert_allclose(actual, expected_state, atol=1e-12, rtol=0)
    # The unnamed position 2 is summed out; evidence is a conjunction.
    observed = np.abs(actual[good]) ** 2
    conditional = observed / observed.sum()
    np.testing.assert_allclose(
        conditional, probabilities[good] / probabilities[good].sum(), atol=1e-12
    )
    assert original.qubits == [Qubit(7), Qubit(2), Qubit(11), Qubit(4)]


def test_decode_query_order_evidence_overlap_and_unmeasured_hidden_qubit():
    sampler = ScriptedSampler([{"001": 2, "011": 3, "101": 4, "111": 1}])
    algorithm = QBayesian(Circuit([7, 2, 11, 4]), {"a": 0, "b": 1, "c": 3}, sampler)
    result = algorithm.infer(
        ["c", "a", "b"],
        {"b": 1, "c": 0},
        target_accepted=4,
        shots_per_round=10,
        amplification_powers=[0],
    )
    assert sampler.calls[0][0].qubits == (Qubit(0), Qubit(1), Qubit(3))
    assert result.counts == {(0, 1, 1): 3}
    assert result.probabilities == {(0, 1, 1): 1.0}
    assert result.status is QBayesianStatus.BUDGET_EXHAUSTED


def test_partial_rounds_overshoot_stop_and_query_counters():
    sampler = ScriptedSampler(
        [
            {"00": 10},
            {"00": 8, "10": 2},
            {"00": 5, "10": 2, "11": 3},
        ]
    )
    result = model(sampler).infer(
        ["q"],
        {"e": 1},
        target_accepted=4,
        shots_per_round=10,
        amplification_powers=[0, 2, 2, 0],
    )
    assert len(sampler.calls) == 3
    assert result.accepted == 7
    assert result.counts == {(0,): 4, (1,): 3}
    assert result.probabilities == {(0,): 4 / 7, (1,): 3 / 7}
    assert result.attempted_shots == 30
    assert result.num_grover_queries == 40
    assert result.num_state_preparation_queries == 110
    assert result.rounds == (
        QBayesianRound(0, 10, 0),
        QBayesianRound(2, 10, 2),
        QBayesianRound(2, 10, 5),
    )
    assert result.status is QBayesianStatus.TARGET_REACHED
    assert not hasattr(result, "evidence_probability")


def test_partial_budget_exhaustion_is_an_empirical_estimate():
    result = model(ScriptedSampler([{"10": 2, "00": 3}])).infer(
        ["q"],
        {"e": True},
        target_accepted=3,
        shots_per_round=5,
        amplification_powers=[0],
    )
    assert result.probabilities == {(0,): 1.0}
    assert result.status is QBayesianStatus.BUDGET_EXHAUSTED


def test_impossible_evidence_consumes_exact_schedule_without_claiming_impossibility():
    result = model(StatevectorSampler()).infer(
        ["q"],
        {"e": 1},
        amplification_powers=[0, 1, 2],
        shots_per_round=7,
    )
    assert result.accepted == 0
    assert result.counts == {}
    assert result.probabilities is None
    assert result.attempted_shots == 21
    assert result.num_grover_queries == 21
    assert result.status is QBayesianStatus.BUDGET_EXHAUSTED


def test_empty_evidence_skips_grover_and_uses_effective_budget(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("empty evidence must not construct Grover")

    monkeypatch.setattr(
        "cqlib_algorithms.algorithms.qbayesian.GroverOperator", forbidden
    )
    sampler = ScriptedSampler([{"0": 2, "1": 3}])
    result = model(sampler).infer(
        ["q"],
        target_accepted=4,
        shots_per_round=5,
        amplification_powers=[32, 32],
        max_grover_queries=1,
    )
    assert result.probabilities == {(0,): 0.4, (1,): 0.6}
    assert result.num_grover_queries == 0
    assert result.num_state_preparation_queries == 5
    assert result.rounds == (QBayesianRound(0, 5, 5),)


@pytest.mark.integration
def test_real_sampler_deterministic_conditional_outcome_and_p_one():
    circuit = Circuit(2)
    circuit.x(0)
    circuit.x(1)
    result = model(StatevectorSampler(), circuit).infer(
        ["e", "q"],
        {"e": 1},
        target_accepted=7,
        shots_per_round=8,
        amplification_powers=[1],
    )
    assert result.counts == {(1, 1): 8}
    assert result.status is QBayesianStatus.TARGET_REACHED


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_accepted": 0},
        {"target_accepted": True},
        {"shots_per_round": 0},
        {"shots_per_round": 1.2},
        {"max_rounds": 0},
        {"max_total_shots": False},
        {"max_grover_queries": 0},
        {"max_grover_power": -1},
        {"max_grover_power": True},
        {"amplification_powers": []},
        {"amplification_powers": "012"},
        {"amplification_powers": iter([0])},
        {"amplification_powers": [True]},
        {"amplification_powers": [-1]},
        {"amplification_powers": [0, 1.2]},
        {"amplification_powers": [0, 1025]},
        {"amplification_powers": [0, 0], "max_rounds": 1},
        {"amplification_powers": [0, 0], "max_total_shots": 511},
        {"amplification_powers": [0, 4], "max_grover_queries": 1023},
    ],
)
def test_invalid_complete_schedule_rejected_before_any_call_or_composition(
    kwargs, monkeypatch
):
    algorithm = model()
    monkeypatch.setattr(
        QBayesian, "build_circuit", lambda *a, **k: pytest.fail("preflight failed")
    )
    with pytest.raises((TypeError, ValueError)):
        algorithm.infer(["q"], {"e": 1}, **kwargs)


@pytest.mark.parametrize(
    "query,evidence",
    [
        ([], None),
        ("q", None),
        (["q", "q"], None),
        (["missing"], None),
        ([1], None),
        ([""], None),
        (["q"], []),
        (["q"], {"missing": 1}),
        (["q"], {"e": 2}),
        (["q"], {"e": 0.5}),
        (["q"], {1: 0}),
    ],
)
def test_invalid_queries_and_evidence_do_not_execute(query, evidence):
    sampler = ScriptedSampler([])
    with pytest.raises((TypeError, ValueError)):
        model(sampler).infer(query, evidence)
    assert sampler.calls == []


@pytest.mark.parametrize(
    "mapping",
    [
        {},
        {"": 0},
        {1: 0},
        {"x": True},
        {"x": -1},
        {"x": 2},
        {"x": 0.5},
        {"x": 0, "y": 0},
        [],
    ],
)
def test_invalid_variable_mapping(mapping):
    with pytest.raises((TypeError, ValueError)):
        QBayesian(Circuit(2), mapping, ScriptedSampler([]))


@pytest.mark.parametrize("kind", ["symbol", "measure", "reset", "delay", "nested"])
def test_invalid_static_preparation_envelope(kind):
    circuit = Circuit(1)
    if kind == "symbol":
        circuit.ry(0, Parameter("x"))
    elif kind == "measure":
        circuit.measure(0)
    elif kind == "reset":
        circuit.reset(0)
    elif kind == "delay":
        circuit.delay(0, 1.0)
    else:
        inner = Circuit(1)
        inner.measure(0)
        circuit.compose(inner, circuit.qubits)
    with pytest.raises(ValueError):
        QBayesian(circuit, {"q": 0}, ScriptedSampler([]))


def test_snapshot_mutation_isolation_and_barriers():
    source = Circuit(2)
    source.barrier([0, 1])
    mapping = {"q": 0, "e": 1}
    algorithm = QBayesian(source, mapping, ScriptedSampler([]))
    source.x(0)
    mapping["q"] = 1
    exposed = algorithm.state_preparation
    exposed.x(1)
    np.testing.assert_allclose(
        algorithm.build_circuit().to_matrix(), np.eye(4), atol=1e-12
    )
    assert algorithm.variable_qubits == {"q": 0, "e": 1}
    with pytest.raises(TypeError):
        algorithm.variable_qubits["q"] = 1
    with pytest.raises(ValueError, match="max_grover_power"):
        algorithm.build_circuit({"e": 1}, grover_power=1025)
    assert algorithm.build_circuit(grover_power=0, max_grover_power=0).num_qubits == 2


@pytest.mark.parametrize(
    "kind",
    ["container", "length", "type", "order", "shots", "total", "string", "status"],
)
def test_raw_sampler_contract_is_validated_before_filtering(kind):
    class BadSampler(Sampler):
        def run(self, requests):
            request = requests[0]
            if kind == "container":
                return ()
            if kind == "length":
                return []
            if kind == "type":
                return [object()]
            if kind == "status":
                result = ExecutionResult(
                    "failed", list(request.qubits), request.shots, len(request.qubits)
                )
                result.fail("test failure", 1)
                return [result]
            return [
                ExecutionResult.from_counts(
                    "bad",
                    list(reversed(request.qubits))
                    if kind == "order"
                    else list(request.qubits),
                    3 if kind == "shots" else request.shots,
                    len(request.qubits),
                    {
                        "xx" if kind == "string" else "00": 1
                        if kind == "total"
                        else request.shots
                    },
                )
            ]

    with pytest.raises((TypeError, ValueError)):
        model(BadSampler()).infer(
            ["q"], {"e": 1}, amplification_powers=[0], shots_per_round=2
        )


def valid_result():
    return QBayesianResult(
        ("q",),
        {"e": 1},
        {(0,): 2},
        {(0,): 1.0},
        2,
        3,
        2,
        (QBayesianRound(1, 3, 2),),
        QBayesianStatus.TARGET_REACHED,
    )


def test_result_mapping_copy_frozen_dataclass_and_replace():
    source = {(0,): 2}
    result = replace(valid_result(), counts=source)
    source[(0,)] = 9
    assert result.counts == {(0,): 2}
    with pytest.raises(TypeError):
        result.counts[(0,)] = 9
    with pytest.raises(TypeError):
        result.evidence["e"] = 0
    with pytest.raises(TypeError):
        result.probabilities[(0,)] = 0
    with pytest.raises(FrozenInstanceError):
        result.accepted = 8
    assert not hasattr(result, "__dict__")


@pytest.mark.parametrize(
    "changes",
    [
        {"query": ()},
        {"counts": {(0, 1): 2}},
        {"counts": {(True,): 2}},
        {"counts": {(0,): 0}},
        {"counts": {(0,): 3}},
        {"probabilities": None},
        {"probabilities": {(0,): 0.5}},
        {"probabilities": {(1,): 1.0}},
        {"probabilities": {(0,): math.nan}},
        {"accepted": 1},
        {"attempted_shots": 4},
        {"status": QBayesianStatus.BUDGET_EXHAUSTED},
        {"status": "target_reached"},
        {"rounds": ()},
        {
            "rounds": (QBayesianRound(0, 3, 2), QBayesianRound(0, 3, 0)),
            "attempted_shots": 6,
        },
        {"evidence": {"q": 1}},
        {"evidence": {}},
    ],
)
def test_public_result_rejects_inconsistent_construction(changes):
    with pytest.raises((TypeError, ValueError)):
        replace(valid_result(), **changes)


@pytest.mark.parametrize(
    "values", [(-1, 2, 1), (True, 2, 1), (0, 0, 0), (0, 2, -1), (0, 2, 3)]
)
def test_round_validation(values):
    with pytest.raises((TypeError, ValueError)):
        QBayesianRound(*values)
