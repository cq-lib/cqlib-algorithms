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

"""Component fidelity and overlap against independent state and matrix references."""

import math

import numpy as np
import pytest
from cqlib import Circuit, Parameter

from cqlib_algorithms.operators import (
    QFT,
    ArbitraryStatePreparation,
    BasisStateInitialization,
    PauliMeasurementPlanner,
    RealAmplitudesAnsatz,
    StatevectorSampler,
    UniformSuperposition,
    UserCircuitPreparation,
)
from cqlib_algorithms.operators.state_fidelity import (
    CircuitFidelityRequest,
    ComputeUncomputeFidelity,
    UnitaryOverlap,
)
from cqlib_algorithms.protocols import Sampler, StatePreparation


def test_unrelated_components_are_rejected_before_building(monkeypatch):
    source = QFT(1)
    reference = BasisStateInitialization(1, 0)

    def unexpected_build(*args, **kwargs):
        pytest.fail("an unrelated component must not be built implicitly")

    monkeypatch.setattr(QFT, "build_circuit", unexpected_build)
    with pytest.raises(TypeError):
        UnitaryOverlap(source, reference)
    with pytest.raises(TypeError):
        CircuitFidelityRequest(reference, source, 32)
    with pytest.raises(TypeError):
        ComputeUncomputeFidelity(StatevectorSampler()).run(source, reference, shots=32)
    with pytest.raises(TypeError):
        PauliMeasurementPlanner(source, [])
    with pytest.raises(TypeError):
        UserCircuitPreparation.from_component(source, {})


class _CachedPreparation(StatePreparation):
    def __init__(self, circuit, *, width=None, slots=None):
        self.circuit = circuit
        self.width = circuit.num_qubits if width is None else width
        self.slots = len(circuit.used_symbols) if slots is None else slots
        self.builds = 0

    def validate(self):
        pass

    def num_qubits(self):
        return self.width

    def num_parameters(self):
        return self.slots

    def build_circuit(self, prefix):
        self.builds += 1
        return self.circuit


@pytest.mark.parametrize(
    "first, second, probability",
    [
        (UniformSuperposition(2), UniformSuperposition(2), 1.0),
        (BasisStateInitialization(2, 1), BasisStateInitialization(2, 2), 0.0),
        (UniformSuperposition(1), BasisStateInitialization(1, 0), 0.5),
        (
            ArbitraryStatePreparation([1 / math.sqrt(2), 1j / math.sqrt(2)]),
            UniformSuperposition(1),
            0.5,
        ),
    ],
)
def test_direct_component_fidelity_matches_independent_state_overlap(
    first, second, probability
):
    result = ComputeUncomputeFidelity(StatevectorSampler()).run(
        first, second, shots=8192, seed=17
    )
    assert result.fidelity == pytest.approx(probability, abs=0.035)
    assert result.variance == pytest.approx(
        result.fidelity * (1 - result.fidelity) / 8191
    )


def test_ansatz_pair_uses_independent_parameters_and_matches_legacy_batch():
    component = RealAmplitudesAnsatz(1, reps=0)
    options = {
        "parameter_bindings_a": {"a_0": 0.3},
        "parameter_bindings_b": {"b_0": 1.2},
    }
    evaluator = ComputeUncomputeFidelity(StatevectorSampler())
    direct = evaluator.run(component, component, shots=8192, seed=91, **options)
    legacy = CircuitFidelityRequest(
        component.build_circuit("a"),
        component.build_circuit("b"),
        8192,
        options["parameter_bindings_a"],
        options["parameter_bindings_b"],
        91,
    )
    request = CircuitFidelityRequest(component, component, 8192, seed=91, **options)
    assert evaluator.run([request, legacy]) == [direct, direct]
    assert direct.fidelity == pytest.approx(math.cos((0.3 - 1.2) / 2) ** 2, abs=0.035)
    assert evaluator.run([]) == []


def test_same_explicit_symbol_can_have_different_values_on_each_fidelity_side():
    circuit = Circuit(1)
    circuit.ry(0, Parameter("shared"))
    component = _CachedPreparation(circuit)
    result = ComputeUncomputeFidelity(StatevectorSampler()).run(
        component,
        component,
        shots=32,
        parameter_bindings_a={"shared": 0.0},
        parameter_bindings_b={"shared": math.pi},
    )
    assert result.fidelity == 0.0
    assert component.builds == 2
    assert circuit.used_symbols == ["shared"]
    with pytest.raises(ValueError, match="disjoint"):
        UnitaryOverlap(component, component)


def test_overlap_preserves_noncommuting_product_and_global_phase():
    first = Circuit(1)
    first.h(0)
    first.set_global_phase(0.37)
    second = Circuit(1)
    second.rz(0, 0.8)
    second.set_global_phase(-0.29)
    actual = UnitaryOverlap(_CachedPreparation(first), _CachedPreparation(second))
    expected = second.to_matrix().conj().T @ first.to_matrix()
    np.testing.assert_allclose(actual.build_circuit().to_matrix(), expected, atol=1e-14)


def test_symbolic_overlap_builds_both_ansatzes_without_implicit_parameter_sharing():
    component = RealAmplitudesAnsatz(1, reps=0)
    overlap = UnitaryOverlap(component, component, prefix_a="left", prefix_b="right")
    circuit = overlap.build_circuit()
    assert set(circuit.used_symbols) == {"left_0", "right_0"}
    matrix = circuit.assign_parameters({"left_0": 0.2, "right_0": 0.9}).to_matrix()
    angle = 0.2 - 0.9
    expected = np.array(
        [
            [math.cos(angle / 2), -math.sin(angle / 2)],
            [math.sin(angle / 2), math.cos(angle / 2)],
        ]
    )
    np.testing.assert_allclose(matrix, expected, atol=1e-14)


def test_component_registers_are_associated_by_insertion_position():
    first = Circuit([9, 2])
    first.x(9)
    second = Circuit([40, 6])
    second.x(40)
    left, right = _CachedPreparation(first), _CachedPreparation(second)
    evaluator = ComputeUncomputeFidelity(StatevectorSampler())
    assert evaluator.run(left, right, shots=32).fidelity == 1.0
    np.testing.assert_allclose(
        UnitaryOverlap(left, right).build_circuit().to_matrix(), np.eye(4), atol=1e-14
    )
    second.x(40)
    second.x(6)
    assert evaluator.run(left, right, shots=32).fidelity == 0.0


def test_cached_component_and_binding_snapshots_survive_source_mutation():
    circuit = Circuit(1)
    circuit.ry(0, Parameter("angle"))
    component = _CachedPreparation(circuit)
    bindings = {"angle": 0.0}
    request = CircuitFidelityRequest(
        component, BasisStateInitialization(1, 0), 32, bindings
    )
    overlap = UnitaryOverlap(component, BasisStateInitialization(1, 0))
    bindings["angle"] = math.pi
    circuit.x(0)
    evaluator = ComputeUncomputeFidelity(StatevectorSampler())
    assert evaluator.run([request])[0].fidelity == 1.0
    np.testing.assert_allclose(
        overlap.build_circuit().assign_parameters({"angle": 0.0}).to_matrix(),
        np.eye(2),
        atol=1e-14,
    )
    with pytest.raises(TypeError):
        request.parameter_bindings_a["angle"] = 2.0


@pytest.mark.parametrize("kind", ["width", "slots", "return", "reset"])
@pytest.mark.parametrize("surface", ["overlap", "request", "run"])
def test_invalid_component_contracts_are_rejected_before_sampling(kind, surface):
    circuit = Circuit(1)
    if kind == "reset":
        circuit.reset(0)
    if kind == "slots":
        circuit.ry(0, Parameter("angle"))
    component = _CachedPreparation(
        circuit,
        width=2 if kind == "width" else 1,
        slots=0 if kind == "slots" else len(circuit.used_symbols),
    )
    if kind == "return":
        component.circuit = object()

    class NeverSampler(Sampler):
        def run(self, requests):
            pytest.fail("invalid components must fail before sampling")

    with pytest.raises((TypeError, ValueError)):
        if surface == "overlap":
            UnitaryOverlap(component, BasisStateInitialization(1, 0))
        elif surface == "request":
            CircuitFidelityRequest(component, BasisStateInitialization(1, 0), 32)
        else:
            ComputeUncomputeFidelity(NeverSampler()).run(
                component, BasisStateInitialization(1, 0), shots=32
            )


@pytest.mark.parametrize(
    "options",
    [
        {"shots": 0},
        {"shots": True},
        {"seed": -1},
        {"prefix_a": 1},
        {"parameter_bindings_a": {"unexpected": 0.1}},
    ],
)
def test_direct_fidelity_rejects_invalid_options(options):
    kwargs = {"shots": 16, **options}
    with pytest.raises((TypeError, ValueError)):
        ComputeUncomputeFidelity(StatevectorSampler()).run(
            UniformSuperposition(1), UniformSuperposition(1), **kwargs
        )


def test_single_and_batch_options_are_not_silently_ignored():
    evaluator = ComputeUncomputeFidelity(StatevectorSampler())
    request = CircuitFidelityRequest(
        UniformSuperposition(1), UniformSuperposition(1), 16
    )
    with pytest.raises(ValueError, match="requires"):
        evaluator.run(UniformSuperposition(1))
    with pytest.raises(ValueError, match="requires"):
        evaluator.run(UniformSuperposition(1), UniformSuperposition(1))
    with pytest.raises(ValueError, match="own"):
        evaluator.run([request], shots=16)
    with pytest.raises(ValueError, match="own"):
        evaluator.run([request], prefix_a="features")


def test_component_substitution_changes_the_actual_fidelity():
    evaluator = ComputeUncomputeFidelity(StatevectorSampler())
    reference = BasisStateInitialization(1, 0)
    assert (
        evaluator.run(reference, BasisStateInitialization(1, 0), shots=32).fidelity == 1
    )
    assert (
        evaluator.run(reference, BasisStateInitialization(1, 1), shots=32).fidelity == 0
    )


@pytest.mark.parametrize("surface", ["overlap", "request", "run"])
def test_component_width_mismatch_is_rejected(surface):
    first, second = UniformSuperposition(1), UniformSuperposition(2)
    with pytest.raises(ValueError, match="same"):
        if surface == "overlap":
            UnitaryOverlap(first, second)
        elif surface == "request":
            CircuitFidelityRequest(first, second, 32)
        else:
            ComputeUncomputeFidelity(StatevectorSampler()).run(first, second, shots=32)


def test_custom_prefixes_and_a_single_shot_preserve_their_semantics():
    ansatz = RealAmplitudesAnsatz(1, reps=0)
    result = ComputeUncomputeFidelity(StatevectorSampler()).run(
        ansatz,
        ansatz,
        shots=1,
        seed=3,
        prefix_a="left",
        prefix_b="right",
        parameter_bindings_a={"left_0": 0.4},
        parameter_bindings_b={"right_0": 0.4},
    )
    assert result.fidelity == 1
    assert result.variance is None
