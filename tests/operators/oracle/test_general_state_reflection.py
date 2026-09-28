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

"""Tests for reflection about a circuit-prepared state."""

from __future__ import annotations

import copy
from dataclasses import is_dataclass
from typing import cast

import numpy as np
import pytest
from cqlib.circuit import ClassicalExpr

from cqlib import Circuit, Parameter
from cqlib_algorithms.operators.oracle import GeneralStateReflection
from cqlib_algorithms.protocols import StatePreparation


def test_reflection_is_a_slotted_encapsulation_with_read_only_public_views():
    reflection = GeneralStateReflection(Circuit(1))
    assert not is_dataclass(reflection)
    assert not hasattr(reflection, "__dict__")
    with pytest.raises(AttributeError):
        reflection.state_preparation = Circuit(2)
    with pytest.raises(AttributeError):
        reflection.num_qubits = 2
    derived = GeneralStateReflection(reflection.state_preparation)
    np.testing.assert_allclose(
        derived.build_circuit().to_matrix(),
        reflection.build_circuit().to_matrix(),
        rtol=0.0,
        atol=1e-12,
    )


class _Preparation(StatePreparation):
    def __init__(
        self,
        circuit: Circuit,
        *,
        num_qubits: object | None = None,
        num_parameters: object = 0,
    ) -> None:
        self.circuit = circuit
        self.declared_qubits = circuit.num_qubits if num_qubits is None else num_qubits
        self.declared_parameters = num_parameters
        self.prefixes: list[str] = []
        self.validated = False

    def validate(self) -> None:
        self.validated = True

    def build_circuit(self, prefix: str) -> Circuit:
        self.prefixes.append(prefix)
        return copy.deepcopy(self.circuit)

    def num_parameters(self) -> int:
        return cast(int, self.declared_parameters)

    def num_qubits(self) -> int:
        return cast(int, self.declared_qubits)


class _StructuralPreparation:
    """Provide the right methods without joining the nominal ABC hierarchy."""

    def __init__(self, circuit: Circuit) -> None:
        self.circuit = circuit

    def validate(self) -> None:
        return None

    def build_circuit(self, prefix: str) -> Circuit:
        return copy.deepcopy(self.circuit)

    def num_parameters(self) -> int:
        return 0

    def num_qubits(self) -> int:
        return int(self.circuit.num_qubits)


def _zero_reflection_reference(num_qubits: int) -> np.ndarray:
    result = -np.eye(1 << num_qubits, dtype=np.complex128)
    result[0, 0] = 1.0
    return result


def _general_reflection_reference(preparation: np.ndarray) -> np.ndarray:
    zero_reflection = _zero_reflection_reference(
        (preparation.shape[0]).bit_length() - 1
    )
    return preparation @ zero_reflection @ preparation.conj().T


@pytest.mark.parametrize("kind", ["hadamard", "rotation", "entangled"])
def test_general_reflection_matches_independent_matrix_formula(kind):
    if kind == "hadamard":
        preparation = Circuit(1)
        preparation.h(0)
    elif kind == "rotation":
        preparation = Circuit(1)
        preparation.ry(0, 0.731)
    else:
        preparation = Circuit(2)
        preparation.h(0)
        preparation.cx(0, 1)

    preparation_matrix = preparation.to_matrix()
    actual = GeneralStateReflection(preparation).build_circuit().to_matrix()

    np.testing.assert_allclose(
        actual,
        _general_reflection_reference(preparation_matrix),
        rtol=0.0,
        atol=1e-12,
    )


def test_preparation_global_phase_cancels_without_changing_reflection():
    with_phase = Circuit(1)
    with_phase.ry(0, 0.4)
    with_phase.set_global_phase(0.917)
    without_phase = Circuit(1)
    without_phase.ry(0, 0.4)

    actual = GeneralStateReflection(with_phase).build_circuit().to_matrix()
    expected = GeneralStateReflection(without_phase).build_circuit().to_matrix()

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_parameterized_preparation_can_be_bound_after_building_reflection():
    theta = Parameter("theta")
    preparation = Circuit(1)
    preparation.ry(0, theta)
    reflection = GeneralStateReflection(preparation).build_circuit()

    value = 0.623
    actual = reflection.assign_parameters({"theta": value}).to_matrix()
    bound_preparation = preparation.assign_parameters({"theta": value}).to_matrix()
    expected = _general_reflection_reference(bound_preparation)

    assert reflection.used_symbols == ["theta"]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_non_contiguous_source_ids_are_normalized_by_insertion_order():
    preparation = Circuit([7, 2])
    preparation.h(7)
    preparation.cx(7, 2)
    source_matrix = preparation.to_matrix([7, 2])

    reflection = GeneralStateReflection(preparation)
    actual = reflection.build_circuit().to_matrix([0, 1])

    assert [qubit.index for qubit in reflection.state_preparation.qubits] == [0, 1]
    np.testing.assert_allclose(
        actual,
        _general_reflection_reference(source_matrix),
        rtol=0.0,
        atol=1e-12,
    )


def test_constructor_defensively_copies_input_and_does_not_modify_it():
    preparation = Circuit(1)
    preparation.h(0)
    snapshot = copy.deepcopy(preparation)
    expected = _general_reflection_reference(snapshot.to_matrix())

    reflection = GeneralStateReflection(preparation)
    assert preparation == snapshot
    preparation.x(0)

    np.testing.assert_allclose(
        reflection.build_circuit().to_matrix(), expected, rtol=0.0, atol=1e-12
    )


def test_exposed_preparation_and_build_results_are_independent_copies():
    preparation = Circuit(1)
    preparation.h(0)
    reflection = GeneralStateReflection(preparation)
    exposed = reflection.state_preparation
    first = reflection.build_circuit()
    second = reflection.build_circuit()

    exposed.x(0)
    first.x(0)

    assert first is not second
    assert len(reflection.state_preparation.operations) == 1
    assert len(second.operations) + 1 == len(first.operations)


def test_general_reflection_is_hermitian_unitary_and_involutory():
    preparation = Circuit(2)
    preparation.ry(0, 0.2)
    preparation.rx(1, -0.7)
    preparation.cx(0, 1)
    matrix = GeneralStateReflection(preparation).build_circuit().to_matrix()
    identity = np.eye(4)

    np.testing.assert_allclose(matrix, matrix.conj().T, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(matrix.conj().T @ matrix, identity, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(matrix @ matrix, identity, rtol=0.0, atol=1e-12)


def test_barriers_are_allowed_and_preserved_through_inverse():
    preparation = Circuit(1)
    preparation.h(0)
    preparation.barrier([0])

    reflection = GeneralStateReflection(preparation).build_circuit()

    np.testing.assert_allclose(
        reflection.to_matrix(), np.array([[0.0, 1.0], [1.0, 0.0]]), rtol=0.0, atol=1e-12
    )
    assert [operation.name for operation in reflection.operations].count("barrier") == 2


@pytest.mark.parametrize("operation", ["measure", "reset", "delay", "control"])
def test_non_invertible_top_level_operations_are_rejected(operation):
    preparation = Circuit(1)
    if operation == "measure":
        preparation.measure(0)
    elif operation == "reset":
        preparation.reset(0)
    elif operation == "delay":
        preparation.delay(0, 1.0)
    else:
        preparation.if_(ClassicalExpr.bool_literal(True), lambda body: body.x(0))

    with pytest.raises(ValueError, match="only valid invertible operations"):
        GeneralStateReflection(preparation)


def test_non_invertible_nested_circuit_gate_is_rejected():
    nested = Circuit(1)
    nested.reset(0)
    preparation = Circuit(1)
    preparation.append_circuit_gate(nested.to_gate("non_unitary"), [0])

    with pytest.raises(ValueError, match="only valid invertible operations"):
        GeneralStateReflection(preparation)


def test_general_reflection_rejects_wrong_type_and_empty_circuit():
    with pytest.raises(TypeError, match="state_preparation must be a Circuit"):
        GeneralStateReflection(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one qubit"):
        GeneralStateReflection(Circuit(0))


@pytest.mark.parametrize("prefix", ["", "state"])
def test_from_state_preparation_accepts_abc_component(prefix):
    circuit = Circuit(1)
    circuit.h(0)
    component = _Preparation(circuit)

    reflection = GeneralStateReflection.from_state_preparation(
        component,
        prefix=prefix,
    )

    assert component.validated is True
    assert component.prefixes == [prefix]
    np.testing.assert_allclose(
        reflection.build_circuit().to_matrix(),
        _general_reflection_reference(circuit.to_matrix()),
        rtol=0.0,
        atol=1e-12,
    )


def test_from_state_preparation_accepts_declared_parameter_slots():
    circuit = Circuit(1)
    circuit.ry(0, Parameter("data_0"))
    component = _Preparation(circuit, num_parameters=1)

    reflection = GeneralStateReflection.from_state_preparation(
        component,
        prefix="data",
    )

    assert reflection.build_circuit().used_symbols == ["data_0"]


@pytest.mark.parametrize("prefix", [None, 1, True])
def test_from_state_preparation_rejects_non_string_prefix(prefix):
    with pytest.raises(TypeError, match="prefix must be a string"):
        GeneralStateReflection.from_state_preparation(
            _Preparation(Circuit(1)),
            prefix=prefix,  # type: ignore[arg-type]
        )


def test_from_state_preparation_rejects_structural_only_component():
    component = _StructuralPreparation(Circuit(1))

    with pytest.raises(TypeError, match="must be a StatePreparation"):
        GeneralStateReflection.from_state_preparation(
            component,  # type: ignore[arg-type]
            prefix="state",
        )


@pytest.mark.parametrize("declared_qubits", [True, 1.5, "1"])
def test_from_state_preparation_rejects_invalid_qubit_count_types(declared_qubits):
    component = _Preparation(Circuit(1), num_qubits=declared_qubits)

    with pytest.raises(TypeError, match="num_qubits must be an integer"):
        GeneralStateReflection.from_state_preparation(
            component,
            prefix="state",
        )


def test_from_state_preparation_rejects_non_positive_or_mismatched_qubit_count():
    for declared_qubits in (0, -1):
        component = _Preparation(Circuit(1), num_qubits=declared_qubits)
        with pytest.raises(ValueError, match="greater than zero"):
            GeneralStateReflection.from_state_preparation(
                component,
                prefix="state",
            )

    with pytest.raises(ValueError, match="built circuit qubit count"):
        GeneralStateReflection.from_state_preparation(
            _Preparation(Circuit(1), num_qubits=2),
            prefix="state",
        )


@pytest.mark.parametrize("declared_parameters", [True, 1.5, "1"])
def test_from_state_preparation_rejects_invalid_parameter_count_types(
    declared_parameters,
):
    component = _Preparation(Circuit(1), num_parameters=declared_parameters)

    with pytest.raises(TypeError, match="num_parameters must be an integer"):
        GeneralStateReflection.from_state_preparation(
            component,
            prefix="state",
        )


def test_from_state_preparation_rejects_negative_or_too_small_parameter_count():
    with pytest.raises(ValueError, match="non-negative"):
        GeneralStateReflection.from_state_preparation(
            _Preparation(Circuit(1), num_parameters=-1),
            prefix="state",
        )

    circuit = Circuit(1)
    circuit.ry(0, Parameter("data"))
    with pytest.raises(ValueError, match="active symbol count"):
        GeneralStateReflection.from_state_preparation(
            _Preparation(circuit, num_parameters=0),
            prefix="state",
        )


def test_from_state_preparation_rejects_non_circuit_and_noninvertible_results():
    component = _Preparation(Circuit(1))
    component.build_circuit = lambda prefix: object()  # type: ignore[assignment]
    with pytest.raises(TypeError, match="must return a Circuit"):
        GeneralStateReflection.from_state_preparation(
            component,
            prefix="state",
        )

    noninvertible = Circuit(1)
    noninvertible.measure(0)
    with pytest.raises(ValueError, match="only valid invertible operations"):
        GeneralStateReflection.from_state_preparation(
            _Preparation(noninvertible),
            prefix="state",
        )


def test_from_state_preparation_propagates_validation_failure():
    component = _Preparation(Circuit(1))

    def fail_validation() -> None:
        raise ValueError("invalid component")

    component.validate = fail_validation  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="invalid component"):
        GeneralStateReflection.from_state_preparation(
            component,
            prefix="state",
        )
