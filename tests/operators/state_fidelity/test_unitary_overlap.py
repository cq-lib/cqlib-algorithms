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

"""Reference, contract, boundary, and integration tests for unitary overlap."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from cqlib import Circuit, Parameter, Qubit
from cqlib.circuit import ClassicalExpr, ClassicalType
from cqlib.circuit.gates import UnitaryGate

from cqlib_algorithms.operators import UnitaryOverlap
from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.operators.state_fidelity import (
    CircuitFidelityRequest,
    ComputeUncomputeFidelity,
)
from cqlib_algorithms.operators.state_fidelity.unitary_overlap import (
    UnitaryOverlap as ModuleUnitaryOverlap,
)


def _one_qubit_pair() -> tuple[Circuit, Circuit]:
    first = Circuit(1)
    first.h(0)
    first.rz(0, 0.37)
    first.set_global_phase(0.19)
    second = Circuit(1)
    second.rx(0, -0.53)
    second.ry(0, 0.29)
    second.set_global_phase(-0.11)
    return first, second


def _two_qubit_pair() -> tuple[Circuit, Circuit]:
    first = Circuit(2)
    first.h(0)
    first.ry(1, 0.43)
    first.cx(0, 1)
    first.rz(0, -0.17)
    first.set_global_phase(0.31)
    second = Circuit(2)
    second.rx(1, 0.61)
    second.cz(0, 1)
    second.ry(0, -0.27)
    second.cx(1, 0)
    second.set_global_phase(-0.23)
    return first, second


def _three_qubit_pair() -> tuple[Circuit, Circuit]:
    first = Circuit(3)
    first.h(2)
    first.rx(0, 0.22)
    first.cx(2, 1)
    first.ry(1, -0.48)
    first.cz(0, 2)
    second = Circuit(3)
    second.rz(2, 0.72)
    second.h(0)
    second.cx(0, 1)
    second.rx(1, -0.34)
    second.cx(1, 2)
    return first, second


@pytest.mark.reference
@pytest.mark.parametrize(
    "factory",
    [_one_qubit_pair, _two_qubit_pair, _three_qubit_pair],
)
def test_matrix_matches_independent_numpy_product(
    factory: Callable[[], tuple[Circuit, Circuit]],
) -> None:
    unitary1, unitary2 = factory()
    first_matrix = unitary1.to_matrix()
    second_matrix = unitary2.to_matrix()

    actual = UnitaryOverlap(unitary1, unitary2).build_circuit().to_matrix()
    expected = second_matrix.conj().T @ first_matrix

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)


@pytest.mark.reference
def test_noncommuting_order_is_exactly_unitary2_adjoint_times_unitary1() -> None:
    unitary1 = Circuit(1)
    unitary1.h(0)
    unitary2 = Circuit(1)
    unitary2.rz(0, math.pi / 3.0)

    actual = UnitaryOverlap(unitary1, unitary2).build_circuit().to_matrix()
    expected = unitary2.to_matrix().conj().T @ unitary1.to_matrix()
    reversed_order = unitary1.to_matrix() @ unitary2.to_matrix().conj().T

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)
    assert not np.allclose(actual, reversed_order, rtol=0.0, atol=1e-10)


def test_global_phase_difference_is_preserved_exactly() -> None:
    unitary1 = Circuit(1)
    unitary1.h(0)
    unitary1.set_global_phase(0.73)
    unitary2 = Circuit(1)
    unitary2.x(0)
    unitary2.set_global_phase(-0.41)

    output = UnitaryOverlap(unitary1, unitary2).build_circuit()
    expected = unitary2.to_matrix().conj().T @ unitary1.to_matrix()

    np.testing.assert_allclose(output.to_matrix(), expected, rtol=0.0, atol=1e-14)


def test_parameterized_inputs_preserve_disjoint_symbols_and_bind_correctly() -> None:
    alpha = Parameter("alpha")
    beta = Parameter("beta")
    unitary1 = Circuit(2)
    unitary1.ry(0, alpha)
    unitary1.cx(0, 1)
    unitary2 = Circuit(2)
    unitary2.rx(1, beta)
    unitary2.cz(0, 1)
    component = UnitaryOverlap(unitary1, unitary2)

    output = component.build_circuit()
    assert output.used_symbols == ["alpha", "beta"]
    bound = output.assign_parameters({"alpha": 0.28, "beta": -0.67})
    first_bound = unitary1.assign_parameters({"alpha": 0.28})
    second_bound = unitary2.assign_parameters({"beta": -0.67})
    expected = second_bound.to_matrix().conj().T @ first_bound.to_matrix()

    np.testing.assert_allclose(bound.to_matrix(), expected, rtol=0.0, atol=1e-14)
    assert unitary1.used_symbols == ["alpha"]
    assert unitary2.used_symbols == ["beta"]


def test_shared_active_parameter_names_are_rejected() -> None:
    unitary1 = Circuit(1)
    unitary1.rx(0, Parameter("theta"))
    unitary2 = Circuit(1)
    unitary2.ry(0, Parameter("theta"))

    with pytest.raises(ValueError, match=r"disjoint.*shared=\['theta'\]"):
        UnitaryOverlap(unitary1, unitary2)


def test_qubits_are_compacted_and_associated_by_insertion_position() -> None:
    unitary1 = Circuit([7, 2])
    unitary1.h(Qubit(7))
    unitary1.cx(Qubit(7), Qubit(2))
    unitary2 = Circuit([11, 5])
    unitary2.ry(Qubit(11), 0.33)
    unitary2.cx(Qubit(5), Qubit(11))

    output = UnitaryOverlap(unitary1, unitary2).build_circuit()
    expected = unitary2.to_matrix([11, 5]).conj().T @ unitary1.to_matrix([7, 2])

    assert output.qubits == [Qubit(0), Qubit(1)]
    np.testing.assert_allclose(output.to_matrix(), expected, rtol=0.0, atol=1e-14)


def test_barrier_option_is_typed_nonsemantic_and_reported() -> None:
    unitary1, unitary2 = _two_qubit_pair()
    plain = UnitaryOverlap(unitary1, unitary2)
    separated = UnitaryOverlap(unitary1, unitary2, insert_barrier=True)
    output = separated.build_circuit()

    assert plain.insert_barrier is False
    assert separated.insert_barrier is True
    assert sum(operation.name == "barrier" for operation in output.operations) == 1
    np.testing.assert_allclose(
        output.to_matrix(),
        plain.build_circuit().to_matrix(),
        rtol=0.0,
        atol=1e-14,
    )


def test_inputs_are_snapshotted_and_each_build_is_independent() -> None:
    unitary1, unitary2 = _two_qubit_pair()
    component = UnitaryOverlap(unitary1, unitary2)
    expected = component.build_circuit().to_matrix().copy()

    unitary1.x(0)
    unitary2.h(1)
    first_output = component.build_circuit()
    first_output.x(0)
    second_output = component.build_circuit()

    np.testing.assert_allclose(second_output.to_matrix(), expected, rtol=0.0, atol=0.0)
    assert not np.allclose(first_output.to_matrix(), expected, rtol=0.0, atol=1e-10)
    assert first_output is not second_output


def test_output_can_be_composed_and_inverted() -> None:
    unitary1, unitary2 = _two_qubit_pair()
    overlap = UnitaryOverlap(unitary1, unitary2).build_circuit()
    destination = Circuit([9, 3, 12])
    destination.compose(overlap, [Qubit(12), Qubit(9)])
    inverse = overlap.inverse()

    expected_overlap = unitary2.to_matrix().conj().T @ unitary1.to_matrix()
    np.testing.assert_allclose(overlap.to_matrix(), expected_overlap, atol=1e-14)
    np.testing.assert_allclose(
        inverse.to_matrix(), expected_overlap.conj().T, rtol=0.0, atol=1e-14
    )
    assert destination.num_qubits == 3


@pytest.mark.integration
def test_integrates_with_compute_uncompute_and_statevector_sampler() -> None:
    unitary1 = Circuit(1)
    unitary1.x(0)
    unitary2 = Circuit(1)
    overlap = UnitaryOverlap(unitary1, unitary2).build_circuit()

    result = ComputeUncomputeFidelity(StatevectorSampler()).run(
        [CircuitFidelityRequest(Circuit(1), overlap, shots=32)]
    )[0]

    assert result.fidelity == 0.0
    assert result.variance == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"unitary1": object()}, "unitary1"),
        ({"unitary2": object()}, "unitary2"),
        ({"insert_barrier": 1}, "insert_barrier"),
    ],
)
def test_rejects_invalid_nominal_types(
    kwargs: dict[str, object],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "unitary1": Circuit(1),
        "unitary2": Circuit(1),
    }
    arguments.update(kwargs)

    with pytest.raises(TypeError, match=message):
        UnitaryOverlap(**arguments)  # type: ignore[arg-type]


def test_rejects_zero_and_different_widths() -> None:
    with pytest.raises(ValueError, match="at least one"):
        UnitaryOverlap(Circuit(0), Circuit(0))
    with pytest.raises(ValueError, match="same qubit count"):
        UnitaryOverlap(Circuit(1), Circuit(2))


def _with_direct_operation(operation: str) -> Circuit:
    circuit = Circuit(1)
    if operation == "measure":
        circuit.measure(0)
    elif operation == "reset":
        circuit.reset(0)
    elif operation == "delay":
        circuit.delay(0, 1.0)
    else:
        raise AssertionError("unsupported test operation")
    return circuit


@pytest.mark.parametrize("operation", ["measure", "reset", "delay"])
@pytest.mark.parametrize("side", ["unitary1", "unitary2"])
def test_rejects_direct_nonunitary_and_delayed_operations(
    operation: str,
    side: str,
) -> None:
    arguments = {"unitary1": Circuit(1), "unitary2": Circuit(1)}
    arguments[side] = _with_direct_operation(operation)

    with pytest.raises(ValueError, match=side):
        UnitaryOverlap(**arguments)


@pytest.mark.parametrize("operation", ["measure", "reset", "delay"])
def test_rejects_nested_nonunitary_and_delayed_operations(operation: str) -> None:
    inner = _with_direct_operation(operation)
    nested = Circuit(1)
    nested.append_circuit_gate(inner.to_gate(f"hidden_{operation}"), [0])

    with pytest.raises(ValueError, match="unitary2.*static unitary"):
        UnitaryOverlap(Circuit(1), nested)


@pytest.mark.parametrize("operation", ["classical_data", "classical_control"])
@pytest.mark.parametrize("side", ["unitary1", "unitary2"])
def test_rejects_classical_data_and_control(operation: str, side: str) -> None:
    invalid = Circuit(1)
    variable = invalid.var(ClassicalType.bool())
    if operation == "classical_data":
        invalid.store(variable, ClassicalExpr.bool_literal(True))
    else:
        invalid.if_(variable.expr(), lambda body: body.x(0))
    invalid.validate()
    arguments = {"unitary1": Circuit(1), "unitary2": Circuit(1)}
    arguments[side] = invalid

    with pytest.raises(ValueError, match=side):
        UnitaryOverlap(**arguments)


@pytest.mark.parametrize("side", ["unitary1", "unitary2"])
def test_rejects_core_operations_without_static_inverse(side: str) -> None:
    unsupported = Circuit(3)
    unsupported.append_unitary_gate(
        UnitaryGate("identity-3q", 3).with_matrix(np.eye(8, dtype=complex)),
        [0, 1, 2],
    )
    arguments = {"unitary1": Circuit(3), "unitary2": Circuit(3)}
    arguments[side] = unsupported

    with pytest.raises(ValueError, match=rf"{side}.*statically invertible"):
        UnitaryOverlap(**arguments)


def test_public_exports_are_identical() -> None:
    assert UnitaryOverlap is ModuleUnitaryOverlap
