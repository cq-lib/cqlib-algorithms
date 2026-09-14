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
"""Tests for the variational ansatz structural protocol."""

from __future__ import annotations

import inspect
from collections.abc import Callable

import pytest
from cqlib.circuit import Circuit, Parameter
from cqlib.circuit.ansatz import (
    BasicEntanglerLayers,
    PauliEvolutionAnsatz,
    QAOAAnsatz,
    TwoLocal,
)
from cqlib.qis import Hamiltonian, PauliString

from cqlib_algorithms.protocols.ansatz import Ansatz


class _SingleParameterAnsatz:
    """Minimal non-inheriting implementation used to verify structural use."""

    def validate(self) -> None:
        return None

    def build_circuit(self, prefix: str) -> Circuit:
        circuit = Circuit(1)
        circuit.ry(0, Parameter(f"{prefix}_0"))
        return circuit

    def num_parameters(self) -> int:
        return 1

    def num_qubits(self) -> int:
        return 1


def _assert_component_contract(component: Ansatz, prefix: str) -> None:
    component.validate()
    num_qubits = component.num_qubits()
    num_parameters = component.num_parameters()

    assert type(num_qubits) is int
    assert type(num_parameters) is int
    assert num_qubits > 0
    assert num_parameters >= 0

    first = component.build_circuit(prefix)
    second = component.build_circuit(prefix)

    assert isinstance(first, Circuit)
    assert isinstance(second, Circuit)
    assert first is not second
    assert first.num_qubits == num_qubits
    assert second.num_qubits == num_qubits
    assert first == second
    assert len(set(first.used_symbols)) == len(first.used_symbols)
    assert len(first.used_symbols) <= num_parameters

    second_length = len(second.operations)
    first.x(first.qubits[0])
    assert len(second.operations) == second_length


def _zz_hamiltonian() -> Hamiltonian:
    return Hamiltonian.from_pauli(PauliString.from_str("ZZ"))


def test_protocol_declares_only_the_four_minimal_methods():
    public_callables = {
        name
        for name, value in Ansatz.__dict__.items()
        if not name.startswith("_") and callable(value)
    }

    assert public_callables == {
        "build_circuit",
        "num_parameters",
        "num_qubits",
        "validate",
    }
    assert (
        inspect.signature(Ansatz.build_circuit).parameters["prefix"].default
        is inspect.Parameter.empty
    )


def test_protocol_is_deliberately_not_runtime_checkable():
    with pytest.raises(TypeError, match="runtime_checkable"):
        isinstance(_SingleParameterAnsatz(), Ansatz)


def test_non_inheriting_python_implementation_is_consumable():
    ansatz = _SingleParameterAnsatz()

    _assert_component_contract(ansatz, "theta")
    assert ansatz.build_circuit("").used_symbols == ["_0"]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TwoLocal(2),
        lambda: BasicEntanglerLayers(2),
        lambda: QAOAAnsatz(_zz_hamiltonian()),
    ],
    ids=["two-local", "basic-entangler-layers", "qaoa"],
)
def test_representative_core_builders_satisfy_the_contract(
    factory: Callable[[], Ansatz],
):
    _assert_component_contract(factory(), "theta")


def test_explicit_core_parameter_name_may_take_precedence_over_prefix():
    ansatz = PauliEvolutionAnsatz(_zz_hamiltonian()).with_time_param_name("tau")
    first = ansatz.build_circuit("first")
    second = ansatz.build_circuit("second")

    assert first.used_symbols == ["tau"]
    assert second.used_symbols == ["tau"]
    assert ansatz.num_parameters() == 1


def test_core_validation_error_is_exposed_as_value_error():
    with pytest.raises(ValueError):
        TwoLocal(0).validate()

