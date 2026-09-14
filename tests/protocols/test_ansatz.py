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

"""Tests for the variational ansatz abstract base class."""

from __future__ import annotations

import inspect

import pytest
from cqlib.circuit import Circuit, Parameter

from cqlib_algorithms.protocols.ansatz import Ansatz


class _SingleParameterAnsatz(Ansatz):
    """Minimal concrete implementation used to verify nominal use."""

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


def test_base_class_declares_only_the_four_minimal_methods():
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


def test_ansatz_is_an_abstract_nominal_base_class():
    assert inspect.isabstract(Ansatz)
    with pytest.raises(TypeError, match="abstract"):
        Ansatz()  # type: ignore[abstract]
    assert isinstance(_SingleParameterAnsatz(), Ansatz)


def test_inheriting_python_implementation_is_consumable():
    ansatz = _SingleParameterAnsatz()

    _assert_component_contract(ansatz, "theta")
    assert ansatz.build_circuit("").used_symbols == ["_0"]
