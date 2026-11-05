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

"""Contract checks for the shared ansatz template implementation."""

from __future__ import annotations

import pytest
from cqlib import Circuit, Parameter
from cqlib.circuit.gates import StandardGate

from cqlib_algorithms._utils.numeric import (
    validate_positive_qubit_count,
    validate_repetitions,
)
from cqlib_algorithms._utils.parameters import (
    validate_parameter_prefix as validate_prefix,
)
from cqlib_algorithms.operators.ansatz import BaseAnsatz
from cqlib_algorithms.operators.ansatz._validation import (
    snapshot_rotation_gates,
    validate_boolean,
    validate_entanglement_gate,
)
from cqlib_algorithms.protocols import Ansatz


class _ExampleAnsatz(BaseAnsatz):
    __slots__ = ("validation_calls",)

    def __init__(self) -> None:
        self.validation_calls = 0

    def _validate_configuration(self) -> None:
        self.validation_calls += 1

    def _build_circuit(self, prefix: str) -> Circuit:
        circuit = Circuit(1)
        circuit.ry(0, Parameter(f"{prefix}_0"))
        return circuit

    def _parameter_count(self) -> int:
        return 1

    def _qubit_count(self) -> int:
        return 1


def test_base_preserves_nominal_hierarchy() -> None:
    assert isinstance(_ExampleAnsatz(), Ansatz)


def test_shared_build_flow_validates_prefix_configuration_and_result() -> None:
    component = _ExampleAnsatz()
    first = component.build_circuit("theta")
    second = component.build_circuit("theta")

    assert component.validation_calls == 2
    assert first.id != second.id
    assert first.num_qubits == component.num_qubits() == 1
    assert len(first.used_symbols) == component.num_parameters() == 1
    with pytest.raises(TypeError, match="prefix"):
        component.build_circuit(1)  # type: ignore[arg-type]


def test_validate_helpers_only_check_and_return_none() -> None:
    assert validate_positive_qubit_count(2) is None
    assert validate_repetitions(0) is None
    assert validate_prefix("") is None
    assert validate_boolean(False, "option") is None
    assert validate_entanglement_gate(StandardGate.CX) is None


def test_rotation_gate_snapshot_is_explicitly_named_and_immutable() -> None:
    gates = [StandardGate.RY, StandardGate.RZ]
    snapshot = snapshot_rotation_gates(gates)
    gates[0] = StandardGate.RX

    assert snapshot == (StandardGate.RY, StandardGate.RZ)
