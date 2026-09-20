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

"""Tests for fixed two-local ansatz specializations."""

from __future__ import annotations

import pytest
from cqlib.circuit.ansatz import EntanglementTopology
from cqlib.circuit.gates import StandardGate

from cqlib_algorithms.operators.ansatz import (
    EfficientSU2Ansatz,
    HardwareEfficientAnsatz,
    RealAmplitudesAnsatz,
)
from cqlib_algorithms.protocols import Ansatz


@pytest.mark.parametrize(
    ("component", "expected_gates", "parameters_per_layer"),
    [
        (RealAmplitudesAnsatz(2, reps=1), ["RY", "RY"], 2),
        (EfficientSU2Ansatz(2, reps=1), ["RY", "RZ", "RY", "RZ"], 4),
        (
            HardwareEfficientAnsatz(2, reps=1),
            ["RZ", "RX", "RZ", "RZ", "RX", "RZ"],
            6,
        ),
    ],
)
def test_two_local_specialization_layout(
    component, expected_gates, parameters_per_layer
):
    circuit = component.build_circuit("theta")
    operations = circuit.operations

    assert isinstance(component, Ansatz)
    assert component.validate() is None
    assert component.num_qubits() == 2
    assert component.num_parameters() == 2 * parameters_per_layer
    assert [
        operation.name for operation in operations[:parameters_per_layer]
    ] == expected_gates
    assert operations[parameters_per_layer].name == "CX"
    assert [
        operation.name for operation in operations[parameters_per_layer + 1 :]
    ] == expected_gates
    assert circuit.used_symbols == [
        f"theta_{index}" for index in range(2 * parameters_per_layer)
    ]


@pytest.mark.parametrize(
    "component_type",
    [RealAmplitudesAnsatz, EfficientSU2Ansatz, HardwareEfficientAnsatz],
)
def test_specializations_forward_topology_and_skip_final(component_type):
    kwargs = {
        "reps": 1,
        "entanglement": EntanglementTopology.circular(),
        "skip_final_rotation_layer": True,
    }
    component = component_type(3, **kwargs)
    circuit = component.build_circuit("p")

    assert [
        [qubit.index for qubit in operation.qubits]
        for operation in circuit.operations
        if operation.num_qubits == 2
    ] == [[0, 1], [1, 2], [2, 0]]


def test_hardware_efficient_supports_core_entanglers():
    circuit = HardwareEfficientAnsatz(
        2,
        reps=1,
        entanglement_gate=StandardGate.CY,
        skip_final_rotation_layer=True,
    ).build_circuit("h")

    assert [operation.name for operation in circuit.operations][-1] == "CY"


@pytest.mark.parametrize("component_type", [RealAmplitudesAnsatz, EfficientSU2Ansatz])
def test_specializations_reject_invalid_common_options(component_type):
    with pytest.raises(ValueError, match="reps must be non-negative"):
        component_type(2, reps=-1)
    with pytest.raises(TypeError, match="EntanglementTopology or None"):
        component_type(2, entanglement="linear")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="skip_final_rotation_layer must be a boolean"):
        component_type(2, skip_final_rotation_layer=1)  # type: ignore[arg-type]


def test_hardware_efficient_rejects_invalid_entangler():
    with pytest.raises(ValueError, match="unbound CX, CY, or CZ"):
        HardwareEfficientAnsatz(2, entanglement_gate=StandardGate.SWAP)
