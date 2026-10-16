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

"""Tests for frozen random-layer variational circuits."""

from __future__ import annotations

import math

import numpy as np
import pytest
from cqlib.circuit.gates import StandardGate

from cqlib_algorithms.operators.ansatz import RandomLayersAnsatz


def _rotation(name: str, theta: float) -> np.ndarray:
    cosine = math.cos(theta / 2)
    sine = math.sin(theta / 2)
    if name == "RX":
        return np.array(
            [[cosine, -1j * sine], [-1j * sine, cosine]],
            dtype=np.complex128,
        )
    if name == "RY":
        return np.array([[cosine, -sine], [sine, cosine]], dtype=np.complex128)
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])


def _single_qubit_matrix(gate: np.ndarray, target: int, width: int) -> np.ndarray:
    matrix = np.zeros((2**width, 2**width), dtype=np.complex128)
    for column in range(2**width):
        input_bit = (column >> target) & 1
        for output_bit in (0, 1):
            row = (column & ~(1 << target)) | (output_bit << target)
            matrix[row, column] = gate[output_bit, input_bit]
    return matrix


def _cx_matrix(control: int, target: int, width: int) -> np.ndarray:
    matrix = np.zeros((2**width, 2**width), dtype=np.complex128)
    for column in range(2**width):
        row = column ^ (1 << target) if (column >> control) & 1 else column
        matrix[row, column] = 1.0
    return matrix


def _snapshot(component: RandomLayersAnsatz) -> list[tuple[str, list[int]]]:
    return [
        (operation.name, [qubit.index for qubit in operation.qubits])
        for operation in component.build_circuit("p").operations
    ]


def test_seeded_architecture_snapshot_is_stable():
    component = RandomLayersAnsatz(3, reps=2, seed=4)

    assert component.num_parameters() == 6
    assert _snapshot(component) == [
        ("CX", [0, 2]),
        ("RY", [0]),
        ("CX", [0, 2]),
        ("RZ", [1]),
        ("RX", [0]),
        ("RY", [1]),
        ("RX", [1]),
        ("CX", [0, 2]),
        ("RY", [0]),
    ]
    assert component.build_circuit("p").used_symbols == [
        f"p_{index}" for index in range(6)
    ]


def test_seeded_matrix_matches_independent_gate_application():
    component = RandomLayersAnsatz(3, reps=2, seed=4)
    circuit = component.build_circuit("p")
    values = tuple(0.1 + 0.09 * index for index in range(6))
    actual = circuit.assign_parameters(
        {f"p_{index}": value for index, value in enumerate(values)}
    ).to_matrix()

    expected = np.eye(8, dtype=np.complex128)
    parameter_index = 0
    for name, qubits in _snapshot(component):
        if name == "CX":
            gate = _cx_matrix(qubits[0], qubits[1], 3)
        else:
            gate = _single_qubit_matrix(
                _rotation(name, values[parameter_index]),
                qubits[0],
                3,
            )
            parameter_index += 1
        expected = gate @ expected

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        actual.conj().T @ actual,
        np.eye(8),
        rtol=0.0,
        atol=1e-12,
    )


def test_architecture_is_frozen_for_repeated_builds_and_equal_seeds():
    first_component = RandomLayersAnsatz(4, reps=2, seed=91)
    second_component = RandomLayersAnsatz(4, reps=2, seed=91)
    first = first_component.build_circuit("p")
    repeated = first_component.build_circuit("p")

    assert first == repeated == second_component.build_circuit("p")
    assert first is not repeated
    first.x(0)
    assert first != repeated


def test_none_seed_is_still_frozen_within_one_instance():
    component = RandomLayersAnsatz(3, reps=2, seed=None)
    assert component.build_circuit("p") == component.build_circuit("p")


def test_one_qubit_disables_entanglement_and_input_sequence_is_snapshotted():
    rotations = [StandardGate.RX]
    component = RandomLayersAnsatz(
        1,
        reps=2,
        rotations_per_layer=3,
        entanglement_ratio=0.999,
        rotation_gates=rotations,
        seed=5,
    )
    rotations[0] = StandardGate.RZ

    assert component.num_parameters() == 6
    assert [
        operation.name for operation in component.build_circuit("p").operations
    ] == ["RX"] * 6


@pytest.mark.parametrize("ratio", [1.0, np.nextafter(1.0, 0.0)])
def test_maximal_entanglement_ratio_has_strict_gate_bound_and_frozen_builds(ratio):
    reps = 2
    rotations_per_layer = 4
    component = RandomLayersAnsatz(
        3,
        reps=reps,
        rotations_per_layer=rotations_per_layer,
        entanglement_ratio=ratio,
        seed=9,
    )
    first = component.build_circuit("p")
    second = component.build_circuit("p")

    assert component.num_parameters() == reps * rotations_per_layer
    assert len(first.operations) == 2 * reps * rotations_per_layer
    assert sum(operation.name == "CX" for operation in first.operations) == (
        reps * rotations_per_layer
    )
    assert first.used_symbols == [
        f"p_{index}" for index in range(reps * rotations_per_layer)
    ]
    assert first == second
    assert first is not second


def test_zero_repetitions_return_identity_without_parameters():
    component = RandomLayersAnsatz(3, reps=0)
    assert component.num_parameters() == 0
    assert component.build_circuit("p").used_symbols == []
    np.testing.assert_array_equal(component.build_circuit("p").to_matrix(), np.eye(8))


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"rotations_per_layer": True}, TypeError, "integer or None"),
        ({"rotations_per_layer": 0}, ValueError, "greater than zero"),
        ({"entanglement_ratio": True}, TypeError, "real number"),
        ({"entanglement_ratio": float("nan")}, ValueError, "finite"),
        ({"entanglement_ratio": -0.1}, ValueError, "0 <= value <= 1"),
        ({"entanglement_ratio": 1.0001}, ValueError, "0 <= value <= 1"),
        ({"rotation_gates": []}, ValueError, "must not be empty"),
        ({"rotation_gates": [StandardGate.H]}, ValueError, "RX, RY, or RZ"),
        ({"entanglement_gate": StandardGate.H}, ValueError, "CX, CY, or CZ"),
        ({"seed": True}, TypeError, "integer or None"),
        ({"seed": -1}, ValueError, "non-negative"),
    ],
)
def test_invalid_random_layer_configuration(kwargs, error, message):
    with pytest.raises(error, match=message):
        RandomLayersAnsatz(2, **kwargs)
