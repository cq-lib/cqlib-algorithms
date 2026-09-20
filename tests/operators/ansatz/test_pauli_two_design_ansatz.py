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

"""Tests for the Pauli two-design ansatz."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cqlib_algorithms.operators.ansatz import PauliTwoDesignAnsatz
from cqlib_algorithms.protocols import Ansatz


def _rx(theta: float) -> np.ndarray:
    cosine = math.cos(theta / 2)
    sine = math.sin(theta / 2)
    return np.array([[cosine, -1j * sine], [-1j * sine, cosine]], dtype=np.complex128)


def _ry(theta: float) -> np.ndarray:
    cosine = math.cos(theta / 2)
    sine = math.sin(theta / 2)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.complex128)


def _snapshot(component: PauliTwoDesignAnsatz) -> list[tuple[str, list[int]]]:
    return [
        (operation.name, [qubit.index for qubit in operation.qubits])
        for operation in component.build_circuit("t").operations
    ]


def test_pauli_two_design_seeded_operation_snapshot():
    component = PauliTwoDesignAnsatz(4, reps=2, seed=5)
    circuit = component.build_circuit("t")

    assert isinstance(component, Ansatz)
    assert component.num_qubits() == 4
    assert component.num_parameters() == 12
    assert _snapshot(component) == [
        *(("RY", [qubit]) for qubit in range(4)),
        ("RZ", [0]),
        ("RY", [1]),
        ("RZ", [2]),
        ("RY", [3]),
        ("CZ", [0, 1]),
        ("CZ", [2, 3]),
        ("CZ", [1, 2]),
        ("RZ", [0]),
        ("RZ", [1]),
        ("RZ", [2]),
        ("RZ", [3]),
        ("CZ", [0, 1]),
        ("CZ", [2, 3]),
        ("CZ", [1, 2]),
        ("RX", [0]),
        ("RY", [1]),
        ("RX", [2]),
        ("RZ", [3]),
    ]
    assert circuit.used_symbols == [f"t_{index}" for index in range(12)]


def test_pauli_two_design_seed_and_build_determinism():
    first_component = PauliTwoDesignAnsatz(5, reps=2, seed=73)
    second_component = PauliTwoDesignAnsatz(5, reps=2, seed=73)

    first = first_component.build_circuit("p")
    repeated = first_component.build_circuit("p")
    second = second_component.build_circuit("p")

    assert first == repeated == second
    assert first is not repeated
    first.x(0)
    assert first != repeated


def test_pauli_two_design_one_qubit_matrix_is_independent_reference():
    theta = 0.37
    circuit = PauliTwoDesignAnsatz(1, reps=0, seed=1).build_circuit("p")

    assert [operation.name for operation in circuit.operations] == ["RY", "RX"]
    actual = circuit.assign_parameters({"p_0": theta}).to_matrix()
    expected = _rx(theta) @ _ry(math.pi / 4)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        actual.conj().T @ actual, np.eye(2), rtol=0.0, atol=1e-12
    )


@pytest.mark.parametrize("seed", [True, 1.5, "1", object()])
def test_pauli_two_design_rejects_invalid_seed_types(seed):
    with pytest.raises(TypeError, match="seed must be an integer or None"):
        PauliTwoDesignAnsatz(2, seed=seed)  # type: ignore[arg-type]


def test_pauli_two_design_rejects_negative_seed():
    with pytest.raises(ValueError, match="seed must be non-negative"):
        PauliTwoDesignAnsatz(2, seed=-1)


def test_pauli_two_design_validates_common_fields_and_prefix():
    with pytest.raises(TypeError, match="num_qubits must be an integer"):
        PauliTwoDesignAnsatz(True)
    with pytest.raises(ValueError, match="greater than zero"):
        PauliTwoDesignAnsatz(0)
    with pytest.raises(TypeError, match="reps must be an integer"):
        PauliTwoDesignAnsatz(2, reps=True)
    with pytest.raises(ValueError, match="reps must be non-negative"):
        PauliTwoDesignAnsatz(2, reps=-1)
    with pytest.raises(TypeError, match="prefix must be a string"):
        PauliTwoDesignAnsatz(2, seed=1).build_circuit(1)  # type: ignore[arg-type]
