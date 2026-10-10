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

"""Tests for the deterministic simplified two-design-style ansatz."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cqlib_algorithms.operators.ansatz import SimplifiedTwoDesignAnsatz


def _ry(theta: float) -> np.ndarray:
    cosine = math.cos(theta / 2)
    sine = math.sin(theta / 2)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.complex128)


def test_operation_order_and_parameter_count():
    component = SimplifiedTwoDesignAnsatz(4, reps=1)
    circuit = component.build_circuit("t")

    assert component.num_parameters() == 10
    assert circuit.used_symbols == [f"t_{index}" for index in range(10)]
    assert [
        (operation.name, [qubit.index for qubit in operation.qubits])
        for operation in circuit.operations
    ] == [
        ("RY", [0]),
        ("RY", [1]),
        ("RY", [2]),
        ("RY", [3]),
        ("CZ", [0, 1]),
        ("RY", [0]),
        ("RY", [1]),
        ("CZ", [2, 3]),
        ("RY", [2]),
        ("RY", [3]),
        ("CZ", [1, 2]),
        ("RY", [1]),
        ("RY", [2]),
    ]


def test_two_qubit_matrix_matches_independent_reference():
    values = (0.11, -0.22, 0.37, -0.48)
    component = SimplifiedTwoDesignAnsatz(2)
    actual = (
        component.build_circuit("p")
        .assign_parameters({f"p_{index}": value for index, value in enumerate(values)})
        .to_matrix()
    )
    identity = np.eye(2, dtype=np.complex128)
    cz = np.diag([1.0, 1.0, 1.0, -1.0])
    initial = np.kron(_ry(values[1]), _ry(values[0]))
    expected = (
        np.kron(_ry(values[3]), identity)
        @ np.kron(identity, _ry(values[2]))
        @ cz
        @ initial
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        actual.conj().T @ actual,
        np.eye(4),
        rtol=0.0,
        atol=1e-12,
    )


def test_zero_repetitions_and_single_qubit_have_only_initial_rotations():
    zero = SimplifiedTwoDesignAnsatz(3, reps=0)
    single = SimplifiedTwoDesignAnsatz(1, reps=4)

    assert zero.num_parameters() == 3
    assert [operation.name for operation in zero.build_circuit("p").operations] == [
        "RY",
        "RY",
        "RY",
    ]
    assert single.num_parameters() == 1
    assert [operation.name for operation in single.build_circuit("p").operations] == [
        "RY"
    ]


def test_repeated_builds_are_independent():
    component = SimplifiedTwoDesignAnsatz(3, reps=2)
    first = component.build_circuit("p")
    second = component.build_circuit("p")

    assert first == second
    assert first is not second
    first.x(0)
    assert first != second


def test_common_validation_boundaries():
    with pytest.raises(TypeError, match="num_qubits must be an integer"):
        SimplifiedTwoDesignAnsatz(True)
    with pytest.raises(ValueError, match="greater than zero"):
        SimplifiedTwoDesignAnsatz(0)
    with pytest.raises(TypeError, match="reps must be an integer"):
        SimplifiedTwoDesignAnsatz(2, reps=True)
    with pytest.raises(ValueError, match="reps must be non-negative"):
        SimplifiedTwoDesignAnsatz(2, reps=-1)
    with pytest.raises(TypeError, match="prefix must be a string"):
        SimplifiedTwoDesignAnsatz(2).build_circuit(1)
