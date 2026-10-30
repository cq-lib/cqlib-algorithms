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

"""Tests for fixed-width two-qubit unitary parameterizations."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import pytest

from cqlib_algorithms.operators.ansatz import (
    BaseAnsatz,
    TwoQubitUnitaryAnsatz,
)
from cqlib_algorithms.protocols import Ansatz

_Matrix = np.ndarray
_Gate = tuple[str, int | tuple[int, int], float]


def _rx(theta: float) -> _Matrix:
    cosine = math.cos(theta / 2)
    sine = math.sin(theta / 2)
    return np.array(
        [[cosine, -1j * sine], [-1j * sine, cosine]],
        dtype=np.complex128,
    )


def _ry(theta: float) -> _Matrix:
    cosine = math.cos(theta / 2)
    sine = math.sin(theta / 2)
    return np.array(
        [[cosine, -sine], [sine, cosine]],
        dtype=np.complex128,
    )


def _rz(theta: float) -> _Matrix:
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)]).astype(np.complex128)


def _single_qubit(matrix: _Matrix, qubit: int) -> _Matrix:
    identity = np.eye(2, dtype=np.complex128)
    return np.kron(matrix, identity) if qubit == 1 else np.kron(identity, matrix)


def _cx(control: int, target: int) -> _Matrix:
    matrix = np.zeros((4, 4), dtype=np.complex128)
    for basis in range(4):
        result = basis ^ (1 << target) if basis & (1 << control) else basis
        matrix[result, basis] = 1.0
    return matrix


def _reference_matrix(gates: Sequence[_Gate]) -> _Matrix:
    matrix = np.eye(4, dtype=np.complex128)
    rotations: dict[str, Callable[[float], _Matrix]] = {
        "RX": _rx,
        "RY": _ry,
        "RZ": _rz,
    }
    for name, qubits, angle in gates:
        if name == "CX":
            control, target = qubits  # type: ignore[misc]
            gate = _cx(control, target)
        else:
            gate = _single_qubit(rotations[name](angle), int(qubits))
        matrix = gate @ matrix
    return matrix


def _mode_gates(
    parameterization: str,
    structure: str,
    parameters: Sequence[float],
) -> list[_Gate]:
    if parameterization == "simple":
        first_gate = "RY" if structure == "real" else "RX"
        return [
            (first_gate, 0, parameters[0]),
            ("RY", 1, parameters[1]),
            ("CX", (0, 1), 0.0),
        ]
    if structure == "real":
        return [
            ("RZ", 0, math.pi / 2),
            ("RZ", 1, math.pi / 2),
            ("RY", 1, math.pi / 2),
            ("CX", (1, 0), 0.0),
            ("RZ", 0, parameters[0]),
            ("RY", 0, parameters[1]),
            ("RZ", 0, parameters[2]),
            ("RZ", 1, parameters[3]),
            ("RY", 1, parameters[4]),
            ("RZ", 1, parameters[5]),
            ("CX", (1, 0), 0.0),
            ("RY", 1, -math.pi / 2),
            ("RZ", 0, -math.pi / 2),
            ("RZ", 1, -math.pi / 2),
        ]
    return [
        ("RZ", 0, parameters[0]),
        ("RY", 0, parameters[1]),
        ("RZ", 0, parameters[2]),
        ("RZ", 1, parameters[3]),
        ("RY", 1, parameters[4]),
        ("RZ", 1, parameters[5] + math.pi / 2),
        ("CX", (1, 0), 0.0),
        ("RZ", 0, 2 * parameters[6] - math.pi / 2),
        ("RY", 1, math.pi / 2 - 2 * parameters[7]),
        ("CX", (0, 1), 0.0),
        ("RY", 1, 2 * parameters[8] - math.pi / 2),
        ("CX", (1, 0), 0.0),
        ("RZ", 1, parameters[9]),
        ("RY", 1, parameters[10]),
        ("RZ", 1, parameters[11]),
        ("RZ", 0, parameters[12] - math.pi / 2),
        ("RY", 0, parameters[13]),
        ("RZ", 0, parameters[14]),
    ]


def _bound_matrix(component: TwoQubitUnitaryAnsatz, values: Sequence[float]) -> _Matrix:
    circuit = component.build_circuit("theta")
    assignments = {f"theta_{index}": value for index, value in enumerate(values)}
    return circuit.assign_parameters(assignments).to_matrix()


@pytest.mark.parametrize(
    ("parameterization", "structure", "parameter_count", "expected_operations"),
    [
        (
            "simple",
            "real",
            2,
            [("RY", [0]), ("RY", [1]), ("CX", [0, 1])],
        ),
        (
            "simple",
            "complex",
            2,
            [("RX", [0]), ("RY", [1]), ("CX", [0, 1])],
        ),
        (
            "general",
            "real",
            6,
            [
                ("RZ", [0]),
                ("RZ", [1]),
                ("RY", [1]),
                ("CX", [1, 0]),
                ("RZ", [0]),
                ("RY", [0]),
                ("RZ", [0]),
                ("RZ", [1]),
                ("RY", [1]),
                ("RZ", [1]),
                ("CX", [1, 0]),
                ("RY", [1]),
                ("RZ", [0]),
                ("RZ", [1]),
            ],
        ),
        (
            "general",
            "complex",
            15,
            [
                ("RZ", [0]),
                ("RY", [0]),
                ("RZ", [0]),
                ("RZ", [1]),
                ("RY", [1]),
                ("RZ", [1]),
                ("CX", [1, 0]),
                ("RZ", [0]),
                ("RY", [1]),
                ("CX", [0, 1]),
                ("RY", [1]),
                ("CX", [1, 0]),
                ("RZ", [1]),
                ("RY", [1]),
                ("RZ", [1]),
                ("RZ", [0]),
                ("RY", [0]),
                ("RZ", [0]),
            ],
        ),
    ],
)
def test_mode_gate_order_and_parameter_contract(
    parameterization,
    structure,
    parameter_count,
    expected_operations,
):
    component = TwoQubitUnitaryAnsatz(
        parameterization=parameterization,
        structure=structure,
    )
    circuit = component.build_circuit("theta")

    assert component.num_qubits() == 2
    assert component.num_parameters() == parameter_count
    assert circuit.used_symbols == [
        f"theta_{index}" for index in range(parameter_count)
    ]
    assert [
        (operation.name, [qubit.index for qubit in operation.qubits])
        for operation in circuit.operations
    ] == expected_operations


@pytest.mark.parametrize(
    ("parameterization", "structure", "values"),
    [
        ("simple", "real", (0.17, -0.41)),
        ("simple", "complex", (-0.29, 0.53)),
        ("general", "real", (0.11, -0.23, 0.37, -0.41, 0.59, -0.67)),
        (
            "general",
            "complex",
            tuple(0.07 + 0.11 * index for index in range(15)),
        ),
    ],
)
@pytest.mark.reference
def test_mode_matrix_matches_independent_little_endian_reference(
    parameterization,
    structure,
    values,
):
    component = TwoQubitUnitaryAnsatz(
        parameterization=parameterization,
        structure=structure,
    )
    actual = _bound_matrix(component, values)
    expected = _reference_matrix(_mode_gates(parameterization, structure, values))

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        actual.conj().T @ actual,
        np.eye(4),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("parameterization", ["simple", "general"])
def test_real_structure_produces_real_orthogonal_matrices(parameterization):
    component = TwoQubitUnitaryAnsatz(
        parameterization=parameterization,
        structure="real",
    )
    values = tuple(0.13 + 0.19 * index for index in range(component.num_parameters()))
    matrix = _bound_matrix(component, values)

    np.testing.assert_allclose(matrix.imag, 0.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        matrix.real.T @ matrix.real,
        np.eye(4),
        rtol=0.0,
        atol=1e-12,
    )
    if parameterization == "general":
        np.testing.assert_allclose(np.linalg.det(matrix), 1.0, atol=1e-12)


@pytest.mark.parametrize(
    ("structure", "values", "expected_rank"),
    [
        ("real", (0.17, -0.31, 0.43, 0.61, -0.79, 0.97), 6),
        (
            "complex",
            tuple(0.09 + 0.137 * index for index in range(15)),
            15,
        ),
    ],
)
@pytest.mark.reference
def test_general_parameterization_has_expected_local_jacobian_rank(
    structure,
    values,
    expected_rank,
):
    component = TwoQubitUnitaryAnsatz(
        parameterization="general",
        structure=structure,
    )
    step = 1e-7
    derivatives = []
    for index in range(len(values)):
        upper = list(values)
        lower = list(values)
        upper[index] += step
        lower[index] -= step
        derivative = (
            _bound_matrix(component, upper) - _bound_matrix(component, lower)
        ) / (2 * step)
        if structure == "real":
            derivatives.append(derivative.real.ravel())
        else:
            derivatives.append(
                np.concatenate((derivative.real.ravel(), derivative.imag.ravel()))
            )

    jacobian = np.stack(derivatives, axis=1)
    assert np.linalg.matrix_rank(jacobian, tol=1e-7) == expected_rank


def test_default_mode_and_nominal_inheritance():
    component = TwoQubitUnitaryAnsatz()

    assert isinstance(component, Ansatz)
    assert isinstance(component, BaseAnsatz)
    assert component.validate() is None
    assert component.num_qubits() == 2
    assert component.num_parameters() == 15


@pytest.mark.parametrize(
    ("keyword", "value", "exception", "message"),
    [
        ("parameterization", 1, TypeError, "parameterization must be a string"),
        ("parameterization", "wide", ValueError, "parameterization must be"),
        ("structure", False, TypeError, "structure must be a string"),
        ("structure", "quaternion", ValueError, "structure must be"),
    ],
)
def test_invalid_mode_options(keyword, value, exception, message):
    with pytest.raises(exception, match=message):
        TwoQubitUnitaryAnsatz(**{keyword: value})  # type: ignore[arg-type]


def test_prefix_validation_is_inherited():
    with pytest.raises(TypeError, match="prefix must be a string"):
        TwoQubitUnitaryAnsatz().build_circuit(1)  # type: ignore[arg-type]


def test_repeated_builds_are_equal_and_independent():
    component = TwoQubitUnitaryAnsatz(
        parameterization="general",
        structure="complex",
    )
    first = component.build_circuit("u")
    second = component.build_circuit("u")

    assert first == second
    assert first is not second
    first.x(0)
    assert first != second
