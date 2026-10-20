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

"""Tests for MPS-, TTN-, and MERA-inspired ansatz circuits."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from cqlib_algorithms.operators.ansatz import MERAAnsatz, MPSAnsatz, TTNAnsatz
from cqlib_algorithms.protocols import Ansatz


def _snapshot(component: Ansatz) -> list[tuple[str, list[int]]]:
    return [
        (operation.name, [qubit.index for qubit in operation.qubits])
        for operation in component.build_circuit("p").operations
    ]


def _block_operations(edges: list[tuple[int, int]]) -> list[tuple[str, list[int]]]:
    operations: list[tuple[str, list[int]]] = []
    for left, right in edges:
        operations.extend([("RY", [left]), ("RY", [right]), ("CX", [left, right])])
    return operations


@pytest.mark.parametrize(
    ("component", "edges"),
    [
        (MPSAnsatz(2), [(0, 1)]),
        (MPSAnsatz(4), [(0, 1), (1, 2), (2, 3)]),
    ],
)
def test_mps_exact_topology(component: MPSAnsatz, edges: list[tuple[int, int]]) -> None:
    assert _snapshot(component) == _block_operations(edges)


@pytest.mark.parametrize(
    ("width", "edges", "root"),
    [
        (2, [(0, 1)], 1),
        (3, [(0, 1), (1, 2)], 2),
        (5, [(0, 1), (2, 3), (1, 3), (3, 4)], 4),
    ],
)
def test_ttn_exact_topology(
    width: int, edges: list[tuple[int, int]], root: int
) -> None:
    assert _snapshot(TTNAnsatz(width)) == _block_operations(edges) + [("RY", [root])]


@pytest.mark.parametrize(
    ("width", "edges", "root"),
    [
        (2, [(0, 1)], 1),
        (4, [(1, 2), (0, 1), (2, 3), (1, 3)], 3),
        (
            8,
            [
                (1, 2),
                (3, 4),
                (5, 6),
                (0, 1),
                (2, 3),
                (4, 5),
                (6, 7),
                (3, 5),
                (1, 3),
                (5, 7),
                (3, 7),
            ],
            7,
        ),
    ],
)
def test_mera_exact_topology(
    width: int, edges: list[tuple[int, int]], root: int
) -> None:
    assert _snapshot(MERAAnsatz(width)) == _block_operations(edges) + [("RY", [root])]


def _ry(theta: float) -> np.ndarray:
    cosine = math.cos(theta / 2)
    sine = math.sin(theta / 2)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.complex128)


def _single_qubit_matrix(num_qubits: int, qubit: int, matrix: np.ndarray) -> np.ndarray:
    factors = [
        matrix if index == qubit else np.eye(2) for index in reversed(range(num_qubits))
    ]
    result = factors[0]
    for factor in factors[1:]:
        result = np.kron(result, factor)
    return result


def _cx_matrix(num_qubits: int, control: int, target: int) -> np.ndarray:
    dimension = 1 << num_qubits
    result = np.zeros((dimension, dimension), dtype=np.complex128)
    for basis in range(dimension):
        output = basis ^ (1 << target) if (basis >> control) & 1 else basis
        result[output, basis] = 1.0
    return result


def _reference_matrix(
    num_qubits: int,
    edges: list[tuple[int, int]],
    values: list[float],
    root: int | None,
) -> np.ndarray:
    result = np.eye(1 << num_qubits, dtype=np.complex128)
    parameter = 0
    for left, right in edges:
        result = _single_qubit_matrix(num_qubits, left, _ry(values[parameter])) @ result
        parameter += 1
        result = (
            _single_qubit_matrix(num_qubits, right, _ry(values[parameter])) @ result
        )
        parameter += 1
        result = _cx_matrix(num_qubits, left, right) @ result
    if root is not None:
        result = _single_qubit_matrix(num_qubits, root, _ry(values[parameter])) @ result
    return result


@pytest.mark.parametrize(
    ("factory", "width", "edges", "root"),
    [
        (MPSAnsatz, 3, [(0, 1), (1, 2)], None),
        (TTNAnsatz, 3, [(0, 1), (1, 2)], 2),
        (MERAAnsatz, 4, [(1, 2), (0, 1), (2, 3), (1, 3)], 3),
    ],
)
def test_bound_matrix_and_unitarity_match_independent_reference(
    factory: Callable[[int], Ansatz],
    width: int,
    edges: list[tuple[int, int]],
    root: int | None,
) -> None:
    component = factory(width)
    values = [-0.41 + 0.073 * index for index in range(component.num_parameters())]
    circuit = component.build_circuit("t")
    actual = circuit.assign_parameters(
        dict(zip(circuit.used_symbols, values))
    ).to_matrix()
    expected = _reference_matrix(width, edges, values, root)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        actual.conj().T @ actual,
        np.eye(1 << width),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("component", "expected_parameters"),
    [
        (MPSAnsatz(2), 2),
        (MPSAnsatz(4), 6),
        (TTNAnsatz(2), 3),
        (TTNAnsatz(5), 9),
        (MERAAnsatz(2), 3),
        (MERAAnsatz(4), 9),
        (MERAAnsatz(8), 23),
    ],
)
def test_protocol_parameter_order_and_repeated_builds(
    component: Ansatz, expected_parameters: int
) -> None:
    assert isinstance(component, Ansatz)
    component.validate()
    assert component.num_parameters() == expected_parameters
    first = component.build_circuit("theta")
    second = component.build_circuit("theta")
    assert first.num_qubits == component.num_qubits()
    assert first.used_symbols == [
        f"theta_{index}" for index in range(expected_parameters)
    ]
    assert first == second
    assert first is not second
    first.x(0)
    assert first != second


@pytest.mark.parametrize("factory", [MPSAnsatz, TTNAnsatz, MERAAnsatz])
@pytest.mark.parametrize("width", [True, 1, 0, -1, 2.0])
def test_rejects_invalid_widths(
    factory: Callable[[int], Ansatz], width: object
) -> None:
    error = TypeError if isinstance(width, (bool, float)) else ValueError
    with pytest.raises(error):
        factory(width)  # type: ignore[arg-type]


@pytest.mark.parametrize("width", [3, 5, 6])
def test_mera_requires_power_of_two(width: int) -> None:
    with pytest.raises(ValueError, match="power of two"):
        MERAAnsatz(width)


def test_prefix_must_be_string() -> None:
    with pytest.raises(TypeError, match="prefix must be a string"):
        MPSAnsatz(2).build_circuit(None)  # type: ignore[arg-type]
