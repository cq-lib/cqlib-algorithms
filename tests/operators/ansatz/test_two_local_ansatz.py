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

"""Tests for the general two-local ansatz."""

from __future__ import annotations

import math

import numpy as np
import pytest
from cqlib.circuit.ansatz import EntanglementTopology
from cqlib.circuit.gates import StandardGate

from cqlib_algorithms.operators.ansatz import TwoLocalAnsatz
from cqlib_algorithms.protocols import Ansatz


def _ry(theta: float) -> np.ndarray:
    cosine = math.cos(theta / 2)
    sine = math.sin(theta / 2)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.complex128)


def _cx_0_to_1() -> np.ndarray:
    matrix = np.zeros((4, 4), dtype=np.complex128)
    for column in range(4):
        row = column ^ 0b10 if column & 0b01 else column
        matrix[row, column] = 1.0
    return matrix


def _operation_snapshot(ansatz: TwoLocalAnsatz) -> list[tuple[str, list[int]]]:
    return [
        (operation.name, [qubit.index for qubit in operation.qubits])
        for operation in ansatz.build_circuit("theta").operations
    ]


def test_two_local_implements_ansatz_and_default_layout():
    ansatz = TwoLocalAnsatz(2)

    assert isinstance(ansatz, Ansatz)
    assert ansatz.validate() is None
    assert ansatz.num_qubits() == 2
    assert ansatz.num_parameters() == 8
    assert _operation_snapshot(ansatz) == [
        ("RY", [0]),
        ("RY", [1]),
        ("CX", [0, 1]),
        ("RY", [0]),
        ("RY", [1]),
        ("CX", [0, 1]),
        ("RY", [0]),
        ("RY", [1]),
        ("CX", [0, 1]),
        ("RY", [0]),
        ("RY", [1]),
    ]


def test_two_local_parameter_order_and_independent_matrix():
    ansatz = TwoLocalAnsatz(2, reps=1, skip_final_rotation_layer=True)
    circuit = ansatz.build_circuit("p")
    theta_0, theta_1 = 0.31, -0.47

    assert circuit.used_symbols == ["p_0", "p_1"]
    actual = circuit.assign_parameters({"p_0": theta_0, "p_1": theta_1}).to_matrix()
    rotations = np.kron(_ry(theta_1), _ry(theta_0))
    expected = _cx_0_to_1() @ rotations

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_two_local_uses_custom_topology_and_gate():
    ansatz = TwoLocalAnsatz(
        3,
        reps=1,
        rotation_gates=(),
        entanglement_gate=StandardGate.CZ,
        entanglement=EntanglementTopology.custom([(2, 0)]),
    )

    assert ansatz.num_parameters() == 0
    assert _operation_snapshot(ansatz) == [("CZ", [2, 0])]


def test_two_local_degenerate_zero_repetition_configurations():
    final_only = TwoLocalAnsatz(2, reps=0)
    empty = TwoLocalAnsatz(2, reps=0, skip_final_rotation_layer=True)

    assert final_only.num_parameters() == 2
    assert [op.name for op in final_only.build_circuit("p").operations] == ["RY", "RY"]
    assert empty.num_parameters() == 0
    assert empty.build_circuit("p").operations == []


@pytest.mark.parametrize("value", [True, 1.5, "2", None])
def test_two_local_rejects_invalid_qubit_count_types(value):
    with pytest.raises(TypeError, match="num_qubits must be an integer"):
        TwoLocalAnsatz(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1])
def test_two_local_rejects_non_positive_qubit_counts(value):
    with pytest.raises(ValueError, match="greater than zero"):
        TwoLocalAnsatz(value)


@pytest.mark.parametrize("value", [True, 1.5, "2", None])
def test_two_local_rejects_invalid_repetition_types(value):
    with pytest.raises(TypeError, match="reps must be an integer"):
        TwoLocalAnsatz(2, reps=value)  # type: ignore[arg-type]


def test_two_local_rejects_negative_repetitions():
    with pytest.raises(ValueError, match="reps must be non-negative"):
        TwoLocalAnsatz(2, reps=-1)


@pytest.mark.parametrize("value", [0, 1, "no", None])
def test_two_local_rejects_invalid_skip_flag(value):
    with pytest.raises(TypeError, match="skip_final_rotation_layer must be a boolean"):
        TwoLocalAnsatz(2, skip_final_rotation_layer=value)  # type: ignore[arg-type]


def test_two_local_rejects_invalid_rotation_and_entanglement_gates():
    with pytest.raises(TypeError, match="rotation_gates must be a sequence"):
        TwoLocalAnsatz(2, rotation_gates=StandardGate.RY)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="contain only StandardGate"):
        TwoLocalAnsatz(2, rotation_gates=("RY",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="only unbound RX, RY, or RZ"):
        TwoLocalAnsatz(2, rotation_gates=(StandardGate.RY(0.2),))
    with pytest.raises(TypeError, match="entanglement_gate must be a StandardGate"):
        TwoLocalAnsatz(2, entanglement_gate="CX")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unbound CX, CY, or CZ"):
        TwoLocalAnsatz(2, entanglement_gate=StandardGate.SWAP)


def test_two_local_rejects_invalid_topology_and_custom_edges():
    with pytest.raises(TypeError, match="EntanglementTopology or None"):
        TwoLocalAnsatz(2, entanglement="linear")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="self-loop"):
        TwoLocalAnsatz(2, entanglement=EntanglementTopology.custom([(0, 0)]))
    with pytest.raises(ValueError, match="out-of-bounds"):
        TwoLocalAnsatz(2, entanglement=EntanglementTopology.custom([(0, 2)]))
    with pytest.raises(ValueError, match="duplicate edge"):
        TwoLocalAnsatz(
            2,
            entanglement=EntanglementTopology.custom([(0, 1), (1, 0)]),
        )


def test_two_local_validates_prefix_and_returns_independent_circuits():
    ansatz = TwoLocalAnsatz(2, reps=1)
    first = ansatz.build_circuit("x")
    second = ansatz.build_circuit("x")

    assert first == second
    assert first is not second
    first.x(0)
    assert first != second
    with pytest.raises(TypeError, match="prefix must be a string"):
        ansatz.build_circuit(1)  # type: ignore[arg-type]
