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

"""Tests for the shared block-encoding base class."""

from __future__ import annotations

import pytest
from cqlib.circuit.gates import StandardGate

from cqlib import Circuit
from cqlib_algorithms.operators.oracle import BlockEncoding


class _ExampleBlockEncoding(BlockEncoding):
    def __init__(
        self,
        *,
        alpha: float = 2.0,
        ancilla_qubits: int = 1,
        epsilon: float = 0.0,
        matrix_shape: tuple[int, int] = (2, 2),
        system_qubits: int | None = None,
    ) -> None:
        super().__init__(
            alpha=alpha,
            ancilla_qubits=ancilla_qubits,
            epsilon=epsilon,
            matrix_shape=matrix_shape,
            system_qubits=system_qubits,
        )

    def build_circuit(self) -> Circuit:
        circuit = Circuit(self.total_qubits)
        if self.total_qubits > 0:
            circuit.append_gate(StandardGate.RY(0.125), [0])
        if self.total_qubits > 1:
            circuit.append_gate(StandardGate.RZ(-0.3), [self.total_qubits - 1])
        return circuit


def test_block_encoding_exposes_standard_specification_properties() -> None:
    component = _ExampleBlockEncoding(
        alpha=3.5,
        ancilla_qubits=2,
        epsilon=1e-6,
        matrix_shape=(3, 2),
    )

    assert component.alpha == pytest.approx(3.5)
    assert component.ancilla_qubits == 2
    assert component.epsilon == pytest.approx(1e-6)
    assert component.matrix_shape == (3, 2)
    assert component.system_qubits == 2
    assert component.total_qubits == 4
    assert component.spec == pytest.approx((3.5, 2, 1e-6))
    assert component.is_standard is True
    assert component.left_projector_dimension == 3
    assert component.right_projector_dimension == 2


def test_block_encoding_default_adjoint_uses_circuit_inverse() -> None:
    component = _ExampleBlockEncoding(matrix_shape=(2, 2))
    circuit = component.build_circuit()
    adjoint = component.build_adjoint_circuit()

    assert circuit is not adjoint
    assert adjoint.to_matrix() == pytest.approx(circuit.to_matrix().conj().T)


def test_block_encoding_allows_explicit_larger_system_registers() -> None:
    component = _ExampleBlockEncoding(matrix_shape=(3, 2), system_qubits=3)

    assert component.system_qubits == 3
    assert component.total_qubits == 4


def test_block_encoding_rejects_invalid_alpha() -> None:
    with pytest.raises(TypeError, match="alpha must be a real finite scalar"):
        _ExampleBlockEncoding(alpha=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="alpha must be greater than zero"):
        _ExampleBlockEncoding(alpha=0.0)


def test_block_encoding_rejects_invalid_ancilla_qubits() -> None:
    with pytest.raises(TypeError, match="ancilla_qubits must be an integer"):
        _ExampleBlockEncoding(ancilla_qubits=True)  # type: ignore[arg-type]
    with pytest.raises(
        ValueError,
        match="ancilla_qubits must be greater than or equal to zero",
    ):
        _ExampleBlockEncoding(ancilla_qubits=-1)


def test_block_encoding_rejects_invalid_epsilon() -> None:
    with pytest.raises(TypeError, match="epsilon must be a real finite scalar"):
        _ExampleBlockEncoding(epsilon=float("nan"))
    with pytest.raises(
        ValueError,
        match="epsilon must be greater than or equal to zero",
    ):
        _ExampleBlockEncoding(epsilon=-1e-6)


def test_block_encoding_rejects_invalid_matrix_shape() -> None:
    with pytest.raises(TypeError, match="matrix_shape must be a tuple of two integers"):
        _ExampleBlockEncoding(matrix_shape=[2, 2])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="matrix_shape\\[0\\] must be greater than zero"):
        _ExampleBlockEncoding(matrix_shape=(0, 2))
    with pytest.raises(ValueError, match="matrix_shape\\[1\\] must be greater than zero"):
        _ExampleBlockEncoding(matrix_shape=(2, 0))


def test_block_encoding_rejects_system_registers_that_are_too_small() -> None:
    with pytest.raises(TypeError, match="system_qubits must be an integer"):
        _ExampleBlockEncoding(system_qubits=True)  # type: ignore[arg-type]
    with pytest.raises(
        ValueError,
        match="system_qubits must be large enough to contain matrix_shape",
    ):
        _ExampleBlockEncoding(matrix_shape=(3, 2), system_qubits=1)
