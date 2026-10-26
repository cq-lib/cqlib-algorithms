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

"""Tests for exact zero-state reflection."""

from __future__ import annotations

from dataclasses import is_dataclass

import numpy as np
import pytest
from cqlib import Circuit, Qubit

from cqlib_algorithms.operators.oracle import ZeroStateReflection


def _zero_reflection_reference(num_qubits: int) -> np.ndarray:
    result = -np.eye(1 << num_qubits, dtype=np.complex128)
    result[0, 0] = 1.0
    return result


@pytest.mark.parametrize("num_qubits", [1, 2, 3, 4])
def test_zero_state_reflection_matches_exact_matrix(num_qubits):
    matrix = ZeroStateReflection(num_qubits).build_circuit().to_matrix()

    np.testing.assert_allclose(
        matrix,
        _zero_reflection_reference(num_qubits),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
def test_zero_state_reflection_is_hermitian_unitary_and_involutory(num_qubits):
    matrix = ZeroStateReflection(num_qubits).build_circuit().to_matrix()
    identity = np.eye(1 << num_qubits)

    np.testing.assert_allclose(matrix, matrix.conj().T, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(matrix.conj().T @ matrix, identity, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(matrix @ matrix, identity, rtol=0.0, atol=1e-12)


def test_zero_state_reflection_retains_required_global_sign():
    matrix = ZeroStateReflection(2).build_circuit().to_matrix()

    assert matrix[0, 0] == pytest.approx(1.0 + 0.0j, abs=1e-12)
    assert matrix[1, 1] == pytest.approx(-1.0 + 0.0j, abs=1e-12)


def test_build_returns_fresh_independent_circuits():
    reflection = ZeroStateReflection(2)
    first = reflection.build_circuit()
    second = reflection.build_circuit()

    assert first is not second
    assert first == second
    second_size = len(second.operations)
    first.x(0)
    assert len(second.operations) == second_size


def test_zero_state_reflection_composes_onto_non_contiguous_qubits():
    reflection = ZeroStateReflection(2).build_circuit()
    destination = Circuit([7, 2])

    destination.compose(reflection, [Qubit(7), Qubit(2)])

    np.testing.assert_allclose(
        destination.to_matrix([7, 2]),
        _zero_reflection_reference(2),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("num_qubits", [True, 1.5, "2"])
def test_zero_state_reflection_rejects_non_integer_counts(num_qubits):
    with pytest.raises(TypeError, match="num_qubits must be an integer"):
        ZeroStateReflection(num_qubits)  # type: ignore[arg-type]


@pytest.mark.parametrize("num_qubits", [0, -1])
def test_zero_state_reflection_rejects_non_positive_counts(num_qubits):
    with pytest.raises(ValueError, match="greater than zero"):
        ZeroStateReflection(num_qubits)


def test_zero_state_reflection_is_a_slotted_component_with_read_only_config():
    reflection = ZeroStateReflection(2)

    assert not is_dataclass(reflection)
    assert not hasattr(reflection, "__dict__")
    assert reflection.num_qubits == 2
    with pytest.raises(AttributeError):
        reflection.num_qubits = 3  # type: ignore[misc]
