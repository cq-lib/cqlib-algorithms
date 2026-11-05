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

"""Tests for computational-basis initialization."""

import numpy as np
import pytest

from cqlib_algorithms.operators.state_preparation import BasisStateInitialization


@pytest.mark.parametrize("num_qubits,basis_index", [(1, 0), (1, 1), (3, 5)])
def test_prepares_little_endian_basis_index(num_qubits: int, basis_index: int) -> None:
    preparation = BasisStateInitialization(num_qubits, basis_index)
    expected = np.zeros(1 << num_qubits, dtype=complex)
    expected[basis_index] = 1.0

    preparation.validate()
    assert preparation.num_qubits() == num_qubits
    assert preparation.num_parameters() == 0
    np.testing.assert_allclose(
        preparation.build_circuit("ignored").to_matrix()[:, 0],
        expected,
        rtol=0.0,
        atol=1e-14,
    )


@pytest.mark.parametrize("num_qubits", [True, 1.0, "2", None])
def test_rejects_non_integer_qubit_count(num_qubits: object) -> None:
    with pytest.raises(TypeError, match="num_qubits must be an int"):
        BasisStateInitialization(num_qubits, 0)  # type: ignore[arg-type]


@pytest.mark.parametrize("num_qubits", [0, -1])
def test_rejects_non_positive_qubit_count(num_qubits: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        BasisStateInitialization(num_qubits, 0)


@pytest.mark.parametrize("basis_index", [True, 1.0, "1", None])
def test_rejects_non_integer_basis_index(basis_index: object) -> None:
    with pytest.raises(TypeError, match="basis_index must be an int"):
        BasisStateInitialization(2, basis_index)  # type: ignore[arg-type]


@pytest.mark.parametrize("basis_index", [-1, 4])
def test_rejects_basis_index_outside_register(basis_index: int) -> None:
    with pytest.raises(ValueError, match="fit within"):
        BasisStateInitialization(2, basis_index)


def test_builds_independent_circuits_and_validates_prefix_type() -> None:
    preparation = BasisStateInitialization(2, 1)
    first = preparation.build_circuit("x")
    first.x(1)
    second = preparation.build_circuit("x")
    assert len(first.operations) == len(second.operations) + 1
    with pytest.raises(TypeError, match="prefix must be a str"):
        preparation.build_circuit(1)  # type: ignore[arg-type]
