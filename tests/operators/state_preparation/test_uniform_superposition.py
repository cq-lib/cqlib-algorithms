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

"""Tests for uniform-superposition preparation."""

import numpy as np
import pytest

from cqlib_algorithms.operators.state_preparation import UniformSuperposition


@pytest.mark.parametrize("num_qubits", [1, 2, 5])
def test_prepares_uniform_positive_amplitudes(num_qubits: int) -> None:
    preparation = UniformSuperposition(num_qubits)
    expected = np.full(1 << num_qubits, 2.0 ** (-num_qubits / 2), dtype=complex)

    preparation.validate()
    assert preparation.num_qubits() == num_qubits
    assert preparation.num_parameters() == 0
    assert [
        operation.name for operation in preparation.build_circuit("").operations
    ] == ["H"] * num_qubits
    np.testing.assert_allclose(
        preparation.build_circuit("unused").to_matrix()[:, 0],
        expected,
        rtol=0.0,
        atol=1e-14,
    )


@pytest.mark.parametrize("num_qubits", [0, -1])
def test_rejects_non_positive_width(num_qubits: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        UniformSuperposition(num_qubits)
