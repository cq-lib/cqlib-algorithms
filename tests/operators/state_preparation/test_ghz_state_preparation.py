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

"""Tests for GHZ-state preparation."""

import numpy as np
import pytest

from cqlib_algorithms.operators.state_preparation import GHZStatePreparation


@pytest.mark.parametrize("num_qubits", [1, 2, 5])
def test_prepares_ghz_state_with_cx_chain(num_qubits: int) -> None:
    preparation = GHZStatePreparation(num_qubits)
    circuit = preparation.build_circuit("unused")
    expected = np.zeros(1 << num_qubits, dtype=complex)
    expected[0] = expected[-1] = 1 / np.sqrt(2)

    preparation.validate()
    assert preparation.num_qubits() == num_qubits
    assert preparation.num_parameters() == 0
    assert [operation.name for operation in circuit.operations] == ["H"] + ["CX"] * (
        num_qubits - 1
    )
    assert [
        [qubit.index for qubit in operation.qubits]
        for operation in circuit.operations[1:]
    ] == [[control, control + 1] for control in range(num_qubits - 1)]
    np.testing.assert_allclose(
        circuit.to_matrix()[:, 0], expected, rtol=0.0, atol=1e-14
    )


def test_rejects_zero_qubits() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        GHZStatePreparation(0)
