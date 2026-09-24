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

"""Tests for user-supplied circuit state preparation."""

import numpy as np
import pytest
from cqlib.circuit.gates import UnitaryGate

from cqlib import Circuit, Parameter
from cqlib_algorithms.operators.state_preparation import UserCircuitPreparation


def test_snapshots_maps_qubits_and_preserves_symbols_and_global_phase() -> None:
    source = Circuit([5, 2])
    source.h(5)
    source.ry(2, Parameter("theta"))
    source.set_global_phase(0.37)
    preparation = UserCircuitPreparation(source)
    source.x(5)

    first = preparation.build_circuit("ignored")
    second = preparation.build_circuit("different")
    assert [qubit.index for qubit in first.qubits] == [0, 1]
    assert first.used_symbols == ["theta"]
    assert second.used_symbols == ["theta"]
    assert preparation.num_qubits() == 2
    assert preparation.num_parameters() == 1
    preparation.validate()

    expected = Circuit(2)
    expected.h(0)
    expected.ry(1, 0.61)
    expected.set_global_phase(0.37)
    np.testing.assert_allclose(
        first.assign_parameters({"theta": 0.61}).to_matrix(),
        expected.to_matrix(),
        rtol=0.0,
        atol=1e-14,
    )


def test_returns_independent_copies_without_renaming_symbols() -> None:
    source = Circuit(1)
    source.rz(0, Parameter("fixed_name"))
    preparation = UserCircuitPreparation(source)
    first = preparation.build_circuit("one")
    first.x(0)
    second = preparation.build_circuit("two")
    assert second.used_symbols == ["fixed_name"]
    assert len(first.operations) == len(second.operations) + 1


def test_rejects_empty_or_noninvertible_circuit() -> None:
    with pytest.raises(ValueError, match="at least one qubit"):
        UserCircuitPreparation(Circuit([]))

    measured = Circuit(1)
    measured.measure(0)
    with pytest.raises(ValueError, match="statically invertible"):
        UserCircuitPreparation(measured)

    reset = Circuit(1)
    reset.reset(0)
    with pytest.raises(ValueError, match="statically invertible"):
        UserCircuitPreparation(reset)


def test_documents_core_three_qubit_numeric_unitary_inverse_boundary() -> None:
    circuit = Circuit(3)
    gate = UnitaryGate("identity-3q", 3).with_matrix(np.eye(8, dtype=complex))
    circuit.append_unitary_gate(gate, [0, 1, 2])
    with pytest.raises(ValueError, match="statically invertible"):
        UserCircuitPreparation(circuit)


def test_rejects_wrong_input_and_prefix_types() -> None:
    with pytest.raises(TypeError, match="circuit must be a Circuit"):
        UserCircuitPreparation(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="prefix must be a str"):
        UserCircuitPreparation(Circuit(1)).build_circuit(None)  # type: ignore[arg-type]
