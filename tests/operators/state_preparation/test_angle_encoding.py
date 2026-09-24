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

"""Tests for the stable Cqlib angle-encoding adapter."""

import copy

import pytest
from cqlib.circuit.gates import StandardGate

from cqlib_algorithms.operators.state_preparation import AngleEncoding


@pytest.mark.parametrize(
    "rotation_gate",
    [StandardGate.RX, StandardGate.RY, StandardGate.RZ, StandardGate.Phase],
)
def test_delegates_supported_unbound_rotation_gate(
    rotation_gate: StandardGate,
) -> None:
    encoding = AngleEncoding(3, rotation_gate)
    first = encoding.build_circuit("feature")
    second = encoding.build_circuit("feature")

    encoding.validate()
    assert encoding.num_qubits() == 3
    assert encoding.num_parameters() == 3
    assert first is not second
    assert first.used_symbols == ["feature_0", "feature_1", "feature_2"]
    assert [operation.name for operation in first.operations] == [
        operation.name for operation in second.operations
    ]


def test_copies_gate_and_rejects_unsupported_or_bound_gate() -> None:
    gate = copy.copy(StandardGate.RX)
    encoding = AngleEncoding(1, gate)
    assert encoding.build_circuit("x").used_symbols == ["x_0"]

    with pytest.raises(ValueError, match="unbound RX"):
        AngleEncoding(1, StandardGate.H)
    with pytest.raises(ValueError, match="unbound RX"):
        AngleEncoding(1, StandardGate.RX(0.2))
    with pytest.raises(TypeError, match="StandardGate"):
        AngleEncoding(1, "RX")  # type: ignore[arg-type]


def test_rejects_invalid_width_and_prefix() -> None:
    with pytest.raises(TypeError, match="num_qubits must be an int"):
        AngleEncoding(True, StandardGate.RY)
    with pytest.raises(ValueError, match="at least 1"):
        AngleEncoding(0, StandardGate.RY)
    with pytest.raises(TypeError, match="prefix must be a str"):
        AngleEncoding(1, StandardGate.RY).build_circuit(None)  # type: ignore[arg-type]
