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

"""Numerical edge tests for private state-synthesis primitives."""

from cqlib.circuit import Circuit
from cqlib.circuit.gates import StandardGate
from cqlib_algorithms.operators.state_preparation._state_synthesis import (
    _append_uniformly_controlled_rotation,
    validate_normalized_vector,
    validate_positive_qubit_count,
    validate_prefix,
    validate_sparse_entries,
)


def test_validation_helpers_only_check_and_return_none() -> None:
    normalized = (1.0 + 0.0j, 0.0 + 0.0j)
    sparse_entries = ((1, 1.0 + 0.0j),)

    assert validate_positive_qubit_count(2) is None
    assert validate_prefix("") is None
    assert validate_normalized_vector(normalized, name="values") is None
    assert validate_sparse_entries(2, sparse_entries) is None

    assert normalized == (1.0 + 0.0j, 0.0 + 0.0j)
    assert sparse_entries == ((1, 1.0 + 0.0j),)


def test_ucr_avoids_walsh_underflow_for_a_representable_nonzero_angle() -> None:
    circuit = Circuit(3)
    tiny_angle = 1e-323
    _append_uniformly_controlled_rotation(
        circuit,
        StandardGate.RY,
        target=0,
        controls=(1, 2),
        conditional_angles=(tiny_angle, 0.0, 0.0, 0.0),
    )

    # Dividing tiny_angle by four in the Walsh transform is zero, whereas
    # the direct controlled RY retains sin(tiny_angle / 2) as a subnormal.
    assert tiny_angle / 4 == 0.0
    assert circuit.to_matrix()[1, 0] != 0.0
    assert any(operation.is_mcgate for operation in circuit.operations)
