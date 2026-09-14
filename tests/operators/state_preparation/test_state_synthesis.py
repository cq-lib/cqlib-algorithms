"""Numerical edge tests for private state-synthesis primitives."""

from cqlib.circuit import Circuit
from cqlib.circuit.gates import StandardGate
from cqlib_algorithms.operators.state_preparation._state_synthesis import (
    _append_uniformly_controlled_rotation,
)


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
