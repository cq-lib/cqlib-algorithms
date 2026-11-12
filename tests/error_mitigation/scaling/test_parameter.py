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

"""Tests for the parameter-perturbation noise-scaling primitive."""

from __future__ import annotations

import numpy as np
import pytest
from cqlib import Circuit, Parameter

from cqlib_algorithms.error_mitigation.scaling import scale_parameters


def _body() -> Circuit:
    """Return a two-qubit body with two rotations and one entangler."""
    circuit = Circuit(2)
    circuit.rx(0, np.pi / 4)
    circuit.ry(1, np.pi / 4)
    circuit.cx(0, 1)
    return circuit


def _rotation_ladder(num_gates: int) -> Circuit:
    """Return one qubit carrying exactly ``num_gates`` rotations."""
    circuit = Circuit(1)
    for index in range(num_gates):
        circuit.rx(0, 0.1 * (index + 1))
    return circuit


def _measured(circuit: Circuit) -> Circuit:
    """Append one measurement per qubit and return the same circuit."""
    for qubit in range(len(circuit.qubits)):
        circuit.measure(qubit)
    return circuit


def _angles(circuit: Circuit) -> list[float]:
    """Return every numeric gate parameter of ``circuit``, in order."""
    return [float(param) for op in circuit.operations for param in op.params]


def _phase(circuit: Circuit) -> complex:
    """Return the global phase of ``circuit`` as a complex scalar."""
    return complex(circuit.global_phase.evaluate({}))


def _measurement_count(circuit: Circuit) -> int:
    """Return how many measurement records ``circuit`` carries."""
    return sum(1 for operation in circuit.operations if operation.is_classical_data)


def test_scale_parameters_at_one_keeps_the_angles() -> None:
    body = _body()
    scaled = scale_parameters(body, 1.0, base_variance=0.01, seed=3)
    assert _angles(scaled) == pytest.approx(_angles(body))


def test_scale_parameters_keeps_the_gate_count() -> None:
    body = _rotation_ladder(4)
    scaled = scale_parameters(body, 3.0, base_variance=0.01, seed=1)
    assert len(scaled.operations) == len(body.operations)


def test_scale_parameters_with_zero_variance_keeps_the_angles() -> None:
    body = _rotation_ladder(4)
    scaled = scale_parameters(body, 4.0, base_variance=0.0, seed=2)
    assert _angles(scaled) == pytest.approx(_angles(body))


def test_scale_parameters_is_deterministic_in_the_seed() -> None:
    first = scale_parameters(_rotation_ladder(4), 3.0, base_variance=0.01, seed=7)
    second = scale_parameters(_rotation_ladder(4), 3.0, base_variance=0.01, seed=7)
    assert _angles(first) == pytest.approx(_angles(second))


def test_scale_parameters_changes_the_angles_when_the_seed_changes() -> None:
    first = scale_parameters(_rotation_ladder(4), 3.0, base_variance=0.01, seed=7)
    second = scale_parameters(_rotation_ladder(4), 3.0, base_variance=0.01, seed=8)
    assert _angles(first) != pytest.approx(_angles(second))


def test_scale_parameters_draws_have_the_requested_deviation() -> None:
    num_gates = 400
    scale_factor = 3.0
    base_variance = 0.01
    body = _rotation_ladder(num_gates)
    scaled = scale_parameters(
        body,
        scale_factor,
        base_variance=base_variance,
        seed=13,
    )
    shifts = np.asarray(_angles(scaled)) - np.asarray(_angles(body))
    expected = np.sqrt((scale_factor - 1.0) * base_variance)
    # A 400-draw sample standard deviation sits within a few percent of the
    # population value, so a 15 percent band keeps the check stable.
    assert float(np.std(shifts)) == pytest.approx(expected, rel=0.15)
    assert float(np.mean(shifts)) == pytest.approx(0.0, abs=0.15 * expected)


def test_scale_parameters_preserves_the_global_phase() -> None:
    body = _body()
    body.set_global_phase(0.2)
    scaled = scale_parameters(body, 1.0, base_variance=0.01, seed=5)
    assert _phase(scaled) == pytest.approx(0.2)


def test_scale_parameters_strips_and_restores_measurements() -> None:
    measured = _measured(_body())
    scaled = scale_parameters(measured, 2.0, base_variance=0.01, seed=5)
    assert _measurement_count(measured) == 2
    assert _measurement_count(scaled) == 2


def test_scale_parameters_rejects_a_scale_below_one() -> None:
    with pytest.raises(ValueError):
        scale_parameters(_body(), 0.5, base_variance=0.01, seed=3)


def test_scale_parameters_rejects_a_negative_variance() -> None:
    with pytest.raises(ValueError):
        scale_parameters(_body(), 3.0, base_variance=-1.0, seed=3)


def test_scale_parameters_rejects_a_symbolic_parameter() -> None:
    circuit = Circuit(1)
    circuit.rx(0, Parameter("theta"))
    with pytest.raises(ValueError):
        scale_parameters(circuit, 2.0, base_variance=0.01, seed=3)
