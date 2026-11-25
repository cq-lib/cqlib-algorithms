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

"""Tests for the reusable phase-estimation circuit component."""

from __future__ import annotations

import math

import numpy as np
import pytest
from cqlib import Circuit, Parameter

from cqlib_algorithms import PhaseEstimationCircuit


def _evaluation_probabilities(circuit: Circuit, width: int) -> np.ndarray:
    dimension = 1 << width
    state = circuit.to_matrix()[:, 0]
    return np.sum(np.abs(state.reshape(-1, dimension)) ** 2, axis=0)


@pytest.mark.reference
def test_exact_eigenphase_maps_to_expected_grid_point() -> None:
    width = 3
    phase = 3 / 8
    unitary = Circuit(1)
    unitary.phase(0, 2.0 * math.pi * phase)
    component = PhaseEstimationCircuit(unitary, width)
    circuit = Circuit(width + 1)
    circuit.x(width)
    circuit.compose(component.build_circuit(), circuit.qubits)

    expected = np.zeros(1 << width)
    expected[3] = 1.0
    np.testing.assert_allclose(
        _evaluation_probabilities(circuit, width),
        expected,
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.reference
def test_global_phase_is_observable_under_quantum_control() -> None:
    width = 3
    unitary = Circuit(1)
    unitary.set_global_phase(2.0 * math.pi / 8.0)

    probabilities = _evaluation_probabilities(
        PhaseEstimationCircuit(unitary, width).build_circuit(),
        width,
    )

    expected = np.zeros(1 << width)
    expected[1] = 1.0
    np.testing.assert_allclose(probabilities, expected, rtol=0.0, atol=1e-12)


@pytest.mark.reference
@pytest.mark.parametrize("system_target", [0, 1])
def test_multi_qubit_unitary_preserves_system_target_order(
    system_target: int,
) -> None:
    width = 4
    phase = 5 / 16
    unitary = Circuit(2)
    unitary.phase(system_target, 2.0 * math.pi * phase)
    component = PhaseEstimationCircuit(unitary, width)
    circuit = Circuit(component.num_qubits)
    circuit.x(width + system_target)
    circuit.compose(component.build_circuit(), circuit.qubits)

    expected = np.zeros(1 << width)
    expected[5] = 1.0
    np.testing.assert_allclose(
        _evaluation_probabilities(circuit, width),
        expected,
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.reference
def test_target_circuit_keeps_absolute_phase_and_system_bit_order() -> None:
    unitary = Circuit(2)
    unitary.phase(0, math.pi / 2)
    unitary.set_global_phase(math.pi / 4)
    component = PhaseEstimationCircuit(unitary, 3)
    forward = component.build_circuit().to_matrix()
    adjoint = (
        PhaseEstimationCircuit(unitary, 3, inverse=True).build_circuit().to_matrix()
    )
    for system_index, grid_point in enumerate((1, 3, 1, 3)):
        basis_in = system_index << 3
        expected = np.zeros(32)
        expected[basis_in + grid_point] = 1
        np.testing.assert_allclose(abs(forward[:, basis_in]) ** 2, expected, atol=1e-12)
    np.testing.assert_allclose(adjoint, forward.conj().T, rtol=0, atol=1e-12)


def test_component_reports_register_widths_and_returns_fresh_circuits() -> None:
    unitary = Circuit(2)
    unitary.cx(0, 1)
    component = PhaseEstimationCircuit(unitary, 3, approximation_degree=1)

    first = component.build_circuit()
    second = component.build_circuit()

    assert component.num_evaluation_qubits == 3
    assert component.num_system_qubits == 2
    assert component.num_qubits == 5
    assert component.approximation_degree == 1
    assert first is not second
    np.testing.assert_allclose(
        first.to_matrix(),
        second.to_matrix(),
        rtol=0.0,
        atol=0.0,
    )


def test_component_snapshots_source_unitary() -> None:
    source = Circuit(1)
    source.z(0)
    component = PhaseEstimationCircuit(source, 2)
    expected = component.build_circuit().to_matrix()

    source.x(0)

    np.testing.assert_allclose(
        component.build_circuit().to_matrix(),
        expected,
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("width", [True, 1.5, "3"])
def test_component_rejects_non_integer_evaluation_width(width: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        PhaseEstimationCircuit(Circuit(1), width)  # type: ignore[arg-type]


@pytest.mark.parametrize("width", [0, -1])
def test_component_rejects_non_positive_evaluation_width(width: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        PhaseEstimationCircuit(Circuit(1), width)


def test_component_rejects_width_beyond_binary64_phase_resolution() -> None:
    with pytest.raises(ValueError, match="must not exceed 52"):
        PhaseEstimationCircuit(Circuit(1), 53)


def test_component_rejects_unbound_and_measured_unitaries() -> None:
    symbolic = Circuit(1)
    symbolic.rz(0, Parameter("theta"))
    with pytest.raises(ValueError, match="fully bound"):
        PhaseEstimationCircuit(symbolic, 2)

    measured = Circuit(1)
    measured.measure(0)
    with pytest.raises(ValueError, match="classical data|static unitary"):
        PhaseEstimationCircuit(measured, 2)


@pytest.mark.reference
def test_inverse_configuration_builds_the_adjoint_of_the_forward_circuit() -> None:
    unitary = Circuit(1)
    unitary.rx(0, 0.7)

    forward = PhaseEstimationCircuit(unitary, 3, approximation_degree=1)
    adjoint = PhaseEstimationCircuit(unitary, 3, approximation_degree=1, inverse=True)

    np.testing.assert_allclose(
        adjoint.build_circuit().to_matrix(),
        forward.build_circuit().to_matrix().conj().T,
        rtol=0.0,
        atol=1e-12,
    )


def test_component_reports_inverse_configuration() -> None:
    assert PhaseEstimationCircuit(Circuit(1), 2).inverse is False
    assert PhaseEstimationCircuit(Circuit(1), 2, inverse=True).inverse is True


@pytest.mark.parametrize("value", [1, "yes", None])
def test_component_rejects_non_boolean_inverse(value: object) -> None:
    with pytest.raises(TypeError, match="must be a boolean"):
        PhaseEstimationCircuit(Circuit(1), 2, inverse=value)  # type: ignore[arg-type]


def test_dense_memory_limit_is_checked_before_matrix_construction() -> None:
    component = PhaseEstimationCircuit(Circuit(11), 1)

    with pytest.raises(ValueError, match="dense controlled-matrix limit"):
        component.build_circuit()
