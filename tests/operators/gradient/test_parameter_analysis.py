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

"""Tests for circuit symbol analysis and shift-rule derivation."""

from __future__ import annotations

import math

import pytest
from cqlib.circuit import FrozenCircuit
from cqlib.circuit.gates import CircuitGate, StandardGate

from cqlib import Circuit, Parameter, Qubit
from cqlib_algorithms.operators.gradient._parameter_analysis import (
    analyze_symbol,
    local_shift_plan,
)


def test_single_rotation_occurrence_spectrum_and_positions() -> None:
    theta = Parameter("theta")
    circuit = Circuit(2)
    circuit.h(0)
    circuit.ry(1, theta)
    occurrences = analyze_symbol(circuit, "theta")
    assert len(occurrences) == 1
    occurrence = occurrences[0]
    assert occurrence.op_index == 1
    assert occurrence.slot == 0
    assert occurrence.qubits == (1,)
    assert occurrence.coefficient == 1.0
    assert occurrence.spectrum == (0.5, -0.5)
    assert not occurrence.zero


def test_phase_occurrence_uses_diagonal_spectrum() -> None:
    phi = Parameter("phi")
    circuit = Circuit(1)
    circuit.phase(0, phi)
    occurrence = analyze_symbol(circuit, "phi")[0]
    assert occurrence.spectrum == (0.0, -1.0)


def test_controlled_rotation_occurrence_spectrum() -> None:
    theta = Parameter("theta")
    circuit = Circuit(2)
    circuit.crx(0, 1, theta)
    occurrence = analyze_symbol(circuit, "theta")[0]
    assert occurrence.spectrum == (0.0, 0.5, -0.5)
    assert occurrence.qubits == (0, 1)


def test_mcgate_rotation_occurrence_uses_control_spectrum() -> None:
    theta = Parameter("theta")
    circuit = Circuit(3)
    circuit.append_mc_gate(StandardGate.RZ(theta).control(2), [0, 1, 2])
    occurrence = analyze_symbol(circuit, "theta")[0]
    assert occurrence.spectrum == (0.0, 0.5, -0.5)
    assert occurrence.qubits == (0, 1, 2)


def test_u_gate_slots_have_distinct_spectra() -> None:
    theta = Parameter("theta")
    phi = Parameter("phi")
    lam = Parameter("lam")
    circuit = Circuit(1)
    circuit.u(0, theta, phi, lam)
    theta_occ = analyze_symbol(circuit, "theta")[0]
    phi_occ = analyze_symbol(circuit, "phi")[0]
    lam_occ = analyze_symbol(circuit, "lam")[0]
    assert theta_occ.slot == 0
    assert theta_occ.spectrum == (0.5, -0.5)
    assert phi_occ.slot == 1
    assert phi_occ.spectrum == (0.0, -1.0)
    assert lam_occ.slot == 2
    assert lam_occ.spectrum == (0.0, -1.0)


def test_gphase_occurrences_are_zero_contribution() -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    circuit.append_gate(StandardGate.GPhase(theta), [])
    occurrences = analyze_symbol(circuit, "theta")
    assert len(occurrences) == 2
    kinds = [occurrence.zero for occurrence in occurrences]
    assert kinds == [False, True]


def test_global_phase_only_symbol_has_no_occurrences() -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.x(0)
    circuit.set_global_phase(theta)
    assert "theta" in circuit.used_symbols
    assert analyze_symbol(circuit, "theta") == ()


def test_affine_expression_scales_spectrum_by_chain_coefficient() -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.ry(0, 2 * theta + 1)
    occurrence = analyze_symbol(circuit, "theta")[0]
    assert occurrence.coefficient == 2.0
    assert occurrence.spectrum == (1.0, -1.0)


def test_shared_symbol_keeps_independent_occurrences() -> None:
    theta = Parameter("theta")
    circuit = Circuit(2)
    circuit.h(0)
    circuit.ry(1, theta)
    circuit.cx(0, 1)
    circuit.phase(1, theta)
    occurrences = analyze_symbol(circuit, "theta")
    assert len(occurrences) == 2
    assert [occurrence.op_index for occurrence in occurrences] == [1, 3]
    assert [local_shift_plan(occurrence)[0] for occurrence in occurrences] == [
        (math.pi / 2, -math.pi / 2),
        (math.pi / 2, -math.pi / 2),
    ]


def test_global_phase_only_symbol_has_no_local_plans() -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.x(0)
    circuit.set_global_phase(theta)
    assert analyze_symbol(circuit, "theta") == ()


def test_classic_rotation_yields_two_point_shift_rule() -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    shifts, weights = local_shift_plan(analyze_symbol(circuit, "theta")[0])
    assert shifts == (math.pi / 2, -math.pi / 2)
    assert weights == pytest.approx((0.5, -0.5), abs=1e-12)


def test_controlled_rotation_uses_four_point_local_rule() -> None:
    theta = Parameter("theta")
    circuit = Circuit(2)
    circuit.crx(0, 1, 3.0 * theta + 0.2)
    occurrence = analyze_symbol(circuit, "theta")[0]
    shifts, weights = local_shift_plan(occurrence)
    assert len(shifts) == 4
    for frequency in (0.5, 1.0):
        matched = sum(
            weight * complex(math.cos(frequency * shift), math.sin(frequency * shift))
            for weight, shift in zip(weights, shifts)
        )
        assert matched == pytest.approx(1j * frequency, abs=1e-10)
    assert occurrence.coefficient == 3.0


@pytest.mark.parametrize(
    "gate_builder",
    [
        lambda c, t: (
            c.xy(0, t) if hasattr(c, "xy") else c.append_gate(StandardGate.XY(t), [0])
        ),
        lambda c, t: c.append_gate(StandardGate.XY2P(t), [0]),
        lambda c, t: c.append_gate(StandardGate.RXY(t, t), [0]),
    ],
)
def test_unsupported_parametric_gates_are_rejected(gate_builder) -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    gate_builder(circuit, theta)
    with pytest.raises(ValueError, match="not supported"):
        analyze_symbol(circuit, "theta")


def test_non_affine_expression_is_rejected() -> None:
    circuit = Circuit(1)
    circuit.ry(0, Parameter.from_expression("sin(theta)"))
    with pytest.raises(ValueError, match="affine"):
        analyze_symbol(circuit, "theta")


def test_measurement_is_rejected() -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    circuit.measure(0)
    with pytest.raises(ValueError, match="unitary"):
        analyze_symbol(circuit, "theta")


def test_reset_and_delay_are_rejected() -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    circuit.reset(0)
    with pytest.raises(ValueError, match="unitary"):
        analyze_symbol(circuit, "theta")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    circuit.delay(0, 1.0)
    with pytest.raises(ValueError, match="unitary"):
        analyze_symbol(circuit, "theta")


def test_circuit_gate_is_rejected() -> None:
    theta = Parameter("theta")
    inner = Circuit(1)
    inner.ry(0, theta)
    gate = CircuitGate("sub", FrozenCircuit([Qubit(0)], inner.operations), ["theta"])
    outer = Circuit(1)
    outer.append_circuit_gate(gate, [0], [theta])
    assert "theta" in outer.used_symbols
    with pytest.raises(ValueError, match="circuit gates"):
        analyze_symbol(outer, "theta")


def test_symbol_requires_string_type() -> None:
    circuit = Circuit(1)
    with pytest.raises(TypeError):
        analyze_symbol(circuit, 1.0)  # type: ignore[arg-type]


def test_barrier_is_skipped_and_indices_stay_stable() -> None:
    theta = Parameter("theta")
    circuit = Circuit(2)
    circuit.h(0)
    circuit.barrier([0, 1])
    circuit.ry(1, theta)
    occurrence = analyze_symbol(circuit, "theta")[0]
    assert occurrence.op_index == 2
    assert occurrence.qubits == (1,)
