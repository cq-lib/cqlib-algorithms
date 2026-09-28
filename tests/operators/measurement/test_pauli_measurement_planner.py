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

"""Pauli planner tests with independent analytical readout probabilities."""

from __future__ import annotations

from math import sqrt

import numpy as np
import pytest
from cqlib.qis import PauliString

from cqlib import Circuit, Qubit
from cqlib_algorithms.operators.measurement import (
    PauliMeasurementPlanner,
)
from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.protocols.sampler import SamplerRequest


def _readout_probabilities(basis_circuit: Circuit) -> np.ndarray:
    """Evaluate a planner-produced basis circuit independently."""
    amplitudes = basis_circuit.to_matrix()[:, 0]
    return np.abs(amplitudes) ** 2


@pytest.mark.parametrize(
    ("axis", "prepare", "expected"),
    [
        ("X", "x_plus", (1.0, 0.0)),
        ("Y", "y_plus", (1.0, 0.0)),
        ("Z", "one", (0.0, 1.0)),
    ],
)
def test_local_basis_measurements_match_pauli_eigenstates(
    axis: str, prepare: str, expected: tuple[float, float]
) -> None:
    circuit = Circuit(1)
    if prepare == "x_plus":
        circuit.h(0)
    elif prepare == "y_plus":
        circuit.h(0)
        circuit.s(0)
    else:
        circuit.x(0)
    plan = PauliMeasurementPlanner(circuit, [PauliString.from_str(axis)]).plan()
    assert plan.groups[0].basis == (axis,)
    np.testing.assert_allclose(
        _readout_probabilities(plan.build_circuit(0)), expected, atol=1e-14
    )


def test_y_negative_eigenstate_and_independent_mixed_state_reference() -> None:
    circuit = Circuit(1)
    circuit.h(0)
    circuit.sdg(0)
    plan = PauliMeasurementPlanner(circuit, [PauliString.from_str("Y")]).plan()
    np.testing.assert_allclose(
        _readout_probabilities(plan.build_circuit(0)), (0.0, 1.0), atol=1e-14
    )

    circuit = Circuit(1)
    circuit.ry(0, np.pi / 3)
    plan = PauliMeasurementPlanner(circuit, [PauliString.from_str("X")]).plan()
    expected_x = (1.0 + sqrt(3.0) / 2.0) / 2.0
    np.testing.assert_allclose(
        _readout_probabilities(plan.build_circuit(0)),
        (expected_x, 1.0 - expected_x),
        atol=1e-14,
    )


def test_joint_qwc_readout_of_independent_pauli_eigenstates() -> None:
    circuit = Circuit(2)
    circuit.h(0)  # |+> on qubit 0
    circuit.h(1)
    circuit.s(1)  # |+i> on qubit 1
    plan = PauliMeasurementPlanner(
        circuit, [PauliString.from_str("IX"), PauliString.from_str("YI")]
    ).plan()
    assert len(plan.groups) == 1
    assert plan.groups[0].basis == ("X", "Y")
    np.testing.assert_allclose(
        _readout_probabilities(plan.build_circuit(0)),
        (1.0, 0.0, 0.0, 0.0),
        atol=1e-14,
    )


def test_identity_terms_are_bookkept_with_original_phase_and_no_circuit() -> None:
    circuit = Circuit(2)
    terms = [
        PauliString.from_str("-iII"),
        PauliString.from_str("XI"),
        PauliString.from_str("+IZ"),
    ]
    plan = PauliMeasurementPlanner(circuit, terms).plan()
    assert plan.identity_indices == (0,)
    assert len(plan.groups) == 1
    assert plan.groups[0].term_indices == (1, 2)
    assert plan.groups[0].basis == ("Z", "X")
    assert plan.term(0).phase.to_complex() == -1j
    plan.term(0).phase = PauliString.from_str("II").phase
    assert plan.term(0).phase.to_complex() == -1j
    assert plan.num_terms == 3

    empty = PauliMeasurementPlanner(circuit, []).plan()
    assert empty.groups == ()
    assert empty.identity_indices == ()
    with pytest.raises(IndexError):
        empty.build_circuit(0)

    identities = PauliMeasurementPlanner(circuit, [PauliString.from_str("-II")]).plan()
    assert identities.groups == ()
    assert identities.identity_indices == (0,)
    assert identities.term(0).phase.to_complex() == -1.0 + 0.0j


def test_reordered_noncontiguous_qubit_ids_map_by_insertion_order() -> None:
    circuit = Circuit([Qubit(9), Qubit(2)])
    circuit.h(Qubit(9))
    terms = [PauliString.from_str("IX"), PauliString.from_str("YI")]
    plan = PauliMeasurementPlanner(circuit, terms).plan()
    assert [qubit.index for qubit in plan.qubits] == [9, 2]
    assert plan.groups[0].basis == ("X", "Y")
    basis_circuit = plan.build_circuit(0)
    assert not basis_circuit.classical_values
    assert [
        (op.name, [q.index for q in op.qubits]) for op in basis_circuit.operations
    ] == [
        ("H", [9]),
        ("H", [9]),
        ("SDG", [2]),
        ("H", [2]),
    ]


def test_planned_basis_circuit_is_consumed_by_statevector_sampler() -> None:
    circuit = Circuit(2)
    circuit.h(0)
    circuit.h(1)
    circuit.s(1)
    plan = PauliMeasurementPlanner(
        circuit, [PauliString.from_str("IX"), PauliString.from_str("YI")]
    ).plan()

    basis_circuit = plan.build_circuit(0)
    result = StatevectorSampler().run([SamplerRequest(basis_circuit, plan.qubits, 32)])[
        0
    ]

    assert not basis_circuit.classical_values
    assert result.qubits == list(plan.qubits)
    assert result.counts == {"00": 32}


def test_snapshot_and_independent_repeated_readout_circuits() -> None:
    circuit = Circuit(1)
    circuit.h(0)
    term = PauliString.from_str("X")
    planner = PauliMeasurementPlanner(circuit, [term])
    circuit.x(0)
    term.set_pauli(0, PauliString.from_str("Z").get_pauli(0))
    first_plan = planner.plan()
    second_plan = planner.plan()
    first = first_plan.build_circuit(0)
    second = first_plan.build_circuit(0)
    third = second_plan.build_circuit(0)
    assert first.id != second.id != third.id
    first.x(0)
    assert len(first.operations) == len(second.operations) + 1
    assert len(second.operations) == len(third.operations)
    assert first_plan.term(0).get_pauli(0) == PauliString.from_str("X").get_pauli(0)
    np.testing.assert_allclose(_readout_probabilities(second), (1.0, 0.0), atol=1e-14)


def test_none_grouping_and_index_validation() -> None:
    circuit = Circuit(1)
    plan = PauliMeasurementPlanner(
        circuit, [PauliString.from_str("X"), PauliString.from_str("Z")], grouping="none"
    ).plan()
    assert tuple(group.term_indices for group in plan.groups) == ((0,), (1,))
    with pytest.raises(TypeError):
        plan.build_circuit(True)
    with pytest.raises(TypeError):
        plan.build_circuit(1.0)  # type: ignore[arg-type]
    with pytest.raises(IndexError):
        plan.build_circuit(-1)
    with pytest.raises(IndexError):
        plan.build_circuit(2)
    with pytest.raises(ValueError):
        PauliMeasurementPlanner(
            circuit, [PauliString.from_str("X")], grouping="commuting"
        )


def test_term_index_rejects_bool_noninteger_negative_and_out_of_range() -> None:
    plan = PauliMeasurementPlanner(
        Circuit(1), [PauliString.from_str("X"), PauliString.from_str("Z")]
    ).plan()
    assert str(plan.term(0)) == "+X"
    assert str(plan.term(1)) == "+Z"
    with pytest.raises(TypeError, match="index must be an integer"):
        plan.term(True)
    with pytest.raises(TypeError, match="index must be an integer"):
        plan.term(1.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="index must be an integer"):
        plan.term(None)  # type: ignore[arg-type]
    with pytest.raises(IndexError, match="index is out of range"):
        plan.term(-1)
    with pytest.raises(IndexError, match="index is out of range"):
        plan.term(2)
    empty = PauliMeasurementPlanner(Circuit(1), []).plan()
    with pytest.raises(IndexError, match="index is out of range"):
        empty.term(0)


def test_reject_invalid_widths_and_prior_nonunitary_operations() -> None:
    with pytest.raises(TypeError):
        PauliMeasurementPlanner("not a circuit", [])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PauliMeasurementPlanner(Circuit(0), [])
    with pytest.raises(ValueError):
        PauliMeasurementPlanner(Circuit(1), [PauliString(2)])

    measured = Circuit(1)
    measured.measure(0)
    with pytest.raises(ValueError):
        PauliMeasurementPlanner(measured, [PauliString.from_str("Z")])
    reset = Circuit(1)
    reset.reset(0)
    with pytest.raises(ValueError):
        PauliMeasurementPlanner(reset, [PauliString.from_str("Z")])


def test_static_delay_is_preserved_for_sampler_execution() -> None:
    delayed = Circuit(1)
    delayed.delay(0, 1.0)
    plan = PauliMeasurementPlanner(delayed, [PauliString.from_str("Z")]).plan()

    basis_circuit = plan.build_circuit(0)
    result = StatevectorSampler().run([SamplerRequest(basis_circuit, plan.qubits, 4)])[
        0
    ]

    assert result.counts == {"0": 4}


def test_reject_nested_measurement_and_reset_in_circuit_gate() -> None:
    inner = Circuit(1)
    inner.measure(0)
    outer = Circuit(1)
    outer.append_circuit_gate(inner.to_gate("nested_measure"), [0])
    with pytest.raises(ValueError):
        PauliMeasurementPlanner(outer, [PauliString.from_str("Z")])

    inner = Circuit(1)
    inner.reset(0)
    outer = Circuit(1)
    outer.append_circuit_gate(inner.to_gate("nested_reset"), [0])
    with pytest.raises(ValueError):
        PauliMeasurementPlanner(outer, [PauliString.from_str("Z")])
