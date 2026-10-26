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

"""Independent matrix and boundary checks for exact estimation."""

from __future__ import annotations

import math

import numpy as np
import pytest
from cqlib import Circuit, Parameter
from cqlib.qis import Hamiltonian, PauliString, Phase

from cqlib_algorithms.operators.estimator import StatevectorEstimator
from cqlib_algorithms.operators.estimator.statevector_estimator import (
    StatevectorEstimator as ModuleStatevectorEstimator,
)
from cqlib_algorithms.protocols.estimator import (
    Estimator,
    EstimatorRequest,
    EstimatorResult,
)


def _matrix_reference(request: EstimatorRequest) -> float:
    """Evaluate a dense matrix reference without using Core expectation."""
    bound = request.circuit.assign_parameters(dict(request.parameter_bindings))
    order = [qubit.index for qubit in bound.qubits]
    state = bound.to_matrix(order)[:, 0]
    matrix = request.observable.to_matrix()
    value = np.vdot(state, matrix @ state)
    assert value.imag == pytest.approx(0.0, abs=1e-12)
    return float(value.real)


def test_local_export_and_abstract_base_class():
    estimator = StatevectorEstimator()
    assert StatevectorEstimator is ModuleStatevectorEstimator
    assert isinstance(estimator, Estimator)
    assert estimator.run([]) == []


def test_exact_z_expectation_and_estimator_variance_are_distinct():
    circuit = Circuit(1)
    circuit.h(0)
    request = EstimatorRequest(circuit, PauliString.from_str("Z"))

    result = StatevectorEstimator().run([request])[0]
    assert result.expectation == pytest.approx(0.0, abs=1e-14)
    assert result.variance == 0.0
    # The physical variance of a Z measurement on |+> is 1, not 0.
    assert 1.0 - result.expectation**2 == pytest.approx(1.0)


@pytest.mark.parametrize("theta", [0.0, 0.37, math.pi / 2, math.pi])
def test_parameter_binding_matches_independent_matrix_reference(theta: float):
    circuit = Circuit(1)
    circuit.ry(0, Parameter("theta"))
    request = EstimatorRequest(
        circuit,
        PauliString.from_str("Z"),
        {"theta": theta},
    )

    result = StatevectorEstimator().run([request])[0]
    assert result.expectation == pytest.approx(math.cos(theta), abs=1e-12)
    assert result.expectation == pytest.approx(_matrix_reference(request), abs=1e-12)
    assert circuit.used_symbols == ["theta"]


def test_pauli_y_sign_and_negative_pauli_phase():
    circuit = Circuit(1)
    circuit.h(0)
    circuit.rz(0, math.pi / 2)
    positive = EstimatorRequest(circuit, PauliString.from_str("Y"))
    negative_pauli = PauliString.from_str("Y")
    negative_pauli.phase = Phase.minus()
    negative = EstimatorRequest(circuit, negative_pauli)

    results = StatevectorEstimator().run([positive, negative])
    assert [result.expectation for result in results] == pytest.approx([1.0, -1.0])
    assert [result.expectation for result in results] == pytest.approx(
        [_matrix_reference(positive), _matrix_reference(negative)],
        abs=1e-12,
    )


@pytest.mark.parametrize(
    "label",
    ["I", "X", "Y", "Z", "XY", "YY", "YZ", "XYY", "YYY", "XYZ"],
)
def test_all_pauli_axes_and_odd_even_y_counts_match_dense_reference(label: str):
    circuit = Circuit(len(label))
    for qubit in range(len(label)):
        circuit.ry(qubit, 0.31 + qubit * 0.22)
        circuit.rz(qubit, -0.43 + qubit * 0.17)
    request = EstimatorRequest(circuit, PauliString.from_str(label))

    result = StatevectorEstimator().run([request])[0]
    assert result.expectation == pytest.approx(_matrix_reference(request), abs=1e-12)


@pytest.mark.parametrize("label", ["Y", "YY", "YYY"])
def test_y_eigenstates_have_positive_expectation_for_odd_and_even_counts(
    label: str,
):
    circuit = Circuit(len(label))
    for qubit in range(len(label)):
        circuit.h(qubit)
        circuit.rz(qubit, math.pi / 2)
    positive = EstimatorRequest(circuit, PauliString.from_str(label))
    negative_pauli = PauliString.from_str(label)
    negative_pauli.phase = Phase.minus()
    negative = EstimatorRequest(circuit, negative_pauli)

    results = StatevectorEstimator().run([positive, negative])
    assert [result.expectation for result in results] == pytest.approx([1.0, -1.0])
    assert [result.expectation for result in results] == pytest.approx(
        [_matrix_reference(positive), _matrix_reference(negative)],
        abs=1e-12,
    )


def test_hamiltonian_expectation_matches_independent_pauli_matrix_sum():
    circuit = Circuit(2)
    circuit.ry(0, 0.67)
    circuit.h(1)
    circuit.cx(1, 0)
    observable = Hamiltonian(2)
    observable.add_term(PauliString.from_str("ZI"), 0.6)
    observable.add_term(PauliString.from_str("IX"), -0.25)
    observable.add_term(PauliString.from_str("YY"), 0.4)
    request = EstimatorRequest(circuit, observable)

    z = np.diag([1.0, -1.0])
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    identity = np.eye(2)
    matrix = (0.6 * np.kron(z, identity) - 0.25 * np.kron(identity, x)).astype(complex)
    matrix += 0.4 * np.kron(y, y)
    state = circuit.to_matrix([q.index for q in circuit.qubits])[:, 0]
    expected = float(np.vdot(state, matrix @ state).real)

    result = StatevectorEstimator().run([request])[0]
    assert result.expectation == pytest.approx(expected, abs=1e-12)
    assert result.expectation == pytest.approx(_matrix_reference(request), abs=1e-12)


def test_effective_phases_and_imaginary_duplicate_terms_cancel_exactly():
    circuit = Circuit(1)
    circuit.x(0)
    plus = PauliString.from_str("Z")
    minus = PauliString.from_str("Z")
    minus.phase = Phase.minus()
    observable = Hamiltonian(1)
    observable.add_term(plus, 1j)
    observable.add_term(minus, 1j)
    phased = PauliString.from_str("Z")
    phased.phase = Phase.i()
    observable.add_term(phased, -0.5j)
    request = EstimatorRequest(circuit, observable)

    result = StatevectorEstimator().run([request])[0]
    assert result.expectation == pytest.approx(-0.5, abs=1e-12)
    assert result.expectation == pytest.approx(_matrix_reference(request), abs=1e-12)


def test_noncontiguous_reordered_qubits_follow_circuit_insertion_order():
    circuit = Circuit([7, 2])
    circuit.h(7)
    circuit.x(2)
    low_logical = EstimatorRequest(circuit, PauliString.from_str("IZ"))
    high_logical = EstimatorRequest(circuit, PauliString.from_str("ZI"))

    assert [q.index for q in circuit.qubits] == [7, 2]
    values = StatevectorEstimator().run([low_logical, high_logical])
    assert [value.expectation for value in values] == pytest.approx([0.0, -1.0])
    assert [value.expectation for value in values] == pytest.approx(
        [_matrix_reference(low_logical), _matrix_reference(high_logical)],
        abs=1e-12,
    )


def test_zero_width_and_empty_hamiltonian_are_exactly_defined():
    circuit = Circuit(0)
    identity = EstimatorRequest(circuit, PauliString(0))
    zero = EstimatorRequest(circuit, Hamiltonian(0))

    values = StatevectorEstimator().run([identity, zero])
    assert values == [EstimatorResult(1.0, 0.0), EstimatorResult(0.0, 0.0)]
    assert [value.expectation for value in values] == pytest.approx(
        [_matrix_reference(identity), _matrix_reference(zero)]
    )


def test_batch_preserves_order_across_different_circuits_and_observables():
    first = EstimatorRequest(Circuit(1), PauliString.from_str("Z"))
    second_circuit = Circuit(1)
    second_circuit.x(0)
    second = EstimatorRequest(second_circuit, PauliString.from_str("Z"))
    third = EstimatorRequest(second_circuit, Hamiltonian(1))

    results = StatevectorEstimator().run([first, second, third])
    assert [result.expectation for result in results] == pytest.approx([1.0, -1.0, 0.0])
    assert all(result.variance == 0.0 for result in results)


def test_positive_shot_request_is_explicitly_rejected():
    request = EstimatorRequest(Circuit(1), PauliString.from_str("Z"), shots=100)

    with pytest.raises(ValueError, match="shot-based estimation is not supported"):
        StatevectorEstimator().run([request])


def test_request_revalidates_mutable_core_objects_before_execution():
    circuit = Circuit(1)
    observable = Hamiltonian.from_pauli(PauliString.from_str("Z"))
    request = EstimatorRequest(circuit, observable)
    observable.add_term(PauliString.from_str("X"), 1e-11j)

    with pytest.raises(ValueError, match="Hermitian"):
        StatevectorEstimator().run([request])


def test_new_symbol_added_after_request_creation_is_revalidated():
    circuit = Circuit(1)
    request = EstimatorRequest(circuit, PauliString.from_str("Z"))
    circuit.ry(0, Parameter("new_theta"))

    with pytest.raises(ValueError, match="missing"):
        StatevectorEstimator().run([request])


@pytest.mark.parametrize("directive", ["measure", "reset"])
def test_top_level_measurement_and_reset_are_rejected(directive: str):
    circuit = Circuit(1)
    if directive == "measure":
        circuit.measure(0)
    else:
        circuit.reset(0)
    request = EstimatorRequest(circuit, PauliString.from_str("Z"))

    with pytest.raises(ValueError, match="unitary circuit"):
        StatevectorEstimator().run([request])


def test_nonunitary_directive_hidden_in_circuit_gate_is_rejected():
    inner = Circuit(1)
    inner.reset(0)
    outer = Circuit(1)
    outer.append_circuit_gate(inner.to_gate("hidden_reset"), [0])
    request = EstimatorRequest(outer, PauliString.from_str("Z"))

    with pytest.raises(ValueError, match="unitary circuit"):
        StatevectorEstimator().run([request])


def test_barriers_are_allowed_but_delay_is_not():
    barrier = Circuit(1)
    barrier.h(0)
    barrier.barrier([0])
    request = EstimatorRequest(barrier, PauliString.from_str("X"))
    assert StatevectorEstimator().run([request])[0].expectation == pytest.approx(1.0)

    delayed = Circuit(1)
    delayed.delay(0, 2.0)
    delayed_request = EstimatorRequest(delayed, PauliString.from_str("Z"))
    with pytest.raises(ValueError, match="unitary circuit"):
        StatevectorEstimator().run([delayed_request])


def test_batch_entries_require_the_frozen_request_type():
    with pytest.raises(TypeError, match="EstimatorRequest instances"):
        StatevectorEstimator().run([object()])  # type: ignore[list-item]
