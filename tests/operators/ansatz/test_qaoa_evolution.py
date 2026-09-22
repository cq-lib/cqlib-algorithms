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

"""Tests for single-layer QAOA Hamiltonian evolutions."""

from __future__ import annotations

import numpy as np
import pytest
from cqlib.circuit.ansatz import EvolutionStrategy
from cqlib.qis import Hamiltonian, PauliString, Phase, TrotterMode

from cqlib_algorithms.operators.ansatz import (
    QAOACostEvolution,
    QAOAMixerEvolution,
)
from cqlib_algorithms.protocols import Ansatz


def _term(paulis: str, coefficient: complex = 1.0) -> Hamiltonian:
    hamiltonian = Hamiltonian(len(paulis))
    hamiltonian.add_term(PauliString.from_str(paulis), coefficient)
    return hamiltonian


def _duplicate_terms(
    coefficients: tuple[complex, ...],
    *,
    phases: tuple[Phase, ...] | None = None,
) -> Hamiltonian:
    hamiltonian = Hamiltonian(1)
    if phases is None:
        phases = tuple(Phase.plus() for _ in coefficients)
    for coefficient, phase in zip(coefficients, phases, strict=True):
        pauli = PauliString.from_str("Z")
        pauli.phase = phase
        hamiltonian.add_term(pauli, coefficient)
    return hamiltonian


def _matrix_exponential_hermitian(matrix: np.ndarray, time: float) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return (eigenvectors * np.exp(-1j * time * eigenvalues)) @ eigenvectors.conj().T


def _x_mixer_matrix(num_qubits: int) -> np.ndarray:
    hamiltonian = np.zeros((2**num_qubits, 2**num_qubits), dtype=np.complex128)
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    identity = np.eye(2, dtype=np.complex128)
    for qubit in range(num_qubits):
        factors = [
            x if index == qubit else identity for index in reversed(range(num_qubits))
        ]
        term = factors[0]
        for factor in factors[1:]:
            term = np.kron(term, factor)
        hamiltonian += term
    return hamiltonian


def test_qaoa_cost_layer_matches_independent_matrix_exponential():
    cost = Hamiltonian(2)
    cost.add_term(PauliString.from_str("ZZ"), 0.5)
    cost.add_term(PauliString.from_str("ZI"), -0.3)
    component = QAOACostEvolution(cost)
    gamma = 0.41

    circuit = component.build_circuit("q")
    actual = circuit.assign_parameters({"q_gamma": gamma}).to_matrix()
    expected = _matrix_exponential_hermitian(cost.to_matrix(), gamma)

    assert isinstance(component, Ansatz)
    assert component.validate() is None
    assert component.num_qubits() == 2
    assert component.num_parameters() == 1
    assert circuit.used_symbols == ["q_gamma"]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_default_qaoa_mixer_matches_sum_of_x_matrix_exponential():
    component = QAOAMixerEvolution(3)
    beta = -0.27

    circuit = component.build_circuit("q")
    actual = circuit.assign_parameters({"q_beta": beta}).to_matrix()
    expected = _matrix_exponential_hermitian(_x_mixer_matrix(3), beta)

    assert isinstance(component, Ansatz)
    assert component.num_qubits() == 3
    assert component.num_parameters() == 1
    assert circuit.used_symbols == ["q_beta"]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_qaoa_cost_takes_defensive_hamiltonian_snapshot():
    original = _term("Z")
    snapshot = original.copy()
    component = QAOACostEvolution(original)
    original.add_term(PauliString.from_str("X"), 2.0)
    gamma = 0.19

    actual = (
        component.build_circuit("p").assign_parameters({"p_gamma": gamma}).to_matrix()
    )
    expected = _matrix_exponential_hermitian(snapshot.to_matrix(), gamma)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_qaoa_mixer_takes_defensive_hamiltonian_snapshot():
    original = _term("Y")
    snapshot = original.copy()
    component = QAOAMixerEvolution(1, mixer_operator=original)
    original.add_term(PauliString.from_str("Z"), 2.0)
    beta = 0.23

    actual = (
        component.build_circuit("p").assign_parameters({"p_beta": beta}).to_matrix()
    )
    expected = _matrix_exponential_hermitian(snapshot.to_matrix(), beta)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    "component", [QAOACostEvolution(_term("Z")), QAOAMixerEvolution(1)]
)
def test_qaoa_layers_return_independent_circuits_and_validate_prefix(component):
    first = component.build_circuit("p")
    second = component.build_circuit("p")

    assert first == second
    assert first is not second
    first.x(0)
    assert first != second
    with pytest.raises(TypeError, match="prefix must be a string"):
        component.build_circuit(1)  # type: ignore[arg-type]


def test_qaoa_layers_accept_explicit_core_evolution_strategy():
    strategy = EvolutionStrategy.trotter(TrotterMode.second_order(), 2)
    cost = Hamiltonian(1)
    cost.add_term(PauliString.from_str("X"), 1.0)
    cost.add_term(PauliString.from_str("Z"), 1.0)

    cost_component = QAOACostEvolution(cost, strategy=strategy)
    mixer_component = QAOAMixerEvolution(1, mixer_operator=cost, strategy=strategy)

    assert cost_component.build_circuit("q").used_symbols == ["q_gamma"]
    assert mixer_component.build_circuit("q").used_symbols == ["q_beta"]


def test_qaoa_cost_rejects_invalid_operator_and_strategy():
    with pytest.raises(TypeError, match="cost_operator must be a Hamiltonian"):
        QAOACostEvolution("Z")  # type: ignore[arg-type]
    with pytest.raises(
        TypeError, match="strategy must be an EvolutionStrategy or None"
    ):
        QAOACostEvolution(_term("Z"), strategy="auto")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty after simplification"):
        QAOACostEvolution(Hamiltonian(1))
    with pytest.raises(ValueError, match="not Hermitian"):
        QAOACostEvolution(_term("Z", 1j))


@pytest.mark.parametrize("coefficient", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("component", ["cost", "mixer"])
def test_qaoa_layers_reject_non_finite_coefficients(component, coefficient):
    operator = _term("Z", coefficient)

    with pytest.raises(
        ValueError, match=rf"{component}_operator coefficients must be finite"
    ):
        if component == "cost":
            QAOACostEvolution(operator)
        else:
            QAOAMixerEvolution(1, mixer_operator=operator)


@pytest.mark.parametrize("component", ["cost", "mixer"])
def test_qaoa_layers_reject_coefficient_overflow_during_aggregation(component):
    maximum = np.finfo(float).max
    operator = _duplicate_terms((maximum, maximum))

    with pytest.raises(
        ValueError, match=rf"{component}_operator coefficients must be finite"
    ):
        if component == "cost":
            QAOACostEvolution(operator)
        else:
            QAOAMixerEvolution(1, mixer_operator=operator)


@pytest.mark.parametrize("component", ["cost", "mixer"])
def test_qaoa_layers_absorb_pauli_phase_into_real_coefficient(component):
    operator = _duplicate_terms((-1j,), phases=(Phase.i(),))
    expected = _term("Z")

    if component == "cost":
        circuit = QAOACostEvolution(operator).build_circuit("q")
        bound = circuit.assign_parameters({"q_gamma": 0.37})
    else:
        circuit = QAOAMixerEvolution(1, mixer_operator=operator).build_circuit("q")
        bound = circuit.assign_parameters({"q_beta": 0.37})

    np.testing.assert_allclose(
        bound.to_matrix(),
        _matrix_exponential_hermitian(expected.to_matrix(), 0.37),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("component", ["cost", "mixer"])
def test_qaoa_layers_accept_duplicate_terms_with_canceling_imaginary_parts(component):
    operator = _duplicate_terms((1.0 + 2.0j, 2.0 - 2.0j))
    expected = _term("Z", 3.0)

    if component == "cost":
        circuit = QAOACostEvolution(operator).build_circuit("q")
        bound = circuit.assign_parameters({"q_gamma": -0.19})
    else:
        circuit = QAOAMixerEvolution(1, mixer_operator=operator).build_circuit("q")
        bound = circuit.assign_parameters({"q_beta": -0.19})

    np.testing.assert_allclose(
        bound.to_matrix(),
        _matrix_exponential_hermitian(expected.to_matrix(), -0.19),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("component", ["cost", "mixer"])
def test_qaoa_layers_strictly_reject_residual_imaginary_coefficient(component):
    operator = _duplicate_terms((1.0 + 1e-15j, 2.0))

    with pytest.raises(ValueError, match=rf"{component}_operator is not Hermitian"):
        if component == "cost":
            QAOACostEvolution(operator)
        else:
            QAOAMixerEvolution(1, mixer_operator=operator)


@pytest.mark.parametrize(
    "operator",
    [
        Hamiltonian(1),
        _duplicate_terms((1.0, -1.0)),
        _duplicate_terms((1.0, 1.0), phases=(Phase.plus(), Phase.minus())),
    ],
    ids=["empty", "coefficient-cancellation", "phase-cancellation"],
)
@pytest.mark.parametrize("component", ["cost", "mixer"])
def test_qaoa_layers_reject_operators_empty_after_simplification(component, operator):
    with pytest.raises(
        ValueError,
        match=rf"{component}_operator must not be empty after simplification",
    ):
        if component == "cost":
            QAOACostEvolution(operator)
        else:
            QAOAMixerEvolution(1, mixer_operator=operator)


def test_qaoa_cost_rejects_zero_qubit_identity_hamiltonian():
    cost = Hamiltonian(0)
    cost.add_term(PauliString(0), 1.0)

    with pytest.raises(
        ValueError, match="cost_operator.num_qubits must be greater than zero"
    ):
        QAOACostEvolution(cost)


def test_qaoa_mixer_rejects_invalid_inputs():
    with pytest.raises(TypeError, match="num_qubits must be an integer"):
        QAOAMixerEvolution(True)
    with pytest.raises(ValueError, match="greater than zero"):
        QAOAMixerEvolution(0)
    with pytest.raises(TypeError, match="mixer_operator must be a Hamiltonian or None"):
        QAOAMixerEvolution(1, mixer_operator="X")  # type: ignore[arg-type]
    with pytest.raises(
        TypeError, match="strategy must be an EvolutionStrategy or None"
    ):
        QAOAMixerEvolution(1, strategy="auto")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="act on num_qubits qubits"):
        QAOAMixerEvolution(2, mixer_operator=_term("X"))
    with pytest.raises(ValueError, match="not Hermitian"):
        QAOAMixerEvolution(1, mixer_operator=_term("X", 1j))


def test_qaoa_mixer_reports_strategy_error_before_custom_width_error():
    with pytest.raises(
        TypeError, match="strategy must be an EvolutionStrategy or None"
    ):
        QAOAMixerEvolution(
            2,
            mixer_operator=_term("X"),
            strategy="auto",  # type: ignore[arg-type]
        )


def test_qaoa_mixer_reports_custom_width_error_before_coefficient_error():
    with pytest.raises(ValueError, match="act on num_qubits qubits"):
        QAOAMixerEvolution(2, mixer_operator=_term("X", float("nan")))
