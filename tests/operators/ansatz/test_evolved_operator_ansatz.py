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

"""Tests for ordered variational operator evolution."""

from __future__ import annotations

import numpy as np
import pytest
from cqlib.circuit.ansatz import EvolutionStrategy
from cqlib.qis import Hamiltonian, PauliString

from cqlib_algorithms.operators.ansatz import EvolvedOperatorAnsatz
from cqlib_algorithms.protocols import Ansatz


def _hamiltonian(*terms: tuple[str, complex]) -> Hamiltonian:
    width = len(terms[0][0])
    operator = Hamiltonian(width)
    for label, coefficient in terms:
        operator.add_term(PauliString.from_str(label), coefficient)
    return operator


def _evolution(operator: Hamiltonian, time: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(operator.to_matrix())
    return (vectors * np.exp(-1j * time * values)) @ vectors.conj().T


def test_ordered_repeated_evolution_matches_independent_matrix_reference():
    first = _hamiltonian(("ZI", 0.7), ("IZ", -0.2))
    second = _hamiltonian(("XX", 0.4))
    component = EvolvedOperatorAnsatz([first, second], reps=2)
    values = (0.13, -0.24, 0.31, 0.42)
    bindings = {f"p_{index}": value for index, value in enumerate(values)}

    actual = component.build_circuit("p").assign_parameters(bindings).to_matrix()
    expected = np.eye(4, dtype=np.complex128)
    for operator, time in zip((first, second, first, second), values):
        expected = _evolution(operator, time) @ expected

    assert isinstance(component, Ansatz)
    assert component.num_qubits() == 2
    assert component.num_parameters() == 4
    assert component.build_circuit("p").used_symbols == [
        "p_0",
        "p_1",
        "p_2",
        "p_3",
    ]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_identity_removal_and_exact_global_phase_retention():
    identity = _hamiltonian(("II", 1.7))
    removed = EvolvedOperatorAnsatz(identity)
    retained = EvolvedOperatorAnsatz(identity, remove_identity_operators=False)

    assert removed.num_qubits() == 2
    assert removed.num_parameters() == 0
    assert removed.build_circuit("p").used_symbols == []
    np.testing.assert_array_equal(removed.build_circuit("p").to_matrix(), np.eye(4))

    theta = -0.27
    retained_circuit = retained.build_circuit("p")
    assert retained.num_parameters() == 1
    assert retained_circuit.used_symbols == ["p_0"]
    np.testing.assert_allclose(
        retained_circuit.assign_parameters({"p_0": theta}).to_matrix(),
        np.exp(-1j * 1.7 * theta) * np.eye(4),
        rtol=0.0,
        atol=1e-12,
    )


def test_zero_repetitions_return_width_preserving_identity():
    component = EvolvedOperatorAnsatz(_hamiltonian(("XYZ", 0.5)), reps=0)

    assert component.num_parameters() == 0
    assert component.build_circuit("p").num_qubits == 3
    np.testing.assert_array_equal(component.build_circuit("p").to_matrix(), np.eye(8))


def test_operator_inputs_are_defensively_snapshotted():
    original = _hamiltonian(("Z", 1.0))
    component = EvolvedOperatorAnsatz(original)
    original.add_term(PauliString.from_str("X"), 4.0)

    actual = component.build_circuit("p").assign_parameters({"p_0": 0.2}).to_matrix()
    expected = _evolution(_hamiltonian(("Z", 1.0)), 0.2)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_builds_are_independent_and_strategy_is_supported():
    component = EvolvedOperatorAnsatz(
        _hamiltonian(("XX", 1.0), ("ZZ", 0.5)),
        strategy=EvolutionStrategy.exact(),
    )
    first = component.build_circuit("p")
    second = component.build_circuit("p")

    assert first == second
    assert first is not second
    first.x(0)
    assert first != second


def test_rejects_operator_removed_by_core_simplification_tolerance():
    with pytest.raises(ValueError, match="must not be empty after simplification"):
        EvolvedOperatorAnsatz(_hamiltonian(("Z", 1e-10)))


def test_exact_strategy_rejects_noncommuting_operator_at_construction():
    noncommuting = _hamiltonian(("X", 1.0), ("Z", 0.5))

    with pytest.raises(ValueError, match="Exact requires all Hamiltonian terms"):
        EvolvedOperatorAnsatz(
            noncommuting,
            strategy=EvolutionStrategy.exact(),
        )

    automatic = EvolvedOperatorAnsatz(noncommuting)
    assert automatic.num_parameters() == 1
    assert automatic.build_circuit("p").used_symbols == ["p_0"]


@pytest.mark.parametrize(
    ("factory", "error", "message"),
    [
        (lambda: EvolvedOperatorAnsatz([]), ValueError, "must not be empty"),
        (
            lambda: EvolvedOperatorAnsatz([_hamiltonian(("Z", 1.0)), "X"]),
            TypeError,
            "only Hamiltonian",
        ),
        (
            lambda: EvolvedOperatorAnsatz(
                [_hamiltonian(("Z", 1.0)), _hamiltonian(("ZZ", 1.0))]
            ),
            ValueError,
            "same number of qubits",
        ),
        (
            lambda: EvolvedOperatorAnsatz(_hamiltonian(("Z", 1j))),
            ValueError,
            "not Hermitian",
        ),
        (
            lambda: EvolvedOperatorAnsatz(_hamiltonian(("Z", 1.0)), strategy="exact"),
            TypeError,
            "EvolutionStrategy",
        ),
        (
            lambda: EvolvedOperatorAnsatz(
                _hamiltonian(("Z", 1.0)), remove_identity_operators=1
            ),
            TypeError,
            "must be a boolean",
        ),
    ],
)
def test_invalid_inputs_are_rejected(factory, error, message):
    with pytest.raises(error, match=message):
        factory()


def test_prefix_and_repetition_validation():
    with pytest.raises(TypeError, match="reps must be an integer"):
        EvolvedOperatorAnsatz(_hamiltonian(("Z", 1.0)), reps=True)
    with pytest.raises(ValueError, match="reps must be non-negative"):
        EvolvedOperatorAnsatz(_hamiltonian(("Z", 1.0)), reps=-1)
    with pytest.raises(TypeError, match="prefix must be a string"):
        EvolvedOperatorAnsatz(_hamiltonian(("Z", 1.0))).build_circuit(1)
