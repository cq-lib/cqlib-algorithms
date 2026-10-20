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

"""Tests for commuting-block Hamiltonian variational evolution."""

from __future__ import annotations

import numpy as np
import pytest
from cqlib.qis import Hamiltonian, PauliString

from cqlib_algorithms.operators.ansatz import HamiltonianVariationalAnsatz


def _hamiltonian(*terms: tuple[str, complex]) -> Hamiltonian:
    operator = Hamiltonian(len(terms[0][0]))
    for label, coefficient in terms:
        operator.add_term(PauliString.from_str(label), coefficient)
    return operator


def _evolution(operator: Hamiltonian, time: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(operator.to_matrix())
    return (vectors * np.exp(-1j * time * values)) @ vectors.conj().T


def test_automatic_stable_commuting_groups_match_matrix_reference():
    hamiltonian = _hamiltonian(
        ("XI", 0.7),
        ("IX", -0.2),
        ("ZI", 0.5),
        ("IZ", 0.9),
    )
    x_block = _hamiltonian(("XI", 0.7), ("IX", -0.2))
    z_block = _hamiltonian(("ZI", 0.5), ("IZ", 0.9))
    component = HamiltonianVariationalAnsatz(hamiltonian)

    actual = (
        component.build_circuit("h")
        .assign_parameters({"h_0": 0.23, "h_1": -0.41})
        .to_matrix()
    )
    expected = _evolution(z_block, -0.41) @ _evolution(x_block, 0.23)

    assert component.num_qubits() == 2
    assert component.num_parameters() == 2
    assert component.build_circuit("h").used_symbols == ["h_0", "h_1"]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_explicit_block_order_and_repetitions_are_preserved():
    first = _hamiltonian(("ZI", 1.0), ("IZ", 0.3))
    second = _hamiltonian(("XX", -0.8), ("YY", 0.2))
    component = HamiltonianVariationalAnsatz([second, first], reps=2)
    values = (0.1, 0.2, 0.3, 0.4)

    actual = (
        component.build_circuit("p")
        .assign_parameters({f"p_{index}": value for index, value in enumerate(values)})
        .to_matrix()
    )
    expected = np.eye(4, dtype=np.complex128)
    for block, time in zip((second, first, second, first), values):
        expected = _evolution(block, time) @ expected

    assert component.num_parameters() == 4
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_identity_only_hamiltonian_becomes_parameter_free_identity():
    component = HamiltonianVariationalAnsatz(_hamiltonian(("II", 2.0)), reps=3)

    assert component.num_qubits() == 2
    assert component.num_parameters() == 0
    np.testing.assert_array_equal(component.build_circuit("h").to_matrix(), np.eye(4))


def test_automatic_input_is_defensively_snapshotted():
    original = _hamiltonian(("Z", 1.0))
    component = HamiltonianVariationalAnsatz(original)
    original.add_term(PauliString.from_str("X"), 1.0)

    actual = component.build_circuit("h").assign_parameters({"h_0": 0.3}).to_matrix()
    np.testing.assert_allclose(
        actual,
        _evolution(_hamiltonian(("Z", 1.0)), 0.3),
        rtol=0.0,
        atol=1e-12,
    )


def test_explicit_blocks_must_commute_internally():
    with pytest.raises(ValueError, match="must commute internally"):
        HamiltonianVariationalAnsatz([_hamiltonian(("X", 1.0), ("Z", 1.0))])


def test_rejects_hamiltonian_removed_by_core_simplification_tolerance():
    with pytest.raises(ValueError, match="must not be empty after simplification"):
        HamiltonianVariationalAnsatz(_hamiltonian(("Z", 1e-10)))


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        ([], ValueError, "must not be empty"),
        (["Z"], TypeError, "only Hamiltonian"),
        (
            [_hamiltonian(("Z", 1.0)), _hamiltonian(("ZZ", 1.0))],
            ValueError,
            "same number of qubits",
        ),
        (_hamiltonian(("Z", 1j)), ValueError, "not Hermitian"),
        ("Z", TypeError, "Hamiltonian or a sequence"),
    ],
)
def test_invalid_hamiltonians_are_rejected(value, error, message):
    with pytest.raises(error, match=message):
        HamiltonianVariationalAnsatz(value)


def test_zero_repetitions_and_build_independence():
    component = HamiltonianVariationalAnsatz(_hamiltonian(("Z", 1.0)), reps=0)
    first = component.build_circuit("h")
    second = component.build_circuit("h")

    assert component.num_parameters() == 0
    assert first == second
    assert first is not second
    np.testing.assert_array_equal(first.to_matrix(), np.eye(2))
