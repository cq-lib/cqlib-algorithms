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

"""Tests for quantum subspace expansion."""

from __future__ import annotations

import math

import numpy as np
import pytest
from cqlib import Circuit
from cqlib.qis import Hamiltonian, PauliString, Phase

from cqlib_algorithms.error_mitigation.core import QemExecutor
from cqlib_algorithms.error_mitigation.inference import ConvergenceWarning
from cqlib_algorithms.error_mitigation.techniques.qse import (
    execute_with_qse,
    generalized_eigenvalue,
    qse_executor,
)
from cqlib_algorithms.protocols import EstimatorRequest

_ATOL = 1e-9


def _two_state_subspace() -> tuple[tuple[PauliString, ...], Hamiltonian, float]:
    """Return the check operator, code Hamiltonian, and exact ground energy.

    The code Hamiltonian :math:`Z_0 + 0.5 X_0` acts on the single qubit whose
    span of :math:`\\{|00\\rangle, |10\\rangle\\}` already contains its exact
    ground state, so a subspace expanded by :math:`X_0` can reach it.
    """
    code_hamiltonian = Hamiltonian(2)
    code_hamiltonian.add_term(PauliString.from_str("ZI"), 1.0)
    code_hamiltonian.add_term(PauliString.from_str("XI"), 0.5)
    code_hamiltonian.simplify()
    check_operators = (PauliString.from_str("XI"),)
    ground = float(np.linalg.eigvalsh(code_hamiltonian.to_matrix())[0])
    return check_operators, code_hamiltonian, ground


def _gram_entry(
    executor: QemExecutor,
    circuit: Circuit,
    left: PauliString,
    right: PauliString,
    /,
) -> complex:
    """Return the exact subspace overlap of two basis operators."""
    product = left * right
    phase = complex(product.phase.to_complex())
    free = product.copy()
    free.phase = Phase.plus()
    value = executor.estimate(
        [EstimatorRequest(circuit=circuit, observable=free)]
    )[0].expectation
    return phase * value


def test_a_two_by_two_subspace_matches_the_analytic_diagonalisation() -> None:
    hamiltonian = np.array([[1.0, 0.5], [0.5, -0.5]])
    overlap = np.array([[1.0, 0.25], [0.25, 1.0]])

    value = generalized_eigenvalue(hamiltonian, overlap)

    # det(H - E S) is the quadratic a E^2 + b E + c, and its smaller root is
    # the corrected energy the generalized problem reports.
    a = overlap[0, 0] * overlap[1, 1] - overlap[0, 1] * overlap[1, 0]
    b = -(
        hamiltonian[0, 0] * overlap[1, 1]
        + hamiltonian[1, 1] * overlap[0, 0]
        - hamiltonian[0, 1] * overlap[1, 0]
        - hamiltonian[1, 0] * overlap[0, 1]
    )
    c = hamiltonian[0, 0] * hamiltonian[1, 1] - hamiltonian[0, 1] * hamiltonian[1, 0]
    expected = (-b - math.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)

    assert value == pytest.approx(expected, abs=_ATOL)


def test_the_output_equals_the_ground_energy_when_the_subspace_contains_it(
    exact_executor,
) -> None:
    check_operators, code_hamiltonian, ground = _two_state_subspace()
    circuit = Circuit(2)

    # The subspace contains the exact ground state, so the projected energy
    # functional attains it and the mitigation returns the ground energy.
    value = execute_with_qse(
        circuit,
        exact_executor,
        code_hamiltonian,
        check_operators=check_operators,
        code_hamiltonian=code_hamiltonian,
    )

    assert value == pytest.approx(ground, abs=_ATOL)


def test_the_overlap_matrix_is_hermitian_and_positive_definite(
    exact_executor,
) -> None:
    check_operators, _, _ = _two_state_subspace()
    circuit = Circuit(2)
    basis = (PauliString.from_str("II"), check_operators[0])

    # The subspace overlap is the Gram matrix of the vectors B_j |psi>, so it
    # is Hermitian and its spectrum is non-negative for independent vectors.
    overlap = np.array(
        [
            [
                _gram_entry(exact_executor, circuit, left, right)
                for right in basis
            ]
            for left in basis
        ]
    )

    assert np.allclose(overlap, overlap.conj().T, atol=_ATOL)
    assert float(np.linalg.eigvalsh(overlap)[0]) > 0.0


def test_a_non_positive_overlap_eigenvalue_is_regularised_and_warns() -> None:
    hamiltonian = np.array([[1.0, 0.5], [0.5, -0.5]])
    # The diagonal is one but the off-diagonal is larger, so this overlap is
    # indefinite in the way a sampling-noise-dominated estimate can be.
    overlap = np.array([[1.0, 1.5], [1.5, 1.0]])

    with pytest.warns(ConvergenceWarning, match="positive definite"):
        value = generalized_eigenvalue(hamiltonian, overlap)

    assert math.isfinite(value)


def test_check_operators_and_code_hamiltonian_are_required(
    exact_executor,
) -> None:
    _, code_hamiltonian, _ = _two_state_subspace()
    circuit = Circuit(2)
    observable = PauliString.from_str("ZI")

    # Both selections are keyword-only and mandatory, so omitting either is a
    # call-site error.
    with pytest.raises(TypeError):
        execute_with_qse(  # type: ignore[call-arg]
            circuit, exact_executor, observable
        )

    # An empty selection cannot span a subspace.
    with pytest.raises(ValueError, match="at least one"):
        qse_executor(
            exact_executor,
            check_operators=[],
            code_hamiltonian=code_hamiltonian,
        )
