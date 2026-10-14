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

"""Tests for Hamiltonian phase estimation."""

from __future__ import annotations

import math

import numpy as np
import pytest
from cqlib import Circuit
from cqlib.qis import Hamiltonian, PauliString, TrotterMode

from cqlib_algorithms import (
    HamiltonianPhaseEstimation,
    HamiltonianPhaseEstimationRequest,
)
from cqlib_algorithms.operators import StatevectorSampler


def _z_hamiltonian() -> Hamiltonian:
    return Hamiltonian.from_pauli(PauliString.from_str("Z"))


@pytest.mark.integration
@pytest.mark.parametrize(
    ("prepare_one", "expected"),
    [(False, 1.0), (True, -1.0)],
)
def test_hpe_recovers_positive_and_negative_z_eigenvalues(
    prepare_one: bool,
    expected: float,
) -> None:
    preparation = Circuit(1)
    if prepare_one:
        preparation.x(0)
    result = HamiltonianPhaseEstimation(StatevectorSampler()).run(
        [
            HamiltonianPhaseEstimationRequest(
                _z_hamiltonian(),
                preparation,
                3,
                32,
                bound=2.0,
            )
        ]
    )[0]

    assert result.eigenvalue == pytest.approx(expected, abs=1e-12)
    assert dict(result.eigenvalue_probabilities) == {expected: 1.0}
    assert result.bound == 2.0
    assert result.scale == math.pi / 2.0


@pytest.mark.integration
def test_hpe_preserves_identity_shift_and_batch_order() -> None:
    hamiltonian = Hamiltonian(1)
    hamiltonian.add_term(PauliString.from_str("I"), 2.0)
    hamiltonian.add_term(PauliString.from_str("Z"), 1.0)
    prepare_zero = Circuit(1)
    prepare_one = Circuit(1)
    prepare_one.x(0)

    results = HamiltonianPhaseEstimation(StatevectorSampler()).run(
        [
            HamiltonianPhaseEstimationRequest(
                hamiltonian,
                preparation,
                3,
                1,
                bound=4.0,
            )
            for preparation in (prepare_zero, prepare_one)
        ]
    )

    assert [result.eigenvalue for result in results] == [3.0, 1.0]
    assert [dict(result.eigenvalue_probabilities) for result in results] == [
        {3.0: 1.0},
        {1.0: 1.0},
    ]


@pytest.mark.integration
def test_hpe_superposition_samples_both_eigenvalues() -> None:
    preparation = Circuit(1)
    preparation.h(0)

    result = HamiltonianPhaseEstimation(StatevectorSampler()).run(
        [
            HamiltonianPhaseEstimationRequest(
                _z_hamiltonian(),
                preparation,
                3,
                256,
                bound=2.0,
            )
        ]
    )[0]

    assert set(result.eigenvalue_probabilities) == {-1.0, 1.0}
    assert sum(result.eigenvalue_probabilities.values()) == pytest.approx(1.0)


@pytest.mark.reference
def test_noncommuting_product_formula_approaches_dense_exponential() -> None:
    hamiltonian = Hamiltonian(1)
    hamiltonian.add_term(PauliString.from_str("X"), 0.37)
    hamiltonian.add_term(PauliString.from_str("Z"), -0.61)
    circuit, bound, scale = HamiltonianPhaseEstimation.build_unitary(
        hamiltonian,
        bound=1.5,
        trotter_steps=64,
        trotter_mode=TrotterMode.second_order(),
    )
    matrix = np.asarray(hamiltonian.to_matrix(), dtype=np.complex128)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    expected = (
        eigenvectors @ np.diag(np.exp(1j * scale * eigenvalues)) @ eigenvectors.conj().T
    )

    assert bound == 1.5
    np.testing.assert_allclose(
        circuit.to_matrix(),
        expected,
        rtol=0.0,
        atol=2e-4,
    )


def test_automatic_bound_is_sum_of_absolute_pauli_coefficients() -> None:
    hamiltonian = Hamiltonian(1)
    hamiltonian.add_term(PauliString.from_str("X"), 0.25)
    hamiltonian.add_term(PauliString.from_str("Z"), -0.75)

    _, bound, scale = HamiltonianPhaseEstimation.build_unitary(hamiltonian)

    assert bound == 1.0
    assert scale == math.pi


def test_request_snapshots_hamiltonian_and_state_preparation() -> None:
    hamiltonian = _z_hamiltonian()
    preparation = Circuit(1)
    request = HamiltonianPhaseEstimationRequest(
        hamiltonian,
        preparation,
        3,
        16,
        bound=2.0,
    )
    hamiltonian.add_term(PauliString.from_str("X"), 1.0)
    preparation.x(0)

    assert request.hamiltonian.num_terms == 1
    assert request.state_preparation.operations == []


def test_hpe_rejects_invalid_hamiltonian_and_bound() -> None:
    non_hermitian = Hamiltonian(1)
    non_hermitian.add_term(PauliString.from_str("X"), 1.0j)
    with pytest.raises(ValueError, match="not Hermitian"):
        HamiltonianPhaseEstimationRequest(
            non_hermitian,
            Circuit(1),
            3,
            1,
        )
    with pytest.raises(ValueError, match="positive and finite"):
        HamiltonianPhaseEstimationRequest(
            _z_hamiltonian(),
            Circuit(1),
            3,
            1,
            bound=0.0,
        )


def test_automatic_bound_converts_accumulation_overflow_to_value_error() -> None:
    hamiltonian = Hamiltonian(1)
    hamiltonian.add_term(PauliString.from_str("X"), 1e308)
    hamiltonian.add_term(PauliString.from_str("Z"), 1e308)

    with pytest.raises(ValueError, match="coefficient bound must be finite"):
        HamiltonianPhaseEstimation.build_unitary(hamiltonian)


def test_hpe_rejects_mismatched_state_width() -> None:
    with pytest.raises(ValueError, match="same qubit count"):
        HamiltonianPhaseEstimationRequest(
            _z_hamiltonian(),
            Circuit(2),
            3,
            1,
        )
