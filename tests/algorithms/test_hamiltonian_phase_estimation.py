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
from cqlib_algorithms.operators import BasisStateInitialization, StatevectorSampler
from cqlib_algorithms.operators.state_preparation import UserCircuitPreparation


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
                UserCircuitPreparation(preparation),
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
@pytest.mark.parametrize("width", [2, 3])
def test_hpe_default_bound_resolves_both_z_endpoints(width: int) -> None:
    solver = HamiltonianPhaseEstimation(StatevectorSampler())
    hamiltonian = _z_hamiltonian()
    preparations = [BasisStateInitialization(1, index) for index in (0, 1)]
    single = [
        solver.run(
            hamiltonian, state_preparation=prep, num_evaluation_qubits=width, shots=16
        )
        for prep in preparations
    ]
    batch = solver.run(
        [
            HamiltonianPhaseEstimationRequest(hamiltonian, prep, width, 16)
            for prep in preparations
        ]
    )

    assert [result.eigenvalue for result in single] == [1.0, -1.0]
    assert [result.phase_result.phase for result in single] == [0.25, 0.75]
    assert batch == single


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
                UserCircuitPreparation(preparation),
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
                UserCircuitPreparation(preparation),
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


def test_automatic_bound_is_twice_the_sum_of_absolute_pauli_coefficients() -> None:
    hamiltonian = Hamiltonian(1)
    hamiltonian.add_term(PauliString.from_str("X"), 0.25)
    hamiltonian.add_term(PauliString.from_str("Z"), -0.75)

    _, bound, scale = HamiltonianPhaseEstimation.build_unitary(hamiltonian)

    assert bound == 2.0
    assert scale == math.pi / 2.0


def test_request_snapshots_hamiltonian_and_state_preparation() -> None:
    hamiltonian = _z_hamiltonian()
    preparation = Circuit(1)
    request = HamiltonianPhaseEstimationRequest(
        hamiltonian,
        UserCircuitPreparation(preparation),
        3,
        16,
        bound=2.0,
    )
    hamiltonian.add_term(PauliString.from_str("X"), 1.0)
    preparation.x(0)

    assert request.hamiltonian.num_terms == 1
    assert request.preparation_circuit.operations == []


def test_hpe_rejects_invalid_hamiltonian_and_bound() -> None:
    non_hermitian = Hamiltonian(1)
    non_hermitian.add_term(PauliString.from_str("X"), 1.0j)
    with pytest.raises(ValueError, match="not Hermitian"):
        HamiltonianPhaseEstimationRequest(
            non_hermitian,
            UserCircuitPreparation(Circuit(1)),
            3,
            1,
        )
    with pytest.raises(ValueError, match="positive and finite"):
        HamiltonianPhaseEstimationRequest(
            _z_hamiltonian(),
            UserCircuitPreparation(Circuit(1)),
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


def test_automatic_bound_rejects_overflow_when_doubling_a_finite_sum() -> None:
    hamiltonian = Hamiltonian(1)
    hamiltonian.add_term(PauliString.from_str("Z"), 1e308)

    with pytest.raises(ValueError, match="bound.*finite"):
        HamiltonianPhaseEstimation.build_unitary(hamiltonian)


def test_hpe_rejects_mismatched_state_width() -> None:
    with pytest.raises(ValueError, match="same qubit count"):
        HamiltonianPhaseEstimationRequest(
            _z_hamiltonian(),
            UserCircuitPreparation(Circuit(2)),
            3,
            1,
        )


@pytest.mark.integration
def test_hpe_single_and_mixed_batch_share_signed_energy_mapping(monkeypatch):
    sampler = StatevectorSampler()
    submissions = []
    original_run = StatevectorSampler.run

    def record(self, requests):
        submissions.append(len(requests))
        return original_run(self, requests)

    monkeypatch.setattr(StatevectorSampler, "run", record)
    solver = HamiltonianPhaseEstimation(sampler)
    hamiltonian = _z_hamiltonian()
    prepared_one = BasisStateInitialization(1, 1)
    single = solver.run(
        hamiltonian,
        state_preparation=prepared_one,
        num_evaluation_qubits=3,
        shots=16,
        bound=2.0,
    )
    raw_one = Circuit(1)
    raw_one.x(0)
    batch = solver.run(
        [
            HamiltonianPhaseEstimationRequest(
                hamiltonian, BasisStateInitialization(1, 0), 3, 16, bound=2.0
            ),
            HamiltonianPhaseEstimationRequest(
                hamiltonian, UserCircuitPreparation(raw_one), 3, 16, bound=2.0
            ),
        ]
    )
    assert submissions == [1, 2]
    assert [result.eigenvalue for result in batch] == [1.0, -1.0]
    assert single == batch[1]
    assert single.phase_result.phase == 0.75


def test_hpe_single_requires_preparation_and_batch_owns_its_settings():
    solver = HamiltonianPhaseEstimation(StatevectorSampler())
    with pytest.raises(TypeError, match="requires state_preparation"):
        solver.run(_z_hamiltonian(), num_evaluation_qubits=2, shots=1)
    with pytest.raises(TypeError, match="batch requests"):
        solver.run([], trotter_steps=1)
    with pytest.raises(ValueError, match="trotter_steps"):
        solver.run(
            _z_hamiltonian(),
            state_preparation=BasisStateInitialization(1, 0),
            num_evaluation_qubits=2,
            shots=1,
            trotter_steps=0,
        )
    with pytest.raises(ValueError, match="same qubit count"):
        solver.run(
            _z_hamiltonian(),
            state_preparation=BasisStateInitialization(2, 0),
            num_evaluation_qubits=2,
            shots=1,
        )
