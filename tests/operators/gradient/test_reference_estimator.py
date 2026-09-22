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

"""Correctness sentinels for the dense-matrix reference estimator."""

from __future__ import annotations

import math

import numpy as np
import pytest
from cqlib import Circuit, Parameter, Qubit
from cqlib.qis import Hamiltonian, PauliString

from cqlib_algorithms.protocols import EstimatorRequest


def _pauli(axis: str) -> np.ndarray:
    return {
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }[axis]


def _statevector_in_insertion_order(circuit) -> np.ndarray:
    qubit_order = [qubit.index for qubit in circuit.qubits]
    return circuit.to_matrix(qubit_order)[:, 0]


@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
def test_single_qubit_eigenstate_expectations_are_exact(
    axis: str, matrix_estimator_factory
) -> None:
    estimator = matrix_estimator_factory()
    circuit = Circuit(1)
    if axis == "X":
        circuit.h(0)
    elif axis == "Y":
        circuit.h(0)
        circuit.s(0)
    request = EstimatorRequest(circuit, PauliString.from_str(axis))
    result = estimator.run([request])[0]
    assert result.expectation == pytest.approx(1.0, abs=1e-12)


def test_odd_y_pauli_sign_matches_independent_matrix_product(
    matrix_estimator_factory,
) -> None:
    """Guard against Core's odd-Y expectation sign defect."""
    estimator = matrix_estimator_factory()
    circuit = Circuit(1)
    circuit.h(0)
    circuit.rz(0, math.pi / 2)
    state = _statevector_in_insertion_order(circuit)
    reference = float(np.real(np.vdot(state, _pauli("Y") @ state)))
    assert reference == pytest.approx(1.0, abs=1e-12)

    request = EstimatorRequest(circuit, PauliString.from_str("Y"))
    result = estimator.run([request])[0]
    assert result.expectation == pytest.approx(reference, abs=1e-12)


def test_mixed_odd_y_two_qubit_expectation_matches_manual_kron(
    matrix_estimator_factory,
) -> None:
    estimator = matrix_estimator_factory()
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.sdg(1)
    state = _statevector_in_insertion_order(circuit)
    observable = np.kron(_pauli("Y"), _pauli("X"))
    reference = float(np.real(np.vdot(state, observable @ state)))

    request = EstimatorRequest(circuit, PauliString.from_str("YX"))
    result = estimator.run([request])[0]
    assert result.expectation == pytest.approx(reference, abs=1e-12)


def test_hamiltonian_weighted_sum_matches_manual_combination(
    matrix_estimator_factory,
) -> None:
    estimator = matrix_estimator_factory()
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    bound = circuit.assign_parameters({"theta": 0.6})
    state = _statevector_in_insertion_order(bound)
    reference = float(
        np.real(np.vdot(state, (_pauli("Z") + 0.5 * _pauli("X")) @ state))
    )

    hamiltonian = Hamiltonian.from_list(
        [(PauliString.from_str("Z"), 1.0), (PauliString.from_str("X"), 0.5)]
    )
    request = EstimatorRequest(circuit, hamiltonian, {"theta": 0.6})
    result = estimator.run([request])[0]
    assert result.expectation == pytest.approx(reference, abs=1e-12)


def test_reference_estimator_rejects_shot_requests(matrix_estimator_factory) -> None:
    circuit = Circuit(1)
    request = EstimatorRequest(circuit, PauliString.from_str("Z"), shots=100)
    estimator = matrix_estimator_factory()
    with pytest.raises(ValueError):
        estimator.run([request])


def test_reference_estimator_handles_parametric_binding(
    matrix_estimator_factory,
) -> None:
    estimator = matrix_estimator_factory()
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    request = EstimatorRequest(circuit, PauliString.from_str("Z"), {"theta": 0.0})
    assert estimator.run([request])[0].expectation == pytest.approx(1.0, abs=1e-12)
    request = EstimatorRequest(circuit, PauliString.from_str("Z"), {"theta": math.pi})
    assert estimator.run([request])[0].expectation == pytest.approx(-1.0, abs=1e-12)


def test_reference_estimator_respects_reordered_qubit_insertion(
    matrix_estimator_factory,
) -> None:
    estimator = matrix_estimator_factory()
    theta = Parameter("theta")
    circuit = Circuit([Qubit(1), Qubit(0)])
    circuit.ry(Qubit(1), theta)
    request = EstimatorRequest(
        circuit,
        PauliString.from_str("IZ"),
        {"theta": 0.4},
    )

    result = estimator.run([request])[0]

    assert result.expectation == pytest.approx(math.cos(0.4), abs=1e-12)
