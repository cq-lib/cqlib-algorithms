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

"""Tests for vibrational modal-occupation observables."""

import numpy as np
import pytest

from cqlib_algorithms.extension.vibration import (
    ModalOccupation,
    ModalSpace,
    OccupiedModals,
    OneHotLeakage,
)


def test_modal_occupation_is_the_expected_physical_projector():
    space = ModalSpace((2, 3))
    observable = ModalOccupation(space, 1, 2)
    expected = np.diag(
        [
            float(space.configuration_from_basis_index(index)[1] == 2)
            for index in range(space.dimension)
        ]
    )
    isometry = space.one_hot_isometry()
    mapped = observable.to_qubit_hamiltonian().to_matrix()

    np.testing.assert_array_equal(
        observable.to_vibrational_op().to_modal_matrix(), expected
    )
    np.testing.assert_array_equal(isometry.conjugate().T @ mapped @ isometry, expected)


def test_each_mode_occupation_is_identity_on_the_physical_subspace():
    space = ModalSpace((2, 3, 2))
    observables = OccupiedModals(space)
    isometry = space.one_hot_isometry()

    assert len(observables.to_vibrational_ops()) == space.num_modes
    for operator, hamiltonian in zip(
        observables.to_vibrational_ops(),
        observables.to_qubit_hamiltonians(),
    ):
        np.testing.assert_array_equal(
            operator.to_modal_matrix(),
            np.eye(space.dimension),
        )
        mapped = hamiltonian.to_matrix()
        np.testing.assert_array_equal(
            isometry.conjugate().T @ mapped @ isometry,
            np.eye(space.dimension),
        )
        np.testing.assert_array_equal(mapped @ isometry, isometry)


def test_mode_occupation_detects_zero_and_multiple_hot_qubits():
    space = ModalSpace((2, 2))
    mode_zero = OccupiedModals(space).to_qubit_hamiltonians()[0].to_matrix()

    assert mode_zero[0, 0] == 0.0
    assert mode_zero[1, 1] == 1.0
    assert mode_zero[2, 2] == 1.0
    assert mode_zero[3, 3] == 2.0


def test_one_hot_leakage_is_zero_exactly_on_single_occupation():
    space = ModalSpace((2,))
    occupation = OccupiedModals(space).to_qubit_hamiltonians()[0].to_matrix()
    leakage = OneHotLeakage(space).to_qubit_hamiltonians()[0].to_matrix()

    np.testing.assert_array_equal(np.diag(occupation), (0.0, 1.0, 1.0, 2.0))
    np.testing.assert_array_equal(np.diag(leakage), (1.0, 0.0, 0.0, 1.0))
    np.testing.assert_array_equal(
        leakage,
        (occupation - np.eye(4)) @ (occupation - np.eye(4)),
    )
    np.testing.assert_array_equal(
        OneHotLeakage(space).to_vibrational_ops()[0].to_modal_matrix(),
        np.zeros((2, 2)),
    )
    assert np.all(np.linalg.eigvalsh(leakage) >= 0.0)


def test_mean_occupation_does_not_replace_strict_leakage():
    space = ModalSpace((2,))
    state = np.array((1.0, 0.0, 0.0, 1.0), dtype=np.complex128) / np.sqrt(2.0)
    occupation = OccupiedModals(space).to_qubit_hamiltonians()[0].to_matrix()
    leakage = OneHotLeakage(space).to_qubit_hamiltonians()[0].to_matrix()

    assert np.vdot(state, occupation @ state).real == pytest.approx(1.0)
    assert np.vdot(state, leakage @ state).real == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("operation", "error"),
    [
        (lambda: ModalOccupation(object(), 0, 0), TypeError),
        (lambda: ModalOccupation(ModalSpace((2,)), True, 0), TypeError),
        (lambda: ModalOccupation(ModalSpace((2,)), 0, 2), ValueError),
        (lambda: OccupiedModals(object()), TypeError),
        (lambda: OccupiedModals(ModalSpace((2,))).for_mode(True), TypeError),
        (lambda: OccupiedModals(ModalSpace((2,))).for_mode(1), ValueError),
        (lambda: OneHotLeakage(object()), TypeError),
        (lambda: OneHotLeakage(ModalSpace((2,))).for_mode(True), TypeError),
        (lambda: OneHotLeakage(ModalSpace((2,))).for_mode(1), ValueError),
        (
            lambda: ModalOccupation(ModalSpace((2,)), 0, 0).to_qubit_hamiltonian(
                object()
            ),
            TypeError,
        ),
        (
            lambda: OneHotLeakage(ModalSpace((2,))).to_qubit_hamiltonians(object()),
            TypeError,
        ),
    ],
)
def test_invalid_observable_inputs_are_rejected(operation, error):
    with pytest.raises(error):
        operation()
