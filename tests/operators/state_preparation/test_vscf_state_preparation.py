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

"""Tests for one-hot VSCF reference-state preparation."""

import numpy as np
import pytest

from cqlib_algorithms.extension.vibration import ModalSpace, OccupiedModals
from cqlib_algorithms.operators.state_preparation import VSCFStatePreparation
from cqlib_algorithms.protocols import StatePreparation


def _expected_basis_state(num_qubits: int, basis_index: int) -> np.ndarray:
    state = np.zeros(1 << num_qubits, dtype=np.complex128)
    state[basis_index] = 1.0
    return state


@pytest.mark.parametrize(
    "num_modals",
    [(1,), (2,), (2, 3), (2, 3, 2)],
)
def test_default_vscf_state_is_exactly_modal_zero_of_every_mode(num_modals):
    space = ModalSpace(num_modals)
    preparation = VSCFStatePreparation(space)
    state = preparation.build_circuit("vscf").to_matrix()[:, 0]
    expected_index = space.encoded_basis_index((0,) * space.num_modes)

    assert isinstance(preparation, StatePreparation)
    assert preparation.reference_modals == (0,) * space.num_modes
    assert preparation.basis_index == expected_index
    assert preparation.num_qubits() == space.num_qubits
    assert preparation.num_parameters() == 0
    np.testing.assert_array_equal(
        state,
        _expected_basis_state(space.num_qubits, expected_index),
    )


def test_explicit_reference_is_defensively_copied_and_prepared_exactly():
    space = ModalSpace((3, 2, 4))
    reference = [2, 1, 3]
    preparation = VSCFStatePreparation(space, reference)
    reference[:] = [0, 0, 0]
    state = preparation.build_circuit("").to_matrix()[:, 0]

    assert preparation.reference_modals == (2, 1, 3)
    np.testing.assert_array_equal(
        state,
        _expected_basis_state(
            space.num_qubits,
            space.encoded_basis_index((2, 1, 3)),
        ),
    )


@pytest.mark.reference
def test_every_mode_has_exactly_one_occupied_modal_in_prepared_state():
    space = ModalSpace((2, 3, 2))
    preparation = VSCFStatePreparation(space, (1, 2, 0))
    state = preparation.build_circuit("vscf").to_matrix()[:, 0]

    assert space.is_physical_basis_index(preparation.basis_index)
    for observable in OccupiedModals(space).to_qubit_hamiltonians():
        expectation = np.vdot(state, observable.to_matrix() @ state)
        assert expectation == pytest.approx(1.0, abs=1e-14)


def test_repeated_builds_are_independent_and_prefix_is_validated():
    preparation = VSCFStatePreparation(ModalSpace((2, 2)))
    first = preparation.build_circuit("a")
    first.x(0)
    second = preparation.build_circuit("b")

    assert len(first.operations) == len(second.operations) + 1
    preparation.validate()
    with pytest.raises(TypeError, match="prefix must be a str"):
        preparation.build_circuit(1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("modal_space", "reference", "error"),
    [
        (object(), None, TypeError),
        (ModalSpace((2, 2)), "00", TypeError),
        (ModalSpace((2, 2)), (0,), ValueError),
        (ModalSpace((2, 2)), (0, True), TypeError),
        (ModalSpace((2, 2)), (0, 2), ValueError),
    ],
)
def test_invalid_vscf_inputs_are_rejected(modal_space, reference, error):
    with pytest.raises(error):
        VSCFStatePreparation(modal_space, reference)  # type: ignore[arg-type]
