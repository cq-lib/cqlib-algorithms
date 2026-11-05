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

"""Tests for finite vibrational modal spaces."""

import numpy as np
import pytest

from cqlib_algorithms.extension.vibration import ModalSpace


def test_modal_space_uses_little_endian_mixed_radix_and_contiguous_blocks():
    space = ModalSpace((2, 3, 2))

    assert space.num_modes == 3
    assert space.num_qubits == 7
    assert space.dimension == 12
    assert space.qubit_offsets == (0, 2, 5)
    assert space.qubit_index(1, 2) == 4
    assert space.modal_basis_index((1, 2, 1)) == 11
    assert space.configuration_from_basis_index(11) == (1, 2, 1)
    assert space.encoded_basis_index((1, 2, 1)) == (1 << 1) | (1 << 4) | (1 << 6)


def test_one_hot_isometry_is_an_isometry_with_expected_column_order():
    space = ModalSpace((2, 3))
    isometry = space.one_hot_isometry()

    assert isometry.shape == (32, 6)
    np.testing.assert_array_equal(
        isometry.conjugate().T @ isometry,
        np.eye(space.dimension),
    )
    assert space.physical_basis_indices == (5, 6, 9, 10, 17, 18)
    for column, row in enumerate(space.physical_basis_indices):
        assert isometry[row, column] == 1.0
        assert space.is_physical_basis_index(row)

    assert not space.is_physical_basis_index(0)
    assert not space.is_physical_basis_index(3)


def test_modal_space_defensively_copies_the_modal_counts():
    counts = [2, 3]
    space = ModalSpace(counts)
    counts.clear()

    assert space.num_modals == (2, 3)


def test_modal_space_normalizes_default_and_explicit_reference_modals() -> None:
    space = ModalSpace((2, 3))
    reference = [1, 2]

    assert space.normalize_reference_modals(None) == (0, 0)
    assert space.normalize_reference_modals(reference) == (1, 2)
    reference.clear()
    assert space.normalize_reference_modals((1, 2)) == (1, 2)


@pytest.mark.parametrize(
    ("counts", "error"),
    [
        ((), ValueError),
        ([], ValueError),
        ("23", TypeError),
        ((True,), TypeError),
        ((0,), ValueError),
        ((2, -1), ValueError),
        ((2, 1.5), TypeError),
    ],
)
def test_invalid_modal_counts_are_rejected(counts, error):
    with pytest.raises(error):
        ModalSpace(counts)


@pytest.mark.parametrize(
    ("operation", "error"),
    [
        (lambda space: space.qubit_index(True, 0), TypeError),
        (lambda space: space.qubit_index(0, True), TypeError),
        (lambda space: space.qubit_index(-1, 0), ValueError),
        (lambda space: space.qubit_index(2, 0), ValueError),
        (lambda space: space.qubit_index(0, 2), ValueError),
        (lambda space: space.modal_basis_index((0,)), ValueError),
        (lambda space: space.modal_basis_index((0, True)), TypeError),
        (lambda space: space.modal_basis_index((0, 3)), ValueError),
        (lambda space: space.configuration_from_basis_index(True), TypeError),
        (lambda space: space.configuration_from_basis_index(6), ValueError),
        (lambda space: space.is_physical_basis_index(True), TypeError),
        (lambda space: space.is_physical_basis_index(32), ValueError),
    ],
)
def test_invalid_indices_and_configurations_are_rejected(operation, error):
    with pytest.raises(error):
        operation(ModalSpace((2, 3)))
