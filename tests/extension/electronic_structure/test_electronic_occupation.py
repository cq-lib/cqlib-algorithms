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

"""Tests for spin-resolved electronic occupations."""

from dataclasses import FrozenInstanceError

import pytest

from cqlib_algorithms.extension.electronic_structure import ElectronicOccupation


@pytest.mark.parametrize(
    ("occupation", "occupied", "virtual", "bits"),
    [
        (ElectronicOccupation(2, 0, 0), (), (0, 1, 2, 3), (False,) * 4),
        (ElectronicOccupation(2, 2, 2), (0, 1, 2, 3), (), (True,) * 4),
        (
            ElectronicOccupation(3, 2, 1),
            (0, 1, 2),
            (3, 4, 5),
            (True, True, True, False, False, False),
        ),
        (
            ElectronicOccupation.closed_shell(3, 4),
            (0, 1, 2, 3),
            (4, 5),
            (True, True, True, True, False, False),
        ),
    ],
)
def test_interleaved_index_ordered_occupation(occupation, occupied, virtual, bits):
    assert occupation.num_spin_orbitals == 2 * occupation.num_spatial_orbitals
    assert occupation.num_electrons == occupation.num_alpha + occupation.num_beta
    assert occupation.occupied_spin_orbitals == occupied
    assert occupation.virtual_spin_orbitals == virtual
    assert occupation.spin_orbital_occupation == bits


@pytest.mark.parametrize("value", [True, 1.0, "1", None])
def test_occupation_rejects_non_integer_values(value):
    with pytest.raises(TypeError, match="int"):
        ElectronicOccupation(value, 0, 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="int"):
        ElectronicOccupation(2, value, 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="int"):
        ElectronicOccupation(2, 0, value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "arguments",
    [(0, 0, 0), (2, -1, 0), (2, 0, -1), (2, 3, 0), (2, 0, 3)],
)
def test_occupation_rejects_invalid_ranges(arguments):
    with pytest.raises(ValueError):
        ElectronicOccupation(*arguments)


@pytest.mark.parametrize("num_electrons", [1, 3, 5])
def test_closed_shell_rejects_odd_electron_counts(num_electrons):
    with pytest.raises(ValueError, match="even"):
        ElectronicOccupation.closed_shell(3, num_electrons)


def test_closed_shell_rejects_capacity_and_preserves_vacuum():
    assert ElectronicOccupation.closed_shell(2, 0) == ElectronicOccupation(2, 0, 0)
    with pytest.raises(ValueError, match="capacity"):
        ElectronicOccupation.closed_shell(2, 6)


@pytest.mark.parametrize("value", [True, 2.0, "2", None])
def test_closed_shell_rejects_non_integer_inputs(value):
    with pytest.raises(TypeError, match="int"):
        ElectronicOccupation.closed_shell(value, 2)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="int"):
        ElectronicOccupation.closed_shell(2, value)  # type: ignore[arg-type]


def test_occupation_is_frozen():
    occupation = ElectronicOccupation(2, 1, 1)
    with pytest.raises(FrozenInstanceError):
        occupation.num_alpha = 2  # type: ignore[misc]
