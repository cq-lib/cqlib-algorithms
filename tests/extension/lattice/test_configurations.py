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

"""Tests for immutable lattice basis configurations."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from cqlib_algorithms.extension.lattice import (
    FermionicLatticeConfiguration,
    FermionicSector,
    SpinLatticeConfiguration,
    SpinMagnetizationSector,
)


def test_fermionic_configuration_uses_interleaved_little_endian_modes():
    configuration = FermionicLatticeConfiguration(
        4,
        occupied_up_sites=[3, 0],
        occupied_down_sites=[2, 0],
    )

    assert configuration.occupied_up == (0, 3)
    assert configuration.occupied_down == (0, 2)
    assert configuration.occupation == (1, 1, 0, 0, 0, 1, 1, 0)
    assert configuration.basis_index == 0b01100011
    assert configuration.sector == FermionicSector(4, 2, 2)


@pytest.mark.parametrize(
    ("num_sites", "start_with_up", "up", "down"),
    [
        (4, True, (0, 2), (1, 3)),
        (4, False, (1, 3), (0, 2)),
        (5, True, (0, 2, 4), (1, 3)),
        (5, False, (1, 3), (0, 2, 4)),
    ],
)
def test_fermionic_neel_configuration(
    num_sites: int,
    start_with_up: bool,
    up: tuple[int, ...],
    down: tuple[int, ...],
):
    configuration = FermionicLatticeConfiguration.neel(
        num_sites,
        start_with_up=start_with_up,
    )

    assert configuration.occupied_up == up
    assert configuration.occupied_down == down
    assert sum(configuration.occupation) == num_sites


def test_fermionic_configuration_defensively_copies_input():
    up = [0]
    down = [1]
    configuration = FermionicLatticeConfiguration(2, up, down)

    up.append(1)
    down.clear()

    assert configuration.occupied_up == (0,)
    assert configuration.occupied_down == (1,)


def test_spin_configuration_basis_convention_and_factories():
    configuration = SpinLatticeConfiguration(4, [2, 0])

    assert configuration.occupied_up == (0, 2)
    assert configuration.basis_index == 0b1010
    assert configuration.sector == SpinMagnetizationSector(4, 2)
    assert SpinLatticeConfiguration.all_up(3).basis_index == 0
    assert SpinLatticeConfiguration.all_down(3).basis_index == 0b111
    assert SpinLatticeConfiguration.neel(5).occupied_up == (0, 2, 4)
    assert SpinLatticeConfiguration.neel(
        5,
        start_with_up=False,
    ).occupied_up == (1, 3)


def test_configurations_are_frozen_standard_dataclasses():
    fermionic = FermionicLatticeConfiguration(2, [0], [1])
    spin = SpinLatticeConfiguration(2, [0])

    with pytest.raises(FrozenInstanceError):
        fermionic.num_sites = 3  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        spin.num_sites = 3  # type: ignore[misc]
    assert replace(fermionic, num_sites=3).num_sites == 3
    assert replace(spin, num_sites=3).num_sites == 3


@pytest.mark.parametrize(
    ("factory", "error", "message"),
    [
        (lambda: FermionicLatticeConfiguration(True, (), ()), TypeError, "num_sites"),
        (lambda: FermionicLatticeConfiguration(0, (), ()), ValueError, "num_sites"),
        (
            lambda: FermionicLatticeConfiguration(2, "0", ()),
            TypeError,
            "occupied_up_sites",
        ),
        (
            lambda: FermionicLatticeConfiguration(2, (True,), ()),
            TypeError,
            "occupied_up_sites",
        ),
        (
            lambda: FermionicLatticeConfiguration(2, (0, 0), ()),
            ValueError,
            "duplicate",
        ),
        (
            lambda: FermionicLatticeConfiguration(2, (), (2,)),
            ValueError,
            "out of range",
        ),
        (lambda: SpinLatticeConfiguration(2, "0"), TypeError, "up_sites"),
        (lambda: SpinLatticeConfiguration(2, (-1,)), ValueError, "out of range"),
        (
            lambda: FermionicLatticeConfiguration.neel(2, start_with_up=1),
            TypeError,
            "start_with_up",
        ),
        (
            lambda: SpinLatticeConfiguration.neel(2, start_with_up=1),
            TypeError,
            "start_with_up",
        ),
    ],
)
def test_configuration_input_errors(factory, error, message: str):
    with pytest.raises(error, match=message):
        factory()
