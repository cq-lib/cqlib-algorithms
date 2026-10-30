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

"""Tests for the unified electronic-system input value."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from cqlib_algorithms.extension.chemistry import (
    ActiveSpace,
    Atom,
    ElectronicIntegrals,
    ElectronicOccupation,
    ElectronicStructureData,
    ElectronicSystem,
    MolecularSpecification,
    SpinResolvedActiveSpace,
)


def _data() -> ElectronicStructureData:
    one_body = np.diag([-1.0, -0.4, 0.3])
    two_body = np.zeros((3, 3, 3, 3))
    return ElectronicStructureData(
        MolecularSpecification(
            atoms=(Atom("He", (0.0, 0.0, 0.0)),),
            basis="minimal",
        ),
        2,
        2,
        np.array([-0.8, -0.2, 0.5]),
        np.eye(3),
        -2.1,
        ElectronicIntegrals(0.7, one_body, two_body),
    )


def test_full_system_from_driver_data_is_consistent_and_immutable():
    data = _data()
    system = ElectronicSystem.from_data(data)

    assert system.integrals is data.integrals
    assert system.occupation == ElectronicOccupation.closed_shell(3, 4)
    assert system.num_spatial_orbitals == 3
    assert system.num_spin_orbitals == 6
    assert system.num_electrons == 4
    assert system.hartree_fock_energy == -2.1
    assert system.nuclear_repulsion_energy == 0.7
    assert system.inactive_energy == 0.0
    np.testing.assert_array_equal(system.orbital_energies, [-0.8, -0.2, 0.5])
    assert system.orbital_energies is not data.orbital_energies
    assert not system.orbital_energies.flags.writeable
    with pytest.raises(FrozenInstanceError):
        system.hartree_fock_energy = 0.0  # type: ignore[misc]


def test_active_system_restricts_integrals_energies_and_electron_count():
    data = _data()
    system = ElectronicSystem.from_data(
        data,
        ActiveSpace.manual(frozen_orbitals=(0,), active_orbitals=(1, 2)),
    )

    assert system.num_spatial_orbitals == 2
    assert system.num_electrons == 2
    assert system.occupation == ElectronicOccupation.closed_shell(2, 2)
    np.testing.assert_array_equal(system.orbital_energies, [-0.2, 0.5])
    assert system.integrals.constant == pytest.approx(-1.3)
    assert system.nuclear_repulsion_energy == pytest.approx(0.7)
    assert system.inactive_energy == pytest.approx(-2.0)
    assert system.hartree_fock_energy == -2.1


def test_open_shell_system_and_active_space_preserve_spin_counts():
    molecule = MolecularSpecification(
        atoms=(Atom("He", (0.0, 0.0, 0.0)), Atom("H", (0.0, 0.0, 1.4))),
        basis="minimal",
        multiplicity=2,
    )
    integrals = ElectronicIntegrals(
        0.4,
        np.diag([-1.2, -0.3, 0.5]),
        np.zeros((3,) * 4),
    )
    data = ElectronicStructureData(
        molecule,
        2,
        1,
        np.array([-0.9, -0.1, 0.6]),
        np.eye(3),
        -2.0,
        integrals,
    )

    full = ElectronicSystem.from_data(data)
    active = ElectronicSystem.from_data(
        data,
        ActiveSpace.manual(frozen_orbitals=(0,), active_orbitals=(1, 2)),
    )

    assert full.occupation == ElectronicOccupation(3, 2, 1)
    assert active.occupation == ElectronicOccupation(2, 1, 0)
    assert active.num_electrons == 1
    np.testing.assert_array_equal(active.orbital_energies, [-0.1, 0.6])


@pytest.mark.parametrize(
    ("args", "error", "message"),
    [
        ((object(), ElectronicOccupation(1, 0, 0)), TypeError, "integrals"),
        (
            (ElectronicIntegrals(0.0, np.eye(1), np.zeros((1,) * 4)), object()),
            TypeError,
            "occupation",
        ),
        (
            (
                ElectronicIntegrals(0.0, np.eye(2), np.zeros((2,) * 4)),
                ElectronicOccupation(1, 0, 0),
            ),
            ValueError,
            "same orbital count",
        ),
    ],
)
def test_system_rejects_invalid_core_fields(args, error, message):
    with pytest.raises(error, match=message):
        ElectronicSystem(*args)


def test_system_rejects_invalid_optional_metadata():
    integrals = ElectronicIntegrals(0.0, np.eye(2), np.zeros((2,) * 4))
    occupation = ElectronicOccupation(2, 1, 1)
    with pytest.raises(ValueError, match="orbital_energies shape"):
        ElectronicSystem(integrals, occupation, np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        ElectronicSystem(integrals, occupation, np.array([0.0, np.inf]))
    with pytest.raises(ValueError, match="finite"):
        ElectronicSystem(integrals, occupation, hartree_fock_energy=np.nan)


def test_from_data_rejects_invalid_arguments():
    with pytest.raises(TypeError, match="data"):
        ElectronicSystem.from_data(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="active_space"):
        ElectronicSystem.from_data(_data(), object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="restricted data"):
        ElectronicSystem.from_data(_data(), SpinResolvedActiveSpace.full(3))
