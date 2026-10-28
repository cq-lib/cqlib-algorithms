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

"""Tests for immutable chemistry input and driver data."""

import numpy as np
import pytest

from cqlib_algorithms.extension.chemistry import (
    Atom,
    ElectronicIntegrals,
    ElectronicStructureData,
    MolecularSpecification,
)


def _molecule() -> MolecularSpecification:
    return MolecularSpecification(
        atoms=(Atom("H", (0.0, 0.0, 0.0)), Atom("H", (0.0, 0.0, 0.7))),
        basis="sto-3g",
    )


def test_integrals_are_defensive_immutable_snapshots():
    one_body = np.array([[1.0, 0.25], [0.25, 2.0]])
    two_body = np.zeros((2, 2, 2, 2))
    integrals = ElectronicIntegrals(0.5, one_body, two_body)

    one_body[0, 0] = 99.0
    two_body[0, 0, 0, 0] = 99.0

    assert integrals.one_body[0, 0] == 1.0
    assert integrals.two_body[0, 0, 0, 0] == 0.0
    assert not integrals.one_body.flags.writeable
    with pytest.raises(ValueError):
        integrals.one_body[0, 0] = 2.0
    with pytest.raises(ValueError):
        integrals.one_body.flags.writeable = True


@pytest.mark.parametrize(
    ("one_body", "two_body", "message"),
    [
        (np.ones((2, 3)), np.zeros((2, 2, 2, 2)), "square"),
        (np.eye(2), np.zeros((2, 2, 2, 3)), "shape"),
        (np.array([[0.0, 1.0], [0.0, 0.0]]), np.zeros((2,) * 4), "symmetric"),
    ],
)
def test_integral_shape_and_symmetry_errors(one_body, two_body, message):
    with pytest.raises(ValueError, match=message):
        ElectronicIntegrals(0.0, one_body, two_body)


def test_integrals_reject_complex_and_nonfinite_values():
    with pytest.raises(TypeError, match="real"):
        ElectronicIntegrals(0.0, np.array([[1.0j]]), np.ones((1,) * 4))
    with pytest.raises(ValueError, match="finite"):
        ElectronicIntegrals(0.0, np.array([[np.inf]]), np.ones((1,) * 4))
    with pytest.raises(ValueError, match="finite"):
        ElectronicIntegrals(float("nan"), np.eye(1), np.ones((1,) * 4))


@pytest.mark.parametrize(
    "one_body",
    [
        [[True]],
        np.array([[np.bool_(False)]]),
        np.array([[True]], dtype=object),
    ],
)
def test_integrals_reject_boolean_array_entries(one_body):
    with pytest.raises(TypeError, match="without booleans"):
        ElectronicIntegrals(0.0, one_body, np.ones((1,) * 4))


def test_integrals_reject_ragged_input_with_stable_type_and_message():
    with pytest.raises(TypeError, match="one_body must be a rectangular real array"):
        ElectronicIntegrals(
            0.0,
            [[1.0, 2.0], [3.0]],
            np.zeros((2, 2, 2, 2)),
        )


def test_molecular_and_driver_data_validation_and_snapshots():
    molecule = _molecule()
    integrals = ElectronicIntegrals(0.7, np.eye(2), np.zeros((2,) * 4))
    energies = np.array([-1.0, 0.5])
    coefficients = np.eye(2)
    result = ElectronicStructureData(
        molecule,
        1,
        1,
        energies,
        coefficients,
        -1.1,
        integrals,
    )
    energies[0] = 10.0
    coefficients[0, 0] = 10.0

    assert result.num_spatial_orbitals == 2
    assert result.num_alpha == 1
    assert result.num_beta == 1
    assert result.num_electrons == 2
    assert result.is_closed_shell
    assert result.orbital_energies.tolist() == [-1.0, 0.5]
    assert np.array_equal(result.molecular_orbitals, np.eye(2))
    assert not result.orbital_energies.flags.writeable

    with pytest.raises(ValueError, match="multiplicity"):
        ElectronicStructureData(molecule, 1, 0, [-1.0, 0.5], np.eye(2), -1.0, integrals)
    with pytest.raises(ValueError, match="columns"):
        ElectronicStructureData(molecule, 1, 1, [-1.0, 0.5], np.eye(3), -1.0, integrals)


def test_open_shell_driver_data_tracks_spin_counts_and_multiplicity():
    molecule = MolecularSpecification(
        atoms=(Atom("H", (0.0, 0.0, 0.0)),),
        basis="sto-3g",
        multiplicity=2,
    )
    integrals = ElectronicIntegrals(0.0, np.array([[-0.5]]), np.ones((1,) * 4))
    result = ElectronicStructureData(
        molecule,
        1,
        0,
        np.array([-0.5]),
        np.eye(1),
        -0.5,
        integrals,
    )

    assert result.num_electrons == 1
    assert not result.is_closed_shell


@pytest.mark.parametrize(
    ("num_alpha", "num_beta", "message"),
    [
        (0, 0, "at least one electron"),
        (2, 0, "multiplicity"),
        (1, 2, "multiplicity"),
    ],
)
def test_driver_data_rejects_invalid_spin_counts(num_alpha, num_beta, message):
    molecule = MolecularSpecification(
        atoms=(Atom("H", (0.0, 0.0, 0.0)),),
        basis="sto-3g",
        multiplicity=2,
    )
    integrals = ElectronicIntegrals(0.0, np.array([[-0.5]]), np.ones((1,) * 4))
    with pytest.raises(ValueError, match=message):
        ElectronicStructureData(
            molecule,
            num_alpha,
            num_beta,
            np.array([-0.5]),
            np.eye(1),
            -0.5,
            integrals,
        )


def test_driver_data_rejects_spin_count_beyond_orbital_capacity():
    molecule = MolecularSpecification(
        atoms=(Atom("He", (0.0, 0.0, 0.0)),),
        basis="sto-3g",
        multiplicity=1,
    )
    integrals = ElectronicIntegrals(0.0, np.array([[-0.5]]), np.ones((1,) * 4))
    with pytest.raises(ValueError, match="num_alpha exceeds"):
        ElectronicStructureData(
            molecule,
            2,
            2,
            np.array([-0.5]),
            np.eye(1),
            -0.5,
            integrals,
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Atom("", (0.0, 0.0, 0.0)),
        lambda: Atom("H", (0.0, 0.0)),
        lambda: MolecularSpecification((_molecule().atoms[0],), ""),
        lambda: MolecularSpecification((_molecule().atoms[0],), "sto-3g", unit="nm"),
    ],
)
def test_molecular_input_errors(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()
