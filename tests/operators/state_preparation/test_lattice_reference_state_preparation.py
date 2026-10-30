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

"""Tests for lattice computational-basis reference preparation."""

from __future__ import annotations

import numpy as np
import pytest

from cqlib_algorithms.extension.lattice import (
    FermiHubbardModel,
    FermionicLatticeConfiguration,
    FermionicSector,
    HeisenbergModel,
    LatticeSystem,
    LineLattice,
    SpinLatticeConfiguration,
    SpinMagnetizationSector,
)
from cqlib_algorithms.operators.state_preparation import (
    LatticeReferenceStatePreparation,
)
from cqlib_algorithms.protocols import StatePreparation


def _fermionic_system(num_sites: int = 3) -> LatticeSystem:
    model = FermiHubbardModel(
        LineLattice(num_sites, edge_parameter=-1.0),
        onsite_interaction=2.0,
    )
    return LatticeSystem(model, FermionicSector(num_sites, 2, 1))


def _spin_system(*, sector: bool = True) -> LatticeSystem:
    model = HeisenbergModel(
        LineLattice(3, edge_parameter=1.0),
        coupling=(1.0, 1.0, 0.5),
    )
    selected = SpinMagnetizationSector(3, 2) if sector else None
    return LatticeSystem(model, selected)


@pytest.mark.parametrize(
    ("system", "configuration", "width", "basis_index"),
    [
        (
            _fermionic_system(),
            FermionicLatticeConfiguration(3, (0, 2), (1,)),
            6,
            (1 << 0) | (1 << 4) | (1 << 3),
        ),
        (
            _spin_system(),
            SpinLatticeConfiguration(3, (0, 2)),
            3,
            1 << 1,
        ),
        (
            _spin_system(sector=False),
            SpinLatticeConfiguration(3, (1,)),
            3,
            (1 << 0) | (1 << 2),
        ),
    ],
)
def test_reference_prepares_exact_basis_amplitudes(
    system: LatticeSystem,
    configuration,
    width: int,
    basis_index: int,
):
    preparation = LatticeReferenceStatePreparation(system, configuration)
    first = preparation.build_circuit("ignored")
    second = preparation.build_circuit("also_ignored")

    expected = np.zeros(1 << width, dtype=np.complex128)
    expected[basis_index] = 1.0
    np.testing.assert_allclose(first.to_matrix()[:, 0], expected, atol=0.0)
    np.testing.assert_allclose(second.to_matrix()[:, 0], expected, atol=0.0)
    assert first is not second
    assert preparation.num_qubits() == width
    assert preparation.num_parameters() == 0
    assert isinstance(preparation, StatePreparation)


def test_reference_exposes_immutable_inputs():
    system = _fermionic_system()
    configuration = FermionicLatticeConfiguration(3, (0, 2), (1,))
    preparation = LatticeReferenceStatePreparation(system, configuration)

    assert preparation.system is system
    assert preparation.configuration is configuration


@pytest.mark.parametrize(
    ("system", "configuration", "error", "message"),
    [
        (
            _fermionic_system(),
            SpinLatticeConfiguration(3, (0, 2)),
            TypeError,
            "FermionicLatticeConfiguration",
        ),
        (
            _spin_system(),
            FermionicLatticeConfiguration(3, (0, 2), (1,)),
            TypeError,
            "SpinLatticeConfiguration",
        ),
        (
            _fermionic_system(),
            FermionicLatticeConfiguration(2, (0,), (1,)),
            ValueError,
            "same number of sites",
        ),
        (
            _fermionic_system(),
            FermionicLatticeConfiguration(3, (0,), (1,)),
            ValueError,
            "system sector",
        ),
        (
            _spin_system(),
            SpinLatticeConfiguration(3, (0,)),
            ValueError,
            "system sector",
        ),
    ],
)
def test_reference_rejects_incompatible_configuration(
    system: LatticeSystem,
    configuration,
    error,
    message: str,
):
    with pytest.raises(error, match=message):
        LatticeReferenceStatePreparation(system, configuration)


def test_reference_rejects_wrong_types():
    with pytest.raises(TypeError, match="system"):
        LatticeReferenceStatePreparation(
            object(),  # type: ignore[arg-type]
            SpinLatticeConfiguration(2, (0,)),
        )
