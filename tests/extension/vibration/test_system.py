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

"""Tests for complete finite-modal vibrational systems."""

import numpy as np
import pytest

from cqlib_algorithms.extension.vibration import (
    ModalIntegrals,
    ModalSpace,
    ModalTransition,
    VibrationalEnergy,
    VibrationalSystem,
)


def _system() -> VibrationalSystem:
    space = ModalSpace((2, 3))
    integrals = ModalIntegrals.from_terms(
        space,
        (
            ((ModalTransition(0, 0, 0),), 1.0),
            ((ModalTransition(1, 1, 1),), 2.0),
        ),
    )
    return VibrationalSystem(VibrationalEnergy(integrals))


def test_system_exposes_consistent_energy_space_and_constraints():
    system = _system()

    assert system.num_modes == 2
    assert system.num_qubits == 5
    assert system.modal_space == ModalSpace((2, 3))
    assert len(system.occupied_modals.to_vibrational_ops()) == 2
    np.testing.assert_array_equal(
        system.to_vibrational_op().to_modal_matrix(),
        system.modal_space.one_hot_isometry().conjugate().T
        @ system.to_qubit_hamiltonian().to_matrix()
        @ system.modal_space.one_hot_isometry(),
    )


def test_physical_configuration_requires_exactly_one_valid_modal_per_mode():
    system = _system()

    assert system.is_physical_configuration((0, 0))
    assert system.is_physical_configuration((1, 2))
    assert not system.is_physical_configuration((0,))
    assert not system.is_physical_configuration((0, 3))
    assert not system.is_physical_configuration((0, True))


def test_physical_statevector_rejects_leakage_and_bad_normalization():
    system = _system()
    physical = np.zeros(1 << system.num_qubits, dtype=np.complex128)
    physical[system.modal_space.encoded_basis_index((1, 2))] = 1.0
    leaked = physical.copy()
    leaked[0] = 1e-3
    leaked /= np.linalg.norm(leaked)

    assert system.is_physical_statevector(physical)
    assert not system.is_physical_statevector(physical * 0.5)
    assert not system.is_physical_statevector(leaked)


@pytest.mark.parametrize(
    ("statevector", "error"),
    [
        (np.zeros(31), ValueError),
        (np.zeros((32, 1)), ValueError),
        (np.full(32, np.nan), ValueError),
    ],
)
def test_invalid_statevector_inputs_are_rejected(statevector, error):
    with pytest.raises(error):
        _system().is_physical_statevector(statevector)


def test_system_rejects_wrong_energy_type():
    with pytest.raises(TypeError, match="energy must be"):
        VibrationalSystem(object())  # type: ignore[arg-type]
