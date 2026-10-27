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

"""Tests for mapped Hartree--Fock computational-basis preparation."""

import numpy as np
import pytest
from cqlib.qis import JordanWignerMapper

from cqlib_algorithms.extension.chemistry import (
    ElectronicEnergy,
    ElectronicIntegrals,
    ElectronicOccupation,
)
from cqlib_algorithms.operators.state_preparation import (
    HartreeFockStatePreparation,
)
from cqlib_algorithms.protocols import StatePreparation


def _expected_basis_state(num_qubits: int, basis_index: int) -> np.ndarray:
    state = np.zeros(1 << num_qubits, dtype=np.complex128)
    state[basis_index] = 1.0
    return state


def test_default_mapping_prepares_h2_little_endian_occupation():
    preparation = HartreeFockStatePreparation(ElectronicOccupation.closed_shell(2, 2))
    assert isinstance(preparation, StatePreparation)
    assert preparation.num_qubits() == 4
    assert preparation.num_parameters() == 0
    assert np.array_equal(
        preparation.build_circuit("hf").to_matrix()[:, 0],
        _expected_basis_state(4, 0b0011),
    )


def test_default_mapping_supports_open_shell_occupation():
    preparation = HartreeFockStatePreparation(ElectronicOccupation(3, 2, 1))
    assert np.array_equal(
        preparation.build_circuit("").to_matrix()[:, 0],
        _expected_basis_state(6, 0b000111),
    )


def test_custom_mapper_is_called_once_and_defensively_snapshotted():
    mapped = [True, False, True]

    class RecordingMapper:
        def __init__(self) -> None:
            self.calls = 0
            self.received: tuple[int, ...] | None = None

        def map_occupation(self, occupation):
            self.calls += 1
            self.received = tuple(occupation)
            return mapped

    mapper = RecordingMapper()
    occupation = ElectronicOccupation(2, 1, 1)
    preparation = HartreeFockStatePreparation(occupation, mapper)
    mapped[:] = [False, False, False]

    first = preparation.build_circuit("a")
    first.x(0)
    second = preparation.build_circuit("b")
    assert mapper.calls == 1
    assert mapper.received == (1, 1, 0, 0)
    assert preparation.num_qubits() == 3
    assert np.array_equal(second.to_matrix()[:, 0], _expected_basis_state(3, 0b101))
    assert len(first.operations) == len(second.operations) + 1


@pytest.mark.parametrize("mapped", [(), [], [1], [True, 0], [True, "False"]])
def test_rejects_invalid_mapper_results(mapped):
    class InvalidMapper:
        def map_occupation(self, occupation):
            return mapped

    expected_error = ValueError if mapped == [] else TypeError
    with pytest.raises(expected_error):
        HartreeFockStatePreparation(ElectronicOccupation(1, 1, 0), InvalidMapper())


def test_rejects_invalid_mapper_and_occupation_types():
    with pytest.raises(TypeError, match="ElectronicOccupation"):
        HartreeFockStatePreparation(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="OccupationStateMapper"):
        HartreeFockStatePreparation(ElectronicOccupation(1, 1, 0), object())


@pytest.mark.reference
def test_one_orbital_hf_expectation_matches_analytic_energy():
    constant = 0.4
    one_body = -1.2
    interaction = 0.7
    energy = ElectronicEnergy(
        ElectronicIntegrals(constant, [[one_body]], [[[[interaction]]]])
    )
    preparation = HartreeFockStatePreparation(
        ElectronicOccupation.closed_shell(1, 2), JordanWignerMapper()
    )
    state = preparation.build_circuit("hf").to_matrix()[:, 0]
    matrix = energy.to_qubit_hamiltonian().to_matrix()
    expectation = np.vdot(state, matrix @ state).real
    assert expectation == pytest.approx(
        constant + 2.0 * one_body + interaction,
        abs=1e-12,
    )


def test_validate_and_prefix_contract_are_delegated():
    preparation = HartreeFockStatePreparation(ElectronicOccupation(1, 1, 0))
    preparation.validate()
    with pytest.raises(TypeError, match="prefix must be a str"):
        preparation.build_circuit(1)  # type: ignore[arg-type]


def test_custom_mapper_can_use_a_different_register_width():
    class CompressedMapper:
        def map_occupation(self, occupation):
            return [bool(sum(occupation) % 2)]

    preparation = HartreeFockStatePreparation(
        ElectronicOccupation(2, 1, 0), CompressedMapper()
    )
    assert preparation.num_qubits() == 1
    assert np.array_equal(
        preparation.build_circuit("").to_matrix()[:, 0],
        _expected_basis_state(1, 1),
    )
