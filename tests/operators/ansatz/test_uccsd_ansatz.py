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

"""Reference, conservation, and boundary tests for ordered UCCSD."""

from __future__ import annotations

import numpy as np
import pytest
from cqlib.qis import FermionicOp, Hamiltonian, JordanWignerMapper

from cqlib_algorithms.extension.chemistry import (
    ElectronicOccupation,
    FermionicExcitation,
    generate_uccsd_excitations,
)
from cqlib_algorithms.operators.ansatz import UCCSDAnsatz
from cqlib_algorithms.operators.state_preparation import BasisStateInitialization
from cqlib_algorithms.protocols import Ansatz


def _apply_ladder(state: int, mode: int, *, create: bool) -> tuple[int, float] | None:
    occupied = bool(state & (1 << mode))
    if occupied == create:
        return None
    sign = -1.0 if (state & ((1 << mode) - 1)).bit_count() % 2 else 1.0
    output = state | (1 << mode) if create else state & ~(1 << mode)
    return output, sign


def _generator(excitation: FermionicExcitation) -> np.ndarray:
    size = 1 << excitation.num_spin_orbitals
    transition = np.zeros((size, size), dtype=np.complex128)
    operations = tuple((True, target) for target in excitation.targets) + tuple(
        (False, source) for source in reversed(excitation.sources)
    )
    for basis in range(size):
        output = basis
        amplitude = 1.0
        for create, mode in reversed(operations):
            applied = _apply_ladder(output, mode, create=create)
            if applied is None:
                break
            output, sign = applied
            amplitude *= sign
        else:
            transition[output, basis] = amplitude
    return 1j * (transition - transition.conj().T)


def _evolution(generator: np.ndarray, theta: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(generator)
    return (vectors * np.exp(-1j * theta * values)) @ vectors.conj().T


def _number_and_spin_z(num_spin_orbitals: int) -> tuple[np.ndarray, np.ndarray]:
    size = 1 << num_spin_orbitals
    number = np.diag([basis.bit_count() for basis in range(size)])
    spin_z = np.diag(
        [
            0.5
            * (
                sum(
                    bool(basis & (1 << mode)) for mode in range(0, num_spin_orbitals, 2)
                )
                - sum(
                    bool(basis & (1 << mode)) for mode in range(1, num_spin_orbitals, 2)
                )
            )
            for basis in range(size)
        ]
    )
    return number, spin_z


def test_h2_uccsd_has_exact_pool_names_and_zero_point_hf_state():
    component = UCCSDAnsatz(ElectronicOccupation.closed_shell(2, 2))
    circuit = component.build_circuit("ucc")
    assert isinstance(component, Ansatz)
    assert component.num_qubits() == 4
    assert component.num_parameters() == 3
    assert circuit.used_symbols == [
        "ucc_theta_0",
        "ucc_theta_1",
        "ucc_theta_2",
    ]
    zero = {name: 0.0 for name in circuit.used_symbols}
    expected = np.zeros(16, dtype=np.complex128)
    expected[0b0011] = 1.0
    np.testing.assert_allclose(
        circuit.assign_parameters(zero).to_matrix()[:, 0],
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_h2_random_parameters_do_not_leave_fixed_alpha_beta_sector():
    circuit = UCCSDAnsatz(ElectronicOccupation.closed_shell(2, 2)).build_circuit("u")
    values = (0.23, -0.41, 0.67)
    bindings = dict(zip(circuit.used_symbols, values, strict=True))
    state = circuit.assign_parameters(bindings).to_matrix()[:, 0]
    for basis, amplitude in enumerate(state):
        if abs(amplitude) <= 1e-12:
            continue
        num_alpha = sum(bool(basis & (1 << mode)) for mode in (0, 2))
        num_beta = sum(bool(basis & (1 << mode)) for mode in (1, 3))
        assert (num_alpha, num_beta) == (1, 1)


@pytest.mark.reference
def test_two_electron_two_orbital_active_pool_order_and_conservation():
    occupation = ElectronicOccupation.closed_shell(2, 2)
    excitations = generate_uccsd_excitations(occupation)
    component = UCCSDAnsatz(
        occupation,
        initial_state=BasisStateInitialization(4, 0),
    )
    circuit = component.build_circuit("active")
    values = (0.17, -0.29, 0.43)
    actual = circuit.assign_parameters(
        dict(zip(circuit.used_symbols, values, strict=True))
    ).to_matrix()
    expected = np.eye(16, dtype=np.complex128)
    for excitation, theta in zip(excitations, values, strict=True):
        expected = _evolution(_generator(excitation), theta) @ expected
    number, spin_z = _number_and_spin_z(4)

    assert component.num_parameters() == 3
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(actual @ number - number @ actual, 0.0, atol=1e-12)
    np.testing.assert_allclose(actual @ spin_z - spin_z @ actual, 0.0, atol=1e-12)


def test_same_mapper_instance_is_used_for_default_reference_and_operators():
    class RecordingMapper:
        def __init__(self) -> None:
            self.occupation_calls = 0
            self.operator_calls = 0

        def num_qubits(self, num_modes: int) -> int:
            return num_modes

        def map_occupation(self, occupation):
            self.occupation_calls += 1
            return JordanWignerMapper().map_occupation(occupation)

        def map_operator(self, operator: FermionicOp) -> Hamiltonian:
            self.operator_calls += 1
            return JordanWignerMapper().map_operator(operator)

    mapper = RecordingMapper()
    component = UCCSDAnsatz(ElectronicOccupation.closed_shell(2, 2), mapper=mapper)
    component.build_circuit("u")
    component.build_circuit("u")
    assert mapper.occupation_calls == 1
    assert mapper.operator_calls == 3


def test_operator_only_mapper_is_allowed_with_custom_fixed_initial_state():
    class OperatorOnlyMapper:
        def num_qubits(self, num_modes: int) -> int:
            return num_modes

        def map_operator(self, operator: FermionicOp) -> Hamiltonian:
            return JordanWignerMapper().map_operator(operator)

    mapper = OperatorOnlyMapper()
    occupation = ElectronicOccupation.closed_shell(2, 2)
    component = UCCSDAnsatz(
        occupation,
        mapper=mapper,
        initial_state=BasisStateInitialization(4, 3),
    )
    assert component.num_parameters() == 3
    with pytest.raises(TypeError, match="OccupationStateMapper"):
        UCCSDAnsatz(occupation, mapper=mapper)


@pytest.mark.parametrize(
    "occupation",
    [
        ElectronicOccupation(2, 0, 0),
        ElectronicOccupation(2, 2, 2),
    ],
)
def test_empty_uccsd_pool_is_rejected(occupation):
    with pytest.raises(ValueError, match="at least one UCCSD excitation"):
        UCCSDAnsatz(occupation)


def test_invalid_occupation_mapper_and_initial_state_are_rejected():
    with pytest.raises(TypeError, match="ElectronicOccupation"):
        UCCSDAnsatz(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="FermionToQubitMapper"):
        UCCSDAnsatz(ElectronicOccupation.closed_shell(2, 2), mapper=object())
    with pytest.raises(ValueError, match="same width"):
        UCCSDAnsatz(
            ElectronicOccupation.closed_shell(2, 2),
            initial_state=BasisStateInitialization(3, 0),
        )


def test_prefix_and_build_independence_are_delegated():
    component = UCCSDAnsatz(ElectronicOccupation.closed_shell(2, 2))
    with pytest.raises(TypeError, match="prefix"):
        component.build_circuit(1)  # type: ignore[arg-type]
    first = component.build_circuit("u")
    second = component.build_circuit("u")
    first.x(0)
    assert first != second
