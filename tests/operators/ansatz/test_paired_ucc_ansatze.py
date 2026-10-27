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

"""Independent matrix, spin-symmetry, and boundary tests for paired UCC."""

from __future__ import annotations

import numpy as np
import pytest
from cqlib.qis import FermionicOp, Hamiltonian, JordanWignerMapper

from cqlib_algorithms.extension.chemistry import (
    ElectronicOccupation,
    FermionicExcitation,
    GroupedFermionicGenerator,
)
from cqlib_algorithms.operators.ansatz import (
    PUCCDAnsatz,
    PUCCSDAnsatz,
    UCCAnsatz,
)
from cqlib_algorithms.operators.state_preparation import BasisStateInitialization


def _apply_ladder(state: int, mode: int, *, create: bool) -> tuple[int, float] | None:
    occupied = bool(state & (1 << mode))
    if occupied == create:
        return None
    sign = -1.0 if (state & ((1 << mode) - 1)).bit_count() % 2 else 1.0
    output = state | (1 << mode) if create else state & ~(1 << mode)
    return output, sign


def _operator_matrix(
    num_modes: int,
    operations: tuple[tuple[bool, int], ...],
) -> np.ndarray:
    size = 1 << num_modes
    result = np.zeros((size, size), dtype=np.complex128)
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
            result[output, basis] = amplitude
    return result


def _transition(excitation: FermionicExcitation) -> np.ndarray:
    operations = tuple((True, target) for target in excitation.targets) + tuple(
        (False, source) for source in reversed(excitation.sources)
    )
    return _operator_matrix(excitation.num_spin_orbitals, operations)


def _generator(excitation: FermionicExcitation) -> np.ndarray:
    transition = _transition(excitation)
    return 1j * (transition - transition.conj().T)


def _group_generator(group: GroupedFermionicGenerator) -> np.ndarray:
    return sum(
        (
            coefficient * _generator(excitation)
            for excitation, coefficient in group.terms
        ),
        np.zeros(
            (1 << group.num_spin_orbitals, 1 << group.num_spin_orbitals),
            dtype=np.complex128,
        ),
    )


def _evolution(generator: np.ndarray, theta: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(generator)
    return (vectors * np.exp(-1j * theta * values)) @ vectors.conj().T


def _spin_operators(
    num_spatial_orbitals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_modes = 2 * num_spatial_orbitals
    size = 1 << num_modes
    number = np.diag([basis.bit_count() for basis in range(size)]).astype(np.complex128)
    spin_z = np.diag(
        [
            0.5
            * (
                sum(bool(basis & (1 << (2 * p))) for p in range(num_spatial_orbitals))
                - sum(
                    bool(basis & (1 << (2 * p + 1)))
                    for p in range(num_spatial_orbitals)
                )
            )
            for basis in range(size)
        ]
    ).astype(np.complex128)
    spin_plus = np.zeros((size, size), dtype=np.complex128)
    for orbital in range(num_spatial_orbitals):
        spin_plus += _operator_matrix(
            num_modes,
            ((True, 2 * orbital), (False, 2 * orbital + 1)),
        )
    spin_minus = spin_plus.conj().T
    spin_squared = spin_z @ spin_z + 0.5 * (
        spin_plus @ spin_minus + spin_minus @ spin_plus
    )
    return number, spin_z, spin_plus, spin_minus, spin_squared


def _commutator(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left @ right - right @ left


def _h2_generators() -> tuple[GroupedFermionicGenerator, FermionicExcitation]:
    group = GroupedFermionicGenerator(
        (
            (FermionicExcitation(4, (0,), (2,)), 1.0),
            (FermionicExcitation(4, (1,), (3,)), 1.0),
        )
    )
    paired = FermionicExcitation(4, (0, 1), (2, 3))
    return group, paired


@pytest.mark.reference
def test_grouped_single_and_paired_double_match_ordered_independent_exponentials():
    group, paired = _h2_generators()
    first_theta = 0.23
    second_theta = -0.41
    first_generator = _group_generator(group)
    second_generator = _generator(paired)
    assert np.max(np.abs(_commutator(first_generator, second_generator))) > 0.5

    actual = (
        UCCAnsatz(
            (group, paired),
            initial_state=BasisStateInitialization(4, 0),
        )
        .build_circuit("u")
        .assign_parameters({"u_theta_0": first_theta, "u_theta_1": second_theta})
        .to_matrix()
    )
    expected = _evolution(second_generator, second_theta) @ _evolution(
        first_generator, first_theta
    )
    reversed_order = _evolution(first_generator, first_theta) @ _evolution(
        second_generator, second_theta
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    assert not np.allclose(actual, reversed_order, rtol=0.0, atol=1e-6)


@pytest.mark.reference
def test_each_paired_ucc_factor_is_a_spin_scalar_and_sensitivity_is_nonzero():
    group, paired = _h2_generators()
    group_matrix = _group_generator(group)
    paired_matrix = _generator(paired)
    number, spin_z, spin_plus, spin_minus, spin_squared = _spin_operators(2)
    for generator in (group_matrix, paired_matrix):
        for symmetry in (number, spin_z, spin_plus, spin_minus, spin_squared):
            np.testing.assert_allclose(
                _commutator(generator, symmetry),
                0.0,
                rtol=0.0,
                atol=1e-12,
            )

    alpha_only = _generator(FermionicExcitation(4, (0,), (2,)))
    assert np.max(np.abs(_commutator(alpha_only, spin_squared))) > 0.5
    assert np.max(np.abs(_commutator(alpha_only, spin_plus))) > 0.5


@pytest.mark.reference
@pytest.mark.parametrize("component_type", [PUCCDAnsatz, PUCCSDAnsatz])
def test_random_paired_ucc_unitary_conserves_number_and_total_spin(component_type):
    occupation = ElectronicOccupation.closed_shell(2, 2)
    component = component_type(
        occupation,
        initial_state=BasisStateInitialization(4, 0),
    )
    circuit = component.build_circuit("p")
    values = tuple(0.17 + 0.13 * index for index in range(component.num_parameters()))
    unitary = circuit.assign_parameters(
        dict(zip(circuit.used_symbols, values, strict=True))
    ).to_matrix()
    for symmetry in _spin_operators(2):
        np.testing.assert_allclose(
            _commutator(unitary, symmetry),
            0.0,
            rtol=0.0,
            atol=1e-12,
        )


@pytest.mark.reference
@pytest.mark.parametrize("component_type", [PUCCDAnsatz, PUCCSDAnsatz])
def test_default_closed_shell_random_state_remains_a_singlet(component_type):
    component = component_type(ElectronicOccupation.closed_shell(2, 2))
    circuit = component.build_circuit("p")
    values = tuple(-0.29 + 0.21 * index for index in range(component.num_parameters()))
    state = circuit.assign_parameters(
        dict(zip(circuit.used_symbols, values, strict=True))
    ).to_matrix()[:, 0]
    number, spin_z, spin_plus, spin_minus, spin_squared = _spin_operators(2)
    assert np.vdot(state, number @ state).real == pytest.approx(2.0, abs=1e-12)
    assert np.vdot(state, spin_z @ state).real == pytest.approx(0.0, abs=1e-12)
    assert np.vdot(state, spin_plus @ state) == pytest.approx(0.0, abs=1e-12)
    assert np.vdot(state, spin_minus @ state) == pytest.approx(0.0, abs=1e-12)
    assert np.vdot(state, spin_squared @ state).real == pytest.approx(0.0, abs=1e-12)


def test_parameter_counts_names_shared_group_symbol_and_zero_point_reference():
    occupation = ElectronicOccupation.closed_shell(2, 2)
    puccd = PUCCDAnsatz(occupation)
    puccsd = PUCCSDAnsatz(occupation)
    assert puccd.num_parameters() == 1
    assert puccsd.num_parameters() == 2
    assert puccd.build_circuit("p").used_symbols == ["p_theta_0"]
    assert puccsd.build_circuit("p").used_symbols == ["p_theta_0", "p_theta_1"]

    expected = np.zeros(16, dtype=np.complex128)
    expected[0b0011] = 1.0
    for component in (puccd, puccsd):
        circuit = component.build_circuit("p")
        zero = {name: 0.0 for name in circuit.used_symbols}
        np.testing.assert_allclose(
            circuit.assign_parameters(zero).to_matrix()[:, 0],
            expected,
            rtol=0.0,
            atol=1e-12,
        )


@pytest.mark.parametrize(
    "component_type",
    [PUCCDAnsatz, PUCCSDAnsatz],
)
def test_mapper_is_called_once_per_factor_and_default_reference_once(component_type):
    class RecordingMapper:
        def __init__(self) -> None:
            self.operator_calls = 0
            self.occupation_calls = 0

        def num_qubits(self, num_modes: int) -> int:
            return num_modes

        def map_operator(self, operator: FermionicOp) -> Hamiltonian:
            self.operator_calls += 1
            return JordanWignerMapper().map_operator(operator)

        def map_occupation(self, occupation):
            self.occupation_calls += 1
            return JordanWignerMapper().map_occupation(occupation)

    mapper = RecordingMapper()
    component = component_type(ElectronicOccupation.closed_shell(2, 2), mapper=mapper)
    expected_factors = 1 if component_type is PUCCDAnsatz else 2
    component.build_circuit("a")
    component.build_circuit("b")
    assert mapper.operator_calls == expected_factors
    assert mapper.occupation_calls == 1


def test_grouped_mapper_result_is_snapshotted():
    group, _ = _h2_generators()

    class RecordingMapper:
        def __init__(self) -> None:
            self.output: Hamiltonian | None = None

        def num_qubits(self, num_modes: int) -> int:
            return num_modes

        def map_operator(self, operator: FermionicOp) -> Hamiltonian:
            self.output = JordanWignerMapper().map_operator(operator)
            return self.output

    mapper = RecordingMapper()
    component = UCCAnsatz((group,), mapper=mapper)
    before = component.build_circuit("u").assign_parameters({"u_theta_0": 0.3})
    assert mapper.output is not None
    mapper.output.add_term(
        JordanWignerMapper().map_operator(FermionicOp.number(4, 0)).terms[0][0], 8.0
    )
    after = component.build_circuit("u").assign_parameters({"u_theta_0": 0.3})
    np.testing.assert_allclose(after.to_matrix(), before.to_matrix(), atol=0.0)


@pytest.mark.parametrize("component_type", [PUCCDAnsatz, PUCCSDAnsatz])
def test_paired_ansatze_reject_open_shell_and_invalid_primary_inputs(component_type):
    with pytest.raises(TypeError, match="ElectronicOccupation"):
        component_type(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="closed shell"):
        component_type(ElectronicOccupation(2, 1, 0))
    with pytest.raises(TypeError, match="FermionToQubitMapper"):
        component_type(ElectronicOccupation.closed_shell(2, 2), mapper=object())
    with pytest.raises(ValueError, match="same width"):
        component_type(
            ElectronicOccupation.closed_shell(2, 2),
            initial_state=BasisStateInitialization(3, 0),
        )


def test_degenerate_paired_pools_are_rejected():
    with pytest.raises(ValueError, match="paired double"):
        PUCCDAnsatz(ElectronicOccupation(2, 0, 0))
    with pytest.raises(ValueError, match="paired double"):
        PUCCDAnsatz(ElectronicOccupation(2, 2, 2))
    with pytest.raises(ValueError, match="nonempty, non-full closed shell"):
        PUCCSDAnsatz(ElectronicOccupation.closed_shell(1, 0))


@pytest.mark.parametrize("num_spatial_orbitals", [2, 3])
@pytest.mark.parametrize("filled", [False, True])
def test_puccsd_rejects_vacuum_and_full_occupations(
    num_spatial_orbitals: int,
    filled: bool,
):
    electrons = 2 * num_spatial_orbitals if filled else 0
    occupation = ElectronicOccupation.closed_shell(num_spatial_orbitals, electrons)
    with pytest.raises(ValueError, match="nonempty, non-full closed shell"):
        PUCCSDAnsatz(
            occupation,
            initial_state=BasisStateInitialization(2 * num_spatial_orbitals, 0),
        )


@pytest.mark.parametrize("component_type", [PUCCDAnsatz, PUCCSDAnsatz])
def test_operator_only_mapper_requires_explicit_fixed_initial_state(component_type):
    class OperatorOnlyMapper:
        def num_qubits(self, num_modes: int) -> int:
            return num_modes

        def map_operator(self, operator: FermionicOp) -> Hamiltonian:
            return JordanWignerMapper().map_operator(operator)

    mapper = OperatorOnlyMapper()
    occupation = ElectronicOccupation.closed_shell(2, 2)
    with pytest.raises(TypeError, match="OccupationStateMapper"):
        component_type(occupation, mapper=mapper)
    component = component_type(
        occupation,
        mapper=mapper,
        initial_state=BasisStateInitialization(4, 3),
    )
    first = component.build_circuit("u")
    second = component.build_circuit("u")
    first.x(0)
    assert first != second
