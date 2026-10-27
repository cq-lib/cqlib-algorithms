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

"""Independent matrix and boundary tests for vibrational coupled cluster."""

from __future__ import annotations

import numpy as np
import pytest

from cqlib_algorithms.extension.vibration import (
    ModalOccupation,
    ModalSpace,
    ModalTransition,
    VibrationalExcitation,
    generate_vibrational_double_excitations,
    generate_vibrational_single_excitations,
    vscf_initial_point,
)
from cqlib_algorithms.operators.ansatz import UVCCAnsatz, UVCCSDAnsatz
from cqlib_algorithms.operators.estimator import StatevectorEstimator
from cqlib_algorithms.operators.gradient import ParameterShiftGradient
from cqlib_algorithms.operators.state_preparation import (
    BasisStateInitialization,
    VSCFStatePreparation,
)
from cqlib_algorithms.protocols import EstimatorRequest, GradientRequest


def _transition_matrix(excitation: VibrationalExcitation) -> np.ndarray:
    """Construct the hard-core modal transition without using the mapper."""
    space = excitation.modal_space
    size = 1 << space.num_qubits
    transition = np.zeros((size, size), dtype=np.complex128)
    operations = tuple(
        operation
        for item in excitation.transitions
        for operation in (
            (True, space.qubit_index(item.mode, item.bra_modal)),
            (False, space.qubit_index(item.mode, item.ket_modal)),
        )
    )
    for basis in range(size):
        output = basis
        for create, qubit in reversed(operations):
            mask = 1 << qubit
            occupied = bool(output & mask)
            if occupied == create:
                break
            output = output | mask if create else output & ~mask
        else:
            transition[output, basis] = 1.0
    return transition


def _generator(excitation: VibrationalExcitation) -> np.ndarray:
    transition = _transition_matrix(excitation)
    return 1j * (transition - transition.conj().T)


def _evolution(generator: np.ndarray, theta: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(generator)
    return (vectors * np.exp(-1j * theta * values)) @ vectors.conj().T


def _excitation(
    space: ModalSpace,
    *transitions: tuple[int, int, int],
) -> VibrationalExcitation:
    return VibrationalExcitation.from_transitions(
        space,
        (ModalTransition(*transition) for transition in transitions),
    )


@pytest.mark.reference
@pytest.mark.parametrize(
    "excitation",
    [
        _excitation(ModalSpace((2,)), (0, 1, 0)),
        _excitation(ModalSpace((3,)), (0, 2, 0)),
        _excitation(ModalSpace((2, 2)), (0, 1, 0), (1, 1, 0)),
        _excitation(
            ModalSpace((2, 2, 2)),
            (0, 1, 0),
            (1, 1, 0),
            (2, 1, 0),
        ),
    ],
)
def test_each_rank_matches_independent_hard_core_exponential(excitation):
    theta = 0.37
    actual = (
        UVCCAnsatz((excitation,))
        .build_circuit("uv")
        .assign_parameters({"uv_theta_0": theta})
        .to_matrix()
    )
    expected = _evolution(_generator(excitation), theta)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.reference
def test_noncommuting_factors_and_repetitions_use_exact_time_order():
    space = ModalSpace((3,))
    first_excitation = _excitation(space, (0, 1, 0))
    second_excitation = _excitation(space, (0, 2, 0))
    first_generator = _generator(first_excitation)
    second_generator = _generator(second_excitation)
    commutator = first_generator @ second_generator - second_generator @ first_generator
    assert np.max(np.abs(commutator)) > 0.5

    values = (0.17, -0.31, 0.43, -0.29)
    component = UVCCAnsatz((first_excitation, second_excitation), reps=2)
    actual = (
        component.build_circuit("u")
        .assign_parameters(
            {f"u_theta_{index}": value for index, value in enumerate(values)}
        )
        .to_matrix()
    )
    expected = np.eye(1 << space.num_qubits, dtype=np.complex128)
    for generator, value in zip(
        (first_generator, second_generator) * 2,
        values,
    ):
        expected = _evolution(generator, value) @ expected

    assert component.num_parameters() == 4
    assert component.build_circuit("u").used_symbols == [
        "u_theta_0",
        "u_theta_1",
        "u_theta_2",
        "u_theta_3",
    ]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_duplicate_factors_remain_independently_parameterized():
    excitation = _excitation(ModalSpace((2,)), (0, 1, 0))
    component = UVCCAnsatz((excitation, excitation))
    assert component.num_parameters() == 2
    assert component.build_circuit("x").used_symbols == [
        "x_theta_0",
        "x_theta_1",
    ]


@pytest.mark.reference
def test_uvcc_preserves_the_entire_one_hot_physical_subspace():
    space = ModalSpace((2, 2))
    excitations = (
        *generate_vibrational_single_excitations(space),
        *generate_vibrational_double_excitations(space),
    )
    unitary = (
        UVCCAnsatz(excitations)
        .build_circuit("p")
        .assign_parameters(
            {
                "p_theta_0": 0.23,
                "p_theta_1": -0.41,
                "p_theta_2": 0.37,
            }
        )
        .to_matrix()
    )
    evolved_physical_basis = unitary @ space.one_hot_isometry()
    forbidden = [
        index
        for index in range(unitary.shape[0])
        if not space.is_physical_basis_index(index)
    ]
    np.testing.assert_allclose(
        evolved_physical_basis[forbidden, :],
        0.0,
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.reference
def test_uvccsd_pool_zero_point_and_random_state_remain_one_hot():
    space = ModalSpace((2, 3, 2))
    reference = (1, 0, 1)
    singles = generate_vibrational_single_excitations(space, reference)
    doubles = generate_vibrational_double_excitations(space, reference)
    excitations = (*singles, *doubles)
    component = UVCCSDAnsatz(space, reference_modals=reference, reps=2)
    circuit = component.build_circuit("v")
    zero = vscf_initial_point(excitations, prefix="v", reps=2)

    assert len(singles) == 4
    assert len(doubles) == 5
    assert component.num_parameters() == 18
    assert tuple(circuit.used_symbols) == tuple(zero)
    zero_state = circuit.assign_parameters(dict(zero)).to_matrix()[:, 0]
    expected = np.zeros(1 << space.num_qubits, dtype=np.complex128)
    expected[space.encoded_basis_index(reference)] = 1.0
    np.testing.assert_allclose(zero_state, expected, rtol=0.0, atol=1e-12)

    values = {
        symbol: 0.07 * (index - 4) for index, symbol in enumerate(circuit.used_symbols)
    }
    state = circuit.assign_parameters(values).to_matrix()[:, 0]
    forbidden = [
        index for index in range(state.size) if not space.is_physical_basis_index(index)
    ]
    np.testing.assert_allclose(state[forbidden], 0.0, rtol=0.0, atol=1e-12)
    assert np.linalg.norm(state) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.reference
def test_uvccsd_matrix_matches_independent_ordered_product():
    space = ModalSpace((2, 2))
    excitations = (
        *generate_vibrational_single_excitations(space),
        *generate_vibrational_double_excitations(space),
    )
    values = (0.21, -0.34, 0.46)
    actual = (
        UVCCSDAnsatz(space)
        .build_circuit("s")
        .assign_parameters(
            {f"s_theta_{index}": value for index, value in enumerate(values)}
        )
        .to_matrix()
    )
    initial = VSCFStatePreparation(space).build_circuit("").to_matrix()
    expected = initial
    for excitation, value in zip(excitations, values):
        expected = _evolution(_generator(excitation), value) @ expected
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_parameter_shift_matches_independent_finite_difference():
    space = ModalSpace((2,))
    circuit = UVCCSDAnsatz(space).build_circuit("v")
    observable = ModalOccupation(space, 0, 1).to_qubit_hamiltonian()
    binding = {"v_theta_0": 0.31}
    request = GradientRequest(
        EstimatorRequest(circuit, observable, binding),
        ("v_theta_0",),
    )
    actual = (
        ParameterShiftGradient().run(StatevectorEstimator(), [request])[0].values[0]
    )

    def expectation(theta: float) -> float:
        state = circuit.assign_parameters({"v_theta_0": theta}).to_matrix()[:, 0]
        return float(np.vdot(state, observable.to_matrix() @ state).real)

    step = 1e-6
    expected = (expectation(0.31 + step) - expectation(0.31 - step)) / (2 * step)
    assert abs(expected) > 0.1
    assert actual == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize(
    ("factory", "error", "message"),
    [
        (lambda: UVCCAnsatz(()), ValueError, "must not be empty"),
        (
            lambda: UVCCAnsatz("bad"),  # type: ignore[arg-type]
            TypeError,
            "sequence",
        ),
        (
            lambda: UVCCAnsatz((object(),)),  # type: ignore[arg-type]
            TypeError,
            "VibrationalExcitation",
        ),
        (
            lambda: UVCCAnsatz(
                (
                    _excitation(ModalSpace((2,)), (0, 1, 0)),
                    _excitation(ModalSpace((3,)), (0, 1, 0)),
                )
            ),
            ValueError,
            "same modal space",
        ),
        (
            lambda: UVCCAnsatz(
                (_excitation(ModalSpace((2,)), (0, 1, 0)),),
                reps=True,  # type: ignore[arg-type]
            ),
            TypeError,
            "reps",
        ),
        (
            lambda: UVCCAnsatz(
                (_excitation(ModalSpace((2,)), (0, 1, 0)),),
                reps=-1,
            ),
            ValueError,
            "reps",
        ),
        (
            lambda: UVCCAnsatz(
                (_excitation(ModalSpace((2,)), (0, 1, 0)),),
                initial_state=object(),  # type: ignore[arg-type]
            ),
            TypeError,
            "initial_state",
        ),
        (
            lambda: UVCCAnsatz(
                (_excitation(ModalSpace((2,)), (0, 1, 0)),),
                initial_state=BasisStateInitialization(3, 0),
            ),
            ValueError,
            "same width",
        ),
        (
            lambda: UVCCSDAnsatz(object()),  # type: ignore[arg-type]
            TypeError,
            "ModalSpace",
        ),
        (
            lambda: UVCCSDAnsatz(ModalSpace((1,))),
            ValueError,
            "at least one",
        ),
        (
            lambda: UVCCSDAnsatz(ModalSpace((2, 2)), reference_modals=(0,)),
            ValueError,
            "length",
        ),
        (
            lambda: UVCCSDAnsatz(ModalSpace((2,)), reference_modals="bad"),  # type: ignore[arg-type]
            TypeError,
            "reference_modals",
        ),
    ],
)
def test_invalid_uvcc_inputs_are_rejected(factory, error, message):
    with pytest.raises(error, match=message):
        factory()


def test_zero_repetitions_leave_only_the_fixed_reference():
    space = ModalSpace((2, 2))
    component = UVCCSDAnsatz(space, reps=0)
    circuit = component.build_circuit("z")
    expected = VSCFStatePreparation(space).build_circuit("").to_matrix()
    assert component.num_parameters() == 0
    assert circuit.used_symbols == []
    np.testing.assert_allclose(circuit.to_matrix(), expected, rtol=0.0, atol=1e-12)
