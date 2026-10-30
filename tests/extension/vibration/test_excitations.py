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

"""Tests for reference-relative vibrational excitations."""

import numpy as np
import pytest

from cqlib_algorithms.extension.many_body import OneHotVibrationalMapper
from cqlib_algorithms.extension.vibration import (
    ModalSpace,
    ModalTransition,
    VibrationalExcitation,
    generate_vibrational_double_excitations,
    generate_vibrational_single_excitations,
    vscf_initial_point,
)


def _basis_vector(width: int, index: int) -> np.ndarray:
    vector = np.zeros(1 << width, dtype=np.complex128)
    vector[index] = 1.0
    return vector


def test_default_single_excitations_are_complete_and_deterministic():
    space = ModalSpace((2, 3, 1))

    excitations = generate_vibrational_single_excitations(space)

    assert tuple(excitation.transitions for excitation in excitations) == (
        (ModalTransition(0, 1, 0),),
        (ModalTransition(1, 1, 0),),
        (ModalTransition(1, 2, 0),),
    )
    assert all(excitation.rank == 1 for excitation in excitations)
    assert tuple(excitation.apply((0, 0, 0)) for excitation in excitations) == (
        (1, 0, 0),
        (0, 1, 0),
        (0, 2, 0),
    )


def test_custom_reference_generates_all_other_targets_in_each_mode():
    space = ModalSpace((3, 2))

    excitations = generate_vibrational_single_excitations(space, (1, 1))

    assert tuple(excitation.transitions for excitation in excitations) == (
        (ModalTransition(0, 0, 1),),
        (ModalTransition(0, 2, 1),),
        (ModalTransition(1, 0, 1),),
    )
    outputs = tuple(excitation.apply((1, 1)) for excitation in excitations)
    assert outputs == (
        (0, 1),
        (2, 1),
        (1, 0),
    )


def test_double_excitations_change_two_distinct_modes_only():
    space = ModalSpace((2, 3, 2))
    reference = (0, 0, 0)

    doubles = generate_vibrational_double_excitations(space, reference)

    # Per-mode alternative counts are (1, 2, 1), hence 1*2 + 1*1 + 2*1.
    assert len(doubles) == 5
    assert all(excitation.rank == 2 for excitation in doubles)
    assert all(
        len({transition.mode for transition in excitation.transitions}) == 2
        for excitation in doubles
    )
    for excitation in doubles:
        output = excitation.apply(reference)
        assert sum(before != after for before, after in zip(reference, output)) == 2
        assert space.is_physical_basis_index(space.encoded_basis_index(output))


@pytest.mark.reference
def test_custom_excitation_operator_maps_source_basis_state_to_target_exactly():
    space = ModalSpace((3, 2, 2))
    source = (2, 0, 1)
    excitation = VibrationalExcitation.from_transitions(
        space,
        (
            ModalTransition(0, 1, 2),
            ModalTransition(2, 0, 1),
        ),
    )
    target = (1, 0, 0)
    mapped = (
        OneHotVibrationalMapper()
        .map_operator(excitation.to_vibrational_op())
        .to_matrix()
    )
    source_state = _basis_vector(space.num_qubits, space.encoded_basis_index(source))
    expected = _basis_vector(space.num_qubits, space.encoded_basis_index(target))

    assert excitation.sources == ((0, 2), (2, 1))
    assert excitation.targets == ((0, 1), (2, 0))
    assert excitation.apply(source) == target
    np.testing.assert_array_equal(mapped @ source_state, expected)
    np.testing.assert_array_equal(
        OneHotVibrationalMapper()
        .map_operator(excitation.adjoint().to_vibrational_op())
        .to_matrix()
        @ expected,
        source_state,
    )


def test_generated_excitations_never_leave_the_physical_modal_space():
    space = ModalSpace((2, 3, 2))
    references = tuple(
        space.configuration_from_basis_index(index) for index in range(space.dimension)
    )

    for reference in references:
        excitations = (
            *generate_vibrational_single_excitations(space, reference),
            *generate_vibrational_double_excitations(space, reference),
        )
        assert space.is_physical_basis_index(space.encoded_basis_index(reference))
        for excitation in excitations:
            target = excitation.apply(reference)
            assert excitation.to_vibrational_op().preserves_one_hot_subspace()
            assert space.is_physical_basis_index(space.encoded_basis_index(target))


def test_vscf_initial_point_matches_excitation_order_and_is_immutable():
    space = ModalSpace((2, 3))
    excitations = (
        *generate_vibrational_single_excitations(space),
        *generate_vibrational_double_excitations(space),
    )

    point = vscf_initial_point(excitations, prefix="vib", reps=2)

    assert point == {f"vib_theta_{index}": 0.0 for index in range(2 * len(excitations))}
    with pytest.raises(TypeError):
        point["vib_theta_0"] = 1.0  # type: ignore[index]
    assert vscf_initial_point(()) == {}


@pytest.mark.parametrize(
    ("operation", "error"),
    [
        (
            lambda: VibrationalExcitation.from_transitions(ModalSpace((2,)), ()),
            ValueError,
        ),
        (
            lambda: VibrationalExcitation.from_transitions(
                ModalSpace((2,)),
                (ModalTransition(0, 0, 0),),
            ),
            ValueError,
        ),
        (
            lambda: VibrationalExcitation.from_transitions(
                ModalSpace((2, 2)),
                (ModalTransition(0, 1, 0), ModalTransition(0, 0, 1)),
            ),
            ValueError,
        ),
        (
            lambda: VibrationalExcitation.from_transitions(
                ModalSpace((2,)),
                (ModalTransition(0, 2, 0),),
            ),
            ValueError,
        ),
        (
            lambda: generate_vibrational_single_excitations(
                ModalSpace((2,)),
                (2,),
            ),
            ValueError,
        ),
        (
            lambda: generate_vibrational_double_excitations(
                ModalSpace((2, 2)),
                (0,),
            ),
            ValueError,
        ),
        (
            lambda: vscf_initial_point([object()]),
            TypeError,
        ),
        (
            lambda: vscf_initial_point(
                [
                    VibrationalExcitation.from_transitions(
                        ModalSpace((2,)),
                        (ModalTransition(0, 1, 0),),
                    ),
                    VibrationalExcitation.from_transitions(
                        ModalSpace((3,)),
                        (ModalTransition(0, 1, 0),),
                    ),
                ]
            ),
            ValueError,
        ),
    ],
)
def test_invalid_excitation_inputs_are_rejected(operation, error):
    with pytest.raises(error):
        operation()


def test_excitation_application_requires_its_source_to_be_occupied():
    excitation = VibrationalExcitation.from_transitions(
        ModalSpace((3,)),
        (ModalTransition(0, 2, 1),),
    )
    with pytest.raises(ValueError, match="source modal"):
        excitation.apply((0,))


def test_vscf_initial_point_preserves_duplicate_slots_and_validates_options():
    excitation = VibrationalExcitation.from_transitions(
        ModalSpace((2,)),
        (ModalTransition(0, 1, 0),),
    )
    assert vscf_initial_point((excitation, excitation)) == {
        "uvcc_theta_0": 0.0,
        "uvcc_theta_1": 0.0,
    }
    with pytest.raises(TypeError, match="prefix"):
        vscf_initial_point((excitation,), prefix=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="reps"):
        vscf_initial_point((excitation,), reps=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reps"):
        vscf_initial_point((excitation,), reps=-1)
