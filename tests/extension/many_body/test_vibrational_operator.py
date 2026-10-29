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

"""Tests for the sparse vibrational operator representation."""

import numpy as np
import pytest

from cqlib_algorithms.extension.many_body import VibrationalLadder, VibrationalOp
from cqlib_algorithms.extension.vibration import ModalSpace


def _transition(
    space: ModalSpace,
    output_configuration: tuple[int, ...],
    input_configuration: tuple[int, ...],
    coefficient: complex = 1.0,
) -> VibrationalOp:
    factors: list[VibrationalLadder] = []
    for mode, (output_modal, input_modal) in enumerate(
        zip(output_configuration, input_configuration)
    ):
        factors.extend(
            (
                VibrationalLadder("+", mode, output_modal),
                VibrationalLadder("-", mode, input_modal),
            )
        )
    return VibrationalOp.from_terms(space, ((factors, coefficient),))


def test_ladder_adjoint_and_validation():
    creation = VibrationalLadder("+", 1, 2)

    assert creation.adjoint() == VibrationalLadder("-", 1, 2)
    assert creation.adjoint().adjoint() == creation
    with pytest.raises(ValueError, match="kind"):
        VibrationalLadder("x", 0, 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="mode"):
        VibrationalLadder("+", True, 0)
    with pytest.raises(ValueError, match="modal"):
        VibrationalLadder("+", 0, -1)


def test_construction_canonicalizes_commuting_factors_and_merges_duplicates():
    space = ModalSpace((2, 2))
    first = VibrationalLadder("+", 0, 1)
    second = VibrationalLadder("-", 1, 0)
    mutable = [second, first]
    entries = [(mutable, 2.0), ((first, second), -0.5)]

    operator = VibrationalOp.from_terms(space, entries)
    mutable.clear()
    entries.clear()

    assert operator.terms == (((first, second), 1.5 + 0.0j),)


def test_same_modal_factor_order_is_not_discarded():
    space = ModalSpace((2,))
    creation = VibrationalLadder("+", 0, 0)
    annihilation = creation.adjoint()
    occupied_projector = VibrationalOp.from_terms(
        space,
        (((creation, annihilation), 1.0),),
    )
    empty_projector = VibrationalOp.from_terms(
        space,
        (((annihilation, creation), 1.0),),
    )

    assert occupied_projector.terms != empty_projector.terms
    np.testing.assert_array_equal(
        occupied_projector.to_modal_matrix(),
        np.array([[1.0, 0.0], [0.0, 0.0]]),
    )
    np.testing.assert_array_equal(
        empty_projector.to_modal_matrix(),
        np.array([[0.0, 0.0], [0.0, 1.0]]),
    )


def test_add_scale_subtract_multiply_and_adjoint_match_modal_linear_algebra():
    space = ModalSpace((3,))
    raising = _transition(space, (2,), (0,), 1.0 + 2.0j)
    lowering = raising.adjoint()
    hermitian = raising + lowering
    product = raising @ lowering

    expected_raising = np.zeros((3, 3), dtype=np.complex128)
    expected_raising[2, 0] = 1.0 + 2.0j
    np.testing.assert_array_equal(raising.to_modal_matrix(), expected_raising)
    np.testing.assert_array_equal(
        lowering.to_modal_matrix(),
        expected_raising.conjugate().T,
    )
    np.testing.assert_allclose(
        hermitian.to_modal_matrix(),
        expected_raising + expected_raising.conjugate().T,
        atol=0.0,
    )
    np.testing.assert_allclose(
        product.to_modal_matrix(),
        expected_raising @ expected_raising.conjugate().T,
        atol=0.0,
    )
    assert hermitian.is_hermitian()
    assert not raising.is_hermitian()
    assert ((2.0 * raising - raising) * (0.5 - 0.5j)).equiv(raising.scaled(0.5 - 0.5j))


def test_hermiticity_respects_exact_local_ladder_identities():
    space = ModalSpace((2,))
    creation = VibrationalLadder("+", 0, 0)
    nilpotent_word = VibrationalOp.from_terms(
        space,
        (((creation, creation), 1.0 + 2.0j),),
    )

    assert not nilpotent_word.coefficient_equiv(nilpotent_word.adjoint())
    np.testing.assert_array_equal(
        nilpotent_word.to_modal_matrix(),
        np.zeros((2, 2), dtype=np.complex128),
    )
    assert nilpotent_word.equiv(nilpotent_word.adjoint())
    assert nilpotent_word.is_hermitian()


def test_normalization_drops_only_the_requested_numerical_residuals():
    space = ModalSpace((2,))
    small = _transition(space, (1,), (0,), 1e-13)
    large = _transition(space, (0,), (1,), 2e-10)
    operator = small + large

    assert operator.normalized().num_terms == 2
    assert operator.normalized(atol=1e-12).equiv(large)
    with pytest.raises(ValueError, match="non-negative"):
        operator.normalized(atol=-1.0)
    with pytest.raises(TypeError, match="real number"):
        operator.normalized(atol=True)


def test_mode_occupancy_conservation_identifies_one_hot_preserving_terms():
    space = ModalSpace((2, 2))
    preserving = _transition(space, (1, 0), (0, 1))
    non_preserving = VibrationalOp.from_terms(
        space,
        (((VibrationalLadder("+", 0, 1),), 1.0),),
    )

    assert preserving.preserves_one_hot_subspace()
    assert VibrationalOp.identity(space).preserves_one_hot_subspace()
    assert not non_preserving.preserves_one_hot_subspace()


@pytest.mark.parametrize(
    ("factory", "error"),
    [
        (lambda: VibrationalOp.from_terms(object(), ()), TypeError),
        (lambda: VibrationalOp.from_terms(ModalSpace((2,)), 42), TypeError),
        (lambda: VibrationalOp.from_terms(ModalSpace((2,)), (((),),)), ValueError),
        (
            lambda: VibrationalOp.from_terms(ModalSpace((2,)), (((object(),), 1.0),)),
            TypeError,
        ),
        (
            lambda: VibrationalOp.from_terms(
                ModalSpace((2,)),
                (((VibrationalLadder("+", 0, 2),), 1.0),),
            ),
            ValueError,
        ),
        (lambda: VibrationalOp.from_terms(ModalSpace((2,)), (((), True),)), TypeError),
        (
            lambda: VibrationalOp.from_terms(
                ModalSpace((2,)),
                (((), complex(float("nan"), 0.0)),),
            ),
            ValueError,
        ),
    ],
)
def test_invalid_operator_inputs_are_rejected(factory, error):
    with pytest.raises(error):
        factory()


def test_arithmetic_rejects_incompatible_spaces_and_wrong_operands():
    left = VibrationalOp.identity(ModalSpace((2,)))
    right = VibrationalOp.identity(ModalSpace((3,)))

    with pytest.raises(ValueError, match="same modal space"):
        _ = left + right
    with pytest.raises(TypeError, match="other must be"):
        _ = left @ object()  # type: ignore[operator]
    with pytest.raises(TypeError, match="factor"):
        _ = left * object()  # type: ignore[operator]
