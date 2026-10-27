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

"""Tests for sparse Hermitian modal integrals."""

import numpy as np
import pytest

from cqlib_algorithms.extension.vibration import (
    ModalIntegrals,
    ModalSpace,
    ModalTransition,
)


def _transition(
    mode: int,
    bra_modal: int,
    ket_modal: int,
) -> ModalTransition:
    return ModalTransition(mode, bra_modal, ket_modal)


def test_one_two_and_higher_mode_terms_share_one_sparse_representation():
    space = ModalSpace((2, 3, 2))
    one_mode = (_transition(0, 1, 1),)
    two_mode = (_transition(2, 1, 0), _transition(0, 0, 1))
    three_mode = (
        _transition(2, 0, 1),
        _transition(0, 1, 0),
        _transition(1, 2, 1),
    )
    integrals = ModalIntegrals.from_terms(
        space,
        (
            (one_mode, 2.0),
            (two_mode, 0.3 + 0.4j),
            (tuple(transition.adjoint() for transition in two_mode), 0.3 - 0.4j),
            (three_mode, -0.25j),
            (
                tuple(transition.adjoint() for transition in three_mode),
                0.25j,
            ),
        ),
        constant=-1.5,
    )

    assert integrals.constant == -1.5
    assert integrals.max_order == 3
    assert len(integrals.terms_of_order(1)) == 1
    assert len(integrals.terms_of_order(2)) == 2
    assert len(integrals.terms_of_order(3)) == 2
    assert all(
        tuple(transition.mode for transition in term)
        == tuple(sorted(t.mode for t in term))
        for term, _ in integrals.terms
    )
    assert integrals.is_hermitian(atol=0.0)


def test_duplicate_terms_are_canonicalized_merged_and_zero_removed():
    space = ModalSpace((2, 2))
    left = (_transition(0, 0, 1), _transition(1, 1, 0))
    reordered = tuple(reversed(left))
    adjoint = tuple(transition.adjoint() for transition in left)
    integrals = ModalIntegrals.from_terms(
        space,
        (
            (left, 0.4),
            (reordered, 0.6),
            (adjoint, 1.0),
            ((_transition(0, 0, 0),), 2.0),
            ((_transition(0, 0, 0),), -2.0),
        ),
    )

    assert integrals.terms == (
        (
            (_transition(0, 0, 1), _transition(1, 1, 0)),
            1.0 + 0.0j,
        ),
        (
            (_transition(0, 1, 0), _transition(1, 0, 1)),
            1.0 + 0.0j,
        ),
    )


def test_bounded_conjugate_residual_is_projected_to_exact_hermitian_terms():
    space = ModalSpace((2,))
    forward = (_transition(0, 1, 0),)
    backward = (_transition(0, 0, 1),)
    integrals = ModalIntegrals.from_terms(
        space,
        (
            (forward, 0.3 + 0.4j),
            (backward, 0.3 - 0.4j + 4e-13j),
        ),
    )

    coefficients = dict(integrals.terms)
    expected = 0.3 + 0.4j - 2e-13j
    assert coefficients[forward] == expected
    assert coefficients[backward] == expected.conjugate()
    assert coefficients[backward] == coefficients[forward].conjugate()
    assert integrals.is_hermitian(atol=0.0)


def test_bounded_missing_adjoint_is_completed_by_hermitian_projection():
    forward = (_transition(0, 1, 0),)
    backward = (_transition(0, 0, 1),)
    integrals = ModalIntegrals.from_terms(
        ModalSpace((2,)),
        ((forward, 6e-13j),),
    )

    assert dict(integrals.terms) == {
        forward: 3e-13j,
        backward: -3e-13j,
    }
    assert integrals.is_hermitian(atol=0.0)


def test_bounded_imaginary_diagonal_is_projected_to_an_exact_real_value():
    diagonal = (_transition(0, 1, 1),)
    integrals = ModalIntegrals.from_terms(
        ModalSpace((2,)),
        ((diagonal, 2.5 + 4e-13j),),
    )

    assert integrals.terms == ((diagonal, 2.5 + 0.0j),)
    assert integrals.is_hermitian(atol=0.0)


def test_duplicate_coefficient_aggregation_rejects_floating_point_overflow():
    diagonal = (_transition(0, 1, 1),)
    largest = np.finfo(np.float64).max

    with pytest.raises(ValueError, match="combined.*finite"):
        ModalIntegrals.from_terms(
            ModalSpace((2,)),
            (
                (diagonal, largest),
                (diagonal, largest),
            ),
        )


def test_large_finite_conjugate_pair_uses_an_overflow_safe_projection():
    forward = (_transition(0, 1, 0),)
    backward = (_transition(0, 0, 1),)
    largest = np.finfo(np.float64).max
    coefficient = complex(largest, largest)
    integrals = ModalIntegrals.from_terms(
        ModalSpace((2,)),
        (
            (forward, coefficient),
            (backward, coefficient.conjugate()),
        ),
    )

    assert dict(integrals.terms) == {
        forward: coefficient,
        backward: coefficient.conjugate(),
    }
    assert integrals.is_hermitian(atol=0.0)


def test_truncation_retains_constant_and_complete_lower_orders():
    space = ModalSpace((2, 2, 2))
    diagonal_terms = tuple(
        (
            tuple(_transition(mode, 1, 1) for mode in range(order)),
            float(order),
        )
        for order in (1, 2, 3)
    )
    integrals = ModalIntegrals.from_terms(space, diagonal_terms, constant=0.75)

    truncated = integrals.truncated(2)

    assert truncated.constant == 0.75
    assert truncated.max_order == 2
    assert truncated.terms == diagonal_terms[:2]


@pytest.mark.parametrize(
    "terms",
    [
        (((_transition(0, 1, 0),), 1.0),),
        (((_transition(0, 0, 0),), 1.0j),),
        (
            ((_transition(0, 1, 0),), 1.0),
            ((_transition(0, 0, 1),), 1.0 + 1e-6j),
        ),
    ],
)
def test_non_hermitian_integrals_are_rejected(terms):
    with pytest.raises(ValueError, match="Hermitian conjugate symmetry"):
        ModalIntegrals.from_terms(ModalSpace((2,)), terms)


def test_conjugate_residual_above_tolerance_is_rejected():
    forward = (_transition(0, 1, 0),)
    backward = (_transition(0, 0, 1),)
    with pytest.raises(ValueError, match="Hermitian conjugate symmetry"):
        ModalIntegrals.from_terms(
            ModalSpace((2,)),
            (
                (forward, 1.0),
                (backward, 1.0 + 1.1e-12j),
            ),
        )


@pytest.mark.parametrize(
    ("operation", "error"),
    [
        (lambda: ModalTransition(True, 0, 0), TypeError),
        (lambda: ModalTransition(0, -1, 0), ValueError),
        (
            lambda: ModalIntegrals.from_terms(
                ModalSpace((2,)),
                (((), 1.0),),
            ),
            ValueError,
        ),
        (
            lambda: ModalIntegrals.from_terms(
                ModalSpace((2,)),
                (((_transition(1, 0, 0),), 1.0),),
            ),
            ValueError,
        ),
        (
            lambda: ModalIntegrals.from_terms(
                ModalSpace((2,)),
                (((_transition(0, 2, 2),), 1.0),),
            ),
            ValueError,
        ),
        (
            lambda: ModalIntegrals.from_terms(
                ModalSpace((2,)),
                (((_transition(0, 0, 0), _transition(0, 1, 1)), 1.0),),
            ),
            ValueError,
        ),
        (
            lambda: ModalIntegrals.from_terms(
                ModalSpace((2,)),
                (((_transition(0, 0, 0),), np.inf),),
            ),
            ValueError,
        ),
        (
            lambda: ModalIntegrals.from_terms(
                ModalSpace((2,)),
                (((_transition(0, 0, 0),), True),),
            ),
            TypeError,
        ),
        (
            lambda: ModalIntegrals.from_terms(
                ModalSpace((2,)),
                (((_transition(0, 0, 0),), 1.0),),
                constant=complex(1.0),  # type: ignore[arg-type]
            ),
            TypeError,
        ),
    ],
)
def test_invalid_integral_data_is_rejected(operation, error):
    with pytest.raises(error):
        operation()


@pytest.mark.parametrize("order", [0, 3])
def test_invalid_truncation_order_is_rejected(order):
    integrals = ModalIntegrals.from_terms(
        ModalSpace((2, 2)),
        (((_transition(0, 0, 0),), 1.0),),
    )
    with pytest.raises(ValueError):
        integrals.truncated(order)


def test_integrals_defensively_copy_input_terms():
    entries = [((_transition(0, 0, 0),), 1.0)]
    integrals = ModalIntegrals.from_terms(ModalSpace((2,)), entries)
    entries.clear()

    assert len(integrals.terms) == 1
