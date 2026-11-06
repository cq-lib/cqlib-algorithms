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

"""Independent matrix and boundary tests for grouped fermionic generators."""

import math

import numpy as np
import pytest
from cqlib.qis import FermionicLadder, JordanWignerMapper

from cqlib_algorithms.extension.electronic_structure import (
    FermionicExcitation,
    GroupedFermionicGenerator,
)


def _apply_ladder(state: int, mode: int, *, create: bool) -> tuple[int, float] | None:
    occupied = bool(state & (1 << mode))
    if occupied == create:
        return None
    sign = -1.0 if (state & ((1 << mode) - 1)).bit_count() % 2 else 1.0
    output = state | (1 << mode) if create else state & ~(1 << mode)
    return output, sign


def _transition(excitation: FermionicExcitation) -> np.ndarray:
    size = 1 << excitation.num_spin_orbitals
    result = np.zeros((size, size), dtype=np.complex128)
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
            result[output, basis] = amplitude
    return result


def _generator(excitation: FermionicExcitation) -> np.ndarray:
    transition = _transition(excitation)
    return 1j * (transition - transition.conj().T)


@pytest.mark.reference
def test_grouped_generator_matches_independent_fock_matrix():
    alpha = FermionicExcitation(4, (0,), (2,))
    beta = FermionicExcitation(4, (1,), (3,))
    grouped = GroupedFermionicGenerator(((alpha, 1.0), (beta, -0.25)))
    expected = _generator(alpha) - 0.25 * _generator(beta)
    actual = (
        JordanWignerMapper().map_operator(grouped.to_fermionic_generator()).to_matrix()
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(actual, actual.conj().T, rtol=0.0, atol=1e-12)
    assert grouped.num_spin_orbitals == 4


@pytest.mark.reference
def test_distinct_generators_with_opposite_coefficients_are_valid():
    alpha = FermionicExcitation(4, (0,), (2,))
    beta = FermionicExcitation(4, (1,), (3,))
    grouped = GroupedFermionicGenerator(((alpha, 1.0), (beta, -1.0)))
    expected = _generator(alpha) - _generator(beta)
    actual = (
        JordanWignerMapper().map_operator(grouped.to_fermionic_generator()).to_matrix()
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("reverse_coefficient", [1.0, 0.25, -0.75])
def test_forward_and_reverse_directions_are_rejected_at_construction(
    reverse_coefficient: float,
):
    forward = FermionicExcitation(4, (0,), (2,))
    reverse = FermionicExcitation(4, (2,), (0,))
    with pytest.raises(ValueError, match="both forward and reverse directions"):
        GroupedFermionicGenerator(((forward, 1.0), (reverse, reverse_coefficient)))


def test_grouped_generator_returns_independent_core_objects():
    grouped = GroupedFermionicGenerator(
        (
            (FermionicExcitation(4, (0,), (2,)), 1.0),
            (FermionicExcitation(4, (1,), (3,)), 1.0),
        )
    )
    first = grouped.to_fermionic_generator()
    second = grouped.to_fermionic_generator()
    first.add_term((FermionicLadder.create(0), FermionicLadder.annihilate(0)), 3.0)
    assert not first.equiv(second, atol=0.0)


@pytest.mark.parametrize(
    ("terms", "error", "message"),
    [
        ([], TypeError, "tuple"),
        ((), ValueError, "must not be empty"),
        (((object(), 1.0),), TypeError, "FermionicExcitation"),
        (((FermionicExcitation(4, (0,), (2,)), True),), TypeError, "real"),
        (((FermionicExcitation(4, (0,), (2,)), 1.0j),), TypeError, "real"),
        (((FermionicExcitation(4, (0,), (2,)), math.inf),), ValueError, "finite"),
        (((FermionicExcitation(4, (0,), (2,)), 0.0),), ValueError, "nonzero"),
        (
            (
                (FermionicExcitation(4, (0,), (2,)), 1.0),
                (FermionicExcitation(4, (0,), (2,)), 2.0),
            ),
            ValueError,
            "unique",
        ),
        (
            (
                (FermionicExcitation(4, (0,), (2,)), 1.0),
                (FermionicExcitation(6, (1,), (3,)), 1.0),
            ),
            ValueError,
            "same width",
        ),
    ],
)
def test_grouped_generator_rejects_invalid_terms(terms, error, message):
    with pytest.raises(error, match=message):
        GroupedFermionicGenerator(terms)  # type: ignore[arg-type]


def test_grouped_generator_rejects_non_tuple_items():
    excitation = FermionicExcitation(4, (0,), (2,))
    with pytest.raises(TypeError, match="tuples"):
        GroupedFermionicGenerator(([excitation, 1.0],))  # type: ignore[arg-type]
