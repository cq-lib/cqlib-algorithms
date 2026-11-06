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

"""Tests for explicit restricted-orbital active-space transformation."""

import numpy as np
import pytest

from cqlib_algorithms.extension.electronic_structure import (
    ActiveSpace,
    ElectronicIntegrals,
    ElectronicOccupation,
    transform_active_space,
)


def _symmetric_eri(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, n, n, n))
    permutations = (
        raw,
        raw.transpose(1, 0, 2, 3),
        raw.transpose(0, 1, 3, 2),
        raw.transpose(1, 0, 3, 2),
        raw.transpose(2, 3, 0, 1),
        raw.transpose(3, 2, 0, 1),
        raw.transpose(2, 3, 1, 0),
        raw.transpose(3, 2, 1, 0),
    )
    return sum(permutations) / 8.0


def _set_chemist_eri(
    tensor: np.ndarray,
    p: int,
    q: int,
    r: int,
    s: int,
    value: float,
) -> None:
    """Set one real-orbital integral and its eight symmetry partners."""
    for indices in {
        (p, q, r, s),
        (q, p, r, s),
        (p, q, s, r),
        (q, p, s, r),
        (r, s, p, q),
        (s, r, p, q),
        (r, s, q, p),
        (s, r, q, p),
    }:
        tensor[indices] = value


def test_full_active_space_is_an_identity_snapshot():
    one_body = np.array([[-1.0, 0.2], [0.2, 0.7]])
    two_body = _symmetric_eri(13, 2)
    source = ElectronicIntegrals(0.3, one_body, two_body)

    result = transform_active_space(
        source,
        ElectronicOccupation.closed_shell(2, 2),
        ActiveSpace.full(2),
    )

    assert result.num_electrons == 2
    assert result.active_orbitals == (0, 1)
    assert result.frozen_orbitals == ()
    assert result.integrals.constant == source.constant
    assert np.array_equal(result.integrals.one_body, source.one_body)
    assert np.array_equal(result.integrals.two_body, source.two_body)
    assert result.integrals.one_body is not source.one_body


def test_manual_active_space_matches_independent_formula():
    one_body = np.array([[-2.0, 0.1, -0.2], [0.1, -0.5, 0.3], [-0.2, 0.3, 0.8]])
    two_body = _symmetric_eri(27, 3)
    source = ElectronicIntegrals(1.25, one_body, two_body)
    selection = ActiveSpace.manual(frozen_orbitals=(0,), active_orbitals=(1, 2))

    result = transform_active_space(
        source,
        ElectronicOccupation.closed_shell(3, 4),
        selection,
    )

    expected_constant = (
        1.25 + 2.0 * one_body[0, 0] + 2.0 * two_body[0, 0, 0, 0] - two_body[0, 0, 0, 0]
    )
    expected_one = np.empty((2, 2))
    for local_p, p in enumerate((1, 2)):
        for local_q, q in enumerate((1, 2)):
            expected_one[local_p, local_q] = (
                one_body[p, q] + 2.0 * two_body[p, q, 0, 0] - two_body[p, 0, 0, q]
            )

    assert result.num_electrons == 2
    assert result.integrals.constant == pytest.approx(expected_constant, abs=1e-14)
    assert np.allclose(result.integrals.one_body, expected_one, rtol=0.0, atol=1e-14)
    assert np.array_equal(result.integrals.two_body, two_body[1:3, 1:3, 1:3, 1:3])


@pytest.mark.reference
def test_two_frozen_orbitals_include_cross_terms_and_both_one_body_shifts():
    one_body = np.diag([-2.0, -1.0, -0.5, 0.8])
    one_body[2, 3] = one_body[3, 2] = 0.2
    two_body = np.zeros((4, 4, 4, 4))

    _set_chemist_eri(two_body, 0, 0, 0, 0, 0.7)
    _set_chemist_eri(two_body, 1, 1, 1, 1, 0.9)
    _set_chemist_eri(two_body, 0, 0, 1, 1, 0.4)
    _set_chemist_eri(two_body, 0, 1, 1, 0, 0.15)

    _set_chemist_eri(two_body, 2, 2, 0, 0, 0.25)
    _set_chemist_eri(two_body, 2, 0, 0, 2, 0.05)
    _set_chemist_eri(two_body, 2, 2, 1, 1, 0.35)
    _set_chemist_eri(two_body, 2, 1, 1, 2, 0.08)
    _set_chemist_eri(two_body, 3, 3, 0, 0, 0.12)
    _set_chemist_eri(two_body, 3, 0, 0, 3, 0.02)
    _set_chemist_eri(two_body, 3, 3, 1, 1, 0.18)
    _set_chemist_eri(two_body, 3, 1, 1, 3, 0.03)
    _set_chemist_eri(two_body, 2, 3, 0, 0, 0.11)
    _set_chemist_eri(two_body, 2, 0, 0, 3, 0.04)
    _set_chemist_eri(two_body, 2, 3, 1, 1, 0.13)
    _set_chemist_eri(two_body, 2, 1, 1, 3, 0.05)

    source = ElectronicIntegrals(0.3, one_body, two_body)
    result = transform_active_space(
        source,
        ElectronicOccupation.closed_shell(4, 6),
        ActiveSpace.manual(frozen_orbitals=(0, 1), active_orbitals=(2, 3)),
    )

    # Hand calculation includes both ordered cross terms (i,j)=(0,1),(1,0):
    # 0.3 + 2*(-2-1) + 0.7 + 0.9 + 2*(2*0.4-0.15) = -2.8.
    assert result.integrals.constant == pytest.approx(-2.8, abs=1e-14)
    assert np.allclose(
        result.integrals.one_body,
        np.array([[0.57, 0.59], [0.59, 1.35]]),
        rtol=0.0,
        atol=1e-14,
    )
    assert result.num_electrons == 2


@pytest.mark.parametrize(
    "selection",
    [
        ActiveSpace.manual(frozen_orbitals=(0,), active_orbitals=(2,)),
        ActiveSpace.manual(frozen_orbitals=(2,), active_orbitals=(0, 1)),
        ActiveSpace.manual(active_orbitals=(0, 3)),
    ],
)
def test_active_space_rejects_invalid_occupancy_or_range(selection):
    source = ElectronicIntegrals(0.0, np.eye(3), np.zeros((3,) * 4))
    with pytest.raises(ValueError):
        transform_active_space(
            source,
            ElectronicOccupation.closed_shell(3, 4),
            selection,
        )


def test_active_space_rejects_unsorted_overlapping_and_boolean_indices():
    with pytest.raises(ValueError, match="strictly increasing"):
        ActiveSpace.manual(active_orbitals=(1, 0))
    with pytest.raises(ValueError, match="disjoint"):
        ActiveSpace.manual(frozen_orbitals=(0,), active_orbitals=(0, 1))
    with pytest.raises(TypeError, match="int"):
        ActiveSpace.manual(active_orbitals=(True,))


def test_rohf_active_space_freezes_only_paired_core_and_preserves_spin():
    one_body = np.diag([-2.0, -0.6, 0.4])
    two_body = np.zeros((3,) * 4)
    _set_chemist_eri(two_body, 0, 0, 0, 0, 0.7)
    _set_chemist_eri(two_body, 1, 1, 0, 0, 0.25)
    _set_chemist_eri(two_body, 1, 0, 0, 1, 0.05)
    _set_chemist_eri(two_body, 2, 2, 0, 0, 0.12)
    _set_chemist_eri(two_body, 2, 0, 0, 2, 0.02)
    source = ElectronicIntegrals(0.3, one_body, two_body)
    occupation = ElectronicOccupation(3, 2, 1)

    result = transform_active_space(
        source,
        occupation,
        ActiveSpace.manual(frozen_orbitals=(0,), active_orbitals=(1, 2)),
    )

    assert result.occupation == ElectronicOccupation(2, 1, 0)
    assert result.num_alpha == 1
    assert result.num_beta == 0
    assert result.num_electrons == 1
    assert result.integrals.constant == pytest.approx(-3.0, abs=1e-14)
    np.testing.assert_allclose(
        result.integrals.one_body,
        np.diag([-0.15, 0.62]),
        rtol=0.0,
        atol=1e-14,
    )


def test_rohf_active_space_rejects_freezing_or_dropping_singly_occupied_orbital():
    source = ElectronicIntegrals(0.0, np.eye(3), np.zeros((3,) * 4))
    occupation = ElectronicOccupation(3, 2, 1)

    with pytest.raises(ValueError, match="doubly occupied"):
        transform_active_space(
            source,
            occupation,
            ActiveSpace.manual(frozen_orbitals=(1,), active_orbitals=(0, 2)),
        )
    with pytest.raises(ValueError, match="every initially occupied"):
        transform_active_space(
            source,
            occupation,
            ActiveSpace.manual(frozen_orbitals=(0,), active_orbitals=(2,)),
        )


def test_active_space_rejects_mismatched_or_invalid_occupation_argument():
    source = ElectronicIntegrals(0.0, np.eye(2), np.zeros((2,) * 4))
    selection = ActiveSpace.full(2)
    with pytest.raises(TypeError, match="ElectronicOccupation"):
        transform_active_space(source, 2, selection)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="same orbital count"):
        transform_active_space(source, ElectronicOccupation(1, 1, 0), selection)
