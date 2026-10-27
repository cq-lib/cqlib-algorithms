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

"""Tests for deterministic spin-conserving excitation generation."""

from math import comb

import pytest

from cqlib_algorithms.extension.chemistry import (
    ElectronicOccupation,
    FermionicExcitation,
    generate_double_excitations,
    generate_single_excitations,
    generate_uccsd_excitations,
)


def _expected_counts(occupation: ElectronicOccupation) -> tuple[int, int]:
    n = occupation.num_spatial_orbitals
    va = n - occupation.num_alpha
    vb = n - occupation.num_beta
    singles = occupation.num_alpha * va + occupation.num_beta * vb
    doubles = (
        comb(occupation.num_alpha, 2) * comb(va, 2)
        + comb(occupation.num_beta, 2) * comb(vb, 2)
        + occupation.num_alpha * occupation.num_beta * va * vb
    )
    return singles, doubles


def _independent_excitation_sets(
    occupation: ElectronicOccupation,
) -> tuple[
    set[tuple[tuple[int, ...], tuple[int, ...]]],
    set[tuple[tuple[int, ...], tuple[int, ...]]],
]:
    occupied = {
        index for index, value in enumerate(occupation.spin_orbital_occupation) if value
    }
    virtual = set(range(occupation.num_spin_orbitals)) - occupied
    singles = {
        ((source,), (target,))
        for source in occupied
        for target in virtual
        if source % 2 == target % 2
    }
    doubles = {
        ((first_source, second_source), (first_target, second_target))
        for first_source in occupied
        for second_source in occupied
        for first_target in virtual
        for second_target in virtual
        if first_source < second_source
        and first_target < second_target
        and (first_source % 2 + second_source % 2)
        == (first_target % 2 + second_target % 2)
    }
    return singles, doubles


def test_h2_excitation_sequence_is_exact():
    occupation = ElectronicOccupation.closed_shell(2, 2)
    singles = generate_single_excitations(occupation)
    doubles = generate_double_excitations(occupation)
    assert singles == (
        FermionicExcitation(4, (0,), (2,)),
        FermionicExcitation(4, (1,), (3,)),
    )
    assert doubles == (FermionicExcitation(4, (0, 1), (2, 3)),)
    assert generate_uccsd_excitations(occupation) == singles + doubles


@pytest.mark.parametrize(
    ("occupation", "expected"),
    [
        (ElectronicOccupation(6, 2, 2), (16, 76)),
        (ElectronicOccupation(5, 1, 1), (8, 16)),
        (
            ElectronicOccupation(5, 3, 2),
            _expected_counts(ElectronicOccupation(5, 3, 2)),
        ),
        (
            ElectronicOccupation(4, 2, 1),
            _expected_counts(ElectronicOccupation(4, 2, 1)),
        ),
    ],
)
def test_excitation_counts_match_combinatorial_formula(occupation, expected):
    assert (
        len(generate_single_excitations(occupation)),
        len(generate_double_excitations(occupation)),
    ) == expected


@pytest.mark.parametrize(
    ("occupation", "expected_spin_sectors"),
    [
        (ElectronicOccupation(6, 2, 2), {0, 1, 2}),
        (ElectronicOccupation(5, 1, 1), {1}),
        (ElectronicOccupation(5, 3, 2), {0, 1, 2}),
    ],
)
def test_excitation_sets_match_independent_oracle(
    occupation,
    expected_spin_sectors,
):
    occupied = set(occupation.occupied_spin_orbitals)
    virtual = set(occupation.virtual_spin_orbitals)
    expected_singles, expected_doubles = _independent_excitation_sets(occupation)
    actual_singles = generate_single_excitations(occupation)
    actual_doubles = generate_double_excitations(occupation)

    assert {(item.sources, item.targets) for item in actual_singles} == expected_singles
    assert {(item.sources, item.targets) for item in actual_doubles} == expected_doubles
    assert all(
        set(item.sources) <= occupied for item in (*actual_singles, *actual_doubles)
    )
    assert all(
        set(item.targets) <= virtual for item in (*actual_singles, *actual_doubles)
    )
    assert {
        sum(index % 2 for index in item.sources) for item in actual_doubles
    } == expected_spin_sectors


@pytest.mark.parametrize(
    "occupation",
    [ElectronicOccupation(3, 0, 0), ElectronicOccupation(3, 3, 3)],
)
def test_vacuum_and_full_occupations_have_no_excitations(occupation):
    assert generate_single_excitations(occupation) == ()
    assert generate_double_excitations(occupation) == ()
    assert generate_uccsd_excitations(occupation) == ()


@pytest.mark.parametrize(
    "occupation",
    [ElectronicOccupation(4, 2, 2), ElectronicOccupation(5, 3, 1)],
)
def test_generated_excitations_are_unique_stable_and_spin_conserving(occupation):
    first = generate_uccsd_excitations(occupation)
    keys = [(item.rank, item.sources, item.targets) for item in first]
    assert first == generate_uccsd_excitations(occupation)
    assert len(first) == len(set(first))
    assert keys == sorted(keys)
    for excitation in first:
        assert sum(mode % 2 == 0 for mode in excitation.sources) == sum(
            mode % 2 == 0 for mode in excitation.targets
        )


def test_generation_rejects_wrong_input_type():
    with pytest.raises(TypeError, match="ElectronicOccupation"):
        generate_uccsd_excitations(object())  # type: ignore[arg-type]
