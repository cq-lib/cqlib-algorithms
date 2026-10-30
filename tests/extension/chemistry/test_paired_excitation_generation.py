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

"""Exact pool tests for paired-double and fixed paired-UCCSD generation."""

from math import comb

import pytest

from cqlib_algorithms.extension.chemistry import (
    ElectronicOccupation,
    FermionicExcitation,
    GroupedFermionicGenerator,
    generate_paired_double_excitations,
    generate_puccsd_generators,
)


def test_h2_paired_and_puccsd_pools_are_exact():
    occupation = ElectronicOccupation.closed_shell(2, 2)
    alpha = FermionicExcitation(4, (0,), (2,))
    beta = FermionicExcitation(4, (1,), (3,))
    paired = FermionicExcitation(4, (0, 1), (2, 3))
    assert generate_paired_double_excitations(occupation) == (paired,)
    assert generate_puccsd_generators(occupation) == (
        GroupedFermionicGenerator(((alpha, 1.0), (beta, 1.0))),
        paired,
    )


def test_paired_doubles_are_occupied_to_virtual_and_lexicographic():
    occupation = ElectronicOccupation.closed_shell(4, 4)
    assert generate_paired_double_excitations(occupation) == (
        FermionicExcitation(8, (0, 1), (4, 5)),
        FermionicExcitation(8, (0, 1), (6, 7)),
        FermionicExcitation(8, (2, 3), (4, 5)),
        FermionicExcitation(8, (2, 3), (6, 7)),
    )


@pytest.mark.parametrize("num_spatial_orbitals", [2, 3, 5])
def test_puccsd_pool_has_all_grouped_singles_then_all_paired_doubles(
    num_spatial_orbitals,
):
    occupation = ElectronicOccupation(
        num_spatial_orbitals,
        min(1, num_spatial_orbitals),
        min(1, num_spatial_orbitals),
    )
    generators = generate_puccsd_generators(occupation)
    pair_count = comb(num_spatial_orbitals, 2)
    assert len(generators) == 2 * pair_count
    assert all(
        isinstance(generator, GroupedFermionicGenerator)
        for generator in generators[:pair_count]
    )
    assert all(
        isinstance(generator, FermionicExcitation)
        for generator in generators[pair_count:]
    )

    expected_pairs = [
        (source, target)
        for source in range(num_spatial_orbitals)
        for target in range(source + 1, num_spatial_orbitals)
    ]
    actual_single_pairs = [
        (generator.terms[0][0].sources[0] // 2, generator.terms[0][0].targets[0] // 2)
        for generator in generators[:pair_count]
        if isinstance(generator, GroupedFermionicGenerator)
    ]
    actual_double_pairs = [
        (generator.sources[0] // 2, generator.targets[0] // 2)
        for generator in generators[pair_count:]
        if isinstance(generator, FermionicExcitation)
    ]
    assert actual_single_pairs == expected_pairs
    assert actual_double_pairs == expected_pairs


def test_puccsd_grouped_singles_share_equal_unscaled_spin_complements():
    generators = generate_puccsd_generators(ElectronicOccupation.closed_shell(3, 2))
    for generator in generators[:3]:
        assert isinstance(generator, GroupedFermionicGenerator)
        assert tuple(coefficient for _, coefficient in generator.terms) == (1.0, 1.0)
        alpha, beta = (excitation for excitation, _ in generator.terms)
        assert beta.sources == (alpha.sources[0] + 1,)
        assert beta.targets == (alpha.targets[0] + 1,)


@pytest.mark.parametrize(
    "factory",
    [generate_paired_double_excitations, generate_puccsd_generators],
)
def test_paired_pool_generation_requires_closed_shell(factory):
    with pytest.raises(TypeError, match="ElectronicOccupation"):
        factory(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="closed shell"):
        factory(ElectronicOccupation(3, 2, 1))


def test_degenerate_one_orbital_and_paired_vacuum_full_pools_are_empty():
    assert generate_puccsd_generators(ElectronicOccupation.closed_shell(1, 0)) == ()
    assert generate_paired_double_excitations(ElectronicOccupation(3, 0, 0)) == ()
    assert generate_paired_double_excitations(ElectronicOccupation(3, 3, 3)) == ()
