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

"""Tests for spin-resolved unrestricted active-space transformation."""

import numpy as np
import pytest

from cqlib_algorithms.extension.electronic_structure import (
    ElectronicEnergy,
    ElectronicOccupation,
    SpinResolvedActiveSpace,
    SpinResolvedElectronicIntegrals,
    transform_spin_resolved_active_space,
)


def _same_spin_tensor(seed: int, n: int) -> np.ndarray:
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


def _mixed_spin_tensor(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, n, n, n))
    return (
        raw
        + raw.transpose(1, 0, 2, 3)
        + raw.transpose(0, 1, 3, 2)
        + raw.transpose(1, 0, 3, 2)
    ) / 4.0


def _integrals(n: int = 4) -> SpinResolvedElectronicIntegrals:
    rng = np.random.default_rng(41)
    one_alpha = rng.normal(size=(n, n))
    one_beta = rng.normal(size=(n, n))
    return SpinResolvedElectronicIntegrals(
        0.37,
        (one_alpha + one_alpha.T) / 2.0,
        (one_beta + one_beta.T) / 2.0,
        _same_spin_tensor(42, n),
        _same_spin_tensor(43, n),
        _mixed_spin_tensor(44, n),
    )


def _embedded_basis(
    selection: SpinResolvedActiveSpace,
    num_active_orbitals: int,
) -> tuple[list[int], list[float]]:
    indices: list[int] = []
    phases: list[float] = []
    frozen_modes = sorted(
        [2 * orbital for orbital in selection.frozen_alpha_orbitals]
        + [2 * orbital + 1 for orbital in selection.frozen_beta_orbitals]
    )
    active_modes = [
        mode
        for pair in zip(
            (2 * orbital for orbital in selection.active_alpha_orbitals),
            (2 * orbital + 1 for orbital in selection.active_beta_orbitals),
            strict=True,
        )
        for mode in pair
    ]
    for active_basis in range(1 << (2 * num_active_orbitals)):
        full_basis = 0
        occupied_sequence = list(frozen_modes)
        for mode in frozen_modes:
            full_basis |= 1 << mode
        for local, mode in enumerate(active_modes):
            if active_basis & (1 << local):
                full_basis |= 1 << mode
                occupied_sequence.append(mode)
        inversions = sum(
            left > right
            for offset, left in enumerate(occupied_sequence)
            for right in occupied_sequence[offset + 1 :]
        )
        indices.append(full_basis)
        phases.append(-1.0 if inversions % 2 else 1.0)
    return indices, phases


def test_full_spin_resolved_active_space_is_an_identity_snapshot():
    source = _integrals(2)
    result = transform_spin_resolved_active_space(
        source,
        ElectronicOccupation(2, 1, 1),
        SpinResolvedActiveSpace.full(2),
    )

    assert result.occupation == ElectronicOccupation(2, 1, 1)
    assert result.integrals.constant == source.constant
    for name in (
        "one_body_alpha",
        "one_body_beta",
        "two_body_alpha_alpha",
        "two_body_beta_beta",
        "two_body_alpha_beta",
    ):
        actual = getattr(result.integrals, name)
        expected = getattr(source, name)
        np.testing.assert_array_equal(actual, expected)
        assert actual is not expected


@pytest.mark.reference
def test_spin_resolved_active_hamiltonian_matches_exact_fock_space_projection():
    source = _integrals()
    occupation = ElectronicOccupation(4, 3, 2)
    full_matrix = ElectronicEnergy(source).to_qubit_hamiltonian().to_matrix()
    cases = (
        (
            SpinResolvedActiveSpace.manual(
                frozen_alpha_orbitals=(0,),
                frozen_beta_orbitals=(0,),
                active_alpha_orbitals=(1, 2),
                active_beta_orbitals=(1, 3),
            ),
            ElectronicOccupation(2, 2, 1),
        ),
        (
            SpinResolvedActiveSpace.manual(
                frozen_alpha_orbitals=(0, 1),
                frozen_beta_orbitals=(0,),
                active_alpha_orbitals=(2,),
                active_beta_orbitals=(1,),
            ),
            ElectronicOccupation(1, 1, 1),
        ),
        (
            SpinResolvedActiveSpace.manual(
                frozen_alpha_orbitals=(0,),
                frozen_beta_orbitals=(0, 1),
                active_alpha_orbitals=(1, 2),
                active_beta_orbitals=(2, 3),
            ),
            ElectronicOccupation(2, 2, 0),
        ),
    )
    for selection, expected_occupation in cases:
        result = transform_spin_resolved_active_space(
            source,
            occupation,
            selection,
        )
        reduced_matrix = (
            ElectronicEnergy(result.integrals).to_qubit_hamiltonian().to_matrix()
        )
        embedded, phases = _embedded_basis(
            selection,
            result.integrals.num_spatial_orbitals,
        )
        projected = full_matrix[np.ix_(embedded, embedded)]
        projected *= np.outer(phases, phases)

        np.testing.assert_allclose(
            reduced_matrix,
            projected,
            rtol=0.0,
            atol=1e-11,
        )
        assert result.occupation == expected_occupation


def test_spin_resolved_constant_and_one_body_terms_match_independent_formula():
    source = _integrals(3)
    occupation = ElectronicOccupation(3, 2, 1)
    selection = SpinResolvedActiveSpace.manual(
        frozen_alpha_orbitals=(0,),
        frozen_beta_orbitals=(0,),
        active_alpha_orbitals=(1, 2),
        active_beta_orbitals=(1, 2),
    )

    result = transform_spin_resolved_active_space(source, occupation, selection)
    aa = source.two_body_alpha_alpha
    bb = source.two_body_beta_beta
    ab = source.two_body_alpha_beta
    expected_constant = (
        source.constant
        + source.one_body_alpha[0, 0]
        + source.one_body_beta[0, 0]
        + 0.5 * (aa[0, 0, 0, 0] - aa[0, 0, 0, 0])
        + 0.5 * (bb[0, 0, 0, 0] - bb[0, 0, 0, 0])
        + ab[0, 0, 0, 0]
    )
    expected_alpha = np.empty((2, 2))
    expected_beta = np.empty((2, 2))
    for local_p, p in enumerate((1, 2)):
        for local_q, q in enumerate((1, 2)):
            expected_alpha[local_p, local_q] = (
                source.one_body_alpha[p, q]
                + aa[p, q, 0, 0]
                - aa[p, 0, 0, q]
                + ab[p, q, 0, 0]
            )
            expected_beta[local_p, local_q] = (
                source.one_body_beta[p, q]
                + bb[p, q, 0, 0]
                - bb[p, 0, 0, q]
                + ab[0, 0, p, q]
            )

    assert result.integrals.constant == pytest.approx(expected_constant, abs=1e-13)
    np.testing.assert_allclose(
        result.integrals.one_body_alpha,
        expected_alpha,
        rtol=0.0,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        result.integrals.one_body_beta,
        expected_beta,
        rtol=0.0,
        atol=1e-13,
    )


def test_spin_resolved_selection_validates_structure_and_occupancy():
    with pytest.raises(ValueError, match="equal lengths"):
        SpinResolvedActiveSpace.manual(
            active_alpha_orbitals=(0,),
            active_beta_orbitals=(0, 1),
        )
    with pytest.raises(ValueError, match="disjoint"):
        SpinResolvedActiveSpace.manual(
            frozen_alpha_orbitals=(0,),
            active_alpha_orbitals=(0,),
            active_beta_orbitals=(0,),
        )
    with pytest.raises(TypeError, match="int"):
        SpinResolvedActiveSpace.manual(
            active_alpha_orbitals=(True,),
            active_beta_orbitals=(0,),
        )

    source = _integrals(3)
    occupation = ElectronicOccupation(3, 2, 1)
    invalid = (
        SpinResolvedActiveSpace.manual(
            frozen_alpha_orbitals=(2,),
            active_alpha_orbitals=(0, 1),
            active_beta_orbitals=(0, 1),
        ),
        SpinResolvedActiveSpace.manual(
            frozen_alpha_orbitals=(0,),
            active_alpha_orbitals=(2,),
            active_beta_orbitals=(0,),
        ),
        SpinResolvedActiveSpace.manual(
            active_alpha_orbitals=(0, 1),
            active_beta_orbitals=(0, 3),
        ),
    )
    for selection in invalid:
        with pytest.raises(ValueError):
            transform_spin_resolved_active_space(source, occupation, selection)


def test_spin_resolved_transform_rejects_invalid_argument_types_and_dimensions():
    source = _integrals(2)
    occupation = ElectronicOccupation(2, 1, 1)
    selection = SpinResolvedActiveSpace.full(2)
    with pytest.raises(TypeError, match="SpinResolvedElectronicIntegrals"):
        transform_spin_resolved_active_space(object(), occupation, selection)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ElectronicOccupation"):
        transform_spin_resolved_active_space(source, object(), selection)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="SpinResolvedActiveSpace"):
        transform_spin_resolved_active_space(source, occupation, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="same orbital count"):
        transform_spin_resolved_active_space(
            source,
            ElectronicOccupation(1, 1, 0),
            selection,
        )
