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

"""Scientific and boundary checks for electronic initial points."""

from types import MappingProxyType

import numpy as np
import pytest

from cqlib_algorithms.extension.electronic_structure import (
    ElectronicEnergy,
    ElectronicIntegrals,
    ElectronicOccupation,
    ElectronicSystem,
    generate_uccsd_excitations,
    hartree_fock_initial_point,
    mp2_initial_point,
)
from cqlib_algorithms.operators.ansatz import UCCSDAnsatz


def _canonical_system(coupling: float = 0.2) -> ElectronicSystem:
    energies = np.array([-0.5, 0.5])
    two_body = np.zeros((2, 2, 2, 2))
    for p, q, r, s in {
        (1, 0, 1, 0),
        (0, 1, 1, 0),
        (1, 0, 0, 1),
        (0, 1, 0, 1),
    }:
        two_body[p, q, r, s] = coupling
    one_body = np.diag(energies)
    occupied = 0
    one_body -= 2.0 * two_body[:, :, occupied, occupied]
    one_body += two_body[:, occupied, occupied, :]
    return ElectronicSystem(
        ElectronicIntegrals(0.4, one_body, two_body),
        ElectronicOccupation.closed_shell(2, 2),
        energies,
        -0.8,
    )


def _apply_ladder(state: int, mode: int, *, create: bool) -> tuple[int, float]:
    occupied = bool(state & (1 << mode))
    assert occupied != create
    sign = -1.0 if (state & ((1 << mode) - 1)).bit_count() % 2 else 1.0
    output = state | (1 << mode) if create else state & ~(1 << mode)
    return output, sign


def _double_excitation_sign() -> float:
    state = 0b0011
    amplitude = 1.0
    for create, mode in ((False, 0), (False, 1), (True, 3), (True, 2)):
        state, sign = _apply_ladder(state, mode, create=create)
        amplitude *= sign
    assert state == 0b1100
    return amplitude


def _excited_basis_and_sign(
    reference: int,
    sources: tuple[int, ...],
    targets: tuple[int, ...],
) -> tuple[int, float]:
    state = reference
    amplitude = 1.0
    operations = (
        *((False, mode) for mode in sources),
        *((True, mode) for mode in reversed(targets)),
    )
    for create, mode in operations:
        state, sign = _apply_ladder(state, mode, create=create)
        amplitude *= sign
    return state, amplitude


def test_hartree_fock_point_matches_uccsd_symbol_order_and_is_immutable():
    system = _canonical_system()
    point = hartree_fock_initial_point(system, prefix="chem")
    circuit = UCCSDAnsatz(system.occupation).build_circuit("chem")

    assert isinstance(point, MappingProxyType)
    assert tuple(point) == tuple(circuit.used_symbols)
    assert point == {
        "chem_theta_0": 0.0,
        "chem_theta_1": 0.0,
        "chem_theta_2": 0.0,
    }
    with pytest.raises(TypeError):
        point["chem_theta_0"] = 1.0  # type: ignore[index]
    state = circuit.assign_parameters(dict(point)).to_matrix()[:, 0]
    np.testing.assert_allclose(state, np.eye(16)[:, 0b0011], rtol=0.0, atol=1e-12)


def test_mp2_double_matches_independent_perturbative_matrix_element():
    system = _canonical_system()
    point = mp2_initial_point(system, prefix="chem")
    matrix = ElectronicEnergy(system.integrals).to_qubit_hamiltonian().to_matrix()
    matrix_element = float(matrix[0b1100, 0b0011].real)
    denominator = 2.0 * (-0.5) - 2.0 * 0.5
    expected_parameter = matrix_element / (denominator * _double_excitation_sign())

    assert point["chem_theta_0"] == 0.0
    assert point["chem_theta_1"] == 0.0
    assert point["chem_theta_2"] == pytest.approx(expected_parameter, abs=1e-12)
    assert point["chem_theta_2"] == pytest.approx(-0.1, abs=1e-12)


def test_mp2_all_spin_channels_match_random_full_hamiltonian_matrix_elements():
    rng = np.random.default_rng(4817)
    num_orbitals = 3
    raw = rng.normal(size=(num_orbitals,) * 4)
    permutations = (
        (0, 1, 2, 3),
        (1, 0, 2, 3),
        (0, 1, 3, 2),
        (1, 0, 3, 2),
        (2, 3, 0, 1),
        (3, 2, 0, 1),
        (2, 3, 1, 0),
        (3, 2, 1, 0),
    )
    two_body = sum(raw.transpose(order) for order in permutations) / 8.0
    energies = np.array([-1.1, 0.2, 0.9])
    one_body = np.diag(energies)
    one_body -= 2.0 * two_body[:, :, 0, 0]
    one_body += two_body[:, 0, 0, :]
    system = ElectronicSystem(
        ElectronicIntegrals(0.0, one_body, two_body),
        ElectronicOccupation.closed_shell(num_orbitals, 2),
        energies,
    )
    point = mp2_initial_point(system, prefix="random")
    matrix = ElectronicEnergy(system.integrals).to_qubit_hamiltonian().to_matrix()
    reference = 0b000011
    spin_energies = np.repeat(energies, 2)

    excitations = generate_uccsd_excitations(system.occupation)
    for index, excitation in enumerate(excitations):
        actual = point[f"random_theta_{index}"]
        if excitation.rank == 1:
            assert actual == 0.0
            continue
        target, transition_sign = _excited_basis_and_sign(
            reference,
            excitation.sources,
            excitation.targets,
        )
        denominator = float(
            sum(spin_energies[mode] for mode in excitation.sources)
            - sum(spin_energies[mode] for mode in excitation.targets)
        )
        expected = matrix[target, reference].real / (denominator * transition_sign)
        assert actual == pytest.approx(expected, abs=1e-12)


def test_mp2_threshold_only_removes_small_amplitudes():
    point = mp2_initial_point(_canonical_system(1e-5), threshold=1e-4)
    assert tuple(point.values()) == (0.0, 0.0, 0.0)


def test_mp2_rejects_missing_noncanonical_or_degenerate_inputs():
    canonical = _canonical_system()
    without_energies = ElectronicSystem(canonical.integrals, canonical.occupation)
    with pytest.raises(ValueError, match="orbital_energies"):
        mp2_initial_point(without_energies)

    one_body = np.array(canonical.integrals.one_body, copy=True)
    one_body[0, 1] = one_body[1, 0] = 0.1
    noncanonical = ElectronicSystem(
        ElectronicIntegrals(
            canonical.integrals.constant,
            one_body,
            canonical.integrals.two_body,
        ),
        canonical.occupation,
        canonical.orbital_energies,
    )
    with pytest.raises(ValueError, match="canonical"):
        mp2_initial_point(noncanonical)

    degenerate = ElectronicSystem(
        ElectronicIntegrals(
            canonical.integrals.constant,
            np.zeros((2, 2)),
            np.zeros((2, 2, 2, 2)),
        ),
        canonical.occupation,
        np.zeros(2),
    )
    with pytest.raises(ValueError, match="denominator"):
        mp2_initial_point(degenerate)


def test_mp2_rejects_open_shell_and_invalid_controls():
    canonical = _canonical_system()
    open_shell = ElectronicSystem(
        canonical.integrals,
        ElectronicOccupation(2, 1, 0),
        canonical.orbital_energies,
    )
    with pytest.raises(ValueError, match="closed-shell"):
        mp2_initial_point(open_shell)
    with pytest.raises(TypeError, match="prefix"):
        mp2_initial_point(canonical, prefix=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="threshold"):
        mp2_initial_point(canonical, threshold=True)
    with pytest.raises(ValueError, match="non-negative"):
        mp2_initial_point(canonical, threshold=-1.0)
    with pytest.raises(ValueError, match="finite"):
        mp2_initial_point(canonical, threshold=np.inf)


def test_initial_points_require_a_nontrivial_uccsd_space():
    system = ElectronicSystem(
        ElectronicIntegrals(0.0, np.zeros((1, 1)), np.zeros((1, 1, 1, 1))),
        ElectronicOccupation.closed_shell(1, 2),
        np.array([0.0]),
    )
    with pytest.raises(ValueError, match="at least one"):
        hartree_fock_initial_point(system)
    with pytest.raises(ValueError, match="at least one"):
        mp2_initial_point(system)
