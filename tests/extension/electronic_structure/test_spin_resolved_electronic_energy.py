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

"""Independent Fock-space checks for unrestricted electronic integrals."""

import numpy as np
import pytest

from cqlib_algorithms.extension.electronic_structure import (
    ElectronicEnergy,
    ElectronicIntegrals,
    SpinResolvedElectronicIntegrals,
)


def _apply_ladder(state: int, mode: int, *, create: bool) -> tuple[int, float] | None:
    occupied = bool(state & (1 << mode))
    if occupied == create:
        return None
    sign = -1.0 if (state & ((1 << mode) - 1)).bit_count() % 2 else 1.0
    output = state | (1 << mode) if create else state & ~(1 << mode)
    return output, sign


def _apply_product(
    state: int,
    operations: tuple[tuple[bool, int], ...],
) -> tuple[int, float] | None:
    output = state
    amplitude = 1.0
    for create, mode in reversed(operations):
        applied = _apply_ladder(output, mode, create=create)
        if applied is None:
            return None
        output, sign = applied
        amplitude *= sign
    return output, amplitude


def _add_term(
    matrix: np.ndarray,
    operations: tuple[tuple[bool, int], ...],
    coefficient: float,
) -> None:
    for basis in range(matrix.shape[0]):
        applied = _apply_product(basis, operations)
        if applied is not None:
            output, sign = applied
            matrix[output, basis] += coefficient * sign


def _independent_fock_matrix(
    integrals: SpinResolvedElectronicIntegrals,
) -> np.ndarray:
    n = integrals.num_spatial_orbitals
    matrix = np.eye(1 << (2 * n), dtype=np.complex128) * integrals.constant
    for p in range(n):
        for q in range(n):
            _add_term(
                matrix,
                ((True, 2 * p), (False, 2 * q)),
                float(integrals.one_body_alpha[p, q]),
            )
            _add_term(
                matrix,
                ((True, 2 * p + 1), (False, 2 * q + 1)),
                float(integrals.one_body_beta[p, q]),
            )
            for r in range(n):
                for s in range(n):
                    _add_term(
                        matrix,
                        (
                            (True, 2 * p),
                            (True, 2 * r),
                            (False, 2 * s),
                            (False, 2 * q),
                        ),
                        0.5 * float(integrals.two_body_alpha_alpha[p, q, r, s]),
                    )
                    _add_term(
                        matrix,
                        (
                            (True, 2 * p + 1),
                            (True, 2 * r + 1),
                            (False, 2 * s + 1),
                            (False, 2 * q + 1),
                        ),
                        0.5 * float(integrals.two_body_beta_beta[p, q, r, s]),
                    )
                    _add_term(
                        matrix,
                        (
                            (True, 2 * p),
                            (True, 2 * r + 1),
                            (False, 2 * s + 1),
                            (False, 2 * q),
                        ),
                        float(integrals.two_body_alpha_beta[p, q, r, s]),
                    )
    return matrix


def _same_spin_tensor(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n,) * 4)
    return (
        sum(
            (
                raw,
                raw.transpose(1, 0, 2, 3),
                raw.transpose(0, 1, 3, 2),
                raw.transpose(1, 0, 3, 2),
                raw.transpose(2, 3, 0, 1),
                raw.transpose(3, 2, 0, 1),
                raw.transpose(2, 3, 1, 0),
                raw.transpose(3, 2, 1, 0),
            )
        )
        / 8.0
    )


def _mixed_tensor(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n,) * 4)
    return (
        raw
        + raw.transpose(1, 0, 2, 3)
        + raw.transpose(0, 1, 3, 2)
        + raw.transpose(1, 0, 3, 2)
    ) / 4.0


def _integrals(n: int = 2) -> SpinResolvedElectronicIntegrals:
    rng = np.random.default_rng(311)
    one_alpha = rng.normal(size=(n, n))
    one_beta = rng.normal(size=(n, n))
    return SpinResolvedElectronicIntegrals(
        0.23,
        (one_alpha + one_alpha.T) / 2.0,
        (one_beta + one_beta.T) / 2.0,
        _same_spin_tensor(312, n),
        _same_spin_tensor(313, n),
        _mixed_tensor(314, n),
    )


@pytest.mark.reference
def test_one_orbital_spectrum_has_full_alpha_beta_interaction():
    integrals = SpinResolvedElectronicIntegrals(
        0.4,
        [[-1.2]],
        [[-0.8]],
        np.zeros((1,) * 4),
        np.zeros((1,) * 4),
        [[[[0.7]]]],
    )

    actual = ElectronicEnergy(integrals).to_qubit_hamiltonian().to_matrix()

    np.testing.assert_allclose(
        actual,
        np.diag([0.4, -0.8, -0.4, -0.9]),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.reference
def test_mapping_matches_independent_spin_resolved_fock_construction():
    integrals = _integrals()
    expected = _independent_fock_matrix(integrals)
    actual = ElectronicEnergy(integrals).to_qubit_hamiltonian().to_matrix()

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(actual, actual.conj().T, rtol=0.0, atol=1e-12)


@pytest.mark.reference
def test_equal_spin_blocks_reduce_exactly_to_restricted_hamiltonian():
    rng = np.random.default_rng(315)
    one_body = rng.normal(size=(2, 2))
    one_body = (one_body + one_body.T) / 2.0
    two_body = _same_spin_tensor(316, 2)
    restricted = ElectronicIntegrals(0.17, one_body, two_body)
    spin_resolved = SpinResolvedElectronicIntegrals(
        0.17,
        one_body,
        one_body,
        two_body,
        two_body,
        two_body,
    )

    expected = ElectronicEnergy(restricted).to_qubit_hamiltonian().to_matrix()
    actual = ElectronicEnergy(spin_resolved).to_qubit_hamiltonian().to_matrix()

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_integrals_are_defensive_immutable_snapshots():
    integrals = _integrals()
    assert not integrals.one_body_alpha.flags.writeable
    assert not integrals.two_body_alpha_beta.flags.writeable
    with pytest.raises(ValueError):
        integrals.one_body_alpha[0, 0] = 3.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("one_body_beta", np.eye(3), "equal shape"),
        ("two_body_alpha_alpha", np.zeros((2, 2, 2, 3)), "shape"),
        (
            "two_body_beta_beta",
            np.arange(16, dtype=float).reshape((2,) * 4),
            "symmetries",
        ),
        (
            "two_body_alpha_beta",
            np.arange(16, dtype=float).reshape((2,) * 4),
            "must satisfy",
        ),
    ],
)
def test_integrals_reject_shape_and_symmetry_errors(field, value, message):
    valid = _integrals()
    arguments = {
        "constant": valid.constant,
        "one_body_alpha": valid.one_body_alpha,
        "one_body_beta": valid.one_body_beta,
        "two_body_alpha_alpha": valid.two_body_alpha_alpha,
        "two_body_beta_beta": valid.two_body_beta_beta,
        "two_body_alpha_beta": valid.two_body_alpha_beta,
    }
    arguments[field] = value
    with pytest.raises(ValueError, match=message):
        SpinResolvedElectronicIntegrals(**arguments)


def test_integrals_reject_complex_and_nonfinite_values():
    valid = _integrals()
    with pytest.raises(TypeError, match="real"):
        SpinResolvedElectronicIntegrals(
            0.0,
            np.eye(2, dtype=np.complex128) * 1j,
            valid.one_body_beta,
            valid.two_body_alpha_alpha,
            valid.two_body_beta_beta,
            valid.two_body_alpha_beta,
        )
    with pytest.raises(ValueError, match="finite"):
        SpinResolvedElectronicIntegrals(
            0.0,
            np.eye(2),
            np.eye(2),
            valid.two_body_alpha_alpha,
            valid.two_body_beta_beta,
            np.full((2,) * 4, np.nan),
        )
