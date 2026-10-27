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

"""Spin-resolved molecular-orbital electronic integral tensors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..validation.numerical import immutable_real_array, validate_finite_real

_SYMMETRY_ATOL = 1e-10


def _snapshot_one_body(
    value: NDArray[np.float64],
    *,
    name: str,
) -> NDArray[np.float64]:
    result = immutable_real_array(value, name=name)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be a square rank-2 array")
    if result.shape[0] < 1:
        raise ValueError(f"{name} must describe at least one orbital")
    if not np.allclose(result, result.T, rtol=0.0, atol=_SYMMETRY_ATOL):
        raise ValueError(f"{name} must be symmetric")
    return result


def _snapshot_same_spin_two_body(
    value: NDArray[np.float64],
    *,
    name: str,
    num_orbitals: int,
) -> NDArray[np.float64]:
    result = immutable_real_array(value, name=name)
    if result.shape != (num_orbitals,) * 4:
        raise ValueError(f"{name} shape must be (n, n, n, n)")
    permutations = (
        result.transpose(1, 0, 2, 3),
        result.transpose(0, 1, 3, 2),
        result.transpose(2, 3, 0, 1),
    )
    if any(
        not np.allclose(result, permutation, rtol=0.0, atol=_SYMMETRY_ATOL)
        for permutation in permutations
    ):
        raise ValueError(
            f"{name} must satisfy the real two-electron integral symmetries"
        )
    return result


def _snapshot_alpha_beta_two_body(
    value: NDArray[np.float64],
    *,
    num_orbitals: int,
) -> NDArray[np.float64]:
    name = "two_body_alpha_beta"
    result = immutable_real_array(value, name=name)
    if result.shape != (num_orbitals,) * 4:
        raise ValueError(f"{name} shape must be (n, n, n, n)")
    if not np.allclose(
        result,
        result.transpose(1, 0, 2, 3),
        rtol=0.0,
        atol=_SYMMETRY_ATOL,
    ):
        raise ValueError(f"{name} must satisfy (pq|rs) = (qp|rs)")
    if not np.allclose(
        result,
        result.transpose(0, 1, 3, 2),
        rtol=0.0,
        atol=_SYMMETRY_ATOL,
    ):
        raise ValueError(f"{name} must satisfy (pq|rs) = (pq|sr)")
    return result


@dataclass(frozen=True, slots=True, eq=False)
class SpinResolvedElectronicIntegrals:
    """Real unrestricted-spin integrals in chemists' index convention.

    Alpha and beta orbitals are separately orthonormal. The mixed tensor
    stores ``(p_alpha q_alpha | r_beta s_beta)``; its pair-exchanged tensor is
    implied and is not duplicated. Arrays are defensive immutable snapshots.

    Args:
        constant: Scalar energy offset.
        one_body_alpha: Alpha-spin one-electron integrals.
        one_body_beta: Beta-spin one-electron integrals.
        two_body_alpha_alpha: Alpha-alpha two-electron integrals.
        two_body_beta_beta: Beta-beta two-electron integrals.
        two_body_alpha_beta: Alpha-beta two-electron integrals.

    Raises:
        TypeError: If an input cannot be interpreted as the required type.
        ValueError: If values are non-finite or tensor shapes and symmetries
            are invalid.
    """

    constant: float
    one_body_alpha: NDArray[np.float64]
    one_body_beta: NDArray[np.float64]
    two_body_alpha_alpha: NDArray[np.float64]
    two_body_beta_beta: NDArray[np.float64]
    two_body_alpha_beta: NDArray[np.float64]

    def __post_init__(self) -> None:
        validate_finite_real(self.constant, name="constant")
        one_alpha = _snapshot_one_body(
            self.one_body_alpha,
            name="one_body_alpha",
        )
        one_beta = _snapshot_one_body(
            self.one_body_beta,
            name="one_body_beta",
        )
        if one_alpha.shape != one_beta.shape:
            raise ValueError("alpha and beta one-body integrals must have equal shape")
        num_orbitals = int(one_alpha.shape[0])
        object.__setattr__(self, "constant", float(self.constant))
        object.__setattr__(self, "one_body_alpha", one_alpha)
        object.__setattr__(self, "one_body_beta", one_beta)
        object.__setattr__(
            self,
            "two_body_alpha_alpha",
            _snapshot_same_spin_two_body(
                self.two_body_alpha_alpha,
                name="two_body_alpha_alpha",
                num_orbitals=num_orbitals,
            ),
        )
        object.__setattr__(
            self,
            "two_body_beta_beta",
            _snapshot_same_spin_two_body(
                self.two_body_beta_beta,
                name="two_body_beta_beta",
                num_orbitals=num_orbitals,
            ),
        )
        object.__setattr__(
            self,
            "two_body_alpha_beta",
            _snapshot_alpha_beta_two_body(
                self.two_body_alpha_beta,
                num_orbitals=num_orbitals,
            ),
        )

    @property
    def num_spatial_orbitals(self) -> int:
        """Return the number of orbitals in each spin sector."""
        return int(self.one_body_alpha.shape[0])


__all__ = ["SpinResolvedElectronicIntegrals"]
