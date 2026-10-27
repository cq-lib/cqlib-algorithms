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

"""Molecular-orbital electronic integral tensors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..validation.numerical import immutable_real_array, validate_finite_real

_SYMMETRY_ATOL = 1e-10


@dataclass(frozen=True, slots=True, eq=False)
class ElectronicIntegrals:
    """Real spatial-orbital integrals in chemists' index convention.

    ``two_body[p, q, r, s]`` stores ``(pq|rs)``. The arrays are defensive,
    immutable float64 snapshots. Complex orbital coefficients are outside the
    scope of this data model.

    Args:
        constant: Scalar energy offset.
        one_body: Real symmetric one-electron integral matrix.
        two_body: Real two-electron tensor in chemists' index order.

    Raises:
        TypeError: If an input cannot be interpreted as the required type.
        ValueError: If values are non-finite or tensor shapes and symmetries
            are invalid.
    """

    constant: float
    one_body: NDArray[np.float64]
    two_body: NDArray[np.float64]

    def __post_init__(self) -> None:
        validate_finite_real(self.constant, name="constant")
        object.__setattr__(self, "constant", float(self.constant))
        object.__setattr__(
            self, "one_body", immutable_real_array(self.one_body, name="one_body")
        )
        object.__setattr__(
            self, "two_body", immutable_real_array(self.two_body, name="two_body")
        )
        if self.one_body.ndim != 2 or self.one_body.shape[0] != self.one_body.shape[1]:
            raise ValueError("one_body must be a square rank-2 array")
        num_orbitals = self.one_body.shape[0]
        if num_orbitals < 1:
            raise ValueError("one_body must describe at least one orbital")
        if self.two_body.shape != (num_orbitals,) * 4:
            raise ValueError("two_body shape must be (n, n, n, n)")
        if not np.allclose(
            self.one_body, self.one_body.T, rtol=0.0, atol=_SYMMETRY_ATOL
        ):
            raise ValueError("one_body must be symmetric")
        if not np.allclose(
            self.two_body,
            self.two_body.transpose(1, 0, 2, 3),
            rtol=0.0,
            atol=_SYMMETRY_ATOL,
        ):
            raise ValueError("two_body must satisfy (pq|rs) = (qp|rs)")
        if not np.allclose(
            self.two_body,
            self.two_body.transpose(0, 1, 3, 2),
            rtol=0.0,
            atol=_SYMMETRY_ATOL,
        ):
            raise ValueError("two_body must satisfy (pq|rs) = (pq|sr)")
        if not np.allclose(
            self.two_body,
            self.two_body.transpose(2, 3, 0, 1),
            rtol=0.0,
            atol=_SYMMETRY_ATOL,
        ):
            raise ValueError("two_body must satisfy (pq|rs) = (rs|pq)")

    @property
    def num_spatial_orbitals(self) -> int:
        """Return the number of spatial orbitals."""
        return int(self.one_body.shape[0])


__all__ = ["ElectronicIntegrals"]
