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

"""General even quadratic fermionic Hamiltonians and stable diagonalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cqlib.qis import FermionicLadder, FermionicOp
from numpy.typing import NDArray
from scipy.linalg import schur  # type: ignore[import-untyped]

from .._numeric import immutable_complex_array, validate_finite_real
from ._bogoliubov_decomposition import majorana_to_bogoliubov

_STRUCTURE_ATOL = 1e-12
_SCHUR_ERROR_FACTOR = 64.0
_FLOAT_EPSILON = np.finfo(np.float64).eps


def _majorana_form(
    constant: float,
    one_body: NDArray[np.complex128],
    pairing: NDArray[np.complex128],
) -> tuple[float, NDArray[np.float64]]:
    """Return ``h0, A`` for ``H=h0+(i/4) gamma^T A gamma``."""
    num_modes = one_body.shape[0]
    # Each entry stores the coefficients of x_p and y_p in a†_p or a_p.
    creation = tuple({2 * mode: 0.5, 2 * mode + 1: -0.5j} for mode in range(num_modes))
    annihilation = tuple(
        {2 * mode: 0.5, 2 * mode + 1: 0.5j} for mode in range(num_modes)
    )
    scalar = complex(constant)
    products: dict[tuple[int, int], complex] = {}

    def accumulate(
        coefficient: complex,
        left: dict[int, complex],
        right: dict[int, complex],
    ) -> None:
        nonlocal scalar
        for first, left_coefficient in left.items():
            for second, right_coefficient in right.items():
                value = coefficient * left_coefficient * right_coefficient
                if first == second:
                    scalar += value
                elif first < second:
                    key = (first, second)
                    products[key] = products.get(key, 0.0j) + value
                else:
                    key = (second, first)
                    products[key] = products.get(key, 0.0j) - value

    for p in range(num_modes):
        for q in range(num_modes):
            accumulate(one_body[p, q], creation[p], annihilation[q])
            accumulate(0.5 * pairing[p, q], creation[p], creation[q])
            accumulate(
                0.5 * pairing[p, q].conjugate(),
                annihilation[q],
                annihilation[p],
            )

    scale = max(
        1.0,
        abs(scalar),
        float(np.max(np.abs(one_body))),
        float(np.max(np.abs(pairing))),
    )
    tolerance = _STRUCTURE_ATOL * scale
    if abs(scalar.imag) > tolerance:
        raise ValueError("quadratic Hamiltonian produced a non-real scalar residual")
    majorana = np.zeros((2 * num_modes, 2 * num_modes), dtype=np.float64)
    for (first, second), coefficient in products.items():
        matrix_value = -2j * coefficient
        if abs(matrix_value.imag) > tolerance:
            raise ValueError(
                "quadratic Hamiltonian produced a non-real Majorana matrix"
            )
        majorana[first, second] = matrix_value.real
        majorana[second, first] = -matrix_value.real
    return float(scalar.real), majorana


def _canonical_majorana_form(
    majorana: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return nonnegative energies and rows of a canonical Majorana map."""
    triangular, vectors = schur(majorana, output="real")
    triangular = np.asarray(triangular, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)
    width = majorana.shape[0]
    majorana_norm = float(np.linalg.norm(majorana, ord="fro"))
    schur_tolerance = _SCHUR_ERROR_FACTOR * width * _FLOAT_EPSILON * majorana_norm

    blocks: list[tuple[float, tuple[int, int]]] = []
    zero_indices: list[int] = []
    index = 0
    while index < width:
        # LAPACK's real Schur form marks a 2x2 block by a nonzero
        # subdiagonal.  This structural marker must not be replaced by a
        # threshold derived from the largest energy: physically independent
        # modes may have arbitrarily different, but still representable,
        # energy scales.
        coupled = index + 1 < width and triangular[index + 1, index] != 0.0
        if not coupled:
            zero_indices.append(index)
            index += 1
            continue
        upper = float(triangular[index, index + 1])
        lower = float(triangular[index + 1, index])
        if abs(upper + lower) > schur_tolerance:
            raise ValueError("real Schur form is not antisymmetric by 2x2 blocks")
        energy = 0.5 * abs(upper - lower)
        oriented = (index, index + 1) if upper >= 0.0 else (index + 1, index)
        blocks.append((energy, oriented))
        index += 2

    if len(zero_indices) % 2 != 0:
        raise ValueError("Majorana zero modes could not be paired canonically")
    blocks.extend(
        (0.0, (zero_indices[offset], zero_indices[offset + 1]))
        for offset in range(0, len(zero_indices), 2)
    )
    blocks.sort(key=lambda item: (item[0], item[1]))
    energies = np.array([item[0] for item in blocks], dtype=np.float64)
    order = [coordinate for _, block in blocks for coordinate in block]
    orthogonal = vectors[:, order].T

    canonical = np.zeros_like(majorana)
    for mode, energy in enumerate(energies):
        canonical[2 * mode, 2 * mode + 1] = energy
        canonical[2 * mode + 1, 2 * mode] = -energy
    reconstructed = orthogonal.T @ canonical @ orthogonal
    residual = float(np.linalg.norm(reconstructed - majorana, ord="fro"))
    matrix_norm = max(
        majorana_norm,
        float(np.linalg.norm(reconstructed, ord="fro")),
    )
    tolerance = _SCHUR_ERROR_FACTOR * width * _FLOAT_EPSILON * matrix_norm
    if residual > tolerance:
        raise ValueError(
            "quadratic Hamiltonian diagonalization failed reconstruction "
            f"(Frobenius residual {residual:.3e})"
        )
    if not np.allclose(
        orthogonal @ orthogonal.T,
        np.eye(width),
        rtol=0.0,
        atol=_SCHUR_ERROR_FACTOR * width * _FLOAT_EPSILON,
    ):
        raise ValueError("quadratic Hamiltonian diagonalization is not orthogonal")
    return energies, orthogonal


@dataclass(frozen=True, slots=True, eq=False)
class QuadraticFermionicHamiltonian:
    """A general even quadratic Hamiltonian on fermionic modes.

    The represented operator is

    ``c + sum M[p,q] a†_p a_q``
    ``+ 1/2 sum (Delta[p,q] a†_p a†_q + Delta*[p,q] a_q a_p)``,

    where ``M`` is Hermitian and ``Delta`` is antisymmetric. Absolute
    structure residuals through ``1e-12`` are canonicalized to these exact
    symmetries at construction; larger residuals are rejected.

    Args:
        constant: Real scalar energy offset.
        one_body: Hermitian number-conserving coefficient matrix ``M``.
        pairing: Antisymmetric pairing matrix ``Delta``.

    Raises:
        TypeError: If an input cannot be interpreted numerically.
        ValueError: If dimensions, finiteness, or algebraic constraints fail.
    """

    constant: float
    one_body: NDArray[np.complex128]
    pairing: NDArray[np.complex128]

    def __post_init__(self) -> None:
        validate_finite_real(self.constant, name="constant")
        object.__setattr__(self, "constant", float(self.constant))
        object.__setattr__(
            self,
            "one_body",
            immutable_complex_array(self.one_body, name="one_body"),
        )
        object.__setattr__(
            self,
            "pairing",
            immutable_complex_array(self.pairing, name="pairing"),
        )
        if self.one_body.ndim != 2 or self.one_body.shape[0] != self.one_body.shape[1]:
            raise ValueError("one_body must be a square rank-2 array")
        if self.one_body.shape[0] < 1:
            raise ValueError("one_body must describe at least one fermionic mode")
        if self.pairing.shape != self.one_body.shape:
            raise ValueError("pairing must have the same square shape as one_body")
        if not np.allclose(
            self.one_body,
            self.one_body.conj().T,
            rtol=0.0,
            atol=_STRUCTURE_ATOL,
        ):
            raise ValueError("one_body must be Hermitian")
        if not np.allclose(
            self.pairing,
            -self.pairing.T,
            rtol=0.0,
            atol=_STRUCTURE_ATOL,
        ):
            raise ValueError("pairing must be antisymmetric")
        # Inputs assembled from floating-point orbital transformations can
        # carry roundoff-sized violations of their exact algebraic structure.
        # Canonicalize accepted residuals once at the construction boundary so
        # every public conversion represents the same mathematical operator.
        object.__setattr__(
            self,
            "one_body",
            immutable_complex_array(
                0.5 * (self.one_body + self.one_body.conj().T),
                name="one_body",
            ),
        )
        object.__setattr__(
            self,
            "pairing",
            immutable_complex_array(
                0.5 * (self.pairing - self.pairing.T),
                name="pairing",
            ),
        )

    @property
    def num_modes(self) -> int:
        """Return the number of fermionic modes."""
        return int(self.one_body.shape[0])

    def to_fermionic_op(self) -> FermionicOp:
        """Return an independent Core fermionic-operator representation."""
        terms: list[tuple[tuple[FermionicLadder, ...], complex]] = []
        if self.constant != 0.0:
            terms.append(((), complex(self.constant)))
        for p in range(self.num_modes):
            for q in range(self.num_modes):
                normal = complex(self.one_body[p, q])
                if normal != 0.0j:
                    terms.append(
                        (
                            (
                                FermionicLadder.create(p),
                                FermionicLadder.annihilate(q),
                            ),
                            normal,
                        )
                    )
                anomalous = 0.5 * complex(self.pairing[p, q])
                if anomalous != 0.0j:
                    terms.append(
                        (
                            (
                                FermionicLadder.create(p),
                                FermionicLadder.create(q),
                            ),
                            anomalous,
                        )
                    )
                    terms.append(
                        (
                            (
                                FermionicLadder.annihilate(q),
                                FermionicLadder.annihilate(p),
                            ),
                            anomalous.conjugate(),
                        )
                    )
        return FermionicOp.from_terms(self.num_modes, terms)

    def diagonalize(
        self,
    ) -> tuple[NDArray[np.complex128], NDArray[np.float64], float]:
        """Return ``([C D], energies, diagonal_constant)``.

        Rows define ``b†_i = sum_p (C[i,p] a†_p + D[i,p] a_p)``. The result
        satisfies ``H = diagonal_constant + sum_i energies[i] b†_i b_i``
        with nonnegative energies. Zero-mode bases and degenerate-mode order
        are deterministic only up to physically equivalent canonical rotations.
        """
        majorana_constant, majorana = _majorana_form(
            self.constant,
            self.one_body,
            self.pairing,
        )
        energies, orthogonal = _canonical_majorana_form(majorana)
        transformation = majorana_to_bogoliubov(orthogonal)
        diagonal_constant = float(majorana_constant - 0.5 * np.sum(energies))
        frozen_energies = np.frombuffer(
            np.asarray(energies, dtype=np.float64).tobytes(), dtype=np.float64
        )
        return transformation, frozen_energies, diagonal_constant


__all__ = ["QuadraticFermionicHamiltonian"]
