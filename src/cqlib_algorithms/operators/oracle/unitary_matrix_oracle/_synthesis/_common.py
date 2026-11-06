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

"""Numerical helpers shared by Block-ZXZ synthesis steps."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt


def is_power_of_two(value: int) -> bool:
    """Return whether ``value`` is a positive power of two."""

    return value > 0 and (value & (value - 1)) == 0


def is_unitary(
    matrix: npt.NDArray[np.complex128],
    *,
    atol: float,
) -> bool:
    """Return whether ``matrix`` is unitary within ``atol``."""

    identity = np.eye(matrix.shape[0], dtype=np.complex128)
    return bool(
        np.allclose(matrix.conj().T @ matrix, identity, atol=atol, rtol=0.0)
    )


def is_diagonal(
    matrix: npt.NDArray[np.complex128],
    *,
    atol: float,
) -> bool:
    """Return whether ``matrix`` is diagonal within ``atol``."""

    diagonal = np.diag(np.diag(matrix))
    return bool(np.allclose(matrix, diagonal, atol=atol, rtol=0.0))


def is_hermitian_for_demux(
    matrix: npt.NDArray[np.complex128],
    *,
    atol: float,
) -> bool:
    """Return whether ``matrix`` should use the Hermitian demux path."""

    return bool(
        np.allclose(
            matrix,
            matrix.conj().T,
            atol=max(atol, 1e-8),
            rtol=1e-5,
        )
    )


def assert_square_unitary(
    matrix: npt.NDArray[np.complex128],
    *,
    atol: float,
) -> None:
    """Validate that ``matrix`` is a square power-of-two unitary."""

    if matrix.ndim != 2:
        raise ValueError("unitary must be a two-dimensional matrix")
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("unitary must be square")
    if not is_power_of_two(rows):
        raise ValueError("unitary dimension must be a power of two")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("unitary must contain only finite entries")
    if not is_unitary(matrix, atol=atol):
        raise ValueError("unitary must be unitary within the requested tolerance")


def polar_from_svd(
    matrix: npt.NDArray[np.complex128],
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Return the polar factors ``S`` and ``U`` of ``matrix``."""

    left, singular_values, right_h = np.linalg.svd(matrix, full_matrices=False)
    sigma = np.diag(singular_values.astype(np.complex128))
    s_factor = left @ sigma @ left.conj().T
    u_factor = left @ right_h
    return (
        np.asarray(s_factor, dtype=np.complex128),
        np.asarray(u_factor, dtype=np.complex128),
    )


def project_to_unitary(
    matrix: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """Return the nearest unitary factor from the polar decomposition."""

    _, unitary_factor = polar_from_svd(matrix)
    return np.asarray(unitary_factor, dtype=np.complex128)


_SUBUNITARY_PROJECTION_THRESHOLD: float = 1e-6
_DEMUX_DIAGONALIZATION_ATOL_FLOOR: float = 1e-7


def safe_project_subunitary(
    matrix: npt.NDArray[np.complex128],
    *,
    atol: float,
    label: str,
) -> npt.NDArray[np.complex128]:
    """Project an intermediate subunitary back to U(n) if the drift is small.

    This helper is only intended for numerically drifting intermediate factors
    produced by the classical Block-ZXZ recursion.  If the distance to the
    nearest unitary is below ``SUBUNITARY_PROJECTION_THRESHOLD`` the polar
    factor is returned; otherwise a ``ValueError`` is raised because the
    factor is seriously non-unitary and indicates a logic/runtime bug.  The
    original ``assert_square_unitary`` entry validation is intentionally left
    strict so that malformed *user-provided* matrices are always rejected.

    Args:
        matrix:: The intermediate subunitary produced by the recursion.
        atol:: The per-call absolute tolerance propagated from the synthesis call.
            This value is used for the final post-projection unitary check while
            the structural drift check uses the module-level threshold.
        label:: Human-readable name used in error messages.

    Returns:
        ``matrix`` if it is already unitary within ``atol``, otherwise the
            nearest unitary matrix (polar factor) when the drift is acceptable.

    Raises:
        ValueError: If ``matrix`` deviates from ``U(n)`` by more than
            ``_SUBUNITARY_PROJECTION_THRESHOLD`` which rules out mere numerical
            round-off.

    """

    if is_unitary(matrix, atol=atol):
        return matrix

    deviation = float(
        np.linalg.norm(
            matrix.conj().T @ matrix
            - np.eye(matrix.shape[0], dtype=np.complex128),
            ord=2,
        )
    )
    if deviation > _SUBUNITARY_PROJECTION_THRESHOLD:
        raise ValueError(
            f"{label} is not unitary (||M†M - I||_2 = {deviation:.3e}); "
            "drift exceeds the structural projection threshold"
        )

    projected = project_to_unitary(matrix)
    if not is_unitary(projected, atol=atol):
        raise ValueError(
            f"{label} failed to project back to unitary within the tolerance"
        )
    return np.asarray(projected, dtype=np.complex128)


def build_unitary_diagonal_sqrt(
    eigenvalues: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """Return a diagonal unitary square root for unit-modulus eigenvalues."""

    normalized = np.divide(
        eigenvalues,
        np.abs(eigenvalues),
        out=np.ones_like(eigenvalues, dtype=np.complex128),
        where=np.abs(eigenvalues) > 0.0,
    )
    phases = np.exp(0.5j * np.angle(normalized))
    return np.diag(phases.astype(np.complex128))


def unitary_eigendecomposition(
    matrix: npt.NDArray[np.complex128],
    *,
    atol: float,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Diagonalize a unitary matrix with a numerically unitary eigenbasis."""

    assert_square_unitary(matrix, atol=atol)
    if is_hermitian_for_demux(matrix, atol=atol):
        raw_eigenvalues, raw_eigenvectors = np.linalg.eigh(matrix)
    else:
        raw_eigenvalues, raw_eigenvectors = np.linalg.eig(matrix)

    unitary_vectors = refine_degenerate_eigenbasis(
        matrix,
        np.asarray(raw_eigenvalues, dtype=np.complex128),
        np.asarray(raw_eigenvectors, dtype=np.complex128),
        atol=atol,
    )
    diagonal_matrix = (
        unitary_vectors.conj().T @ matrix @ unitary_vectors
    ).astype(np.complex128)
    if not is_diagonal(
        diagonal_matrix,
        atol=max(atol, _DEMUX_DIAGONALIZATION_ATOL_FLOOR),
    ):
        raise ValueError("failed to diagonalize unitary demux factor")

    diagonal = np.diag(diagonal_matrix)
    normalized = np.divide(
        diagonal,
        np.abs(diagonal),
        out=np.ones_like(diagonal, dtype=np.complex128),
        where=np.abs(diagonal) > 0.0,
    )
    return unitary_vectors, normalized.astype(np.complex128)


def refine_degenerate_eigenbasis(
    matrix: npt.NDArray[np.complex128],
    raw_eigenvalues: npt.NDArray[np.complex128],
    raw_eigenvectors: npt.NDArray[np.complex128],
    *,
    atol: float,
) -> npt.NDArray[np.complex128]:
    r"""Return an orthonormal eigenbasis of a normal matrix.

    For a normal ``A`` with eigenpair decomposition ``A V = V Lambda``, the
    thin QR factorisation ``V = Q R`` gives ``Q^H A Q = R Lambda R^-1``.
    That product is upper triangular, and a unitary similarity preserves
    normality, so ``Q^H A Q`` is a normal triangular matrix and is therefore
    diagonal.  Hence the orthonormal factor ``Q`` is an exact eigenbasis and
    column ``i`` carries the eigenvalue ``Lambda[i]``.

    Applying one QR factorisation to the whole ordered basis is what makes
    this exact: an eigenvector matrix that is merely close to orthogonal still
    yields ``R = I + E`` with small ``E``, and the ``E Lambda E^-1`` part of
    ``R Lambda R^-1`` cancels to the rounding floor.  Clustering the basis by
    eigenvalue gap before factorising instead leaves the residual at the
    orthogonality defect of the raw basis, which was the failure mode for
    tightly grouped eigenvalues.

    Args:
        matrix: Matrix whose eigenvectors were computed, retained for a
            signature-stable call site.
        raw_eigenvalues: Eigenvalues returned by the dense eigensolver.
        raw_eigenvectors: Eigenvectors returned by the dense eigensolver.
        atol: Diagonalisation tolerance of the caller, retained for a
            signature-stable call site.

    Returns:
        Complex orthonormal eigenbasis ordered by increasing eigenphase.
    """

    del matrix, atol
    order = np.argsort(np.angle(raw_eigenvalues))
    ordered_basis = np.asarray(raw_eigenvectors[:, order], dtype=np.complex128)
    orthonormal_basis, _ = np.linalg.qr(ordered_basis)
    return np.asarray(orthonormal_basis, dtype=np.complex128)


def num_qubits_from_matrix(
    matrix: npt.NDArray[np.complex128],
) -> int:
    """Return the qubit width encoded by a square power-of-two matrix."""

    return int(math.log2(matrix.shape[0]))
