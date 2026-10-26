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

    Parameters
    ----------
    matrix:
        The intermediate subunitary produced by the recursion.
    atol:
        The per-call absolute tolerance propagated from the synthesis call.
        This value is used for the final post-projection unitary check while
        the structural drift check uses the module-level threshold.
    label:
        Human-readable name used in error messages.

    Returns
    -------
    npt.NDArray[np.complex128]
        ``matrix`` if it is already unitary within ``atol``, otherwise the
        nearest unitary matrix (polar factor) when the drift is acceptable.

    Raises
    ------
    ValueError
        If ``matrix`` deviates from ``U(n)`` by more than
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
    """Return an orthonormal eigenbasis refined inside near-degenerate clusters."""

    _ = matrix
    order = np.argsort(np.angle(raw_eigenvalues))
    eigenvalues = np.asarray(raw_eigenvalues[order], dtype=np.complex128)
    eigenvectors = np.asarray(raw_eigenvectors[:, order], dtype=np.complex128)

    orthonormal_columns: list[npt.NDArray[np.complex128]] = []
    cluster_tolerance = max(atol * 10.0, 1e-10)
    start = 0
    while start < eigenvalues.size:
        stop = start + 1
        while (
            stop < eigenvalues.size
            and abs(eigenvalues[stop] - eigenvalues[start]) <= cluster_tolerance
        ):
            stop += 1

        cluster = eigenvectors[:, start:stop]
        q_factor, _ = np.linalg.qr(cluster)
        # Optional refinement kept disabled for now because it noticeably slows
        # down recursive Block-ZXZ synthesis on dilation-structured inputs.
        # projected_cluster = q_factor.conj().T @ matrix @ q_factor
        # _, local_vectors = np.linalg.eig(projected_cluster)
        # refined_cluster = q_factor @ local_vectors
        # refined_cluster, _ = np.linalg.qr(refined_cluster)
        # q_factor = refined_cluster
        for column in range(q_factor.shape[1]):
            orthonormal_columns.append(
                np.asarray(q_factor[:, column], dtype=np.complex128)
            )
        start = stop

    return np.column_stack(orthonormal_columns).astype(np.complex128)


def num_qubits_from_matrix(
    matrix: npt.NDArray[np.complex128],
) -> int:
    """Return the qubit width encoded by a square power-of-two matrix."""

    return int(math.log2(matrix.shape[0]))
