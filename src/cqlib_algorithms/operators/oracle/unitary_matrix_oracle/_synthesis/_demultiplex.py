"""Demultiplexing helpers for the Block-ZXZ decomposition."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._common import (
    build_unitary_diagonal_sqrt,
    is_diagonal,
    is_unitary,
    safe_project_subunitary,
    unitary_eigendecomposition,
)

_DEMUX_RECONSTRUCTION_ATOL_FLOOR: float = 1e-7


def _coerce_demux_reconstruction_tolerance(atol: float) -> float:
    return max(float(atol), _DEMUX_RECONSTRUCTION_ATOL_FLOOR)


def demultiplex_pair(
    first_unitary: npt.NDArray[np.complex128],
    second_unitary: npt.NDArray[np.complex128],
    *,
    atol: float,
) -> tuple[
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
]:
    """Return ``V, D, W`` such that ``U0 = VDW`` and ``U1 = VD†W``."""

    first_unitary = safe_project_subunitary(
        first_unitary,
        atol=atol,
        label="demultiplex first input",
    )
    second_unitary = safe_project_subunitary(
        second_unitary,
        atol=atol,
        label="demultiplex second input",
    )
    if first_unitary.shape != second_unitary.shape:
        raise ValueError("demultiplexed unitaries must have the same shape")

    product = np.asarray(
        first_unitary @ second_unitary.conj().T,
        dtype=np.complex128,
    )
    product = safe_project_subunitary(
        product,
        atol=atol,
        label="demultiplex eig product",
    )
    left_factor, eigenvalues = unitary_eigendecomposition(product, atol=atol)
    diagonal_factor = build_unitary_diagonal_sqrt(eigenvalues)
    right_factor = np.asarray(
        diagonal_factor @ left_factor.conj().T @ second_unitary,
        dtype=np.complex128,
    )

    left_factor = safe_project_subunitary(
        left_factor,
        atol=atol,
        label="demultiplex factor V",
    )
    right_factor = safe_project_subunitary(
        right_factor,
        atol=atol,
        label="demultiplex factor W",
    )

    if not is_unitary(left_factor, atol=atol):
        raise ValueError("demultiplex factor V is not unitary")
    if not is_diagonal(diagonal_factor, atol=atol):
        raise ValueError("demultiplex factor D is not diagonal")
    if not is_unitary(diagonal_factor, atol=atol):
        raise ValueError("demultiplex factor D is not unitary")
    if not is_unitary(right_factor, atol=atol):
        raise ValueError("demultiplex factor W is not unitary")

    reconstruction_atol = _coerce_demux_reconstruction_tolerance(atol)

    reconstructed_first = np.asarray(
        left_factor @ diagonal_factor @ right_factor,
        dtype=np.complex128,
    )
    reconstructed_second = np.asarray(
        left_factor @ diagonal_factor.conj().T @ right_factor,
        dtype=np.complex128,
    )
    if not np.allclose(
        reconstructed_first,
        first_unitary,
        atol=reconstruction_atol,
        rtol=0.0,
    ):
        raise ValueError("demultiplex reconstruction of the first unitary failed")
    if not np.allclose(
        reconstructed_second,
        second_unitary,
        atol=reconstruction_atol,
        rtol=0.0,
    ):
        raise ValueError("demultiplex reconstruction of the second unitary failed")

    return left_factor, diagonal_factor, right_factor


def demultiplex_block_zxz(
    a1_factor: npt.NDArray[np.complex128],
    a2_factor: npt.NDArray[np.complex128],
    b_factor: npt.NDArray[np.complex128],
    c_factor: npt.NDArray[np.complex128],
    *,
    atol: float,
) -> dict[str, tuple[npt.NDArray[np.complex128], ...]]:
    """Return the demultiplexed factors of a Block-ZXZ split."""

    a_branch = demultiplex_pair(a1_factor, a2_factor, atol=atol)
    identity = np.eye(c_factor.shape[0], dtype=np.complex128)
    c_branch = demultiplex_pair(identity, c_factor, atol=atol)
    middle_first = np.asarray(a_branch[2] @ c_branch[0], dtype=np.complex128)
    middle_second = np.asarray(
        a_branch[2] @ b_factor @ c_branch[0],
        dtype=np.complex128,
    )
    middle_branch = demultiplex_pair(middle_first, middle_second, atol=atol)

    return {
        "a_branch": a_branch,
        "c_branch": c_branch,
        "middle_pair": (middle_first, middle_second),
        "middle_branch": middle_branch,
    }
