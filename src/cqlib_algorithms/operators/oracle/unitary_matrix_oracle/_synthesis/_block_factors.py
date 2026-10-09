"""Block factorization helpers for the Block-ZXZ decomposition."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._common import (
    assert_square_unitary,
    is_unitary,
    num_qubits_from_matrix,
    polar_from_svd,
    safe_project_subunitary,
)


def split_unitary_blocks(
    unitary: npt.NDArray[np.complex128],
) -> tuple[
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
]:
    """Split ``unitary`` into four equally sized blocks."""

    half = unitary.shape[0] // 2
    return (
        np.asarray(unitary[:half, :half], dtype=np.complex128),
        np.asarray(unitary[:half, half:], dtype=np.complex128),
        np.asarray(unitary[half:, :half], dtype=np.complex128),
        np.asarray(unitary[half:, half:], dtype=np.complex128),
    )


def compute_c_from_polar(
    u_x: npt.NDArray[np.complex128],
    u_y: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """Return the Block-ZXZ factor ``C``."""

    return np.asarray(-1j * u_x.conj().T @ u_y, dtype=np.complex128)


def compute_a1(
    s_x: npt.NDArray[np.complex128],
    s_y: npt.NDArray[np.complex128],
    u_x: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """Return the Block-ZXZ factor ``A1``."""

    return np.asarray((s_x + 1j * s_y) @ u_x, dtype=np.complex128)


def compute_a2(
    u21: npt.NDArray[np.complex128],
    u22: npt.NDArray[np.complex128],
    u_y: npt.NDArray[np.complex128],
    u_x: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """Return the Block-ZXZ factor ``A2``."""

    return np.asarray(
        u21 + u22 @ (1j * u_y.conj().T @ u_x),
        dtype=np.complex128,
    )


def compute_b(
    a1: npt.NDArray[np.complex128],
    x_block: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """Return the Block-ZXZ factor ``B``."""

    identity = np.eye(x_block.shape[0], dtype=np.complex128)
    return np.asarray(2.0 * a1.conj().T @ x_block - identity, dtype=np.complex128)


def factor_block_zxz(
    unitary: npt.NDArray[np.complex128],
    *,
    atol: float,
) -> tuple[
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
]:
    """Return the ``A1, A2, B, C`` factors of the Block-ZXZ split."""

    assert_square_unitary(unitary, atol=atol)
    if num_qubits_from_matrix(unitary) < 2:
        raise ValueError("Block-ZXZ factorization expects at least a 2-qubit unitary")

    x_block, y_block, u21, u22 = split_unitary_blocks(unitary)
    s_x, u_x = polar_from_svd(x_block)
    s_y, u_y = polar_from_svd(y_block)

    a1_factor = compute_a1(s_x, s_y, u_x)
    a2_factor = compute_a2(u21, u22, u_y, u_x)
    b_factor = compute_b(a1_factor, x_block)
    c_factor = compute_c_from_polar(u_x, u_y)

    a1_factor = safe_project_subunitary(a1_factor, atol=atol, label="factor A1")
    a2_factor = safe_project_subunitary(a2_factor, atol=atol, label="factor A2")
    b_factor = safe_project_subunitary(b_factor, atol=atol, label="factor B")
    c_factor = safe_project_subunitary(c_factor, atol=atol, label="factor C")

    for name, matrix in (
        ("A1", a1_factor),
        ("A2", a2_factor),
        ("B", b_factor),
        ("C", c_factor),
    ):
        if not is_unitary(matrix, atol=atol):
            raise ValueError(f"{name} is not unitary within the requested tolerance")

    return a1_factor, a2_factor, b_factor, c_factor
