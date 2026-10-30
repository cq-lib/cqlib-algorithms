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

"""Numerically guarded generalized eigensolver for electronic qEOM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eig  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True)
class _QEOMSolution:
    energies: NDArray[np.float64]
    amplitudes: NDArray[np.complex128]
    residuals: NDArray[np.float64]
    metric_cutoff: float
    retained_rank: int
    positive_inertia: int
    negative_inertia: int
    discarded_metric_eigenvalues: NDArray[np.float64]


def _norm(matrix: NDArray[np.complex128]) -> float:
    return float(np.linalg.norm(matrix, ord=2))


def _normalize_pencil_matrix(
    matrix: NDArray[np.complex128],
    *,
    name: str,
) -> NDArray[np.complex128]:
    """Validate and return a complex square-matrix view."""
    array = cast(
        NDArray[np.complex128],
        np.asarray(matrix, dtype=np.complex128),
    )
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] < 1:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _fix_phase(vector: NDArray[np.complex128]) -> NDArray[np.complex128]:
    pivot = int(np.argmax(np.abs(vector)))
    value = vector[pivot]
    if value == 0.0j:
        raise ValueError("qEOM eigenvector must not be zero")
    phased = cast(
        NDArray[np.complex128],
        vector * np.exp(-1j * np.angle(value)),
    )
    if phased[pivot].real < 0.0:
        phased = -phased
    return phased


def _solve_qeom_pencil(
    a_matrix: NDArray[np.complex128],
    b_matrix: NDArray[np.complex128],
    *,
    num_roots: int | None,
    atol: float,
    rtol: float,
) -> _QEOMSolution:
    """Solve a singular-aware Hermitian qEOM pencil and keep physical roots."""
    a = _normalize_pencil_matrix(a_matrix, name="A")
    b = _normalize_pencil_matrix(b_matrix, name="B")
    if a.shape != b.shape:
        raise ValueError("A and B must have the same shape")
    if not np.isfinite(atol) or atol <= 0.0:
        raise ValueError("atol must be finite and positive")
    if not np.isfinite(rtol) or rtol <= 0.0:
        raise ValueError("rtol must be finite and positive")
    if num_roots is not None:
        if isinstance(num_roots, bool) or not isinstance(num_roots, int):
            raise TypeError("num_roots must be an integer or None")
        if num_roots < 1:
            raise ValueError("num_roots must be positive")

    a_norm = _norm(a)
    b_norm = _norm(b)
    a_tolerance = max(atol, rtol * a_norm)
    b_tolerance = max(atol, rtol * b_norm)
    if _norm(a - a.conj().T) > a_tolerance:
        raise ValueError("qEOM A matrix must be Hermitian")
    if _norm(b - b.conj().T) > b_tolerance:
        raise ValueError("qEOM B matrix must be Hermitian")

    metric_values, metric_vectors = np.linalg.eigh(b)
    retained_mask = np.abs(metric_values) > b_tolerance
    positive_inertia = int(np.count_nonzero(metric_values > b_tolerance))
    negative_inertia = int(np.count_nonzero(metric_values < -b_tolerance))
    if positive_inertia == 0 or positive_inertia != negative_inertia:
        raise ValueError("qEOM metric must have balanced nonzero inertia")
    retained = metric_vectors[:, retained_mask]
    nullspace = metric_vectors[:, ~retained_mask]
    if nullspace.shape[1] > 0:
        coupling = _norm(retained.conj().T @ a @ nullspace)
        if coupling > a_tolerance:
            raise ValueError("qEOM metric nullspace is coupled by the A matrix")

    retained_values = metric_values[retained_mask]
    inverse_scale = np.diag(1.0 / np.sqrt(np.abs(retained_values)))
    congruence = retained @ inverse_scale
    reduced_a = congruence.conj().T @ a @ congruence
    signature = np.diag(np.sign(retained_values).astype(np.complex128))
    values, reduced_vectors = eig(reduced_a, signature)

    candidates: list[tuple[float, float, NDArray[np.complex128], float]] = []
    for value, reduced_vector in zip(values, reduced_vectors.T, strict=True):
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            continue
        reality_tolerance = max(atol, rtol * max(1.0, abs(value.real)))
        if abs(value.imag) > reality_tolerance:
            continue
        omega = float(value.real)
        vector = np.asarray(congruence @ reduced_vector, dtype=np.complex128)
        vector_norm = float(np.linalg.norm(vector))
        if vector_norm == 0.0 or not np.isfinite(vector_norm):
            continue
        residual_vector = a @ vector - omega * (b @ vector)
        denominator = max(
            atol,
            a_norm * vector_norm + abs(omega) * b_norm * vector_norm,
        )
        relative_residual = float(np.linalg.norm(residual_vector)) / denominator
        if relative_residual > rtol:
            continue
        metric_norm = float(np.vdot(vector, b @ vector).real)
        if not np.isfinite(metric_norm):
            continue
        candidates.append((omega, metric_norm, vector, relative_residual))

    physical: list[tuple[float, NDArray[np.complex128], float]] = []
    for omega, metric_norm, vector, relative_residual in candidates:
        root_tolerance = max(atol, rtol * max(1.0, abs(omega)))
        if omega <= root_tolerance or metric_norm <= b_tolerance:
            continue
        paired = any(
            negative_omega < -root_tolerance
            and abs(negative_omega + omega) <= root_tolerance
            for negative_omega, _, _, _ in candidates
        )
        if not paired:
            continue
        normalized = vector / np.sqrt(metric_norm)
        physical.append((omega, _fix_phase(normalized), relative_residual))

    physical.sort(key=lambda item: item[0])
    if not physical:
        raise ValueError("qEOM pencil has no validated positive physical roots")
    requested = len(physical) if num_roots is None else num_roots
    if requested > len(physical):
        raise ValueError(
            "num_roots exceeds the number of validated positive qEOM roots"
        )
    selected = physical[:requested]
    energies = np.array([item[0] for item in selected], dtype=np.float64)
    amplitudes = np.column_stack([item[1] for item in selected]).astype(
        np.complex128,
        copy=False,
    )
    residuals = np.array([item[2] for item in selected], dtype=np.float64)
    discarded_metric_eigenvalues = np.array(
        metric_values[~retained_mask],
        dtype=np.float64,
        copy=True,
    )
    for array in (
        energies,
        amplitudes,
        residuals,
        discarded_metric_eigenvalues,
    ):
        array.flags.writeable = False
    return _QEOMSolution(
        energies,
        amplitudes,
        residuals,
        b_tolerance,
        int(np.count_nonzero(retained_mask)),
        positive_inertia,
        negative_inertia,
        discarded_metric_eigenvalues,
    )


__all__: list[str] = []
