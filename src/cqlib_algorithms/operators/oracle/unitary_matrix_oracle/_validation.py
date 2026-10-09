"""Validation helpers for unitary-matrix oracle synthesis."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
import numpy.typing as npt


def validate_atol(value: float) -> float:
    """Return a validated non-negative absolute tolerance."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("atol must be a real finite scalar")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise TypeError("atol must be a real finite scalar")
    if scalar < 0.0:
        raise ValueError("atol must be greater than or equal to zero")
    return scalar


def align_global_phase(
    reference: npt.NDArray[np.complex128],
    candidate: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """Return ``candidate`` aligned to ``reference`` up to global phase."""

    overlap = np.vdot(reference.reshape(-1), candidate.reshape(-1))
    if abs(overlap) == 0.0:
        return np.asarray(candidate.copy(), dtype=np.complex128)
    return np.asarray(
        candidate * np.exp(-1j * np.angle(overlap)),
        dtype=np.complex128,
    )


def allclose_up_to_global_phase(
    reference: npt.NDArray[np.complex128],
    candidate: npt.NDArray[np.complex128],
    *,
    atol: float,
) -> bool:
    """Return whether two matrices agree up to global phase."""

    aligned = align_global_phase(reference, candidate)
    return bool(np.allclose(reference, aligned, atol=atol, rtol=0.0))


def reconstruction_error_norm(
    reference: npt.NDArray[np.complex128],
    candidate: npt.NDArray[np.complex128],
) -> float:
    """Return the Frobenius norm after global-phase alignment."""

    aligned = align_global_phase(reference, candidate)
    return float(np.linalg.norm(aligned - reference))
