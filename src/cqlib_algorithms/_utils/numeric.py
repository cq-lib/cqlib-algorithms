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

"""Private numerical validation and snapshot helpers."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


def validate_optional_seed(seed: int | None) -> None:
    """Validate a non-negative optional random seed."""
    if seed is None:
        return
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer or None")
    if seed < 0:
        raise ValueError("seed must be non-negative")


def validate_positive_integer(value: int, name: str) -> None:
    """Validate a strictly positive integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def validate_optional_positive_integer(value: int | None, name: str) -> None:
    """Validate an optional strictly positive integer."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer or None")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def normalize_positive_integer(value: object, name: str) -> int:
    """Validate and return a strictly positive integer in normalized form."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def validate_nonnegative_integer(value: int, *, name: str) -> None:
    """Validate a non-negative integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def validate_positive_qubit_count(num_qubits: int) -> None:
    """Validate a strictly positive qubit count."""
    validate_positive_integer(num_qubits, "num_qubits")


def validate_repetitions(reps: int) -> None:
    """Validate a non-negative repetition count."""
    validate_nonnegative_integer(reps, name="reps")


def normalize_finite_real(value: float, name: str) -> float:
    """Validate and return a finite real value in normalized form."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def normalize_positive_real(value: float, name: str) -> float:
    """Validate and return a finite positive real value in normalized form."""
    normalized = normalize_finite_real(value, name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def normalize_nonnegative_real(value: float, name: str) -> float:
    """Validate and return a finite non-negative real value in normalized form."""
    normalized = normalize_finite_real(value, name)
    if normalized < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def copy_finite_real_array(
    values: ArrayLike,
    name: str,
    ndim: int,
) -> NDArray[np.float64]:
    """Return a defensive finite real array snapshot."""
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular real numeric array") from error
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values, excluding bool")
    if raw.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(raw, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result


__all__: list[str] = []
