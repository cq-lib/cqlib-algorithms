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

"""Numerical validation helpers for immutable electronic-structure data."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def immutable_complex_array(
    value: ArrayLike,
    *,
    name: str,
) -> NDArray[np.complex128]:
    """Return an immutable complex128 snapshot of a finite numeric array."""
    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a rectangular complex array") from exc
    try:
        object_values = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a rectangular complex array") from exc
    if any(
        isinstance(item, (bool, np.bool_, str, bytes)) for item in object_values.flat
    ):
        raise TypeError(f"{name} must contain numeric values without booleans")
    try:
        copied = np.array(source, dtype=np.complex128, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a rectangular complex array") from exc
    if not np.all(np.isfinite(copied)):
        raise ValueError(f"{name} must contain only finite values")
    return np.frombuffer(copied.tobytes(), dtype=np.complex128).reshape(copied.shape)


def immutable_real_array(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    """Return an immutable float64 snapshot of a finite real array."""
    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a rectangular real array") from exc
    if np.iscomplexobj(source):
        raise TypeError(f"{name} must contain real values")
    try:
        object_values = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a rectangular real array") from exc
    if any(
        isinstance(item, (bool, np.bool_, str, bytes)) for item in object_values.flat
    ):
        raise TypeError(f"{name} must contain numeric values without booleans")
    try:
        copied = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a rectangular real array") from exc
    if not np.all(np.isfinite(copied)):
        raise ValueError(f"{name} must contain only finite values")

    immutable = np.frombuffer(copied.tobytes(), dtype=np.float64).reshape(copied.shape)
    return immutable


def validate_strict_int(value: Any, *, name: str, minimum: int | None = 0) -> None:
    """Validate an integer without accepting booleans."""
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def validate_finite_real(value: Any, *, name: str) -> None:
    """Validate a finite real scalar without accepting booleans."""
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    if not np.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
