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

"""Shared numerical normalization for scientific extension models."""

from __future__ import annotations

import math
import numbers

import numpy as np


def normalize_finite_complex(value: complex, *, name: str) -> complex:
    """Validate and return a finite real or complex value."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Complex):
        raise TypeError(f"{name} must be a real or complex number")
    result = complex(value)
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise ValueError(f"{name} must be finite")
    return result


def normalize_vibrational_atol(atol: float) -> float:
    """Validate and return a finite non-negative vibrational tolerance."""
    if isinstance(atol, (bool, np.bool_)) or not isinstance(atol, numbers.Real):
        raise TypeError("atol must be a real number")
    converted = float(atol)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError("atol must be finite and non-negative")
    return converted


__all__: list[str] = []
