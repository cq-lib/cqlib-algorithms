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

"""Amplitude-encoded functional rotations on an auxiliary qubit.

This package is organised into two subpackages that expose the same
high-level shape (a control register plus an auxiliary result qubit)
but use fundamentally different circuit families:

* :mod:`.multiplexed` — uniformly controlled (UCR / Gray-code)
  multiplexors.  Families here build an arbitrary per-control table
  with ``O(2**n)`` gates and include the base gates :class:`UCRX`,
  :class:`UCRY`, :class:`UCRZ` together with the higher-level tables
  :class:`ExactReciprocal` (HHL eigenvalue inversion) and
  :class:`ArbitraryFunction` (user-supplied function table, two
  encoding modes).
* :mod:`.polynomial` — bit-wise controlled-rotation constructions
  whose gate count scales **polynomially** with the control-register
  bit width instead of exponentially.  Currently implements
  :class:`LinearPauliRotation` (``theta(k) = slope*k + offset`` via
  ``O(n)`` single-controlled gates).
"""

from __future__ import annotations

from .multiplexed import (
    UCRX,
    UCRY,
    UCRZ,
    ArbitraryFunction,
    ExactReciprocal,
)
from .polynomial import (
    LinearAmplitudeRotation,
    LinearPauliRotation,
    PiecewiseChebyshev,
    PiecewiseLinearPauliRotation,
    PiecewisePolynomialPauliRotation,
    PolynomialPauliRotation,
)

__all__ = [
    "UCRX",
    "UCRY",
    "UCRZ",
    "ArbitraryFunction",
    "ExactReciprocal",
    "LinearAmplitudeRotation",
    "LinearPauliRotation",
    "PiecewiseChebyshev",
    "PiecewiseLinearPauliRotation",
    "PiecewisePolynomialPauliRotation",
    "PolynomialPauliRotation",
]
