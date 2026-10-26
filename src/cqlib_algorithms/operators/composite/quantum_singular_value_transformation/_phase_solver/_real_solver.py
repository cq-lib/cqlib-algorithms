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

r"""QSP phase synthesis for real Chebyshev polynomials.

This module implements Corollary 10 style phase synthesis: the target is a
real polynomial ``P_Re`` and the solver fits ``Re(P(x)) = P_Re(x)`` where
``P`` is the complex polynomial induced by the phase sequence.

The phase sequence is constrained to the symmetric form
``(phi_0, phi_1, ..., phi_1, phi_0)`` (mirror), which halves the number of
free parameters and is the convention used by the QSP-to-QSVT phase
conversion in :mod:`_converters`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

import numpy as np
from numpy.polynomial.chebyshev import chebval

from ._numerics import (
    _BOUND_TOLERANCE,
    _COEFFICIENT_TOLERANCE,
    _chebyshev_nodes,
    _loss,
    _run_bfgs,
    _symmetric_phases,
    _validate_positive_integer,
    _validate_positive_scalar,
    _wrap_phases,
)


def solve_qsp_phases(
    coefficients: Sequence[float] | np.ndarray,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 300,
    gradient_tolerance: float = 1e-10,
) -> np.ndarray:
    r"""Return a symmetric QSP phase sequence for a real Chebyshev polynomial.

    Parameters
    ----------
    coefficients
        Chebyshev-basis coefficients in ascending order. The implied degree is
        the highest non-zero coefficient index after trimming trailing entries
        smaller than ``1e-14`` in magnitude.
    tolerance
        Target mean-squared error on the Chebyshev-node training grid.
    max_iterations
        Maximum number of BFGS iterations.
    gradient_tolerance
        Infinity-norm tolerance used to stop when the loss surface is locally
        flat enough.

    Returns
    -------
    numpy.ndarray
        Full symmetric phase sequence of length ``degree + 1``.

    Raises
    ------
    TypeError
        If coefficients or solver controls have invalid types.
    ValueError
        If the polynomial violates the QSP parity or boundedness constraints,
        or if the solver controls are outside their valid ranges.
    RuntimeError
        If the internal quasi-Newton solve does not converge.

    References
    ----------
    .. [1] G. H. Low and I. L. Chuang,
       "Hamiltonian Simulation by Qubitization,"
       Quantum 3, 163, 2019. arXiv:1610.06546.
    .. [2] Y. Dong, X. Meng, K. B. Whaley, and L. Lin,
       "Efficient phase-factor evaluation in quantum signal processing,"
       Physical Review A 103, 042419, 2021. arXiv:2002.11649.
    """
    trimmed = _normalize_coefficients(coefficients)
    degree = len(trimmed) - 1

    tolerance_float = _validate_positive_scalar(tolerance, name="tolerance")
    gradient_tolerance_float = _validate_positive_scalar(
        gradient_tolerance,
        name="gradient_tolerance",
    )
    max_iterations_int = _validate_positive_integer(
        max_iterations,
        name="max_iterations",
    )

    _validate_qsp_constraints(trimmed)

    if degree == 0:
        return np.asarray([math.acos(float(trimmed[0]))], dtype=np.float64)
    if degree == 1:
        phase = 0.5 * math.acos(float(trimmed[1]))
        return np.asarray([phase, phase], dtype=np.float64)

    num_free_phases = (degree + 2) // 2
    nodes = _chebyshev_nodes(num_free_phases)
    targets = chebval(nodes, trimmed).astype(np.float64)

    def loss_fn(hat_phases: np.ndarray) -> float:
        full = _symmetric_phases(hat_phases, degree)
        return _loss(full, nodes, targets, real_only=True)

    return _run_bfgs(
        num_free_phases=num_free_phases,
        degree=degree,
        loss_fn=loss_fn,
        tolerance=tolerance_float,
        max_iterations=max_iterations_int,
        gradient_tolerance=gradient_tolerance_float,
        wrap=lambda ph: _wrap_phases(_symmetric_phases(ph, degree)),
    )


def _normalize_coefficients(
    coefficients: Sequence[float] | np.ndarray,
) -> np.ndarray:
    if isinstance(coefficients, np.ndarray):
        flattened = coefficients.reshape(-1).tolist()
    elif isinstance(coefficients, Sequence) and not isinstance(
        coefficients,
        (str, bytes),
    ):
        flattened = list(coefficients)
    else:
        raise TypeError("coefficients must be a one-dimensional real sequence")

    if not flattened:
        raise ValueError("coefficients must not be empty")

    values = np.empty(len(flattened), dtype=np.float64)
    for index, value in enumerate(flattened):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("coefficients must contain real finite scalars")
        scalar = float(value)
        if not math.isfinite(scalar):
            raise TypeError("coefficients must contain real finite scalars")
        values[index] = scalar

    nonzero = np.flatnonzero(np.abs(values) > _COEFFICIENT_TOLERANCE)
    if nonzero.size == 0:
        return np.asarray([0.0], dtype=np.float64)
    return values[: int(nonzero[-1]) + 1]


def _validate_qsp_constraints(coefficients: np.ndarray) -> None:
    degree = len(coefficients) - 1
    parity = degree % 2
    for index, value in enumerate(coefficients):
        if index % 2 != parity and abs(float(value)) > _COEFFICIENT_TOLERANCE:
            raise ValueError(
                "coefficients must have Chebyshev parity matching the degree",
            )

    grid_size = max(129, 16 * degree + 1)
    grid = np.linspace(-1.0, 1.0, grid_size, dtype=np.float64)
    max_magnitude = float(np.max(np.abs(chebval(grid, coefficients))))
    if max_magnitude > 1.0 + _BOUND_TOLERANCE:
        raise ValueError("target polynomial must satisfy |f(x)| <= 1 on [-1, 1]")
