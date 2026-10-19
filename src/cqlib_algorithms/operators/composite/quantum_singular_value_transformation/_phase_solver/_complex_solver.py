r"""QSP phase synthesis for complex Chebyshev polynomials.

This module implements phase synthesis in the setting of Corollary 8 of the
GSLW19 paper (arXiv:1806.01838).  The target is a complex polynomial
``P in C[x]`` of degree ``d`` that follows Corollary 8, i.e. it satisfies the
four conditions

1. ``P`` has parity ``d % 2``;
2. ``|P(x)| <= 1`` on ``[-1, 1]``;
3. ``|P(x)| >= 1`` on ``(-inf, -1] U [1, inf)``;
4. ``P(ix) P*(ix) >= 1`` on ``R`` when ``d`` is even.

The solver fits the full complex ``(0, 0)`` entry of the QSP response matrix
to ``P(x)``.

Unlike the real-polynomial solver, the phase sequence is not constrained to
the symmetric mirror form.  All ``degree + 1`` phases are free parameters,
which is necessary to represent a generic complex polynomial.  The resulting
full phase sequence can still be passed to
:func:`._converters.convert_qsp_phases` to obtain QSVT ``W``-form phases.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.polynomial.chebyshev import chebval

from ._numerics import (
    _BOUND_TOLERANCE,
    _COEFFICIENT_TOLERANCE,
    _chebyshev_nodes,
    _loss,
    _run_bfgs,
    _validate_positive_integer,
    _validate_positive_scalar,
    _wrap_phases,
)


def solve_complex_qsp_phases(
    coefficients: Sequence[complex] | np.ndarray,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 300,
    gradient_tolerance: float = 1e-10,
) -> np.ndarray:
    r"""Return a QSP phase sequence for a complex Chebyshev polynomial.

    The solver fits the full complex ``(0, 0)`` entry of the QSP response
    matrix to ``P(x)`` (Corollary 8).  All ``degree + 1`` phases are treated
    as free real parameters (no symmetry constraint).

    Parameters
    ----------
    coefficients
        Chebyshev-basis coefficients ``(c_0, c_1, ..., c_d)`` in ascending
        order.  Coefficients may be complex.  The implied degree is the
        highest non-zero coefficient index after trimming trailing entries
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
        Full QSP phase sequence of length ``degree + 1`` (real-valued).

    Raises
    ------
    TypeError
        If coefficients or solver controls have invalid types.
    ValueError
        If the polynomial violates any of the Corollary 8 conditions on ``P``,
        or if the solver controls are outside their valid ranges.
    RuntimeError
        If the internal quasi-Newton solve does not converge.

    Notes
    -----
    The target follows Corollary 8 of Ref. [1]: the four conditions on ``P``
    (parity; ``|P| <= 1`` on ``[-1, 1]``; ``|P| >= 1`` outside ``[-1, 1]``;
    and ``P(ix) P*(ix) >= 1`` for ``x in R`` when ``d`` is even) are
    necessary and sufficient for the existence of a phase sequence.  They are
    checked numerically by :func:`_validate_complex_qsp_constraints` before the
    solve, with the unbounded regions sampled on a finite grid.

    References
    ----------
    .. [1] A. Gilyen, Y. Su, G. H. Low, and N. Wiebe,
       "Quantum singular value transformation and beyond: exponential
       improvements for quantum matrix arithmetics,"
       Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of
       Computing, pp. 193-204, 2019. arXiv:1806.01838.
    .. [2] Y. Dong, X. Meng, K. B. Whaley, and L. Lin,
       "Efficient phase-factor evaluation in quantum signal processing,"
       Physical Review A 103, 042419, 2021. arXiv:2002.11649.
    """
    trimmed = _normalize_complex_coefficients(coefficients)
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

    _validate_complex_qsp_constraints(trimmed)

    num_free_phases = degree + 1
    nodes = _chebyshev_nodes(num_free_phases)
    targets = chebval(nodes, trimmed).astype(np.complex128)

    def loss_fn(phases: np.ndarray) -> float:
        return _loss(phases, nodes, targets, real_only=False)

    return _run_bfgs(
        num_free_phases=num_free_phases,
        degree=degree,
        loss_fn=loss_fn,
        tolerance=tolerance_float,
        max_iterations=max_iterations_int,
        gradient_tolerance=gradient_tolerance_float,
        wrap=lambda ph: _wrap_phases(ph),
    )


def _normalize_complex_coefficients(
    coefficients: Sequence[complex] | np.ndarray,
) -> np.ndarray:
    if isinstance(coefficients, np.ndarray):
        flattened = coefficients.reshape(-1).tolist()
    elif isinstance(coefficients, Sequence) and not isinstance(
        coefficients,
        (str, bytes),
    ):
        flattened = list(coefficients)
    else:
        raise TypeError("coefficients must be a one-dimensional sequence")

    if not flattened:
        raise ValueError("coefficients must not be empty")

    values = np.empty(len(flattened), dtype=np.complex128)
    for index, value in enumerate(flattened):
        if isinstance(value, bool):
            raise TypeError("coefficients must contain complex finite scalars")
        scalar = complex(value)
        if not math.isfinite(scalar.real) or not math.isfinite(scalar.imag):
            raise TypeError("coefficients must contain complex finite scalars")
        values[index] = scalar

    nonzero = np.flatnonzero(np.abs(values) > _COEFFICIENT_TOLERANCE)
    if nonzero.size == 0:
        return np.asarray([0.0 + 0.0j], dtype=np.complex128)
    return values[: int(nonzero[-1]) + 1]


def _validate_complex_qsp_constraints(coefficients: np.ndarray) -> None:
    r"""Validate the four Corollary 8 conditions on ``P``.

    Following Corollary 8 of Ref. [1] (equivalently Theorem 4), a degree-``d``
    polynomial ``P in C[x]`` admits a QSP phase sequence if and only if

    1. ``P`` has parity ``d % 2``;
    2. ``|P(x)| <= 1`` for ``x in [-1, 1]``;
    3. ``|P(x)| >= 1`` for ``x in (-inf, -1] U [1, inf)``;
    4. ``P(ix) P*(ix) >= 1`` for ``x in R`` when ``d`` is even.

    The unbounded regions in conditions 3 and 4 are sampled on a finite grid,
    so the checks are numerical rather than symbolic.
    """
    degree = len(coefficients) - 1
    parity = degree % 2
    for index, value in enumerate(coefficients):
        if index % 2 != parity and abs(value) > _COEFFICIENT_TOLERANCE:
            raise ValueError(
                "input polynomial violates the condition: parity d mod 2, "
                "see GSLW19 Corollary 8 condition 1 in detail",
            )

    grid_size = max(129, 16 * degree + 1)
    span = float(grid_size)

    interior = np.linspace(-1.0, 1.0, grid_size, dtype=np.float64)
    max_magnitude = float(np.max(np.abs(chebval(interior, coefficients))))
    if max_magnitude > 1.0 + _BOUND_TOLERANCE:
        raise ValueError(
            "input polynomial violates the condition: |P(x)| <= 1 on "
            "[-1, 1], see GSLW19 Corollary 8 condition 2 in detail",
        )

    exterior = np.linspace(1.0, 1.0 + span, grid_size, dtype=np.float64)
    exterior = np.concatenate((exterior, -exterior))
    min_magnitude = float(np.min(np.abs(chebval(exterior, coefficients))))
    if min_magnitude < 1.0 - _BOUND_TOLERANCE:
        raise ValueError(
            "input polynomial violates the condition: |P(x)| >= 1 outside "
            "[-1, 1], see GSLW19 Corollary 8 condition 3 in detail",
        )

    if degree % 2 == 0:
        real_x = np.linspace(
            -1.0 - span,
            1.0 + span,
            grid_size,
            dtype=np.float64,
        )
        forward = chebval(1j * real_x, coefficients)
        backward = chebval(-1j * real_x, coefficients)
        product = float(np.min(np.real(forward * np.conjugate(backward))))
        if product < 1.0 - _BOUND_TOLERANCE:
            raise ValueError(
                "input polynomial violates the condition: "
                "P(ix) P*(ix) >= 1 for x in R when d is even, "
                "see GSLW19 Corollary 8 condition 4 in detail",
            )
