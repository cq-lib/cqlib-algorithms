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

"""Tests for multivariate Richardson extrapolation used by LRE."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product

import numpy as np
import pytest

from cqlib_algorithms.error_mitigation.inference.multivariate import (
    multivariate_richardson_coefficients,
)

_ATOL = 1e-9


def _graded_exponents(num_layers: int, degree: int, /) -> list[tuple[int, ...]]:
    """List every exponent with :math:`1 \\le |\\alpha| \\le` ``degree``.

    Rebuilt from the definition of the graded monomial basis rather than
    imported, so that the constraints are checked against an independent
    enumeration.
    """
    return [
        term
        for term in product(range(degree + 1), repeat=num_layers)
        if 1 <= sum(term) <= degree
    ]


def _constraint_residual(
    weights: Sequence[float],
    points: Sequence[Sequence[float]],
    degree: int,
    /,
) -> float:
    r"""Return the worst violation of :math:`\sum \gamma = 1` and
    :math:`\sum \gamma \lambda^\alpha = 0`."""
    grid = np.asarray(points, dtype=float)
    gamma = np.asarray(weights, dtype=float)
    worst = abs(float(gamma.sum()) - 1.0)
    for exponent in _graded_exponents(grid.shape[1], degree):
        monomial = np.prod(grid ** np.asarray(exponent), axis=1)
        worst = max(worst, abs(float(gamma @ monomial)))
    return worst


def test_a_single_layer_reduces_to_the_univariate_richardson_weights() -> None:
    points = [(1.0,), (2.0,), (3.0,)]

    weights = multivariate_richardson_coefficients(points)

    assert weights == pytest.approx([3.0, -3.0, 1.0], abs=_ATOL)
    assert _constraint_residual(weights, points, 1) < _ATOL


def test_bilinear_weights_cancel_every_non_constant_monomial() -> None:
    points = [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)]

    weights = multivariate_richardson_coefficients(points)

    # Three points over two layers fix a total degree of one: the constant
    # term survives and both linear monomials cancel.
    assert _constraint_residual(weights, points, 1) < _ATOL


def test_a_two_layer_quadratic_basis_lifts_a_quadratic_response_exactly() -> None:
    points = [
        (1.0, 1.0),
        (2.0, 1.0),
        (1.0, 2.0),
        (3.0, 1.0),
        (1.0, 3.0),
        (2.0, 2.0),
    ]
    grid = np.asarray(points, dtype=float)
    response = (
        1.0
        + grid[:, 0]
        + grid[:, 1]
        + grid[:, 0] * grid[:, 1]
        + grid[:, 0] ** 2
        + grid[:, 1] ** 2
    )

    weights = multivariate_richardson_coefficients(points)

    # Six points over two layers match the size of the degree-2 graded basis,
    # so every non-constant monomial of the response cancels.
    assert _constraint_residual(weights, points, 2) < _ATOL
    assert float(np.asarray(weights) @ response) == pytest.approx(1.0, abs=_ATOL)


def test_repeated_scale_points_are_rejected_as_singular() -> None:
    points = [(1.0,), (1.0,), (2.0,)]

    with pytest.raises(ValueError, match="singular"):
        multivariate_richardson_coefficients(points)


def test_a_point_count_that_matches_no_basis_is_rejected() -> None:
    points = [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (2.0, 2.0)]

    # Four points over two layers are one short of the degree-2 basis, so no
    # degree reproduces the sample matrix as a square system.
    with pytest.raises(ValueError, match="no graded monomial basis"):
        multivariate_richardson_coefficients(points)


def test_a_non_rectangular_collection_is_rejected() -> None:
    with pytest.raises(ValueError, match="rectangular 2-D collection"):
        multivariate_richardson_coefficients([1.0, 2.0, 3.0])
