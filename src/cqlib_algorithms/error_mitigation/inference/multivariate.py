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

r"""Multivariate Richardson extrapolation for layerwise Richardson (LRE).

Standard ZNE assumes a single noise parameter, so every scale factor is a
scalar. LRE scales each layer of the circuit independently, which turns the
noise into a vector :math:`\lambda = (\lambda_1, \dots, \lambda_L)` and the
expectation value into a multivariate polynomial

.. math::

    \langle O(\lambda) \rangle = \sum_{|\alpha| \le d} c_\alpha \lambda^\alpha.

Extrapolating to the noise-free point means choosing weights :math:`\gamma`
such that every non-constant monomial cancels and the constant term survives,

.. math::

    \sum_i \gamma_i = 1, \qquad
    \sum_i \gamma_i \lambda_i^\alpha = 0
    \quad \text{for all } 1 \le |\alpha| \le d .

The weights depend only on the scale points, so they are computed here once
and reused for every expectation value. The graded monomial basis and the
determinant-ratio construction follow Russo and Mari (arXiv:2402.04000),
Section II B.

References
----------
[1] A. Russo and A. Mari, "Quantum error mitigation by layerwise Richardson
    extrapolation," Physical Review A 110, 062420, 2024. arXiv:2402.04000.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product

import numpy as np


def multivariate_richardson_coefficients(
    scale_points: Sequence[Sequence[float]],
    /,
) -> list[float]:
    r"""Return weights satisfying :math:`\sum \gamma = 1` and
    :math:`\sum \gamma \lambda^\alpha = 0` for all :math:`|\alpha| \le d`.

    The degree :math:`d` is not a parameter: it is recovered from the shape
    of the input, since a graded monomial basis in :math:`L` variables with
    total degree at most :math:`d` has exactly
    :math:`\binom{L + d}{d}` terms and the sample matrix must be square.

    Args:
        scale_points: One noise scale vector per measurement, each of length :math:`L`
            (the number of independently scaled layers). All vectors must have
            the same length, and their count :math:`N` must equal the size of a
            graded monomial basis over :math:`L` variables, that is
            :math:`N = \binom{L + d}{d}` for some degree :math:`d \ge 1`. Every
            vector contributes one expectation value, which keeps the weight
            system square.

    Returns:
        One weight per scale point, in the order the points were given.
            Multiplying the measured expectation values by these weights and
            summing cancels the noise to order :math:`d`.

    Raises:
        ValueError: If ``scale_points`` is not a rectangular 2-D collection, if the
            number of points matches no graded monomial basis, if the sample
            matrix is singular (degenerate or repeated scale points), or if its
            determinant is not finite.

    Note:
        Scaling every layer by the same factor reduces the weights to the
        univariate Richardson coefficients.
    """
    points = np.asarray(scale_points, dtype=float)
    if points.ndim != 2:
        raise ValueError(
            "scale_points must be a rectangular 2-D collection of scale "
            f"vectors but had shape {points.shape!r}."
        )
    num_points, num_layers = points.shape
    degree = _infer_degree(num_layers, num_points)
    exponents = _monomial_exponents(num_layers, degree)
    sample = _sample_matrix(points, exponents)

    determinant = float(np.linalg.det(sample))
    if not np.isfinite(determinant):
        raise ValueError(
            "The determinant of the multivariate sample matrix is not "
            "finite; the scale points span too wide a range."
        )
    if determinant == 0.0:
        raise ValueError(
            "The multivariate sample matrix is singular, so the scale "
            "points are degenerate; see Russo and Mari (arXiv:2402.04000) "
            "Section II B for the required linear independence."
        )

    target = np.zeros(num_points, dtype=float)
    target[0] = 1.0
    # Cramer's rule: weight i is the ratio between the determinant of the
    # sample matrix with row i replaced by (1, 0, ..., 0) and the determinant
    # of the sample matrix itself.
    coefficients: list[float] = []
    for index in range(num_points):
        replaced = sample.copy()
        replaced[index] = target
        coefficients.append(float(np.linalg.det(replaced) / determinant))
    return coefficients


def _infer_degree(num_layers: int, num_points: int, /) -> int:
    """Recover the polynomial degree from the basis size.

    The graded monomial basis in ``num_layers`` variables with total degree
    at most ``d`` has ``comb(num_layers + d, d)`` terms, so a valid degree
    exists only when ``num_points`` hits that sequence exactly.
    """
    if num_points < num_layers + 1:
        raise ValueError(
            f"{num_points} scale points over {num_layers} layers are too "
            f"few: degree 1 already needs {num_layers + 1} points."
        )
    terms = num_layers + 1
    degree = 1
    while terms < num_points:
        candidate_degree = degree + 1
        numerator = terms * (num_layers + candidate_degree)
        candidate_terms = numerator // candidate_degree
        if candidate_terms > num_points:
            raise ValueError(
                f"{num_points} scale points over {num_layers} layers match "
                f"no graded monomial basis: degree {degree} needs {terms} "
                f"points and degree {candidate_degree} needs "
                f"{candidate_terms}."
            )
        terms = candidate_terms
        degree = candidate_degree
    return degree


def _monomial_exponents(
    num_layers: int,
    degree: int,
    /,
) -> list[tuple[int, ...]]:
    """List the total-degree exponents of the graded monomial basis.

    Terms are sorted by total degree first and then lexicographically in
    reverse variable order, so the exponents read as in Russo and Mari
    (arXiv:2402.04000) Section II B. For ``num_layers=2`` and ``degree=2``
    this yields ``[(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]``.
    """
    exponents = [
        term
        for term in product(range(degree + 1), repeat=num_layers)
        if sum(term) <= degree
    ]
    exponents.sort(key=lambda term: (sum(term), term[::-1]))
    return exponents


def _sample_matrix(
    scale_points: np.ndarray,
    exponents: Sequence[tuple[int, ...]],
    /,
) -> np.ndarray:
    """Evaluate the monomial basis at every scale point.

    Row ``i`` holds the monomials evaluated at ``scale_points[i]``, and
    column ``j`` holds one basis term across all points.
    """
    sample = np.empty((len(scale_points), len(exponents)), dtype=float)
    for row, point in enumerate(scale_points):
        for column, exponent in enumerate(exponents):
            sample[row, column] = float(
                np.prod([base**power for base, power in zip(point, exponent)])
            )
    return sample
