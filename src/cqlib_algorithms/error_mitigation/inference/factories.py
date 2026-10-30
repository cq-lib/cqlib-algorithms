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

r"""Zero-noise extrapolation factories.

A factory is the classical half of ZNE: the caller measures an expectation
value at one or more noise scale factors :math:`\lambda`, feeds the results
to :meth:`Factory.push`, and asks for the zero-noise limit through
:meth:`Factory.reduce`. The whole branch uses one shape so that interpolation,
annealing, and adaptation differ only in how the model is fitted:

.. code-block:: python

    factory = RichardsonFactory((1.0, 2.0, 3.0))
    while (scale := factory.next_scale_factor()) is not None:
        factory.push(scale, measure(scale))
    limit = factory.reduce()

Every model is fitted with NumPy only. The polynomial route uses the normal
equations :math:`(V^\top V) c = V^\top y` on a Vandermonde matrix; the
exponential route is linearised in the log domain, see :class:`ExpFactory`
for the consequences of that choice.

References
----------
[1] L. F. Richardson and J. A. Gaunt, "The deferred approach to the limit. Part I.
    Single lattice. Part II. Interpenetrating lattices," Philosophical Transactions of
    the Royal Society A 226, 299-361, 1927. doi:10.1098/rsta.1927.0008.
[2] K. Temme, S. Bravyi, and J. M. Gambetta, "Error mitigation for short-depth quantum
    circuits," Physical Review Letters 119, 180509, 2017. arXiv:1612.02058.
[3] A. Kandala, K. Temme, A. D. Corcoles, A. Mezzacapo, J. M. Chow, and J. M. Gambetta,
    "Error mitigation extends the computational reach of a noisy quantum processor,"
    Nature 567, 491-495, 2019. arXiv:1805.04492.
[4] T. Giurgica-Tiron, Y. Hindy, R. LaRose, A. Mari, and W. J. Zeng, "Digital zero noise
    extrapolation for quantum error mitigation," IEEE International Conference on
    Quantum Computing and Engineering (QCE), pp. 306-316, 2020. arXiv:2005.10921.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

_SHIFT_FACTOR = 1.27846
_EPSILON = 1.0e-9


class ExtrapolationError(Exception):
    """Raised when an extrapolation fit cannot be performed."""


class ConvergenceWarning(UserWarning):
    """Warned when a mitigation step is numerically ill-conditioned.

    Ill-conditioning does not invalidate the result: the fit stays finite and
    usable, but it amplifies the error of its inputs, so the mitigated value
    may be far less accurate than the raw measurements. It is reported as a
    warning rather than an error, and it derives from
    :class:`UserWarning` so that callers that filter by category still see it.
    """


class Factory(Protocol):
    """Extrapolation contract shared by every zero-noise factory.

    A factory owns one lifecycle: request the next scale factor, record the
    measured expectation value with :meth:`push`, and turn the accumulated
    data into the zero-noise limit with :meth:`reduce`. Callers annotate a
    factory argument with this class to stay independent of the fitted
    model.
    """

    def push(self, scale_factor: float, value: float, /) -> None:
        """Record one measured expectation value at ``scale_factor``."""
        raise NotImplementedError

    def is_converged(self) -> bool:
        """Return whether enough expectation values have been recorded."""
        raise NotImplementedError

    def reduce(self) -> float:
        """Return the fitted zero-noise limit."""
        raise NotImplementedError

    def next_scale_factor(self) -> float | None:
        """Return the next scale factor to measure, or ``None`` if done."""
        raise NotImplementedError


class _BatchedFactory(Factory):
    """Factory driven by a fixed grid of scale factors.

    The grid is known up front, so the caller may batch every circuit in one
    submission and push the results afterwards in any order. Values are
    re-ordered along the grid when the model is fitted.
    """

    def __init__(self, scale_factors: Sequence[float]) -> None:
        factors = [float(factor) for factor in scale_factors]
        if len(factors) < 2:
            raise ValueError("At least 2 scale factors are necessary.")
        self._scale_factors = factors
        self._values: dict[float, float] = {}

    def push(self, scale_factor: float, value: float, /) -> None:
        self._values[float(scale_factor)] = float(value)

    def is_converged(self) -> bool:
        return all(factor in self._values for factor in self._scale_factors)

    def next_scale_factor(self) -> float | None:
        for factor in self._scale_factors:
            if factor not in self._values:
                return factor
        return None

    def reduce(self) -> float:
        if not self.is_converged():
            missing = len(self._scale_factors) - len(self._values)
            raise ExtrapolationError(
                f"The extrapolation needs {missing} more expectation value(s) "
                "before the zero-noise limit can be fitted."
            )
        measured = [self._values[factor] for factor in self._scale_factors]
        return self._fit(measured)

    def _fit(self, values: Sequence[float]) -> float:
        """Fit the model along the grid and return its value at zero."""
        raise NotImplementedError


class RichardsonFactory(_BatchedFactory):
    r"""Richardson extrapolation through the closed-form weights.

    The weights solve :math:`\sum_i \gamma_i \lambda_i^{\alpha} =
    \delta_{\alpha 0}` for every power :math:`\alpha` below the number of
    points, so a degree-:math:`(M-1)` polynomial is reproduced exactly:

    .. math::

        \gamma_i = \prod_{j \neq i} \frac{\lambda_j}{\lambda_i - \lambda_j}

    For :math:`\lambda = (1, 2, 3)` this is exactly
    :math:`(\gamma_1, \gamma_2, \gamma_3) = (3, -3, 1)`, which satisfies
    :math:`\sum \gamma = 1` and :math:`\sum \gamma \lambda = \sum
    \gamma \lambda^2 = 0`.

    Args:
        scale_factors: At least two noise scale factors for the fixed grid.

    Raises:
        ValueError: If fewer than two scale factors are provided.
    """

    def _fit(self, values: Sequence[float]) -> float:
        weights = _richardson_coefficients(self._scale_factors)
        return float(np.dot(weights, np.asarray(values, dtype=float)))


class PolyFactory(_BatchedFactory):
    r"""Polynomial extrapolation by least squares.

    A polynomial of ``degree`` is fitted through the measured points with
    the normal equations :math:`(V^\top V) c = V^\top y`, where ``V`` is the
    Vandermonde matrix of the scale factors. The zero-noise limit is the
    constant coefficient :math:`c_0 = p(0)`.

    Normal equations square the condition number of the Vandermonde matrix,
    so a high ``degree`` relative to the number of points becomes unstable;
    :class:`RichardsonFactory` avoids the fit entirely by using closed-form
    weights when an exact interpolant is wanted.

    Args:
        scale_factors: At least two noise scale factors for the fixed grid.
        degree: Non-negative polynomial degree smaller than the grid size.

    Raises:
        ValueError: If the grid has fewer than two points or ``degree`` is out
            of range.
    """

    def __init__(self, scale_factors: Sequence[float], /, *, degree: int) -> None:
        super().__init__(scale_factors)
        if degree < 0:
            raise ValueError(f"degree must be >= 0 but was {degree!r}.")
        if degree > len(self._scale_factors) - 1:
            raise ValueError(
                "The extrapolation degree cannot exceed len(scale_factors) - 1 "
                f"but was {degree!r} for {len(self._scale_factors)} scale "
                "factors."
            )
        self._degree = degree

    @property
    def degree(self) -> int:
        """Return the degree of the fitted polynomial."""
        return self._degree

    def _fit(self, values: Sequence[float]) -> float:
        coefficients = _fit_polynomial(self._scale_factors, values, self._degree)
        return float(coefficients[0])


class LinearFactory(PolyFactory):
    r"""Linear extrapolation, i.e. a polynomial fit of degree one.

    Args:
        scale_factors: At least two noise scale factors for the fixed grid.

    Raises:
        ValueError: If fewer than two scale factors are provided.
    """

    def __init__(self, scale_factors: Sequence[float], /) -> None:
        super().__init__(scale_factors, degree=1)


class ExpFactory(_BatchedFactory):
    r"""Exponential extrapolation by log-domain linearisation.

    The model is :math:`y(\lambda) = a_0 + a_1 e^{-c\lambda}` with
    :math:`c > 0`. The asymptote :math:`a_0` is either given by the caller or
    estimated from the last three points, which is exact when they are
    equally spaced. Subtracting it leaves
    :math:`\ln|y - a_0| = \ln|a_1| - c\lambda`, a straight line fitted by the
    same polynomial routine used everywhere else.

    Args:
        scale_factors: Noise scale factors at which expectation values are measured.
        asymptote: Infinite-noise limit :math:`a_0`. When ``None`` it is estimated from
            the last three points, which must then be equally spaced.

    Raises:
        ExtrapolationError: If the residuals :math:`y - a_0` change sign, if their magnitudes
            are not monotonically decaying, or if the fitted decay rate is not
            positive.

    Note:
        The log-linearised route yields a legal but different extrapolated value
        from a nonlinear three-parameter fit of the same ansatz, so numerical
        anchors for exponential extrapolation must be calibrated against this
        route.
    """

    def __init__(
        self,
        scale_factors: Sequence[float],
        /,
        *,
        asymptote: float | None = None,
    ) -> None:
        super().__init__(scale_factors)
        self._asymptote = None if asymptote is None else float(asymptote)

    @property
    def asymptote(self) -> float | None:
        """Return the infinite-noise limit, or ``None`` when estimated."""
        return self._asymptote

    def _fit(self, values: Sequence[float]) -> float:
        asymptote, amplitude, _ = _fit_exponential(
            self._scale_factors,
            values,
            self._asymptote,
        )
        return asymptote + amplitude


class PolyExpFactory(_BatchedFactory):
    r"""Exponential extrapolation whose exponent is a polynomial.

    The model is :math:`y(\lambda) = a_0 + \sigma \exp(z(\lambda))`, where
    :math:`\sigma \in \{-1, +1\}` is read off the residuals and :math:`z` is a
    polynomial of ``degree``. Since the log-linearisation needs the asymptote
    before taking logarithms, ``asymptote`` is required: without it the fit
    would need the nonlinear optimisation this branch avoids.

    Args:
        scale_factors: Noise scale factors at which expectation values are measured.
        degree: Degree of the exponent polynomial :math:`z`.
        asymptote: Infinite-noise limit :math:`a_0`, which must be supplied.

    Raises:
        ValueError: If ``asymptote`` is ``None`` or ``degree`` is not in
            ``[0, len(scale_factors) - 1]``.
        ExtrapolationError: If the residuals :math:`y - a_0` change sign.
    """

    def __init__(
        self,
        scale_factors: Sequence[float],
        /,
        *,
        degree: int,
        asymptote: float | None,
    ) -> None:
        super().__init__(scale_factors)
        if asymptote is None:
            raise ValueError(
                "asymptote is required: the log-linearised route cannot "
                "estimate it without nonlinear optimisation."
            )
        if degree < 0:
            raise ValueError(f"degree must be >= 0 but was {degree!r}.")
        if degree > len(self._scale_factors) - 1:
            raise ValueError(
                "The exponent degree cannot exceed len(scale_factors) - 1 "
                f"but was {degree!r} for {len(self._scale_factors)} scale "
                "factors."
            )
        self._asymptote = float(asymptote)
        self._degree = degree

    def _fit(self, values: Sequence[float]) -> float:
        factors = np.asarray(self._scale_factors, dtype=float)
        measured = np.asarray(values, dtype=float)
        residuals = measured - self._asymptote
        sign = _residual_sign(residuals)
        coefficients = _fit_polynomial(
            factors,
            np.log(np.abs(residuals)),
            self._degree,
        )
        return self._asymptote + sign * float(np.exp(coefficients[0]))


class AdaExpFactory(Factory):
    r"""Adaptive exponential extrapolation with a history-driven schedule.

    The first two scale factors are fixed at :math:`\lambda_1 = 1` and
    :math:`\lambda_2 =` ``scale_factor``; when ``asymptote`` is ``None`` a
    third fixed point :math:`2 \lambda_2 - 1` completes an equally spaced
    seed. Every further point is placed where the current exponential fit is
    least constrained, through the rule

    .. math::

        \lambda_{\text{next}} = \min\!\left(\max\!\left(1 +
        \frac{1.27846}{|c| + 10^{-9}}, \lambda_{\text{last}}\right),
        \lambda_{\max}\right)

    with :math:`c` the decay rate of the running fit. Points therefore crowd
    toward the region where the curve still bends, instead of filling a
    preset grid. The inner :math:`\max` keeps the schedule monotone: a
    proposal below the last sampled point saturates the spacing rather than
    revisiting an already measured region.

    Args:
        steps: Total number of expectation values to collect. At least 3 are
            needed, or 4 when ``asymptote`` is ``None``.
        scale_factor: The second scale factor; the first is always ``1``.
        asymptote: Infinite-noise limit :math:`a_0`. When ``None`` it is estimated once
            from the equally spaced seed triplet and then held fixed, since the
            adaptive points are deliberately not equally spaced.
        max_scale_factor: Upper bound on any adaptively chosen scale factor.

    Raises:
        ValueError: If ``steps`` is too small, or ``scale_factor`` / ``max_scale_factor``
            is not strictly greater than one.
        ExtrapolationError: If the accumulated data cannot be fitted by :class:`ExpFactory`.
    """

    _SHIFT_FACTOR = _SHIFT_FACTOR
    _EPSILON = _EPSILON

    def __init__(
        self,
        /,
        *,
        steps: int,
        scale_factor: float = 2.0,
        asymptote: float | None = None,
        max_scale_factor: float = 6.0,
    ) -> None:
        if scale_factor <= 1.0:
            raise ValueError(
                "scale_factor must be strictly greater than one but was "
                f"{scale_factor!r}."
            )
        if max_scale_factor <= 1.0:
            raise ValueError(
                "max_scale_factor must be strictly greater than one but was "
                f"{max_scale_factor!r}."
            )
        minimum_steps = 3 + int(asymptote is None)
        if steps < minimum_steps:
            raise ValueError(
                f"steps must be >= {minimum_steps} but was {steps!r}."
            )
        self._steps = steps
        self._scale_factor = float(scale_factor)
        self._asymptote = None if asymptote is None else float(asymptote)
        self._estimated_asymptote: float | None = None
        self._max_scale_factor = float(max_scale_factor)
        self._plan: list[float] = []
        self._values: dict[float, float] = {}

    def push(self, scale_factor: float, value: float, /) -> None:
        self._values[float(scale_factor)] = float(value)

    def is_converged(self) -> bool:
        return len(self._values) >= self._steps

    def next_scale_factor(self) -> float | None:
        if self.is_converged():
            return None
        if not self._plan:
            self._plan.append(1.0)
        elif len(self._plan) == 1:
            self._plan.append(self._scale_factor)
        elif len(self._plan) == 2 and self._asymptote is None:
            self._plan.append(2.0 * self._scale_factor - 1.0)
        else:
            self._plan.append(self._suggest())
        return self._plan[-1]

    def reduce(self) -> float:
        measured = [factor for factor in self._plan if factor in self._values]
        if len(measured) < 3:
            raise ExtrapolationError(
                "The adaptive exponential fit needs at least 3 collected "
                f"scale factors but only {len(measured)} are available."
            )
        values = [self._values[factor] for factor in measured]
        asymptote, amplitude, _ = _fit_exponential(
            measured,
            values,
            self._resolved_asymptote(measured),
        )
        return asymptote + amplitude

    def _suggest(self) -> float:
        r"""Place the next point from the decay rate of the running fit."""
        measured = [factor for factor in self._plan if factor in self._values]
        values = [self._values[factor] for factor in measured]
        _, _, decay = _fit_exponential(
            measured,
            values,
            self._resolved_asymptote(measured),
        )
        proposed = 1.0 + self._SHIFT_FACTOR / (abs(decay) + self._EPSILON)
        lower = measured[-1] + self._EPSILON
        return float(min(max(proposed, lower), self._max_scale_factor))

    def _resolved_asymptote(self, measured: Sequence[float]) -> float:
        r"""Return the asymptote, estimated once from the seed triplet.

        The seed triplet is equally spaced, so the closed form of
        :func:`_estimate_asymptote` applies to it. The adaptive points are
        not equally spaced, hence the estimate is cached on first use rather
        than recomputed from the running points.

        Raises:
            ExtrapolationError: If ``asymptote`` was not given and fewer than three seed points
                have been collected.

        """
        if self._asymptote is not None:
            return self._asymptote
        if self._estimated_asymptote is None:
            seeds = measured[:3]
            if len(seeds) < 3:
                raise ExtrapolationError(
                    "At least 3 equally spaced points are necessary to "
                    "estimate the asymptote."
                )
            self._estimated_asymptote = _estimate_asymptote(
                seeds,
                [self._values[factor] for factor in seeds],
            )
        return self._estimated_asymptote


def _richardson_coefficients(scale_factors: Sequence[float], /) -> np.ndarray:
    r"""Return the closed-form Richardson weights for ``scale_factors``.

    Weight :math:`i` is :math:`\prod_{j \neq i} \lambda_j / (\lambda_i -
    \lambda_j)`, which annihilates every power of :math:`\lambda` below the
    number of points while summing to one.
    """
    factors = np.asarray(scale_factors, dtype=float)
    weights = np.empty(factors.size)
    for index in range(factors.size):
        others = np.delete(factors, index)
        weights[index] = np.prod(others / (factors[index] - others))
    return weights


def _fit_polynomial(
    scale_factors: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    degree: int,
    /,
) -> np.ndarray:
    r"""Fit a polynomial of ``degree`` through the points by least squares.

    Returns the coefficients in ascending power order, so entry ``0`` is the
    fit evaluated at zero noise.

    Raises:
        ExtrapolationError: If the normal equations are singular or yield a non-finite solution.

    """
    factors = np.asarray(scale_factors, dtype=float)
    measured = np.asarray(values, dtype=float)
    vandermonde = np.vander(factors, degree + 1, increasing=True)
    gram = vandermonde.T @ vandermonde
    try:
        coefficients = np.linalg.solve(gram, vandermonde.T @ measured)
    except np.linalg.LinAlgError as error:
        raise ExtrapolationError(
            "The polynomial fit is singular; the scale factors are too "
            "close together or too few for the requested degree."
        ) from error
    if not np.all(np.isfinite(coefficients)):
        raise ExtrapolationError(
            "The polynomial fit produced non-finite coefficients."
        )
    return coefficients


def _estimate_asymptote(
    scale_factors: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    /,
) -> float:
    r"""Estimate the infinite-noise limit from the last three points.

    For an equally spaced triple,
    :math:`a_0 = (y_1 y_3 - y_2^2) / (y_1 - 2 y_2 + y_3)` holds exactly for
    any :math:`y(\lambda) = a_0 + a_1 e^{-c\lambda}`. Equally spaced input is
    therefore required, since the closed form does not generalise.

    Raises:
        ExtrapolationError: If the last three scale factors are not equally spaced, or the
            points are collinear and so cannot pin down an asymptote.

    """
    factors = np.asarray(scale_factors, dtype=float)
    measured = np.asarray(values, dtype=float)
    order = np.argsort(factors)
    tail = measured[order][-3:]
    spacing = np.diff(np.sort(factors)[-3:])
    if not np.isclose(spacing[0], spacing[1]):
        raise ExtrapolationError(
            "The asymptote can only be estimated when the last three scale "
            "factors are equally spaced; pass asymptote explicitly instead."
        )
    denominator = float(tail[0] - 2.0 * tail[1] + tail[2])
    if np.isclose(denominator, 0.0):
        raise ExtrapolationError(
            "The last three expectation values are collinear, so they cannot "
            "determine an exponential asymptote."
        )
    return float((tail[0] * tail[2] - tail[1] ** 2) / denominator)


def _residual_sign(residuals: np.ndarray, /) -> float:
    r"""Return ``+1`` or ``-1`` for a residual vector of a single sign.

    Raises:
        ExtrapolationError: If the residuals change sign, in which case no single exponential can
            describe the deviation from the asymptote.

    """
    if np.all(residuals > 0.0):
        return 1.0
    if np.all(residuals < 0.0):
        return -1.0
    raise ExtrapolationError(
        "The expectation values cross the assumed asymptote, so they cannot "
        "be described by a single exponential."
    )


def _fit_exponential(
    scale_factors: Sequence[float],
    values: Sequence[float],
    asymptote: float | None,
    /,
) -> tuple[float, float, float]:
    r"""Fit :math:`a_0 + a_1 e^{-c\lambda}` by log-domain linearisation.

    Returns ``(a_0, a_1, c)``. The asymptote is taken from ``asymptote`` when
    given and estimated from the last three points otherwise; the remaining
    two parameters come from a straight-line fit of
    :math:`\ln|y - a_0|` against :math:`\lambda`.

    Raises:
        ExtrapolationError: If the residuals change sign, their magnitudes grow with the scale
            factor, or the fitted decay rate is not positive.

    """
    factors = np.asarray(scale_factors, dtype=float)
    measured = np.asarray(values, dtype=float)
    order = np.argsort(factors)
    factors = factors[order]
    measured = measured[order]

    if asymptote is None:
        if factors.size < 3:
            raise ExtrapolationError(
                "At least 3 equally spaced points are necessary to estimate "
                "the asymptote."
            )
        asymptote = _estimate_asymptote(factors, measured)

    residuals = measured - asymptote
    sign = _residual_sign(residuals)
    magnitudes = np.abs(residuals)
    if not np.all(np.diff(magnitudes) <= 0.0):
        raise ExtrapolationError(
            "The deviation from the asymptote does not decay monotonically "
            "with the scale factor, so it is not exponential."
        )

    intercept, slope = _fit_polynomial(factors, np.log(magnitudes), 1)
    decay = -float(slope)
    if decay <= 0.0:
        raise ExtrapolationError(
            f"The fitted decay rate must be positive but was {decay!r}."
        )
    return float(asymptote), sign * float(np.exp(intercept)), decay
