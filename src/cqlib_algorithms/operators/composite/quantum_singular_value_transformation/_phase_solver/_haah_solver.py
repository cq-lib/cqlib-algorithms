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

r"""Haah product decomposition of SU(2)-valued Laurent polynomials.

This module implements the algebraic core of Jeongwan Haah's product
decomposition for quantum signal processing [1].  Haah's framework
characterises the SU(2)-valued Laurent polynomials that a QSP circuit can
realise, and shows that every such polynomial admits a unique
factorisation into rank-one primitives

.. math::

    E_P(t) = t P + t^{-1} (I - P).

That factorisation is the deterministic, optimisation-free counterpart of
the iterative BFGS phase solvers in :mod:`._complex_solver`.

Method
------
Given a feasible target ``P``, the QSP unitarity relation

.. math::

    |P(x)|^2 + (1 - x^2) |Q(x)|^2 = 1

fixes the modulus of the complementary polynomial, and the even quotient

.. math::

    g(x) = \frac{1 - |P(x)|^2}{1 - x^2} = Q(x) \bar Q(x)

is spectrally factored on ``[-1, 1]``.  The modulus ``|Q|`` is unique but
its phase is not, and only some representatives make the assembled response
decomposable into rank-one primitives, so every spectral factor of ``g`` is
tried in turn.

The entry point :func:`solve_haah_phases` mirrors
:func:`._complex_solver.solve_complex_qsp_phases`: the same input (complex
Chebyshev coefficients) and the same output (an R-form phase sequence of
length ``d + 1``), but the phases come out of a deterministic factorisation
rather than a quasi-Newton search.

References
----------
[1] J. Haah, "Product Decomposition of Periodic Functions in Quantum Signal Processing,"
    Quantum 3, 190, 2019. arXiv:1806.10236.
[2] A. Gilyen, Y. Su, G. H. Low, and N. Wiebe, "Quantum singular value transformation
    and beyond: exponential improvements for quantum matrix arithmetics," Proceedings of
    the 51st Annual ACM SIGACT Symposium on Theory of Computing, pp. 193-204, 2019.
    arXiv:1806.01838.
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from numpy.polynomial import polynomial as poly
from numpy.polynomial.chebyshev import cheb2poly, chebroots

from ._complex_solver import (
    _normalize_complex_coefficients,
    _validate_complex_qsp_constraints,
)
from ._numerics import _normalize_positive_scalar

_ZERO_TOLERANCE = 1e-12
_SERIES_TRIM_TOLERANCE = 1e-14

# A root whose imaginary part falls below this relative bound counts as real:
# ``chebroots`` splits a double real root by ``~sqrt(eps) ~ 1e-8``.
_REAL_ROOT_TOLERANCE = 1e-7

# Relative slop of the vanishing of ``h`` at ``u = 0`` for an even degree.
# Every other feasibility check on ``P`` comes from
# :func:`._complex_solver._validate_complex_qsp_constraints`.
_FEASIBILITY_TOLERANCE = 1e-7

# Coefficient tolerance of the Haah peel after the assembled series has been
# normalised to unit maximum coefficient.  The rounding floor of the assembly
# grows like ``eps * 2**degree`` (about ``5e-11`` at degree 20), while the
# accuracy of random targets degrades monotonically once the peel starts
# accepting a wrong spectral factor, so ``1e-10`` is the empirical optimum.
_PEEL_TOLERANCE = 1e-10

# Degree from which the conjugate-root pairing of ``chebroots`` starts to
# degrade.  The solver keeps running, but the regime is flagged with a
# ``UserWarning`` because a valid target can then be reported as having no
# QSP-decomposable spectral factor.
_CONJUGATE_PAIRING_DEGREE_LIMIT = 16

_LaurentSeries = dict[int, npt.NDArray[np.complex128]]
r"""Laurent coefficients ``{j: C_j}`` of ``F(t) = \sum_j C_j t^j``."""


def _primitive_matrix(
    projector: npt.NDArray[np.complex128],
    *,
    inverse: bool = False,
) -> _LaurentSeries:
    r"""Return the primitive matrix ``E_P(t)`` of Eq. (2) in Ref. [1].

    The primitive built from a rank-one projector ``P`` and its orthogonal
    complement ``Q = I - P`` is

    .. math::

        E_P(t) = t P + t^{-1} Q,
        \qquad
        E_P(t)^{-1} = E_P(t^{-1}) = t^{-1} P + t Q,

    where the inverse identity follows directly from ``P Q = 0``.

    Args:
        projector: Rank-one projector ``P``.
        inverse: When ``True`` return ``E_P(t)^{-1}`` instead of ``E_P(t)``.

    """
    complement = np.eye(2, dtype=np.complex128) - projector
    if inverse:
        return {-1: projector, 1: complement}
    return {1: projector, -1: complement}


def _laurent_multiply(
    left: _LaurentSeries,
    right: _LaurentSeries,
) -> _LaurentSeries:
    r"""Return the matrix-valued Laurent product of two series."""
    product: _LaurentSeries = {}
    for left_exponent, left_matrix in left.items():
        for right_exponent, right_matrix in right.items():
            exponent = left_exponent + right_exponent
            term = left_matrix @ right_matrix
            if exponent in product:
                product[exponent] = product[exponent] + term
            else:
                product[exponent] = term
    return product


def _laurent_highest_term(
    series: _LaurentSeries,
    *,
    tolerance: float = _ZERO_TOLERANCE,
) -> tuple[int, npt.NDArray[np.complex128]]:
    r"""Return the highest non-negligible exponent and its coefficient.

    The soft tolerance is what turns the exact matrix cancellation exploited
    by Theorem 2 of Ref. [1] into a numerically stable degree reduction.
    """
    significant = [
        exponent
        for exponent, matrix in series.items()
        if float(np.max(np.abs(matrix))) > tolerance
    ]
    if not significant:
        raise ValueError("Laurent series has no non-negligible coefficient")
    exponent = max(significant)
    return exponent, series[exponent]


def _rank_one_projector(
    coefficient: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    r"""Return ``C^\dagger C / Tr(C^\dagger C)`` for a rank-one ``C``.

    Theorem 2 of Ref. [1] proves that the highest-degree coefficient ``C_n``
    of ``F \in E_n`` is singular, so this Gram-matrix normalisation is
    exactly the rank-one projector ``P_n``.
    """
    gram = coefficient.conj().T @ coefficient
    trace = float(np.real(np.trace(gram)))
    if trace <= _ZERO_TOLERANCE:
        raise ValueError("highest-degree coefficient is numerically rank zero")
    return np.asarray(gram / trace, dtype=np.complex128)


def _decompose_product(
    series: _LaurentSeries,
    degree: int,
    *,
    tolerance: float = _ZERO_TOLERANCE,
) -> tuple[npt.NDArray[np.complex128], list[npt.NDArray[np.complex128]]]:
    r"""Factor an ``E_n`` Laurent polynomial into rank-one primitives.

    Implements the recursion of Theorem 2 in Ref. [1]_: the highest-degree
    coefficient ``C_n`` determines the projector

    .. math::

        P_n = C_n^\dagger C_n / \operatorname{Tr}(C_n^\dagger C_n),

    and the residual ``F(t) E_{P_n}(t)^{-1}`` is again SU(2)-valued but has
    degree ``n - 1``.  Repeating the step ``n`` times leaves the constant
    factor ``E_0``.

    Args:
        series: Laurent coefficients of ``F`` as a mapping from exponent to a 2x2
            matrix.  The exponents are expected to share the parity of
            ``degree``.
        degree: Degree ``n`` of the product, i.e. the highest exponent of ``F``.
        tolerance: Magnitude below which a coefficient counts as numerically zero.

    Returns:
        ``(E_0, projectors)`` where ``E_0`` is the constant SU(2) factor and
            ``projectors`` lists ``(P_1, ..., P_n)`` in the order of appearance
            in ``F = E_0 E_{P_1}(t) \cdots E_{P_n}(t)``.

    """
    residual = dict(series)
    peeled: list[npt.NDArray[np.complex128]] = []
    for _ in range(degree):
        _, coefficient = _laurent_highest_term(residual, tolerance=tolerance)
        projector = _rank_one_projector(coefficient)
        peeled.append(projector)
        residual = _laurent_multiply(
            residual,
            _primitive_matrix(projector, inverse=True),
        )

    exponent, constant = _laurent_highest_term(residual, tolerance=tolerance)
    if exponent != 0:
        raise ValueError("decomposition did not reduce the series to a constant")
    return constant, list(reversed(peeled))


def _evaluate_laurent(
    series: _LaurentSeries,
    angle: float,
) -> npt.NDArray[np.complex128]:
    r"""Evaluate a Laurent series at ``t = exp(i * angle)``."""
    total = np.zeros((2, 2), dtype=np.complex128)
    for exponent, matrix in series.items():
        total += matrix * np.exp(1j * exponent * angle)
    return total


# ---------------------------------------------------------------------------
# Complementary polynomial: spectral factor of ``1 - |P|^2``
# ---------------------------------------------------------------------------
def _cheb_mul(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    r"""Multiply two Chebyshev series via ``T_m T_n = (T_{m+n} + T_{|m-n|})/2``."""
    left = np.asarray(left, dtype=np.complex128)
    right = np.asarray(right, dtype=np.complex128)
    product = np.zeros(left.size + right.size - 1, dtype=np.complex128)
    for index, value in enumerate(left):
        if value == 0.0:
            continue
        for other, weight in enumerate(right):
            half = 0.5 * value * weight
            product[index + other] += half
            product[abs(index - other)] += half
    return product


def _divide_even_by_one_minus_x2(f_even: np.ndarray) -> np.ndarray:
    r"""Solve ``(1 - x^2) G = F`` for an even Chebyshev series ``F``.

    ``f_even`` holds the full Chebyshev coefficients of the even series
    ``F``.  The returned vector holds the ``[0, 1]`` Chebyshev coefficients
    of the quotient ``G(x) = h(x^2)``.
    """
    half = np.real(np.asarray(f_even[0::2]))
    order = half.size - 1
    quotient = np.zeros(order + 2, dtype=np.float64)
    for index in range(order, 1, -1):
        quotient[index - 1] = (
            -4.0 * half[index] + 2.0 * quotient[index] - quotient[index + 1]
        )
    if order >= 2:
        quotient[0] = 2.0 * half[0] + 0.5 * quotient[1]
    else:
        quotient[0] = 2.0 * half[0]
    return quotient[:order]


def _conjugate_pairs(roots: np.ndarray) -> list[tuple[complex, ...]]:
    r"""Group a root multiset into conjugate pairs.

    Each returned tuple holds the roots that are interchangeable in the
    Fejer-Riesz factor ``q(u) = \prod (u - r)``: two complex conjugates give
    a genuine binary choice, while a root that is its own conjugate
    (numerically real) gives a single option.  Enumerating the per-pair
    choices reproduces every spectral factor of ``h``, which is exactly the
    gauge freedom left by the amplitude condition ``|Q|^2 = h``.

    A double real root is returned by ``chebroots`` as a near-degenerate
    pair ``r +- delta`` with ``delta ~ sqrt(eps)``.  Such a pair carries no
    gauge freedom: either representative perturbs ``|q|^2`` away from ``h``
    by ``O(delta)``, which is enough to break the peel.  When both members
    are numerically real they are therefore averaged back into the exact
    root and collapsed to a single option.
    """
    remaining = [complex(root) for root in roots]
    pairs: list[tuple[complex, ...]] = []
    while remaining:
        root = remaining.pop(0)
        best: int | None = None
        best_distance = math.inf
        for index, other in enumerate(remaining):
            distance = abs(other - np.conj(root))
            if distance < best_distance:
                best, best_distance = index, distance
        if best is None or best_distance >= 1e-6 * max(1.0, abs(root)):
            pairs.append((root,))
            continue
        partner = remaining.pop(best)
        root_is_real = abs(np.imag(root)) <= _REAL_ROOT_TOLERANCE * max(
            1.0,
            abs(root),
        )
        partner_is_real = abs(np.imag(partner)) <= _REAL_ROOT_TOLERANCE * max(
            1.0,
            abs(partner),
        )
        if root_is_real and partner_is_real:
            pairs.append((complex(0.5 * (root.real + partner.real)),))
        else:
            pairs.append((root, partner))
    return pairs


def _complement_candidates(coefficients: np.ndarray) -> list[np.ndarray]:
    r"""Return every spectral factor ``Q`` of ``1 - |P|^2`` in the power basis.

    This is the built-in completion step of Haah's method: it turns the target
    ``P`` into the complementary ``Q_tilde`` whose product decomposition the
    solver performs next.

    Remark:
    From degree 16 onwards the conjugate-root pairing may fail, so the
    practical degree ceiling of the solver is set by this step rather than by
    the peel that follows.

    ``coefficients`` are the ascending complex Chebyshev coefficients of the
    target ``P``, assumed to already follow the four Corollary 8 conditions
    checked by :func:`._complex_solver._validate_complex_qsp_constraints`.
    The unitarity relation is solved in the Chebyshev basis, which avoids the
    catastrophic cancellation of the monomial route at high degree, and
    ``h(u) = |Q(sqrt(u))|^2`` is factored from its ``[0, 1]`` Chebyshev series
    through the colleague matrix of
    :func:`numpy.polynomial.chebyshev.chebroots`.

    The modulus ``|Q|`` is fixed by ``h`` but its phase is not: choosing a
    different representative from a conjugate pair of roots of ``h`` gives
    another spectral factor with the same modulus on ``[0, 1]``.  Every such
    choice is returned, the default representative (first root of each pair)
    first, because only some of them are QSP decomposable.

    The factorisation itself is a plain Fejer-Riesz one: the roots come from
    the colleague matrix without polishing and the conjugate pairs are matched
    by a fixed real-axis tolerance, so the pair count returned by ``chebroots``
    degrades as the roots cluster -- which is the RuntimeError below.

    Raises:
        ValueError: If an even-degree ``P`` has a complement that cannot be odd.
        RuntimeError: If the spectral factorisation of ``h`` breaks down numerically, i.e.
            if the root at ``u = 0`` is missing, if the root multiset does not
            split into the expected number of conjugate pairs, or if the leading
            coefficient of ``h`` is numerically non-positive.

    Warns:
        UserWarning: If the degree reaches :data:`_CONJUGATE_PAIRING_DEGREE_LIMIT`, at
            which the conjugate-root pairing becomes numerically unreliable.

    """
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    degree = coefficients.size - 1
    if degree >= _CONJUGATE_PAIRING_DEGREE_LIMIT:
        warnings.warn(
            "the Haah spectral factorisation is numerically unreliable from "
            f"degree {_CONJUGATE_PAIRING_DEGREE_LIMIT} onwards; at degree "
            f"{degree} the conjugate-root pairing of chebroots may fail and "
            "report a missing QSP-decomposable factor for a valid target",
            UserWarning,
            stacklevel=2,
        )

    abs_squared = _cheb_mul(coefficients, np.conj(coefficients))
    magnitude = max(1.0, float(np.max(np.abs(np.real(abs_squared)))))

    f_even = np.zeros(abs_squared.size, dtype=np.float64)
    f_even[0::2] = -np.real(abs_squared[0::2])
    f_even[0] += 1.0
    h = _divide_even_by_one_minus_x2(f_even)

    while h.size > 1 and abs(h[-1]) < _ZERO_TOLERANCE * magnitude:
        h = h[:-1]
    h_at_zero = float(h @ ((-1.0) ** np.arange(h.size)))
    if degree % 2 == 0 and abs(h_at_zero) > _FEASIBILITY_TOLERANCE * magnitude:
        # Q must be odd for an even-degree P, hence h(u) = u |q(u)|^2.
        raise ValueError(
            "target is not realizable: for even degree Q must be odd, so "
            "(1 - |P|^2)/(1 - x^2) must vanish at x = 0"
        )

    roots = chebroots(h)
    u_roots = (np.real(roots) + 1.0) / 2.0 + 0.5j * np.imag(roots)
    if degree % 2 == 0:
        nearest = int(np.argmin(np.abs(u_roots)))
        if abs(u_roots[nearest]) > 1e-6:
            raise RuntimeError(
                "spectral factorisation failed: h has no root at u = 0"
            )
        u_roots = np.delete(u_roots, nearest)

    pairs = _conjugate_pairs(u_roots)
    expected_order = (degree - 1) // 2
    if len(pairs) != expected_order:
        raise RuntimeError(
            "spectral factorisation failed: expected "
            f"{expected_order} conjugate pairs of roots of h, found {len(pairs)}"
        )

    # Leading monomial coefficient of ``h`` in ``u``: the Chebyshev-xi term
    # ``c_n T_n(2u - 1)`` has leading coefficient ``c_n * 2^(2n - 1)`` for
    # ``n >= 1`` but plain ``c_0`` for ``n = 0``.
    leading = float(h[-1]) * (2.0 ** (2 * h.size - 3) if h.size > 1 else 1.0)
    if leading <= 0.0:
        raise RuntimeError(
            "spectral factorisation failed: the leading coefficient of "
            "(1 - |P|^2)/(1 - x^2) is numerically non-positive"
        )

    # ``Q`` has the parity opposite to ``P``: odd for even degree, even for odd.
    parity_offset = 1 - degree % 2
    candidates: list[np.ndarray] = []
    for bits in itertools.product(*[range(len(pair)) for pair in pairs]):
        chosen = [pairs[index][bit] for index, bit in enumerate(bits)]
        monic = np.asarray(poly.polyfromroots(chosen), dtype=np.complex128)
        q_u = monic * math.sqrt(leading)
        q_power = np.zeros(2 * len(chosen) + 1 + parity_offset, dtype=np.complex128)
        q_power[parity_offset::2] = q_u
        candidates.append(q_power)
    return candidates


# ---------------------------------------------------------------------------
# Response assembly and phase extraction
# ---------------------------------------------------------------------------
def _scalar_laurent_multiply(
    left: dict[int, complex],
    right: dict[int, complex],
) -> dict[int, complex]:
    r"""Return the Laurent product of two scalar series."""
    product: dict[int, complex] = {}
    for left_exponent, left_value in left.items():
        for right_exponent, right_value in right.items():
            exponent = left_exponent + right_exponent
            product[exponent] = (
                product.get(exponent, 0.0) + left_value * right_value
            )
    return product


def _scalar_laurent_polynomial(
    coefficients: np.ndarray,
    variable: dict[int, complex],
) -> dict[int, complex]:
    r"""Return the Laurent series of ``sum_k c_k * variable**k``.

    Coefficients below ``_SERIES_TRIM_TOLERANCE`` are dropped so that the
    exponents that should be empty stay empty in the assembled response.
    """
    result: dict[int, complex] = {0: 0.0}
    for coefficient in reversed(np.asarray(coefficients, dtype=np.complex128)):
        result = _scalar_laurent_multiply(result, variable)
        result[0] = result.get(0, 0.0) + complex(coefficient)
    return {
        key: value
        for key, value in result.items()
        if abs(value) > _SERIES_TRIM_TOLERANCE
    }


def _response_series(
    p_power: np.ndarray,
    q_power: np.ndarray,
) -> _LaurentSeries:
    r"""Assemble ``U_Phi(t)`` as an SU(2) Laurent series in ``t = e^{i theta}``.

    With ``x = cos(theta) = (t + t^{-1})/2`` and
    ``i sin(theta) = (t - t^{-1})/2``, the response matrix is

    .. math::

        U_\Phi(\theta) =
        \begin{pmatrix} P(x) & i \sin\theta\, Q(x) \\
        i \sin\theta\, \bar Q(x) & \bar P(x) \end{pmatrix},

    which is the signal form produced by the R-form phase sequence returned
    by :func:`solve_haah_phases`.
    """
    cosine: dict[int, complex] = {1: 0.5, -1: 0.5}
    imaginary_sine: dict[int, complex] = {1: 0.5, -1: -0.5}

    p_series = _scalar_laurent_polynomial(p_power, cosine)
    q_series = _scalar_laurent_polynomial(q_power, cosine)
    entry_00 = p_series
    entry_01 = _scalar_laurent_multiply(imaginary_sine, q_series)
    entry_10 = _scalar_laurent_multiply(
        imaginary_sine,
        {key: np.conj(value) for key, value in q_series.items()},
    )
    entry_11 = {key: np.conj(value) for key, value in p_series.items()}

    series: _LaurentSeries = {}
    for exponent in set(entry_00) | set(entry_01) | set(entry_10) | set(entry_11):
        series[exponent] = np.asarray(
            [
                [entry_00.get(exponent, 0.0), entry_01.get(exponent, 0.0)],
                [entry_10.get(exponent, 0.0), entry_11.get(exponent, 0.0)],
            ],
            dtype=np.complex128,
        )
    return series


def _decompose_normalized(
    series: _LaurentSeries,
    degree: int,
    *,
    tolerance: float,
) -> tuple[npt.NDArray[np.complex128], list[npt.NDArray[np.complex128]]]:
    r"""Factor an assembled response series into rank-one primitives.

    The R-form assembly scales every coefficient by ``2**-degree`` (one
    factor of ``1/2`` per signal block), while :func:`_decompose_product`
    compares the coefficients against an absolute tolerance.  Rescaling to
    unit maximum coefficient restores the natural scale: the positive factor
    multiplies the constant ``E_0`` and leaves the extracted phases alone.
    """
    scale = max(float(np.max(np.abs(matrix))) for matrix in series.values())
    if scale > 0.0:
        series = {
            exponent: np.asarray(matrix / scale, dtype=np.complex128)
            for exponent, matrix in series.items()
        }
    return _decompose_product(series, degree, tolerance=tolerance)


def _projectors_to_qsp_phases(
    constant: npt.NDArray[np.complex128],
    projectors: list[npt.NDArray[np.complex128]],
) -> np.ndarray:
    r"""Convert the Haah factorisation into the repository R-form phases.

    The constant factor ``E_0`` carries the global phase, and every primitive
    ``E_{P_k} = \exp(i (theta/2) n_k \cdot \sigma)`` carries the half-angle
    rotation about its own axis ``n_k``.  The rotation axis is read off the
    off-diagonal entry of ``P_k``, and the resulting half-angles are
    accumulated into the sandwich

    .. math::

        U_\Phi(\theta) = A_{\phi_0} W(\theta) A_{\phi_1} \cdots
        W(\theta) A_{\phi_d},

    returning the phase sequence ``(\phi_0, \dots, \phi_d)``.
    """
    total = float(np.angle(complex(constant[0, 0])))
    partial = [
        total + 0.5 * float(np.angle(2.0 * complex(projector[0, 1])))
        for projector in projectors
    ]
    phases: list[float] = []
    previous = 0.0
    for value in partial:
        phases.append(value - previous)
        previous = value
    phases.append(total - previous)
    return np.asarray(phases, dtype=np.float64)


def _select_complement(
    p_power: np.ndarray,
    candidates: list[np.ndarray],
    degree: int,
    *,
    tolerance: float,
) -> tuple[npt.NDArray[np.complex128], list[npt.NDArray[np.complex128]]]:
    r"""Return the first spectral factor whose response decomposes.

    Every candidate has the same modulus on ``[-1, 1]``, so all of them
    satisfy the SU(2) unitarity relation; only the QSP-realizable ones factor
    into rank-one primitives.  The candidates are tried in the order given,
    the default representative first.

    Raises:
        RuntimeError: If no spectral factor of ``1 - |P|^2`` decomposes.  This is the
            deterministic analogue of the non-convergence failure of the
            iterative solvers, signalling an ill-conditioned spectral
            factorisation rather than an invalid argument.

    """
    for q_power in candidates:
        try:
            return _decompose_normalized(
                _response_series(p_power, q_power),
                degree,
                tolerance=tolerance,
            )
        except ValueError:
            continue
    raise RuntimeError("no spectral factor of 1 - |P|^2 is QSP-decomposable")


def solve_haah_phases(
    coefficients: Sequence[complex] | np.ndarray,
    *,
    tolerance: float = _PEEL_TOLERANCE,
) -> np.ndarray:
    r"""Return a QSP phase sequence for a complex polynomial via Haah's method.

    Deterministic counterpart of
    :func:`._complex_solver.solve_complex_qsp_phases`: the same target
    specification is turned into the same R-form convention, but the phases
    come from the product decomposition of Ref. [1] instead of a quasi-Newton
    search, so no initial guess, iteration or training grid is involved.

    Args:
        coefficients: Chebyshev-basis coefficients ``(c_0, c_1, ..., c_d)`` in ascending
            order.  Coefficients may be complex.  The implied degree is the
            highest non-zero coefficient index after trimming trailing entries
            smaller than ``1e-14`` in magnitude.
        tolerance: Magnitude below which a coefficient of the assembled response counts
            as zero during the peel.  The default ``1e-10`` sits above the
            rounding floor of the assembly (``~eps * 2**d``) and keeps the peel
            from accepting a wrong spectral factor.

    Returns:
        R-form phase sequence of length ``degree + 1`` (real-valued).  It can
            be passed to :func:`._converters.convert_qsp_phases` to obtain QSVT
            ``W``-form phases.

    Raises:
        TypeError: If coefficients or solver controls have invalid types.
        ValueError: If the target is infeasible, i.e. it violates one of the four
            conditions of Corollary 8; also if the degree is below one or if
            ``tolerance`` is not a positive scalar.
        RuntimeError: If the spectral factorisation of ``1 - |P|^2`` or the subsequent
            decomposition into rank-one primitives fails, e.g. because no
            spectral factor is QSP decomposable.  This is the deterministic
            analogue of the non-convergence failure of the iterative solvers and
            signals an ill-conditioned factorisation rather than a bad argument.

    Warns:
        UserWarning: If the degree reaches :data:`_CONJUGATE_PAIRING_DEGREE_LIMIT`, at
            which the conjugate-root pairing of the spectral factorisation becomes
            numerically unreliable and a valid target may be rejected.

    Note:
        The target must follow Corollary 8 of Ref. [2].  The four conditions are
        checked up front by
        :func:`._complex_solver._validate_complex_qsp_constraints`, the same
        numerical test used by :func:`._complex_solver.solve_complex_qsp_phases`;
        the constructive sufficiency is then established when
        :func:`_complement_candidates` actually finds a spectral factor.
    """
    trimmed = _normalize_complex_coefficients(coefficients)
    peel_tolerance = _normalize_positive_scalar(tolerance, name="tolerance")
    degree = len(trimmed) - 1
    if degree < 1:
        raise ValueError("degree must be at least one")
    _validate_complex_qsp_constraints(trimmed)

    candidates = _complement_candidates(trimmed)
    constant, projectors = _select_complement(
        cheb2poly(trimmed).astype(np.complex128),
        candidates,
        degree,
        tolerance=peel_tolerance,
    )
    return _projectors_to_qsp_phases(constant, projectors)
