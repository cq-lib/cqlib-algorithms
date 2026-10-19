r"""Basis and phase-form conversion utilities.

This module collects the small classical conversions that sit between the
polynomial specification and the phase solvers / QSVT circuit builder:

* :func:`polynomial_to_chebyshev` — power-basis to Chebyshev-basis
  (supports both real and complex coefficients),
* :func:`convert_qsp_phases` — full QSP phase sequence to QSVT ``W``-form.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

import numpy as np
import numpy.typing as npt


def polynomial_to_chebyshev(
    coefficients: Sequence[float] | Sequence[complex] | np.ndarray,
) -> npt.NDArray[np.float64 | np.complex128]:
    r"""Convert power-basis polynomial coefficients to the Chebyshev basis.

    Parameters
    ----------
    coefficients
        Coefficients ``(a_0, a_1, ..., a_d)`` of
        ``P(x) = a_0 + a_1 x + ... + a_d x^d`` in ascending order.  May be
        real or complex.

    Returns
    -------
    numpy.ndarray
        Chebyshev-basis coefficients in ascending order.  The dtype is
        ``float64`` when the input is real, ``complex128`` otherwise.
    """
    poly = np.asarray(coefficients).reshape(-1)
    if np.iscomplexobj(poly):
        poly = poly.astype(np.complex128, copy=False)
    else:
        poly = poly.astype(np.float64, copy=False)
    return np.asarray(np.polynomial.chebyshev.poly2cheb(poly))


def convert_qsp_phases(
    phases: Sequence[float] | np.ndarray,
) -> tuple[float, ...]:
    r"""Convert full symmetric QSP phases to QSVT ``W``-form phases.

    The output is the ``W``-form QSVT phase sequence
    ``(phi_1, ..., phi_d)`` derived via the ``R``-form phase equivalence
    established in the proof of Corollary 8 in Ref. [1]:

    .. math::

        \phi_1 = \phi'_0 + \phi'_d + \tfrac{1}{2}(d - 1)\pi,
        \qquad
        \phi_k = \phi'_{k-1} - \tfrac{\pi}{2} \ (2 \le k \le d).

    Parameters
    ----------
    phases
        Full QSP phase sequence ``(phi'_0, ..., phi'_d)``.

    Returns
    -------
    tuple[float, ...]
        Converted QSVT phase sequence ``(phi_1, ..., phi_d)``.

    References
    ----------
    .. [1] Y. Dong, X. Meng, K. B. Whaley, and L. Lin,
       "Efficient phase-factor evaluation in quantum signal
       processing," Physical Review A 103, 042419, 2021.
       arXiv:2002.11649.
    .. [2] A. Gilyen, Y. Su, G. H. Low, and N. Wiebe,
       "Quantum singular value transformation and beyond:
       exponential improvements for quantum matrix arithmetics,"
       Proceedings of the 51st Annual ACM SIGACT Symposium on Theory
       of Computing, pp. 193-204, 2019. arXiv:1806.01838.
    """
    normalized = _normalize_phase_sequence(phases)
    degree = len(normalized) - 1
    if degree == 0:
        return ()

    converted = [
        normalized[0] + normalized[-1] + 0.5 * (degree - 1) * math.pi,
    ]
    converted.extend(phase - 0.5 * math.pi for phase in normalized[1:-1])
    return tuple(converted)


def _normalize_phase_sequence(
    phases: Sequence[float] | np.ndarray,
) -> tuple[float, ...]:
    if isinstance(phases, np.ndarray):
        flattened = phases.reshape(-1).tolist()
    elif isinstance(phases, Sequence) and not isinstance(phases, (str, bytes)):
        flattened = list[float](phases)
    else:
        raise TypeError("phases must be a one-dimensional real sequence")

    if not flattened:
        raise ValueError("phases must not be empty")

    normalized: list[float] = []
    for value in flattened:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("phases must contain real finite scalars")
        scalar = float(value)
        if not math.isfinite(scalar):
            raise TypeError("phases must contain real finite scalars")
        normalized.append(scalar)
    return tuple(normalized)
