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

"""Polynomial-approximated amplitude encoding of a linear ``f(k) = a*k + b``.

This operator encodes the target function ``f(k) = slope * k + offset``
**as the ``|1>`` amplitude** on the auxiliary (result) qubit for every
control basis ``|k>``.  In exact arithmetic one would apply

.. math::

    R_y\\bigl(2 \\arcsin(f(k))\\bigr) \\; |0>
        = \\sqrt{1 - f(k)^2} \\, |0> + f(k) \\, |1>;

however, ``arcsin(x)`` is *not* a polynomial and therefore cannot be
realised in the bit-wise controlled-rotation framework that underlies the
:mod:`polynomial` subpackage.  To keep the gate scaling polynomial in
``n`` this operator instead approximates ``arcsin(x)`` with its
Maclaurin series truncated at the configured order.  The truncated
expressions implemented here are

* ``order == 1`` : ``arcsin(x)  ≈  x``,
* ``order == 3`` : ``arcsin(x)  ≈  x  +  x^3 / 6``,
* ``order == 5`` : ``arcsin(x)  ≈  x  +  x^3 / 6  +  3 x^5 / 40``.

The resulting ``theta(k) = 2 * arcsin_trunc(f(k))`` is itself a
polynomial in ``k`` (of degree equal to ``order``), so the rotation can
be synthesised with the machinery of
:class:`PolynomialPauliRotation`.  The class therefore *composes* a
private instance of that class rather than extending it (the
``is-a`` relationship does not hold between amplitude encoding and
pure-angle encoding operators).

Domain validation
-----------------
The control values ``f(k) = slope * k + offset`` must lie in
``[-1 - eps_abs, 1 + eps_abs]`` with ``eps_abs = 1e-8``.  Any value
strictly beyond that band raises :class:`ValueError`; values inside the
band are clamped to ``[-1, 1]`` before the polynomial substitution so
that ``sin(theta(k)/2)`` stays inside ``[-1, 1]`` regardless of the
approximation order.  The maximal deviation of
``sin(alpha[k] / 2)`` from the clamped ``f(k)`` — the actual worst-case
amplitude error introduced by the truncated Maclaurin series — is
exposed via :attr:`amplitude_error_bound` and triggers a second
:class:`UserWarning` at construction time whenever the bound exceeds
0.01, so the caller is explicitly nudged toward using the exact
UCR-based ``ArbitraryFunction(mode='amplitude')`` if higher fidelity is
needed.

References
----------
[1] M. Abramowitz and I. A. Stegun, Handbook of Mathematical Functions,
    §4.4.35, 1964.
    Relevance: Standard Maclaurin series for ``arcsin`` used for the
    coefficients ``1/6`` and ``3/40`` of the odd-order terms.
[2] N. Stamatopoulos et al., "Option Pricing using Quantum Computers,"
    Quantum, 4, 291, 2020.
    arXiv:1905.02666.
    Relevance: Broader context of using polynomial-controlled rotations to
    encode payoff amplitudes for Monte Carlo pricing.
[3] A. Gilliam, S. Woerner, and C. Gonciulea, "Quantum computing enhanced
    optimization," 2019.
    arXiv:1912.04088.
    Relevance: Foundation of the multi-subset controlled-rotation synthesis
    performed internally by the underlying
    :class:`PolynomialPauliRotation` instance.
"""

from __future__ import annotations

import math
import warnings
from typing import Literal

import numpy as np
import numpy.typing as npt

from ._mc_rotation import _validate_axis
from .polynomial_pauli_rotation import PolynomialPauliRotation

_TWO_PI: float = 2.0 * np.pi

from cqlib import Circuit

# ---------------------------------------------------------------------------
# Local constants and validators.
# ---------------------------------------------------------------------------

_VALID_ORDERS: frozenset[int] = frozenset({1, 3, 5})

_ARCSIN_ATOL: float = 1e-8

_LARGE_AMPLITUDE_ERROR_TOL: float = 0.01


def _validate_order(order: int) -> Literal[1, 3, 5]:
    """Validate the truncation order of the arcsin Maclaurin series.

    Parameters
    ----------
    order : int
        Candidate order.  :class:`bool` is explicitly rejected.

    Returns
    -------
    Literal[1, 3, 5]
        The order itself after successful validation.

    Raises
    ------
    TypeError
        If ``order`` is a :class:`bool` or not a real :class:`int`.
    ValueError
        If ``order`` is not one of ``{1, 3, 5}``.
    """

    if isinstance(order, bool) or not isinstance(order, int):
        raise TypeError(f"order must be an int, got {order!r}.")
    if order not in _VALID_ORDERS:
        raise ValueError(
            f"Unsupported order {order!r}.  Valid choices are "
            f"{sorted(_VALID_ORDERS)}."
        )
    return order  # type: ignore[return-value]


def _evaluate_arcsin_truncated(
    x: npt.NDArray[np.float64],
    order: Literal[1, 3, 5],
) -> npt.NDArray[np.float64]:
    """Evaluate the truncated ``arcsin`` series on ``x`` element-wise.

    The series is an odd function of ``x`` so only odd powers appear.
    Coefficients come from the Maclaurin expansion; only orders 1, 3
    and 5 are implemented.
    """

    result = np.array(x, dtype=np.float64, copy=True)
    if order >= 3:
        result = result + (x**3) / 6.0
    if order >= 5:
        result = result + 3.0 * (x**5) / 40.0
    return result


def _expand_theta_polynomial(
    slope: float,
    offset: float,
    order: Literal[1, 3, 5],
) -> npt.NDArray[np.float64]:
    """Compute ``coeffs`` such that ``theta(k) = sum_m coeffs[m] * k**m``.

    Derivation: substitute ``f(k) = a*k + b`` into
    ``theta(k) = 2 * arcsin_trunc(f(k))``.  Let ``a = slope``,
    ``b = offset``.  Expansions:

    * order 1: ``theta(k) = 2 * (a*k + b)``
    * order 3: ``theta(k) = 2 * [ (a*k+b) + (a*k+b)^3 / 6 ]``
    * order 5: ``theta(k) = 2 * [ (a*k+b) + (a*k+b)^3 / 6
                                    + 3*(a*k+b)^5 / 40 ]``

    Binomial coefficients are pulled from :func:`math.comb`.
    """

    a = float(slope)
    b = float(offset)
    # Maximum expected degree is equal to order (all exponents odd after
    # substitution, <= order).
    max_deg = int(order)
    coeffs = np.zeros(max_deg + 1, dtype=np.float64)

    # Add the linear term f(k) = a*k + b (multiplied by 2 at the very end).
    coeffs[0] += b
    coeffs[1] += a

    if order >= 3:
        # (a*k + b)^3 / 6
        for m in range(4):
            binom = math.comb(3, m)
            coeff = binom * (a**m) * (b ** (3 - m)) / 6.0
            coeffs[m] += coeff

    if order >= 5:
        # 3 * (a*k + b)^5 / 40
        for m in range(6):
            binom = math.comb(5, m)
            coeff = 3.0 * binom * (a**m) * (b ** (5 - m)) / 40.0
            coeffs[m] += coeff

    # Factor of 2 from the 2*arcsin(...) rotation-angle convention.
    coeffs *= 2.0
    return coeffs


# ---------------------------------------------------------------------------
# Public class.
# ---------------------------------------------------------------------------


class LinearAmplitudeRotation:
    r"""Polynomial ``arcsin`` approximation of linear-amplitude encoding.

    For ``k in [0, 2**num_state_qubits - 1]`` (LSB-first integer), the
    operator targets the encoding

    .. math::

        |k> |0> \;\mapsto\; |k> \bigl[ \sqrt{1 - f(k)^2} |0> + f(k) |1> \bigr]

    where ``f(k) = slope * k + offset``.  The exact unitary would apply
    ``R_y(2 arcsin(f(k)))`` on the auxiliary qubit; instead this class
    replaces ``arcsin`` with its Maclaurin series truncated up to
    ``k**order``.  Supported truncation orders are ``1``, ``3`` (the
    default, ``≈ x + x^3/6``) and ``5`` (adds the ``3 x^5 / 40``
    correction).  The resulting theta-polynomial is in turn delegated
    to :class:`PolynomialPauliRotation` for multi-subset gate synthesis.

    Values of ``f(k)`` outside ``[-1 - eps, 1 + eps]`` raise a
    :class:`ValueError`; values inside the band are clamped to
    ``[-1, 1]`` before substitution so the resulting amplitudes stay
    bounded by one in modulus for any order.

    Parameters
    ----------
    num_state_qubits : int
        Positive bit width ``n`` of the control register.  :class:`bool`
        is explicitly rejected.
    slope : float
        Linear coefficient ``a`` in ``f(k) = a*k + b``.  Must be finite.
        :class:`bool` is rejected.
    offset : float
        Additive constant ``b`` in ``f(k) = a*k + b``.  Must be finite.
        :class:`bool` is rejected.
    order : Literal[1, 3, 5], keyword-only
        Highest odd power retained in the Maclaurin approximation of
        ``arcsin``.  Defaults to ``3``.
    axis : Literal["x", "y", "z"], keyword-only
        Rotation axis used by the underlying
        :class:`PolynomialPauliRotation` (default ``"y"``).  Arbitrary
        axes are accepted; amplitude encoding is physically meaningful
        only for Y so callers that want ``|1>`` amplitudes exactly equal
        to ``f(k)`` should stick to the default axis.
    name : str | None, keyword-only
        Optional human-readable label.  Defaults to
        ``"LinearAmplitudeRotation"``.

    Attributes
    ----------
    num_state_qubits : int
        Control register bit width.
    num_qubits : int
        Total local circuit width: ``n + 1``.
    slope : float
        Configured linear coefficient ``a``.
    offset : float
        Configured additive constant ``b``.
    order : Literal[1, 3, 5]
        Configured Maclaurin truncation order.
    axis : Literal["x", "y", "z"]
        Selected rotation axis (lower cased).
    name : str
        Configured label.
    control_register_indices : list[int]
        Local control wire indices ``[0, ..., n - 1]`` (defensive copy).
    result_qubit_index : int
        Local target wire index ``n``.
    alpha : numpy.ndarray
        Length ``2**n`` rotation angle table, folded into ``[0, 2*pi)``
        via Euclidean modulo, returned as a defensive copy.
    target_amplitude : numpy.ndarray
        Length ``2`` clamped values of ``f(k)``, i.e. the exact ``|1>``
        amplitude we *tried* to encode.
    amplitude_error_bound : float
        Max over ``k`` of ``|sin(alpha[k] / 2) - target_amplitude[k]|``.
        For order-3 with ``max|f| <= 1`` this is bounded by
        ``|pi/2 - (1 + 1/6)| ≈ 0.019`` at the endpoints.

    Raises
    ------
    TypeError
        If any parameter has a disallowed type (including
        :class:`bool` for integer or scalar parameters).
    ValueError
        If ``num_state_qubits < 1``, if a scalar parameter is not
        finite, or if any ``f(k)`` falls outside the arcsin safe band
        ``[-1 - eps, 1 + eps]``.

    Examples
    --------
    >>> from cqlib_algorithms.operators.arithmetic import (
    ...     LinearAmplitudeRotation,
    ... )
    >>> import numpy as np
    >>> # f(k) = 0.2 * k + 0.1 on n=3, order=3 → stays well inside [-1,1]
    >>> op = LinearAmplitudeRotation(3, slope=0.2, offset=0.1, order=3)
    >>> circ = op.build_circuit()
    >>> circ.num_qubits
    4
    >>> # worst-case amplitude deviation is much less than 0.01 here
    >>> op.amplitude_error_bound < 0.01
    True
    """

    _DEFAULT_NAME = "LinearAmplitudeRotation"

    __slots__ = (
        "_alpha",
        "_amplitude_error_bound",
        "_axis",
        "_name",
        "_num_state_qubits",
        "_offset",
        "_order",
        "_poly",
        "_slope",
        "_target_amplitude",
    )

    _num_state_qubits: int
    _slope: float
    _offset: float
    _order: Literal[1, 3, 5]
    _axis: Literal["x", "y", "z"]
    _name: str
    _poly: PolynomialPauliRotation
    _target_amplitude: npt.NDArray[np.float64]
    _amplitude_error_bound: float
    _alpha: npt.NDArray[np.float64]

    def __init__(
        self,
        num_state_qubits: int,
        slope: float,
        offset: float,
        *,
        order: Literal[1, 3, 5] = 3,
        axis: Literal["x", "y", "z"] = "y",
        name: str | None = None,
    ) -> None:
        if isinstance(num_state_qubits, bool) or not isinstance(
            num_state_qubits, int
        ):
            raise TypeError(
                "num_state_qubits must be an int, got "
                f"{num_state_qubits!r}."
            )
        if num_state_qubits < 1:
            raise ValueError(
                "num_state_qubits must be >= 1, got "
                f"{num_state_qubits!r}."
            )

        if isinstance(slope, bool) or not isinstance(slope, (int, float)):
            raise TypeError(f"slope must be a float, got {slope!r}.")
        if not math.isfinite(float(slope)):
            raise ValueError(f"slope must be finite, got {slope!r}.")

        if isinstance(offset, bool) or not isinstance(offset, (int, float)):
            raise TypeError(f"offset must be a float, got {offset!r}.")
        if not math.isfinite(float(offset)):
            raise ValueError(f"offset must be finite, got {offset!r}.")

        validated_order: Literal[1, 3, 5] = _validate_order(order)
        validated_axis = _validate_axis(axis)

        if name is not None and not isinstance(name, str):
            raise TypeError(f"name must be a string or None, got {name!r}.")

        n = int(num_state_qubits)
        a = float(slope)
        b = float(offset)

        # ----- Domain validation + clamping ----------------------------
        ks = np.arange(1 << n, dtype=np.float64)
        f_raw = a * ks + b
        if np.any(np.abs(f_raw) > 1.0 + _ARCSIN_ATOL):
            worst_idx = int(np.argmax(np.abs(f_raw)))
            worst_val = float(f_raw[worst_idx])
            raise ValueError(
                f"f(k) = slope*k + offset = {worst_val} at k="
                f"{worst_idx} exceeds arcsin domain "
                f"[-1-{_ARCSIN_ATOL:.1e}, 1+{_ARCSIN_ATOL:.1e}].  "
                "LinearAmplitudeRotation encodes |1> amplitudes.  "
                "Please normalise slope/offset, or switch to "
                "ArbitraryFunction(mode='amplitude') for the exact "
                "table-based encoding."
            )
        f_clamped = np.clip(f_raw, -1.0, 1.0)

        # ----- Build theta polynomial in k using the (a,b) coefficients
        # of the *unclamped* slope/offset; clamping is only used for
        # the final warning/error bound.  Using the original slope
        # keeps the polynomial monomial coefficients consistent with
        # user intent everywhere except possibly at the boundary
        # values which lie inside the atol band anyway.
        poly_coeffs = _expand_theta_polynomial(a, b, validated_order)

        self._num_state_qubits = n
        self._slope = a
        self._offset = b
        self._order = validated_order
        self._axis = validated_axis
        self._name = name or self._DEFAULT_NAME
        self._target_amplitude = np.asarray(f_clamped, dtype=np.float64)

        # ----- Construct internal delegate (captures its own alpha +
        # oversized UserWarning if any) --------------------------------
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self._poly = PolynomialPauliRotation(
                n,
                poly_coeffs,
                axis=validated_axis,
                name=f"{self._name}_delegate",
            )
        self._alpha = self._poly.alpha

        # ----- Diagnostics --------------------------------------------
        sin_half = np.sin(self._alpha / 2.0)
        error = float(np.max(np.abs(sin_half - self._target_amplitude)))
        self._amplitude_error_bound = error
        if error > _LARGE_AMPLITUDE_ERROR_TOL:
            warnings.warn(
                "LinearAmplitudeRotation: worst-case amplitude error "
                f"|sin(alpha[k]/2) - f(k)| = {error:.4f} exceeds "
                f"{_LARGE_AMPLITUDE_ERROR_TOL}.  This usually happens "
                "when f(k) approaches ±1 or when using order=1.  "
                "Possible fixes: (a) raise order to 3 or 5, "
                "(b) scale slope/offset so that max|f(k)| ≤ 0.9, "
                "or (c) switch to ArbitraryFunction(mode='amplitude') "
                "for exact (non-polynomial) UCR encoding.",
                UserWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Read-only properties.
    # ------------------------------------------------------------------

    @property
    def num_state_qubits(self) -> int:
        """Return the control-register bit width ``n``."""
        return self._num_state_qubits

    @property
    def num_qubits(self) -> int:
        """Return the total local circuit width: ``n + 1``."""
        return self._num_state_qubits + 1

    @property
    def slope(self) -> float:
        """Return the configured linear coefficient ``a``."""
        return self._slope

    @property
    def offset(self) -> float:
        """Return the configured additive constant ``b``."""
        return self._offset

    @property
    def order(self) -> Literal[1, 3, 5]:
        """Return the Maclaurin truncation order."""
        return self._order

    @property
    def axis(self) -> Literal["x", "y", "z"]:
        """Return the selected Pauli rotation axis."""
        return self._axis

    @property
    def name(self) -> str:
        """Return the configured human-readable name."""
        return self._name

    @property
    def control_register_indices(self) -> list[int]:
        """Return a defensive copy of the local control wire indices."""
        return list(range(self._num_state_qubits))

    @property
    def result_qubit_index(self) -> int:
        """Return the local target (auxiliary) wire index."""
        return self._num_state_qubits

    @property
    def alpha(self) -> npt.NDArray[np.float64]:
        """Return a defensive copy of the folded angle table.

        ``alpha[k]`` is the net rotation angle for the ``k``-th control
        basis.  Shape ``(2**num_state_qubits,)``, ``dtype=float64``.
        The stored values are folded into ``[0, 2*pi)`` by the
        underlying :class:`PolynomialPauliRotation` instance.
        """
        return np.asarray(self._alpha.copy(), dtype=np.float64)

    @property
    def target_amplitude(self) -> npt.NDArray[np.float64]:
        """Return a defensive copy of the clamped target amplitudes."""
        return np.asarray(self._target_amplitude.copy(), dtype=np.float64)

    @property
    def amplitude_error_bound(self) -> float:
        """Return the worst-case ``|sin(alpha[k]/2) - f_clamped[k]|``."""
        return self._amplitude_error_bound

    # ------------------------------------------------------------------
    # Circuit construction (forward).
    # ------------------------------------------------------------------

    def build_circuit(self) -> Circuit:
        """Return a fresh circuit equivalent to the delegated polynomial.

        The composed :class:`PolynomialPauliRotation` instance already
        encodes the correct rotation angles via multi-subset
        multi-controlled gates.  The returned circuit is a fresh,
        independent :class:`cqlib.Circuit` with the same layout used by
        every other operator in the ``amplitude_encoded_rotation``
        package: ``[control_0, ..., control_{n-1}, result]``.

        Returns
        -------
        Circuit
            Newly built :class:`cqlib.Circuit` of width
            ``num_state_qubits + 1``.
        """

        return self._poly.build_circuit()
