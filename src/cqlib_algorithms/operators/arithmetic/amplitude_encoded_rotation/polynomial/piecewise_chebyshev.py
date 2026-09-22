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

"""Piecewise-Chebyshev amplitude-encoded Pauli rotation (Y-axis only).

:class:`PiecewiseChebyshev` is a *high-level convenience wrapper* around
:class:`PiecewisePolynomialPauliRotation`.  The user supplies a target
function ``f_x`` and a shared Chebyshev degree ``degree``; the class
performs the completely classical work of

1. completing the user-supplied ``breakpoints`` into a partition of
   ``[0, 2**num_state_qubits)`` (filling the head/tail with "identity"
   segments that encode the full-amplitude state ``|1>``),
2. building a degree-``degree`` Chebyshev interpolant of ``f_x`` on
   every user segment via
   :func:`numpy.polynomial.chebyshev.Chebyshev.interpolate`,
3. converting every interpolant to monomial (power) basis and
   multiplying the coefficients by ``2`` (matching Qiskit's convention
   ``theta_k = 2 * P_d(k)``).

Circuit construction is then *entirely delegated* to
:class:`PiecewisePolynomialPauliRotation` which runs the shared
comparator-driven incremental accumulator on top of the bit-wise
multi-controlled rotation expansion.  No quantum gate is synthesised
directly in this module.

Fixed conventions (matching Qiskit 2.4.1 ``PiecewiseChebyshev``):

* **Encoding**: amplitude-like (``theta_k = 2 * P_d(k)``).  Identity
  segments use ``theta = pi`` so that ``sin(pi/2) = 1`` (full
  ``|1>``-component amplitude).
* **Axis**: Y-axis Pauli rotations (cannot be changed).
* **Reference**: ``f_x(x)`` accepts a ``float`` argument in the
  physical domain ``[lo, hi-1]`` of each segment; interpolation is
  performed at the standard **Chebyshev-Gauss** (zeros-of-``T``)
  nodes *without* integer rounding, matching
  :func:`numpy.polynomial.chebyshev.Chebyshev.interpolate` and
  Qiskit.  This guarantees the classical exponential convergence
  rate for analytic targets.  The resulting polynomial ``P_d`` is
  finally evaluated at the integer control-register indices.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from numpy.polynomial import Chebyshev, Polynomial

from cqlib import Circuit

from ._mc_rotation import (
    _validate_piecewise_breakpoints as _shared_validate_breakpoints,
)
from .piecewise_polynomial_pauli_rotation import PiecewisePolynomialPauliRotation

_PI: float = math.pi
_TWO_PI: float = 2.0 * math.pi
_DEFAULT_NAME = "pw_cheb"


# ---------------------------------------------------------------------------
# Module-private helpers.
# ---------------------------------------------------------------------------


def _infer_num_state_qubits_from_breakpoints(breakpoints: Sequence[int]) -> int:
    """Infer ``n`` from an *incomplete* breakpoint list (max value bit width).

    Raises
    ------
    ValueError
        If ``breakpoints`` is empty.
    """

    bp_list = list(breakpoints)
    if not bp_list:
        raise ValueError(
            "Cannot infer num_state_qubits from an empty breakpoints list."
        )
    max_val = max(bp_list)
    if max_val == 0:
        return 1
    return int(max_val).bit_length()


def _complete_breakpoints(
    breakpoints: list[int],
    num_state_qubits: int,
) -> list[int]:
    """Complete the user breakpoints into a full ``[0, ..., 2**n]`` partition.

    Head identity: if ``breakpoints[0] > 0`` then ``0`` is prepended (the
    prepended segment is later filled with the identity polynomial ``[pi]``).
    Tail identity: if ``breakpoints[-1] < 2**n`` then ``2**n`` is appended
    (same identity fill).  Duplicates are *not* removed silently; the
    downstream validator will raise.
    """

    upper = 1 << int(num_state_qubits)
    completed = list(breakpoints)
    if completed[0] > 0:
        completed.insert(0, 0)
    if completed[-1] < upper:
        completed.append(upper)
    return completed


def _validate_and_normalize_degree(degree: Any) -> int:
    """Validate the Chebyshev degree argument.

    ``None`` is interpreted as the Qiskit default ``degree=1`` (linear).
    Negative degrees, non-int types, and ``bool`` are rejected.
    """

    if degree is None:
        return 1
    if isinstance(degree, bool) or not isinstance(degree, int):
        raise TypeError(
            f"degree must be an int or None, got {degree!r}."
        )
    if degree < 0:
        raise ValueError(
            f"degree must be >= 0, got {degree!r}."
        )
    return int(degree)


def _validate_and_normalize_fx(
    f_x: Any,
) -> Callable[[float], float]:
    """Normalise ``f_x`` into ``callable(float) -> float``.

    A plain numeric ``float`` / ``int`` (but *not* ``bool``) constant is
    promoted to the zero-argument lambda ``lambda _x: float(const)``.

    The callable is required to accept a ``float`` argument because the
    standard Chebyshev-Gauss interpolant samples the target function
    at *floating-point* abscissae (the zeros of ``T_{d+1}``), not only
    at the integer control-register indices.  Callables that accept
    ``int`` also work because Python widens ``int`` to ``float``
    silently.
    """

    if isinstance(f_x, bool):
        raise TypeError(
            f"f_x must be a float or a callable, got bool {f_x!r}."
        )
    if isinstance(f_x, (int, float)):
        const = float(f_x)
        if not math.isfinite(const):
            raise ValueError(
                f"f_x constant must be finite, got {const!r}."
            )
        return cast(
            Callable[[float], float],
            lambda _x, _c=const: _c,
        )
    if not callable(f_x):
        raise TypeError(
            "f_x must be a float (constant) or a callable accepting a "
            f"single float argument, got {type(f_x).__name__}."
        )
    return cast(Callable[[float], float], f_x)


def _validate_and_normalize_breakpoints(
    breakpoints: Any,
) -> list[int]:
    """Validate the user ``breakpoints`` list (before completion).

    Accepts ``None`` (returns the single-segment placeholder ``[0]``), or
    any non-string ``Sequence`` of non-bool ints.  Strict ascending order
    is enforced *before* head/tail completion so we do not silently mask
    user ordering mistakes.
    """

    if breakpoints is None:
        return [0]
    if isinstance(breakpoints, bool) or not isinstance(
        breakpoints, Sequence
    ) or isinstance(breakpoints, (str, bytes)):
        raise TypeError(
            "breakpoints must be a sequence of ints or None, got "
            f"{breakpoints!r}."
        )
    bp_list = list(breakpoints)
    for idx, value in enumerate(bp_list):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"breakpoints[{idx}] must be an int, got {value!r}."
            )
    for idx in range(len(bp_list) - 1):
        if not bp_list[idx + 1] > bp_list[idx]:
            raise ValueError(
                "breakpoints must be strictly ascending; got "
                f"breakpoints[{idx}]={bp_list[idx]!r} and "
                f"breakpoints[{idx + 1}]={bp_list[idx + 1]!r}."
            )
    if bp_list and bp_list[0] < 0:
        raise ValueError(
            "breakpoints must be non-negative (unsigned integer domain), "
            f"got breakpoints[0]={bp_list[0]!r}."
        )
    return bp_list


def _interpolate_segment(
    f_x: Callable[[float], float],
    degree: int,
    lo: int,
    hi: int,
) -> npt.NDArray[np.float64]:
    """Interpolate ``f_x`` with a degree-``degree`` Chebyshev on ``[lo, hi]``.

    The returned coefficient vector is in *monomial* (power) basis,
    constant term first, already multiplied by ``2`` so that the
    downstream ``PolynomialPauliRotation`` layer sees
    ``theta(k) = 2 * P_d(k)`` directly.

    Standard Chebyshev-Gauss nodes (zeros of ``T_{n+1}``) are used
    *without* integer rounding, matching the convention of
    :func:`numpy.polynomial.chebyshev.Chebyshev.interpolate` and
    Qiskit's :class:`~qiskit.circuit.library.PiecewiseChebyshev`.
    Samples are taken at the exact (floating-point) abscissae
    ``cos((2i+1)*pi/(2*(d+1)))``, which guarantees the classical
    exponential convergence rate for analytic targets and monotonic
    error decay in ``degree`` on sufficiently fine grids.
    """

    segment_length = hi - lo
    if segment_length <= 0:
        raise ValueError(
            f"Degenerate segment [{lo}, {hi}); hi must be strictly greater "
            "than lo."
        )

    if degree == 0 or segment_length == 1:
        mid_sample = float(lo)
        mid_val = float(f_x(mid_sample))
        if not math.isfinite(mid_val):
            raise ValueError(
                f"f_x({mid_sample}) is non-finite; Chebyshev interpolation "
                "requires finite target values."
            )
        scaled = 2.0 * mid_val
        return np.asarray([scaled], dtype=np.float64)

    num_nodes = degree + 1
    zeros_ts = np.cos(
        (2 * np.arange(num_nodes) + 1) * np.pi / (2 * num_nodes)
    )
    a = float(lo)
    b = float(hi - 1)
    xs = 0.5 * (a + b) + 0.5 * (b - a) * zeros_ts
    fs = np.empty(num_nodes, dtype=np.float64)
    for idx in range(num_nodes):
        x_val = float(xs[idx])
        val = float(f_x(x_val))
        if not math.isfinite(val):
            raise ValueError(
                f"f_x({x_val}) is non-finite; Chebyshev interpolation "
                "requires finite target values."
            )
        fs[idx] = val

    cheb = Chebyshev.fit(xs, fs, deg=degree, domain=[a, b])
    poly = cheb.convert(kind=Polynomial)
    coeffs_raw = np.asarray(poly.coef, dtype=np.float64, order="C")
    coeffs_scaled = 2.0 * coeffs_raw
    return np.asarray(coeffs_scaled, dtype=np.float64)


def _warn_if_any_coeff_oversized(
    coeffs: list[npt.NDArray[np.float64]],
    breakpoints: list[int],
    num_state_qubits: int,
) -> None:
    """Emit a single :class:`UserWarning` if any ``|theta(k)|`` exceeds ``2*pi``.

    Mirrors the warning policy of :class:`PiecewisePolynomialPauliRotation`
    / :class:`PolynomialPauliRotation`: we do *not* pre-fold coefficients
    at this layer (folding must happen only at the final angle level, and
    Pauli rotations have a ``4*pi`` period so per-gate folding would be
    incorrect).  The warning is purely informational for the user.
    """

    n = int(num_state_qubits)
    dim = 1 << n
    ks = np.arange(dim, dtype=np.float64)
    num_segments = len(breakpoints) - 1
    raw = np.zeros(dim, dtype=np.float64)
    for j in range(num_segments):
        lo = breakpoints[j]
        hi = breakpoints[j + 1]
        mask = (ks >= lo) & (ks < hi)
        c_j = coeffs[j]
        raw[mask] = np.polyval(c_j[::-1], ks[mask])
    if bool(np.any(np.abs(raw) > _TWO_PI)):
        warnings.warn(
            "One or more theta(k) values produced by the piecewise "
            "Chebyshev interpolant exceed 2*pi in absolute value.  The "
            "underlying PiecewisePolynomialPauliRotation stores angles "
            "mod 2*pi.  To avoid silent aliasing scale the target "
            "function f_x or reduce its magnitude.",
            UserWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Public class.
# ---------------------------------------------------------------------------


class PiecewiseChebyshev:
    r"""Piecewise-Chebyshev amplitude-encoded Pauli rotation (Y-axis).

    The target ``f_x`` (a scalar constant or a callable taking a
    ``float`` value in the physical domain of the active segment) is
    approximated on every user-defined segment by a standard
    Chebyshev interpolant of shared degree ``degree`` sampled on the
    exact (non-integer-rounded) **Chebyshev-Gauss** nodes, matching
    the convention of
    :func:`numpy.polynomial.chebyshev.Chebyshev.interpolate`.  Each
    interpolant is converted to monomial basis and multiplied by
    ``2`` so that the downstream
    :class:`PiecewisePolynomialPauliRotation` layer applies the
    Y-axis rotation

    .. math::

        \theta_k \;=\; 2 \, P_d^{(j)}(k), \qquad k \in [b_j,\, b_{j+1}),

    where ``P_d^{(j)}`` is the Chebyshev interpolating polynomial on the
    ``j``-th segment.  Identity segments (auto-filled when the user
    breakpoint list does not span the whole ``[0, 2**n)`` domain) use
    ``theta = pi``, which yields ``sin(pi/2) = 1`` (full ``|1>``
    amplitude).

    The user breakpoint list is *completed* before being forwarded to
    :class:`PiecewisePolynomialPauliRotation`: a leading ``0`` is
    prepended if necessary, and a trailing ``2**n`` is appended if
    necessary.  Completion matches Qiskit's convention.

    Local wire layout is produced entirely by the delegated
    :class:`PiecewisePolynomialPauliRotation` instance, i.e.

        ``[x_0, ..., x_{n-1}, geq_1, ..., geq_{J-1}, result]``

    where ``J`` is the *completed* number of segments (including any
    auto-filled head/tail identity segments).

    Parameters
    ----------
    f_x : float | callable[[float], float]
        Target function.  A plain numeric constant is promoted to
        ``lambda x: const``.  The callable must accept a ``float``
        argument because the standard Chebyshev-Gauss nodes (zeros
        of the Chebyshev polynomials) lie strictly between the
        integer control-register grid points and the segment
        endpoints.  ``bool`` values are explicitly rejected.
    degree : int | None, optional
        Shared Chebyshev interpolant degree.  ``None`` defaults to
        ``1`` (linear interpolant, matching Qiskit).  Must be a
        non-negative :class:`int` (``bool`` rejected).
    breakpoints : list[int] | None, optional
        User breakpoint list (unsigned state indices, strictly
        ascending).  ``None`` is treated as the single-segment
        placeholder ``[0]`` which after completion becomes
        ``[0, 2**n]``.  The constructor automatically prepends ``0``
        and appends ``2**n`` when necessary (head/tail identity
        segments).
    num_state_qubits : int | None, optional
        Control-register bit width ``n``.  If ``None`` it is inferred
        from the largest breakpoint value via ``bit_length``.  When
        both ``breakpoints`` and ``num_state_qubits`` are ``None`` the
        class cannot be constructed and :class:`ValueError` is raised.
    name : str, optional
        Human-readable label forwarded to the underlying
        :class:`PiecewisePolynomialPauliRotation` instance.  Defaults
        to ``"pw_cheb"``.

    Attributes
    ----------
    num_state_qubits : int
        Control-register bit width (after any inference).
    num_segments : int
        Number of *completed* segments (including auto-filled identity
        segments).
    num_qubits : int
        Total circuit width, identical to the delegated
        :class:`PiecewisePolynomialPauliRotation.num_qubits`.
    breakpoints : list[int]
        Defensive copy of the *completed* breakpoint list.
    polynomials : list[numpy.ndarray]
        Defensive copies of the per-segment monomial coefficient
        vectors (constant term first, already scaled by ``2``).
    degree : int
        Shared Chebyshev degree used on every non-identity segment.
    name : str
        Configured label.
    control_register_indices / geq_register_indices / result_qubit_index : list[int] | int
        Forwarded verbatim from the underlying
        :class:`PiecewisePolynomialPauliRotation`.

    Raises
    ------
    TypeError
        Disallowed types including ``bool`` for any scalar / sequence
        entry, or a non-callable, non-float ``f_x``.
    ValueError
        Negative degree, descending / negative breakpoints, non-finite
        ``f_x`` at the interpolation nodes, an empty breakpoint list,
        or inconsistent ``num_state_qubits`` / ``breakpoints``.

    Examples
    --------
    >>> import numpy as np
    >>> from cqlib_algorithms.operators.arithmetic import PiecewiseChebyshev
    >>> # n=3, degree=4, single segment: interpolate sin(pi*x/8).
    >>> op = PiecewiseChebyshev(
    ...     f_x=lambda x: np.sin(np.pi * x / 8.0),
    ...     degree=4,
    ...     breakpoints=None,
    ...     num_state_qubits=3,
    ... )
    >>> circ = op.build_circuit()
    >>> circ.num_qubits  # n + J = 3 + 1
    4
    """

    _DEFAULT_NAME: str = _DEFAULT_NAME

    __slots__ = (
        "_breakpoints",
        "_degree",
        "_delegate",
        "_f_x",
        "_name",
        "_num_state_qubits",
        "_polynomials",
    )

    _num_state_qubits: int
    _breakpoints: list[int]
    _degree: int
    _name: str
    _f_x: Callable[[float], float]
    _polynomials: list[npt.NDArray[np.float64]]
    _delegate: PiecewisePolynomialPauliRotation

    def __init__(
        self,
        f_x: float | Callable[[float], float],
        degree: int | None = None,
        breakpoints: list[int] | None = None,
        num_state_qubits: int | None = None,
        name: str = _DEFAULT_NAME,
    ) -> None:
        # --------------------------------------------------------------
        # 1. Validate & normalise all user inputs independently.
        # --------------------------------------------------------------
        if not isinstance(name, str):
            raise TypeError(
                f"name must be a string, got {name!r}."
            )
        self._name = name
        self._degree = _validate_and_normalize_degree(degree)
        self._f_x = _validate_and_normalize_fx(f_x)
        user_breakpoints = _validate_and_normalize_breakpoints(breakpoints)

        if num_state_qubits is None:
            if not user_breakpoints or (
                len(user_breakpoints) == 1 and user_breakpoints[0] == 0
            ):
                raise ValueError(
                    "num_state_qubits cannot be inferred when breakpoints "
                    "is None or only contains [0].  Please pass an explicit "
                    "num_state_qubits."
                )
            self._num_state_qubits = _infer_num_state_qubits_from_breakpoints(
                user_breakpoints
            )
        else:
            if isinstance(num_state_qubits, bool) or not isinstance(
                num_state_qubits, int
            ):
                raise TypeError(
                    "num_state_qubits must be an int or None, got "
                    f"{num_state_qubits!r}."
                )
            if num_state_qubits < 1:
                raise ValueError(
                    "num_state_qubits must be >= 1, got "
                    f"{num_state_qubits!r}."
                )
            self._num_state_qubits = int(num_state_qubits)

        n = self._num_state_qubits
        upper = 1 << n
        if user_breakpoints:
            for idx, bp in enumerate(user_breakpoints):
                if bp > upper:
                    raise ValueError(
                        f"breakpoints[{idx}]={bp!r} exceeds the "
                        f"unsigned register upper bound 2**n={upper!r} "
                        f"(n={n}).  Either increase num_state_qubits or "
                        "reduce the largest breakpoint."
                    )

        # --------------------------------------------------------------
        # 2. Complete breakpoints (head/tail identity segments).
        # --------------------------------------------------------------
        completed_bp = _complete_breakpoints(user_breakpoints, n)
        validated_bp = _shared_validate_breakpoints(completed_bp, n)
        self._breakpoints = list(validated_bp)
        num_segments = len(validated_bp) - 1

        # --------------------------------------------------------------
        # 3. Work out which segments are user segments vs auto-filled
        #    identity segments.  Identity segments use theta = pi so
        #    that sin(pi/2) = 1 (full |1> amplitude).
        # --------------------------------------------------------------
        identity_poly = np.asarray([_PI], dtype=np.float64)

        polynomials: list[npt.NDArray[np.float64]] = []
        for j in range(num_segments):
            lo = validated_bp[j]
            hi = validated_bp[j + 1]
            user_bp_nonempty = bool(user_breakpoints)
            # An auto-filled identity segment is recognised by BOTH of
            #
            #   (a) the current segment lies *outside* the span of the
            #       user breakpoints (i.e. head side ``lo < ub[0]`` or
            #       tail side ``hi > ub[-1]``) so it is provably a
            #       completion artifact, and
            #   (b) the completed breakpoint list contains *more than*
            #       one segment (``J >= 2``).
            #
            # The single-segment case ``J == 1`` – which occurs for
            # ``breakpoints=None`` (``user_breakpoints == [0]``) and any
            # explicit user partition that already spans ``[0, 2**n)`` –
            # must therefore *never* be mistaken for an identity fill.
            # This is the root cause that previously collapsed any
            # piecewise-Chebyshev with a single user segment into a
            # flat ``π`` rotation.
            if user_bp_nonempty and num_segments >= 2:
                ub0 = int(user_breakpoints[0])
                ub_last = int(user_breakpoints[-1])
                is_head_identity = (j == 0) and (int(hi) <= ub0)
                is_tail_identity = (
                    j == num_segments - 1 and int(lo) >= ub_last
                )
            else:
                is_head_identity = False
                is_tail_identity = False
            if is_head_identity:
                polynomials.append(
                    np.asarray(identity_poly.copy(), dtype=np.float64)
                )
                continue
            if is_tail_identity:
                polynomials.append(
                    np.asarray(identity_poly.copy(), dtype=np.float64)
                )
                continue
            seg_poly = _interpolate_segment(
                self._f_x, self._degree, lo, hi
            )
            polynomials.append(
                np.asarray(seg_poly.copy(), dtype=np.float64)
            )

        _warn_if_any_coeff_oversized(polynomials, validated_bp, n)

        self._polynomials = polynomials

        # --------------------------------------------------------------
        # 4. Delegate the circuit entirely to PiecewisePolynomial.
        # --------------------------------------------------------------
        self._delegate = PiecewisePolynomialPauliRotation(
            n,
            breakpoints=validated_bp,
            coeffs=polynomials,
            axis="y",
            name=name,
        )

    # ------------------------------------------------------------------
    # Read-only properties.
    # ------------------------------------------------------------------

    @property
    def num_state_qubits(self) -> int:
        """Return the control-register bit width ``n``."""
        return self._num_state_qubits

    @property
    def degree(self) -> int:
        """Return the shared Chebyshev interpolant degree."""
        return self._degree

    @property
    def num_segments(self) -> int:
        """Return the completed segment count (incl. identity segments)."""
        return len(self._breakpoints) - 1

    @property
    def num_qubits(self) -> int:
        """Return the total local circuit width."""
        return self._delegate.num_qubits

    @property
    def breakpoints(self) -> list[int]:
        """Return a defensive copy of the completed breakpoint list."""
        return list(self._breakpoints)

    @property
    def polynomials(self) -> list[npt.NDArray[np.float64]]:
        """Return defensive copies of the per-segment monomial coefficients.

        Each vector is constant-term-first and already multiplied by
        ``2``; this is exactly the format consumed by
        :class:`PiecewisePolynomialPauliRotation`.
        """
        return [
            np.asarray(arr.copy(), dtype=np.float64)
            for arr in self._polynomials
        ]

    @property
    def name(self) -> str:
        """Return the configured human-readable label."""
        return self._name

    @property
    def control_register_indices(self) -> list[int]:
        """Forward x-register local indices from the delegate."""
        return list(self._delegate.control_register_indices)

    @property
    def geq_register_indices(self) -> list[int]:
        """Forward geq-flag local indices from the delegate."""
        return list(self._delegate.geq_register_indices)

    @property
    def result_qubit_index(self) -> int:
        """Forward the local target-wire index from the delegate."""
        return int(self._delegate.result_qubit_index)

    # ------------------------------------------------------------------
    # Circuit construction (delegated).
    # ------------------------------------------------------------------

    def build_circuit(self) -> Circuit:
        """Build and return a fresh circuit (delegated to PiecewisePolynomial).

        Returns
        -------
        Circuit
            Independent :class:`cqlib.Circuit` instance with width
            :attr:`num_qubits`.  Identical in layout and behaviour to
            the circuit produced directly by
            :meth:`PiecewisePolynomialPauliRotation.build_circuit`.
        """
        return self._delegate.build_circuit()
