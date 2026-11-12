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

"""Amplitude-encoded reciprocal via uniformly controlled Y rotations.

This is the standard HHL "ExactReciprocalGate" construction, which
operates on the following convention:

* The control register stores an integer ``k`` in
  ``[0, 2**num_state_qubits - 1]`` (binary integer layout, LSB first).
* The integer ``k`` is interpreted as the **numerator of the binary
  fraction** ``lambda = k / 2**num_state_qubits``, which is the
  convention HHL phase estimation produces when encoding eigenvalues
  into the control register.
* For every non-zero ``k`` the target qubit's ``|1>`` amplitude after
  application equals ``scaling / lambda = scaling * 2**n / k``, where
  ``n == num_state_qubits``. This is the reciprocal-eigenvalue rotation used
  by the HHL construction.
* ``k == 0`` (zero eigenvalue) is treated as identity (zero rotation,
  ``alpha_0 = 0``).

The validity bound ``scaling * 2**n <= 1`` ensures the largest amplitude
(at the smallest non-zero ``k == 1``) does not exceed ``1``.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from cqlib import Circuit

from ._synthesis import _FunctionalPauliRotation

# ---------------------------------------------------------------------------
# Module-private alpha-table construction (reciprocal only).
# ---------------------------------------------------------------------------


_FPU_TOL = 1e-8


def _validate_amplitude_range(
    values: np.ndarray[Any, Any],
    *,
    atol: float = _FPU_TOL,
) -> None:
    """Raise ``ValueError`` when any entry exceeds ``[-1 - atol, 1 + atol]``.

    This is a reciprocal-scoped helper; it is intentionally kept inside
    this module rather than promoted to the generic ``_synthesis`` package
    because each of the four functional rotations uses slightly different
    conventions for ``arcsin`` clamps.
    """

    max_mag = float(np.max(np.abs(values))) if values.size else 0.0
    if max_mag > 1.0 + atol:
        raise ValueError(
            "Computed amplitudes exceed the arcsin domain [-1, 1]; "
            f"max |f(x)| = {max_mag:.6g} > 1 + {atol:g}.  Reduce the "
            "scaling / adjust parameters before constructing the rotation."
        )


def _compute_reciprocal_alpha(
    num_state_qubits: int,
    scaling: float,
    *,
    neg_vals: bool = False,
) -> np.ndarray[Any, Any]:
    """Compute the UCRY angle table for amplitude-encoded ``scaling / x``.

    The control register ``k`` in ``[0, 2**n - 1]`` is interpreted as an
    integer index ("binary" convention).  The division result lives on the
    integer magnitude and is then normalised by ``2**n`` so that
    ``f(k) = s * 2**n / k``, matching the conventional Exact Reciprocal
    semantics used by the downstream test design.  ``k == 0`` defines
    ``alpha_0 = 0`` (identity rotation).

    When ``neg_vals`` is ``True`` the MSB is a sign bit and the magnitude
    ``m`` is the integer encoded in the lower ``(n - 1)`` bits, giving
    ``|f| = s * 2**(n-1) / m``.  Sign bit zero -> positive, sign bit one
    -> negative amplitude.

    Parameters
    ----------
    num_state_qubits : int
        Control register bit width ``n``.
    scaling : float
        Scale factor ``s``.  Valid unsigned range is
        ``0 <= s <= 1 / 2**n``; with signs enabled the range becomes
        ``0 <= s <= 2 / 2**n``.
    neg_vals : bool, keyword-only
        When ``True`` treat the MSB as a sign bit.  Requires
        ``num_state_qubits >= 2``.

    Returns
    -------
    np.ndarray[Any, Any]
        Alpha table of length ``2**num_state_qubits`` with dtype float64.
    """

    if isinstance(num_state_qubits, bool) or not isinstance(num_state_qubits, int):
        raise TypeError(f"num_state_qubits must be an int, got {num_state_qubits!r}.")
    if num_state_qubits < 1:
        raise ValueError(f"num_state_qubits must be >= 1, got {num_state_qubits!r}.")

    if isinstance(scaling, bool) or not isinstance(scaling, (int, float)):
        raise TypeError(f"scaling must be a finite float, got {scaling!r}.")
    if not math.isfinite(float(scaling)):
        raise ValueError(f"scaling must be finite, got {scaling!r}.")
    s = float(scaling)
    if s < 0.0:
        raise ValueError(f"scaling must be >= 0, got {s!r}.")

    if not isinstance(neg_vals, bool):
        raise TypeError(f"neg_vals must be a bool, got {neg_vals!r}.")

    n = num_state_qubits
    dim = 2**n

    if neg_vals:
        if n < 2:
            raise ValueError(
                "num_state_qubits must be >= 2 when neg_vals=True "
                "(1 bit for sign + 1 bit for magnitude)."
            )
        norm_factor = 2 ** (n - 1)
        if s * norm_factor > 1.0 + _FPU_TOL:
            raise ValueError(
                "With neg_vals=True scaling must satisfy "
                f"s * 2**(n-1) <= 1, got s={s!r} vs "
                f"bound={1.0 / norm_factor:.6g}."
            )
        alpha = np.zeros(dim, dtype=np.float64)
        sign_mask = 1 << (n - 1)
        low_mask = sign_mask - 1
        for k in range(dim):
            magnitude_bits = k & low_mask
            sign_bit = k & sign_mask
            if magnitude_bits == 0:
                alpha[k] = 0.0
                continue
            amp = s * norm_factor / magnitude_bits
            _validate_amplitude_range(np.array([abs(amp)]))
            if sign_bit:
                amp = -amp
            alpha[k] = 2.0 * math.asin(max(-1.0, min(1.0, amp)))
        return alpha

    # Unsigned case.
    norm_factor = 2**n
    if s * norm_factor > 1.0 + _FPU_TOL:
        raise ValueError(
            "scaling must satisfy s * 2**n <= 1.  With "
            f"num_state_qubits={n} the bound is s <= {1.0 / norm_factor:.6g}, "
            f"got s={s!r}."
        )
    alpha = np.zeros(dim, dtype=np.float64)
    for k in range(1, dim):
        amp = s * norm_factor / k
        alpha[k] = 2.0 * math.asin(max(-1.0, min(1.0, amp)))
    return alpha


# ---------------------------------------------------------------------------
# Public class.
# ---------------------------------------------------------------------------


class ExactReciprocal(_FunctionalPauliRotation):
    """Amplitude-encoded reciprocal in the HHL convention.

    **Encoding is always ``mode="amplitude"`` (arcsin encoding)**, i.e.
    the target qubit's ``|1>`` amplitude after application equals
    ``scaling / lambda`` for every non-zero control value, where
    ``lambda = k / 2**num_state_qubits`` is the binary-fraction
    eigenvalue representation that HHL phase estimation produces, and
    ``k`` is the integer value stored in the control register.

    Concretely:

    .. math::

        |k\\rangle |0\\rangle
            \\mapsto
        |k\\rangle
            \\left(
                \\sqrt{1 - \\Big(\\frac{s \\cdot 2^n}{k}\\Big)^2}\\, |0\\rangle
                + \\frac{s \\cdot 2^n}{k}\\, |1\\rangle
            \\right)

    for all non-zero ``k``, and ``|0,0> -> |0,0>`` (identity at the
    zero-eigenvalue pole).  Here ``n == num_state_qubits`` and
    ``s == scaling``.  The bound ``s * 2**n <= 1`` must hold so that
    the maximum amplitude (attained at the smallest non-zero ``k == 1``)
    does not exceed ``1``.

    Local wire layout: ``[control_0, ..., control_{n-1}, result]`` where
    wire ``n`` is the target.

    Parameters
    ----------
    num_state_qubits : int
        Positive integer bit width of the control register.  Must not be
        a ``bool``.
    scaling : float
        Scale factor ``s``.  Must satisfy ``0 <= s <= 1 / 2**n`` when
        ``neg_vals`` is ``False`` (or the equivalent sign-adjusted bound
        ``s <= 2 / 2**n`` when signs are enabled).
    neg_vals : bool, keyword-only
        When ``True`` the MSB of the control register is a sign bit.
        Requires ``num_state_qubits >= 2``.
    axis : Literal["x", "y", "z"], keyword-only
        Pauli rotation axis.  Defaults to ``"y"`` which yields the
        standard amplitude-encoding convention.
    name : str | None, keyword-only
        Optional human-readable label.  Defaults to ``"ExactReciprocal"``.

    Attributes
    ----------
    num_state_qubits : int
        Control-register bit width.
    num_qubits : int
        Total circuit width: ``num_state_qubits + 1``.
    control_register_indices : list[int]
        Local wire indices ``[0, ..., n - 1]``.
    result_qubit_index : int
        Local target wire index ``n``.
    scaling : float
        Stored scale factor.
    neg_vals : bool
        Whether sign handling is enabled.
    axis : Literal["x", "y", "z"]
        Selected Pauli axis.
    name : str
        Configured human-readable name.
    alpha : np.ndarray[Any, Any]
        Evaluated rotation angle table (defensive copy).  Length
        ``2**num_state_qubits``.

    Raises
    ------
    TypeError
        If any parameter has a disallowed type (including ``bool`` for
        integer / scalar parameters).
    ValueError
        If ``num_state_qubits < 1``, ``scaling`` out of range, or
        ``neg_vals`` requires ``num_state_qubits >= 2``.

    Examples
    --------
    >>> from cqlib_algorithms.operators.arithmetic import ExactReciprocal
    >>> n = 2
    >>> s = 1.0 / (2 ** n)          # largest valid unsigned scaling
    >>> rec = ExactReciprocal(n, s)
    >>> circ = rec.build_circuit()
    >>> circ.num_qubits
    3
    """

    _DEFAULT_NAME = "ExactReciprocal"

    _scaling: float
    _neg_vals: bool
    _alpha_cached: npt.NDArray[np.float64]

    def __init__(
        self,
        num_state_qubits: int,
        scaling: float,
        *,
        neg_vals: bool = False,
        axis: Literal["x", "y", "z"] = "y",
        name: str | None = None,
    ) -> None:
        super().__init__(num_state_qubits, axis=axis, name=name)
        if isinstance(scaling, bool) or not isinstance(scaling, (int, float)):
            raise TypeError(f"scaling must be a float, got {scaling!r}.")
        if not math.isfinite(float(scaling)):
            raise ValueError(f"scaling must be finite, got {scaling!r}.")
        if float(scaling) < 0.0:
            raise ValueError(f"scaling must be >= 0, got {scaling!r}.")
        if not isinstance(neg_vals, bool):
            raise TypeError(f"neg_vals must be a bool, got {neg_vals!r}.")
        if neg_vals and num_state_qubits < 2:
            raise ValueError(
                "num_state_qubits must be >= 2 when neg_vals=True; "
                f"got num_state_qubits={num_state_qubits!r}."
            )
        self._scaling = float(scaling)
        self._neg_vals = neg_vals
        self._alpha_cached = _compute_reciprocal_alpha(
            self._num_state_qubits,
            self._scaling,
            neg_vals=self._neg_vals,
        )

    # ------------------------------------------------------------------
    # ExactReciprocal-specific properties.
    # ------------------------------------------------------------------

    @property
    def scaling(self) -> float:
        """Return the stored scale factor ``s``."""

        return self._scaling

    @property
    def neg_vals(self) -> bool:
        """Return whether sign handling is enabled."""

        return self._neg_vals

    # ------------------------------------------------------------------
    # Hooks.
    # ------------------------------------------------------------------

    def _compute_alpha(self) -> npt.NDArray[np.float64]:
        return np.asarray(self._alpha_cached.copy(), dtype=np.float64)

    # ------------------------------------------------------------------
    # Explicit override purely for documentation (super works fine).
    # ------------------------------------------------------------------

    def build_circuit(self) -> Circuit:
        """Return a fresh circuit implementing the reciprocal rotation."""

        return super().build_circuit()
