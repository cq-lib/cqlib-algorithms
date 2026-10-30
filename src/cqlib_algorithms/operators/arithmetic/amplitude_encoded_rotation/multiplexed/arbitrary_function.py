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

"""Arbitrary multiplexed function table (amplitude- or angle-encoded).

Given an explicit real-valued per-control table ``f(k)`` of length
``2**num_state_qubits`` (one entry for each integer control value
``k in [0, 2**n - 1]``, LSB-first bit ordering), this module builds the
corresponding uniformly controlled rotation.  Two encoding modes are
supported:

* ``mode="amplitude"`` — the function values are interpreted as
  ``|1>``-component amplitudes, so the target rotation angles are
  ``alpha_k = 2 * arcsin(clamp(f(k), -1, 1))``.  Values strictly
  outside ``[-1 - atol, 1 + atol]`` raise :class:`ValueError`; values
  within the tolerance band are clamped to the domain boundary.
* ``mode="angle"`` — the function values are passed directly as
  rotation angles.  If any entry has magnitude strictly greater than
  ``2*pi``, a :class:`UserWarning` is emitted and all angles are
  folded into ``[0, 2*pi)`` via the Euclidean modulo operation.

The implementation reuses the axis-generic base class
:class:`_FunctionalPauliRotation` from :mod:`._synthesis` and the
three concrete ``UCRX``/``UCRY``/``UCRZ`` gates; it does not duplicate
any Gray-code synthesis logic.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from cqlib import Circuit

from ._synthesis import _FunctionalPauliRotation

# ---------------------------------------------------------------------------
# Module-private helpers.
# ---------------------------------------------------------------------------

_ARBITRARY_TOL = 1e-8
_TWO_PI = 2.0 * math.pi
_VALID_MODES: frozenset[str] = frozenset({"amplitude", "angle"})


def _validate_mode(mode: str) -> Literal["amplitude", "angle"]:
    """Validate ``mode`` and return a lower-cased enum-compatible string.

    Parameters
    ----------
    mode : str
        User-supplied mode.

    Returns
    -------
    Literal["amplitude", "angle"]

    Raises
    ------
    TypeError
        If ``mode`` is not a string.
    ValueError
        If ``mode`` is neither ``"amplitude"`` nor ``"angle"``.
    """

    if not isinstance(mode, str):
        raise TypeError(f"mode must be a string, got {mode!r}.")
    lower = mode.lower()
    if lower not in _VALID_MODES:
        raise ValueError(
            f"Unsupported mode {mode!r}.  Expected one of "
            f"{' '.join(sorted(_VALID_MODES))}."
        )
    return lower  # type: ignore[return-value]


def _validate_values_sequence(
    values: Sequence[float] | np.ndarray[Any, Any],
    num_state_qubits: int,
) -> np.ndarray[Any, Any]:
    """Shape-check and copy the user-provided function table.

    Parameters
    ----------
    values : Sequence[float] | np.ndarray
        User-supplied function table.
    num_state_qubits : int
        Expected length ``2**num_state_qubits``.

    Returns
    -------
    np.ndarray
        One-dimensional ``float64`` copy of the table with length
        ``2**num_state_qubits``.

    Raises
    ------
    TypeError / ValueError
        For wrong type, non-finite entries, wrong dimension, or wrong
        length.
    """

    if isinstance(values, (str, bytes)) or not isinstance(
        values, (Sequence, np.ndarray)
    ):
        raise TypeError(
            "values must be a sequence of floats, got "
            f"{type(values).__name__}."
        )
    if isinstance(values, np.ndarray):
        arr = np.asarray(values, dtype=np.float64)
    else:
        arr = np.asarray(list(values), dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("values must be a one-dimensional array.")
    expected_len = 2**num_state_qubits
    if arr.size != expected_len:
        raise ValueError(
            "values length must equal 2**num_state_qubits "
            f"(expected {expected_len}, got {arr.size})."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("values must contain only finite floats.")
    return arr


def _compute_arbitrary_alpha(
    values: np.ndarray[Any, Any],
    mode: Literal["amplitude", "angle"],
    *,
    atol: float = _ARBITRARY_TOL,
) -> tuple[npt.NDArray[np.float64], bool]:
    """Convert the function table ``values`` into a UCR alpha table.

    Parameters
    ----------
    values : np.ndarray
        Sanitised one-dimensional float array as produced by
        :func:`_validate_values_sequence`.
    mode : Literal["amplitude", "angle"]
        Selected encoding mode.
    atol : float, keyword-only
        Absolute tolerance for ``amplitude`` clamps.

    Returns
    -------
    tuple[np.ndarray, bool]
        ``(alpha, oversized)`` where ``alpha`` is the folded/clamped
        rotation angle table of shape ``values.shape`` and
        ``dtype=float64``; ``oversized`` is ``True`` only in
        ``mode="angle"`` when at least one raw value had magnitude
        strictly larger than ``2*pi``.

    Raises
    ------
    ValueError
        If ``mode=="amplitude"`` and any entry is strictly outside
        ``[-1 - atol, 1 + atol]``.
    """

    if mode == "amplitude":
        max_mag = float(np.max(np.abs(values)))
        if max_mag > 1.0 + atol:
            raise ValueError(
                "Function values exceed the amplitude-encoding domain "
                f"[-1, 1]; max |f(k)| = {max_mag:.6g} > 1 + {atol:.3g}.  Reduce the "
                "function magnitude or use mode=\"angle\" instead."
            )
        clamped = np.clip(values, -1.0, 1.0)
        return np.asarray(2.0 * np.arcsin(clamped), dtype=np.float64), False

    # mode == "angle"
    oversized = bool(np.any(np.abs(values) > _TWO_PI))
    folded = np.mod(values, _TWO_PI)
    return np.asarray(folded, dtype=np.float64), oversized


# ---------------------------------------------------------------------------
# Public class.
# ---------------------------------------------------------------------------


class ArbitraryFunction(_FunctionalPauliRotation):
    r"""Multiplexed rotation from an explicit per-control function table.

    For every integer control value
    ``k in [0, 2**num_state_qubits - 1]`` (LSB-first), the target
    qubit is rotated by ``alpha_k`` where

    * ``mode="amplitude"``:
      ``alpha_k = 2 * arcsin(clamp(f(k), -1, 1))``.
    * ``mode="angle"``:
      ``alpha_k = f(k)  (mod  2*pi)`` with a :class:`UserWarning` if
      any ``|f(k)|`` exceeds ``2*pi``.

    Local wire layout is ``[control_0, ..., control_{n-1}, result]``
    where wire ``n`` is the auxiliary (result) qubit.  The underlying
    circuit is built via the axis-matching ``UCRX``/``UCRY``/``UCRZ``
    Gray-code decomposition from :mod:`._synthesis`.

    Parameters
    ----------
    num_state_qubits : int
        Positive bit width ``n`` of the control register.  Must be a
        real :class:`int`; :class:`bool` is rejected.
    values : Sequence[float] | np.ndarray[Any, Any]
        Length ``2**n`` real-valued function table.  Entry ``values[k]``
        corresponds to the unsigned integer ``k`` stored in the control
        register (LSB-first bit numbering).  ``nan``/``inf`` are
        rejected.
    mode : Literal["amplitude", "angle"], keyword-only
        Encoding convention for ``values``.  Case insensitive.
        Defaults to ``"amplitude"``.
    axis : Literal["x", "y", "z"], keyword-only
        Pauli rotation axis.  Case insensitive.  Defaults to ``"y"``.
    name : str | None, keyword-only
        Optional human-readable label.  Defaults to
        ``"ArbitraryFunction"``.

    Attributes
    ----------
    num_state_qubits : int
        Control-register bit width.
    num_qubits : int
        Total circuit width: ``num_state_qubits + 1``.
    values : np.ndarray[Any, Any]
        Defensive copy of the user-provided function table.
    mode : Literal["amplitude", "angle"]
        Stored encoding mode.
    axis : Literal["x", "y", "z"]
        Selected Pauli axis (lower cased).
    name : str
        Configured human-readable label.
    control_register_indices : list[int]
        Local control wire indices ``[0, ..., n - 1]`` (defensive copy).
    result_qubit_index : int
        Local target wire index ``n``.
    alpha : np.ndarray[Any, Any]
        Length ``2**n`` rotation-angle table as passed to the UCR
        synthesis.  Returned as a defensive copy with ``dtype=float64``.

    Raises
    ------
    TypeError
        If any parameter has a disallowed type (including
        :class:`bool` for the integer parameter).
    ValueError
        If ``num_state_qubits < 1``, the table has the wrong length,
        contains non-finite entries, or ``mode="amplitude"`` receives
        values strictly outside ``[-1 - atol, 1 + atol]``.

    Examples
    --------
    >>> import numpy as np
    >>> from cqlib_algorithms.operators.arithmetic import ArbitraryFunction
    >>> n = 2
    >>> fs = np.array([0.0, 0.25, 0.5, 0.75])
    >>> op = ArbitraryFunction(n, fs)           # amplitude mode, Y axis
    >>> circ = op.build_circuit()
    >>> circ.num_qubits
    3
    """

    _DEFAULT_NAME = "ArbitraryFunction"

    _mode: Literal["amplitude", "angle"]
    _values: npt.NDArray[np.float64]
    _alpha_cached: npt.NDArray[np.float64]

    def __init__(
        self,
        num_state_qubits: int,
        values: Sequence[float] | np.ndarray[Any, Any],
        *,
        mode: Literal["amplitude", "angle"] = "amplitude",
        axis: Literal["x", "y", "z"] = "y",
        name: str | None = None,
    ) -> None:
        super().__init__(num_state_qubits, axis=axis, name=name)
        self._mode = _validate_mode(mode)
        values_arr = _validate_values_sequence(values, self._num_state_qubits)
        self._values = np.asarray(values_arr.copy(), dtype=np.float64)
        alpha_table, oversized = _compute_arbitrary_alpha(
            self._values, self._mode
        )
        if oversized:
            warnings.warn(
                "One or more raw function values passed to "
                "ArbitraryFunction (mode=\"angle\") exceed 2*pi in "
                "absolute value.  Stored angles have been folded into "
                "[0, 2*pi); this may cause silent aliasing if the input "
                "was intended to be small.",
                UserWarning,
                stacklevel=2,
            )
        self._alpha_cached = alpha_table

    # ------------------------------------------------------------------
    # ArbitraryFunction-specific properties.
    # ------------------------------------------------------------------

    @property
    def values(self) -> npt.NDArray[np.float64]:
        """Return a defensive copy of the input function table."""

        return np.asarray(self._values.copy(), dtype=np.float64)

    @property
    def mode(self) -> Literal["amplitude", "angle"]:
        """Return the encoding mode used by the table."""

        return self._mode

    # ------------------------------------------------------------------
    # Hooks.
    # ------------------------------------------------------------------

    def _compute_alpha(self) -> npt.NDArray[np.float64]:
        return np.asarray(self._alpha_cached.copy(), dtype=np.float64)

    # ------------------------------------------------------------------
    # Explicit override purely for documentation (super works fine).
    # ------------------------------------------------------------------

    def build_circuit(self) -> Circuit:
        """Return a fresh local-numbering multiplexed-rotation circuit."""

        return super().build_circuit()
