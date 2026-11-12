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

"""Bit-wise linear Pauli rotation via single-qubit controlled gates.

This module implements a linear-angle functional rotation on an auxiliary
qubit.  For a control register storing an unsigned integer
``k in [0, 2**num_state_qubits - 1]`` (LSB-first indexing), the overall
block-diagonal action on the target qubit is

.. math::

    |k\\rangle |0\\rangle
        \\mapsto
    |k\\rangle \\, R_{\\text{axis}}(s \\cdot k + o) \\, |0\\rangle

where ``s`` is the configured *slope* and ``o`` the configured *offset*.

Angle-addition property of co-axial Pauli rotations is exploited to
decompose the action into exactly ``n`` singly controlled rotation gates
plus one unconditional rotation on the auxiliary qubit.  The construction
is taken from the bit-wise linear payoff encoding of Stamatopoulos et al.
(2020); the resource scaling is ``O(n)`` controlled rotation gates in
contrast with the ``O(2**n)`` Gray-code table method of the uniformly
controlled rotations of Möttönen et al. (2005).

References
----------
[1] N. Stamatopoulos et al., "Option Pricing using Quantum Computers,"
    Quantum, 4, 291, 2020.
    arXiv:1905.02666.
    Relevance: Origin of the bit-wise decomposition used here for linear and
    piecewise-linear payoff functions.
[2] M. Möttönen, J. J. Vartiainen, V. Bergholm, and M. M. Salomaa,
    "Transformation of quantum states using uniformly controlled rotations,"
    Quantum Information and Computation, 5(6), 467–473, 2005.
    arXiv:quant-ph/0407010.
    Relevance: Appendix B of the project design document contrasts the
    bit-wise construction of this module with the UCR table method.
"""

from __future__ import annotations

import math
import warnings
from typing import Literal

import numpy as np
import numpy.typing as npt
from cqlib import Circuit
from cqlib.circuit.gates import StandardGate

# ---------------------------------------------------------------------------
# Module-private numerical helpers.
# ---------------------------------------------------------------------------

_VALID_AXES: frozenset[str] = frozenset({"x", "y", "z"})

_TWO_PI = 2.0 * math.pi


def _validate_axis(axis: str) -> Literal["x", "y", "z"]:
    """Validate that ``axis`` is a supported Pauli rotation axis.

    Parameters
    ----------
    axis : str
        Candidate axis label.  Comparison is case insensitive.

    Returns
    -------
    Literal["x", "y", "z"]
        Lower-case axis string.

    Raises
    ------
    TypeError
        If ``axis`` is not a string.
    ValueError
        If ``axis`` is not one of ``{"x", "y", "z"}``.
    """

    if not isinstance(axis, str):
        raise TypeError(f"axis must be a string, got {axis!r}.")
    lower = axis.lower()
    if lower not in _VALID_AXES:
        raise ValueError(
            f"Unsupported rotation axis {axis!r}.  Expected one of "
            f"{' '.join(sorted(_VALID_AXES))}."
        )
    return lower  # type: ignore[return-value]


def _compute_linear_alpha(
    num_state_qubits: int,
    slope: float,
    offset: float,
) -> tuple[npt.NDArray[np.float64], bool]:
    """Compute the per-control-state rotation angles for a linear map.

    The control register ``k in [0, 2**n - 1]`` is interpreted as an
    unsigned integer (LSB-first bit ordering).  The raw target angle for
    the ``k``-th control basis is ``slope * k + offset``.  Raw angles
    whose absolute value exceeds ``2*pi`` cause the returned warning
    flag to be ``True``; the caller should emit a single ``UserWarning``
    at construction time.  All returned angles are folded into the half
    open interval ``[0, 2*pi)`` via the Euclidean modulo operation.

    Parameters
    ----------
    num_state_qubits : int
        Positive integer bit width ``n`` of the control register.  The
        caller is responsible for the ``isinstance`` / ``bool`` check.
    slope : float
        Linear coefficient ``s``.  Caller guarantees finiteness.
    offset : float
        Additive constant ``o``.  Caller guarantees finiteness.

    Returns
    -------
    tuple[np.ndarray[Any, Any], bool]
        ``(alpha, oversized)`` where ``alpha`` has shape ``(2**n,)`` and
        ``dtype=float64``; ``oversized`` is ``True`` when at least one
        raw angle satisfies ``|raw| > 2*pi``.
    """

    n = int(num_state_qubits)
    dim = 1 << n
    ks = np.arange(dim, dtype=np.float64)
    raw = slope * ks + float(offset)
    oversized = bool(np.any(np.abs(raw) > _TWO_PI))
    mod = np.mod(raw, _TWO_PI)
    return np.asarray(mod, dtype=np.float64), oversized


def _append_axis_rotation(
    circuit: Circuit,
    target: int,
    angle: float,
    *,
    axis: Literal["x", "y", "z"],
) -> None:
    """Append a single-qubit Pauli rotation ``R_axis(angle)``."""

    if axis == "x":
        circuit.rx(target, angle)
    elif axis == "y":
        circuit.ry(target, angle)
    else:  # "z"
        circuit.rz(target, angle)


def _append_controlled_axis_rotation(
    circuit: Circuit,
    control: int,
    target: int,
    angle: float,
    *,
    axis: Literal["x", "y", "z"],
) -> None:
    """Append a singly controlled ``R_axis(angle)`` gate.

    Uses ``StandardGate.<AX>(angle).control(1)`` together with
    ``Circuit.append_mc_gate`` in order to be independent of any
    direct ``crx``/``cry``/``crz`` bindings on the circuit object.
    """

    if axis == "x":
        base = StandardGate.RX(angle)
    elif axis == "y":
        base = StandardGate.RY(angle)
    else:  # "z"
        base = StandardGate.RZ(angle)
    gate = base.control(1)
    circuit.append_mc_gate(gate, [control, target])


# ---------------------------------------------------------------------------
# Public class.
# ---------------------------------------------------------------------------


class LinearPauliRotation:
    r"""Bit-wise linear Pauli rotation on an auxiliary qubit.

    The unsigned integer ``k`` held by the control register (LSB-first
    bit numbering) determines the total rotation applied to the target
    qubit:

    .. math::

        \theta_k = s \cdot k + o \pmod{2\pi}

    where ``s == slope`` and ``o == offset``.  Exploiting the angle
    addition property shared by every one-parameter Pauli rotation
    subgroup, the circuit is built from

    * one unconditional ``R_axis(offset)`` on the target, and
    * ``n`` singly controlled rotations ``CR_axis(s * 2**i)`` with the
      ``i``-th control qubit as control (in order ``i = 0, ..., n-1``).

    Angles larger than ``2*pi`` in magnitude are folded into
    ``[0, 2*pi)`` after emitting a single :class:`UserWarning` at
    construction time.

    Local wire layout is ``[control_0, ..., control_{n-1}, result]``
    where wire ``n`` is the auxiliary (result) qubit.

    Parameters
    ----------
    num_state_qubits : int
        Positive bit width ``n`` of the control register.  Must be a
        real :class:`int`; :class:`bool` is rejected.
    slope : float
        Linear coefficient ``s`` multiplying the control integer.  Must
        be finite.  :class:`bool` is rejected.
    offset : float
        Additive constant ``o`` applied as the unconditional rotation.
        Must be finite.  :class:`bool` is rejected.
    axis : Literal["x", "y", "z"], keyword-only
        Pauli rotation axis.  Case insensitive.  Defaults to ``"y"``.
    name : str | None, keyword-only
        Optional human-readable label.  Defaults to
        ``"LinearPauliRotation"``.

    Attributes
    ----------
    num_state_qubits : int
        Control register bit width.
    num_qubits : int
        Total circuit width: ``num_state_qubits + 1``.
    slope : float
        Configured linear coefficient.
    offset : float
        Configured additive constant.
    axis : Literal["x", "y", "z"]
        Selected rotation axis (lower cased).
    name : str
        Configured label.
    control_register_indices : list[int]
        Local control wire indices ``[0, ..., n - 1]`` (defensive copy).
    result_qubit_index : int
        Local target wire index ``n``.
    alpha : np.ndarray[Any, Any]
        Length ``2**n`` rotation-angle table, folded into ``[0, 2*pi)``
        and returned as a defensive copy.

    Raises
    ------
    TypeError
        If any parameter has a disallowed type (including :class:`bool`
        for integer or scalar parameters).
    ValueError
        If ``num_state_qubits < 1`` or a scalar parameter is not
        finite.

    Examples
    --------
    >>> from cqlib_algorithms.operators.arithmetic import LinearPauliRotation
    >>> op = LinearPauliRotation(2, slope=0.5, offset=0.3)
    >>> circ = op.build_circuit()
    >>> circ.num_qubits
    3
    >>> import numpy as np
    >>> expected = (0.5 * np.arange(4) + 0.3) % (2 * np.pi)
    >>> np.allclose(op.alpha, expected)
    True
    """

    _DEFAULT_NAME = "LinearPauliRotation"

    __slots__ = (
        "_alpha",
        "_axis",
        "_name",
        "_num_state_qubits",
        "_offset",
        "_slope",
    )

    _num_state_qubits: int
    _slope: float
    _offset: float
    _axis: Literal["x", "y", "z"]
    _name: str
    _alpha: npt.NDArray[np.float64]

    def __init__(
        self,
        num_state_qubits: int,
        slope: float,
        offset: float,
        *,
        axis: Literal["x", "y", "z"] = "y",
        name: str | None = None,
    ) -> None:
        if isinstance(num_state_qubits, bool) or not isinstance(num_state_qubits, int):
            raise TypeError(
                f"num_state_qubits must be an int, got {num_state_qubits!r}."
            )
        if num_state_qubits < 1:
            raise ValueError(
                f"num_state_qubits must be >= 1, got {num_state_qubits!r}."
            )

        if isinstance(slope, bool) or not isinstance(slope, (int, float)):
            raise TypeError(f"slope must be a float, got {slope!r}.")
        if not math.isfinite(float(slope)):
            raise ValueError(f"slope must be finite, got {slope!r}.")

        if isinstance(offset, bool) or not isinstance(offset, (int, float)):
            raise TypeError(f"offset must be a float, got {offset!r}.")
        if not math.isfinite(float(offset)):
            raise ValueError(f"offset must be finite, got {offset!r}.")

        self._axis = _validate_axis(axis)

        if name is not None and not isinstance(name, str):
            raise TypeError(f"name must be a string or None, got {name!r}.")

        self._num_state_qubits = int(num_state_qubits)
        self._slope = float(slope)
        self._offset = float(offset)
        self._name = name or self._DEFAULT_NAME

        alpha, oversized = _compute_linear_alpha(
            self._num_state_qubits,
            self._slope,
            self._offset,
        )
        if oversized:
            warnings.warn(
                "One or more raw linear angles slope*k+offset exceed "
                "2*pi in absolute value.  Stored angles are folded "
                "mod 2*pi.  To avoid silent aliasing reduce slope and "
                "offset accordingly.",
                UserWarning,
                stacklevel=2,
            )
        self._alpha = alpha

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
        """Return the stored linear coefficient ``s``."""
        return self._slope

    @property
    def offset(self) -> float:
        """Return the stored additive constant ``o``."""
        return self._offset

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
        basis after applying Euclidean ``mod 2*pi``.  Shape
        ``(2**num_state_qubits,)``, ``dtype=float64``.
        """
        return np.asarray(self._alpha.copy(), dtype=np.float64)

    # ------------------------------------------------------------------
    # Circuit construction.
    # ------------------------------------------------------------------

    def build_circuit(self) -> Circuit:
        """Build a fresh circuit implementing the linear Pauli rotation.

        The returned circuit has width ``num_state_qubits + 1``.  Gates
        are emitted in the following fixed order:

        1. ``R_axis(offset)`` on wire ``result_qubit_index``.
        2. For ``i`` from ``0`` to ``num_state_qubits - 1`` (inclusive),
           a singly controlled gate ``CR_axis(slope * 2**i)`` whose
           control wire is ``i`` and whose target wire is
           ``result_qubit_index``.

        Returns
        -------
        Circuit
            Independent :class:`cqlib.Circuit` instance.
        """

        n = self._num_state_qubits
        target = n
        circuit = Circuit(n + 1)

        const_angle = float(np.mod(self._offset, _TWO_PI))
        _append_axis_rotation(circuit, target, const_angle, axis=self._axis)

        slope = self._slope
        axis = self._axis
        for i in range(n):
            bit_angle = float(np.mod(slope * (1 << i), _TWO_PI))
            _append_controlled_axis_rotation(
                circuit,
                control=i,
                target=target,
                angle=bit_angle,
                axis=axis,
            )

        return circuit
