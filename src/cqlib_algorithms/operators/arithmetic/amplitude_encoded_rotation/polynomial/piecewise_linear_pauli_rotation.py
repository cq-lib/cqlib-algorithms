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

"""Piecewise-linear Pauli rotation via incremental accumulation.

Breakpoints ``breakpoints = [b_0, b_1, ..., b_J]`` partition the unsigned
register domain ``k in [0, 2**num_state_qubits - 1]`` into ``J`` contiguous
segments ``[b_j, b_{j+1})``.  Within each segment ``j`` the rotation angle
applied to the auxiliary (result) qubit is ``slopes[j] * k + offsets[j]``.

The implementation exploits two observations to keep the circuit lean and
composable:

1. **Incremental form** — the piecewise linear map is rewritten as a sum
   over differences applied conditionally at each breakpoint::

       theta(k) = Delta_offset[0] + Delta_slope[0] * k
                + sum_{j=1}^{J-1} 1_{k >= b_j} * (
                    Delta_offset[j] + Delta_slope[j] * k
                )

   where ``Delta_slope[0] == slopes[0]``, ``Delta_offset[0] == offsets[0]``,
   and for ``j >= 1``::

       Delta_slope[j]  = slopes[j]  - slopes[j-1]
       Delta_offset[j] = offsets[j] - offsets[j-1]

   This avoids computing explicit per-segment indicator qubits and the
   associated uncompute passes.

2. **Comparator-wire direct mapping** — each breakpoint flag
   ``geq_j = (k >= b_j)`` is written directly into a dedicated geq qubit by
   reusing the ripple-carry ``x >= value`` builder from
   :mod:`cqlib_algorithms.operators.arithmetic.comparators.integer_comparator`.
   The geq qubits are *not* uncomputed; they only appear as additional
   controls on rotations acting on orthogonal subspaces and therefore
   introduce no computational garbage.

Local wire layout (total ``n + (J-1) + 1 = n + J`` qubits, no helper)::

    [x_0, ..., x_{n-1}, geq_1, ..., geq_{J-1}, result]

Academic references
-------------------
1. Stamatopoulos, Egger, Sun, Zoufal, Iten, Shen, Woerner (2020).
   Option Pricing using Quantum Computers.  *Quantum* 4, 291.
   arXiv:1905.02666.  — Origin of the bit-wise payoff encoding; §3
   describes how piecewise linear payoffs can be implemented by
   combining comparators with linear controlled rotations.
2. Kazlacheuski, Azfar, Malinowski, Rawski (2024).  Efficient explicit
   circuit for quantum state preparation of piece-wise continuous
   functions.  arXiv:2411.01131.  — Systematisation of the
   comparator + oracle superposition pattern.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

from cqlib import Circuit

from ...comparators.integer_comparator import _build_ripple_geq_comparator
from ._mc_rotation import (
    _append_segmented_axis_rotation as _shared_append_segmented,
)
from ._mc_rotation import (
    _validate_axis as _shared_validate_axis,
)
from ._mc_rotation import (
    _validate_float_sequence as _shared_validate_float,
)
from ._mc_rotation import (
    _validate_piecewise_breakpoints as _shared_validate_breakpoints,
)

_TWO_PI: float = 2.0 * np.pi


def _compute_piecewise_linear_alpha(
    num_state_qubits: int,
    breakpoints: list[int],
    slopes: list[float],
    offsets: list[float],
) -> tuple[npt.NDArray[np.float64], bool]:
    """Compute the per-control angle table for a piecewise linear map.

    The alpha table is computed directly by evaluating the original
    piecewise definition on every ``k`` in ``[0, 2**n - 1]``; angles are
    folded into ``[0, 2*pi)`` and the caller is warned if any raw value
    exceeds ``2*pi`` in magnitude.
    """

    n = int(num_state_qubits)
    dim = 1 << n
    ks = np.arange(dim, dtype=np.float64)
    raw = np.zeros(dim, dtype=np.float64)
    num_segments = len(breakpoints) - 1
    for j in range(num_segments):
        lo = breakpoints[j]
        hi = breakpoints[j + 1]
        mask = (ks >= lo) & (ks < hi)
        raw[mask] = float(slopes[j]) * ks[mask] + float(offsets[j])
    oversized = bool(np.any(np.abs(raw) > _TWO_PI))
    folded = np.mod(raw, _TWO_PI)
    return np.asarray(folded, dtype=np.float64), oversized


class PiecewiseLinearPauliRotation:
    r"""Piecewise-linear Pauli rotation on an auxiliary qubit.

    The unsigned integer ``k`` held by the control register (LSB-first
    bit numbering) selects one of ``J`` contiguous segments defined by
    strictly ascending breakpoints ``breakpoints = [b_0, ..., b_J]``
    with ``b_0 == 0`` and ``b_J == 2**num_state_qubits``.  When
    ``k in [b_j, b_{j+1})`` the total rotation applied to the target
    qubit is

    .. math::

        \theta_k \;=\; \texttt{slopes}[j] \cdot k + \texttt{offsets}[j]
                       \pmod{2\pi}.

    The circuit is built in the *incremental accumulation* form (see
    module-level documentation) so that only ``J - 1`` comparator flags
    ``geq_j = (k >= b_j)`` are ever materialised.  Comparator flags
    feed directly as additional controls into each segment's bit-wise
    linear expansion and are never uncomputed.

    Local wire layout is ``[x_0, ..., x_{n-1}, geq_1, ..., geq_{J-1},
    result]`` (total ``num_qubits = n + J`` wires; no helper qubits
    because the ripple-carry geq builder writes into the dedicated geq
    wires directly).

    Parameters
    ----------
    num_state_qubits : int
        Positive bit width ``n`` of the control register.  Must be a
        real :class:`int`; :class:`bool` is rejected.
    breakpoints : Sequence[int]
        Strictly ascending endpoints ``[b_0, ..., b_J]`` with
        ``b_0 == 0`` and ``b_J == 2**num_state_qubits``.  :class:`bool`
        entries are rejected.
    slopes : Sequence[float]
        Linear coefficients for the ``J`` segments (length ``J``).
        Every element must be finite.  :class:`bool` entries are
        rejected.
    offsets : Sequence[float]
        Additive constants for the ``J`` segments (length ``J``).
        Constraints match ``slopes``.
    axis : Literal["x", "y", "z"], keyword-only
        Pauli rotation axis.  Case insensitive.  Defaults to ``"y"``.
    name : str | None, keyword-only
        Optional human-readable label.  Defaults to
        ``"PiecewiseLinearPauliRotation"``.

    Attributes
    ----------
    num_state_qubits : int
        Control register bit width.
    num_qubits : int
        Total circuit width: ``n + J``.
    num_segments : int
        Number of segments ``J == len(breakpoints) - 1``.
    breakpoints : list[int]
        Defensive copy of the configured breakpoints.
    slopes : numpy.ndarray
        Defensive copy, shape ``(J,)``, ``dtype=float64``.
    offsets : numpy.ndarray
        Defensive copy, shape ``(J,)``, ``dtype=float64``.
    axis : Literal["x", "y", "z"]
        Selected rotation axis (lower cased).
    name : str
        Configured label.
    control_register_indices : list[int]
        Local x-register wire indices ``[0, ..., n - 1]``.
    geq_register_indices : list[int]
        Local comparator-flag wire indices ``[n, ..., n + J - 2]``.
        Returns ``[]`` when ``J == 1`` (no interior breakpoints).
    result_qubit_index : int
        Local target wire index ``n + (J - 1)``.
    alpha : numpy.ndarray
        Per-control folded rotation angle table, shape ``(2**n,)``.

    Raises
    ------
    TypeError
        If any parameter has a disallowed type (including
        :class:`bool` for integer / scalar entries).
    ValueError
        If ``num_state_qubits < 1``, breakpoints / lengths / bounds do
        not match the documented contract, any scalar is non-finite, or
        the axis is unknown.

    Examples
    --------
    >>> from cqlib_algorithms.operators.arithmetic import (
    ...     PiecewiseLinearPauliRotation,
    ... )
    >>> import numpy as np
    >>> # n=2, J=2: flat on [0,2); ramp on [2,4).
    >>> op = PiecewiseLinearPauliRotation(
    ...     2,
    ...     breakpoints=[0, 2, 4],
    ...     slopes=[0.5, -0.3],
    ...     offsets=[0.1, 1.0],
    ... )
    >>> circ = op.build_circuit()
    >>> circ.num_qubits  # n + J = 2 + 2
    4
    >>> ks = np.arange(4)
    >>> expected = np.where(
    ...     ks < 2, 0.5 * ks + 0.1, -0.3 * ks + 1.0
    ... ) % (2 * np.pi)
    >>> np.allclose(op.alpha, expected)
    True
    """

    _DEFAULT_NAME = "PiecewiseLinearPauliRotation"

    __slots__ = (
        "_alpha",
        "_axis",
        "_breakpoints",
        "_delta_offsets",
        "_delta_slopes",
        "_name",
        "_num_state_qubits",
        "_offsets",
        "_slopes",
    )

    _num_state_qubits: int
    _breakpoints: list[int]
    _slopes: npt.NDArray[np.float64]
    _offsets: npt.NDArray[np.float64]
    _delta_slopes: list[float]
    _delta_offsets: list[float]
    _axis: Literal["x", "y", "z"]
    _name: str
    _alpha: npt.NDArray[np.float64]

    def __init__(
        self,
        num_state_qubits: int,
        breakpoints: Sequence[int],
        slopes: Sequence[float],
        offsets: Sequence[float],
        *,
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

        bp_list = _shared_validate_breakpoints(breakpoints, num_state_qubits)
        num_segments = len(bp_list) - 1
        slope_list = _shared_validate_float(
            slopes, "slopes", expected_len=num_segments
        )
        offset_list = _shared_validate_float(
            offsets, "offsets", expected_len=num_segments
        )

        self._axis = _shared_validate_axis(axis)

        if name is not None and not isinstance(name, str):
            raise TypeError(f"name must be a string or None, got {name!r}.")

        n = int(num_state_qubits)
        self._num_state_qubits = n
        self._breakpoints = bp_list
        self._slopes = np.asarray(slope_list, dtype=np.float64)
        self._offsets = np.asarray(offset_list, dtype=np.float64)
        self._name = name or self._DEFAULT_NAME

        delta_slopes: list[float] = [float(slope_list[0])]
        delta_offsets: list[float] = [float(offset_list[0])]
        for j in range(1, num_segments):
            delta_slopes.append(
                float(slope_list[j]) - float(slope_list[j - 1])
            )
            delta_offsets.append(
                float(offset_list[j]) - float(offset_list[j - 1])
            )
        self._delta_slopes = delta_slopes
        self._delta_offsets = delta_offsets

        alpha, oversized = _compute_piecewise_linear_alpha(
            n, bp_list, slope_list, offset_list
        )
        if oversized:
            warnings.warn(
                "One or more raw piecewise-linear angles exceed 2*pi "
                "in absolute value.  Stored angles are folded mod 2*pi.  "
                "To avoid silent aliasing reduce the magnitude of the "
                "slopes / offsets accordingly.",
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
    def num_segments(self) -> int:
        """Return the number of segments ``J``."""
        return len(self._breakpoints) - 1

    @property
    def num_qubits(self) -> int:
        """Return the total local circuit width ``n + J``."""
        return self._num_state_qubits + self.num_segments

    @property
    def breakpoints(self) -> list[int]:
        """Return a defensive copy of the breakpoints list."""
        return list(self._breakpoints)

    @property
    def slopes(self) -> npt.NDArray[np.float64]:
        """Return a defensive copy of the per-segment slopes array."""
        return np.asarray(self._slopes.copy(), dtype=np.float64)

    @property
    def offsets(self) -> npt.NDArray[np.float64]:
        """Return a defensive copy of the per-segment offsets array."""
        return np.asarray(self._offsets.copy(), dtype=np.float64)

    @property
    def axis(self) -> Literal["x", "y", "z"]:
        """Return the selected Pauli rotation axis (lower cased)."""
        return self._axis

    @property
    def name(self) -> str:
        """Return the configured human-readable name."""
        return self._name

    @property
    def control_register_indices(self) -> list[int]:
        """Return a defensive copy of the x-register local indices."""
        return list(range(self._num_state_qubits))

    @property
    def geq_register_indices(self) -> list[int]:
        """Return a defensive copy of the geq-flag local indices."""
        n = self._num_state_qubits
        num_geq = self.num_segments - 1
        return list(range(n, n + num_geq))

    @property
    def result_qubit_index(self) -> int:
        """Return the local target (auxiliary) wire index."""
        return self._num_state_qubits + (self.num_segments - 1)

    @property
    def alpha(self) -> npt.NDArray[np.float64]:
        """Return a defensive copy of the folded angle table.

        ``alpha[k]`` is the net rotation for control basis ``k`` after
        Euclidean ``mod 2*pi`` folding.  Shape ``(2**n,)``,
        ``dtype=float64``.
        """
        return np.asarray(self._alpha.copy(), dtype=np.float64)

    # ------------------------------------------------------------------
    # Circuit construction.
    # ------------------------------------------------------------------

    def build_circuit(self) -> Circuit:
        """Build a fresh circuit implementing the piecewise linear map.

        Gate order is fully deterministic:

        1. **Comparator layer** — for each interior breakpoint
           ``b_j`` (``j = 1, ..., J-1``) the ripple-carry ``x >= b_j``
           builder writes its flag into dedicated wire ``geq_{j-1}``
           (local index ``n + j - 1``).  Comparators are applied in
           ascending breakpoint order so repeated X flips applied by
           the ripple builder always commute trivially across disjoint
           flag wires.
        2. **Rotation layer** — for ``j = 0, ..., J-1`` in order:

            * ``j == 0`` (base segment): unconditional
              ``R_axis(Delta_offset[0])``, then for ``i = 0..n-1`` a
              singly controlled ``CR_axis(Delta_slope[0] * 2**i)`` with
              control wire ``x_i``.
            * ``j >= 1`` (incremental segment): the same offset/slope
              template is used, but **every rotation is prepended with
              an extra control on ``geq_{j-1}``**.  The constant term
              therefore becomes a 1-controlled gate and each bit term
              becomes a 2-controlled gate ``(geq_{j-1}, x_i)``.

        Returns
        -------
        Circuit
            Independent :class:`cqlib.Circuit` instance with width
            ``num_qubits``.
        """

        n = self._num_state_qubits
        J = self.num_segments
        num_qubits = self.num_qubits
        target = self.result_qubit_index
        axis = self._axis
        x_indices = list(range(n))
        delta_slopes = self._delta_slopes
        delta_offsets = self._delta_offsets
        bp_list = self._breakpoints

        circuit = Circuit(num_qubits)

        # --------------------------------------------------------------
        # 1. Comparator layer: x >= b_j  ->  geq_{j-1}.
        # --------------------------------------------------------------
        for j in range(1, J):
            bp_value = bp_list[j]
            geq_wire = n + (j - 1)
            _build_ripple_geq_comparator(
                circuit,
                x_indices,
                geq_wire,
                bp_value,
            )

        # --------------------------------------------------------------
        # 2. Rotation layer in incremental form.
        # --------------------------------------------------------------
        for j in range(J):
            extra_ctrl: list[int]
            if j == 0:
                extra_ctrl = []
            else:
                extra_ctrl = [n + (j - 1)]

            const_angle = float(delta_offsets[j])
            _shared_append_segmented(
                circuit,
                list(extra_ctrl),
                target,
                const_angle,
                axis=axis,
            )

            slope_j = float(delta_slopes[j])
            for i in range(n):
                bit_angle = slope_j * (1 << i)
                ctrls = list(extra_ctrl) + [i]
                _shared_append_segmented(
                    circuit,
                    ctrls,
                    target,
                    bit_angle,
                    axis=axis,
                )

        return circuit
