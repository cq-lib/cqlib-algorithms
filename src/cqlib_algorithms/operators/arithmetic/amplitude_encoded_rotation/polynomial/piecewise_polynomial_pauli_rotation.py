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

"""Piecewise-polynomial Pauli rotation via incremental accumulation.

Parallel to :class:`PiecewiseLinearPauliRotation` but each segment may hold
an arbitrary degree polynomial
``theta_j(k) = sum_m coeffs[j][m] * k**m``.  Different segments may use
different polynomial degrees; the implementation pads every segment to the
global maximum degree ``d_max`` before computing the incremental
coefficients ``Delta_coeffs[j]`` that actually drive the multi-controlled
rotations.

The shared bit-wise expansion helpers from :mod:`._mc_rotation` are reused
to turn each padded delta segment into a ``(const_term, gammas)`` pair; the
comparator-driven incremental accumulation layer is identical to the linear
case.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
from cqlib import Circuit

from ...comparators.integer_comparator import _build_ripple_geq_comparator
from ._mc_rotation import (
    _append_segmented_axis_rotation as _shared_append_segmented,
)
from ._mc_rotation import (
    _expand_polynomial_gammas as _shared_expand_gammas,
)
from ._mc_rotation import (
    _sorted_gamma_items as _shared_sorted_gamma_items,
)
from ._mc_rotation import (
    _validate_axis as _shared_validate_axis,
)
from ._mc_rotation import (
    _validate_nested_coefficients as _shared_validate_nested_coeffs,
)
from ._mc_rotation import (
    _validate_piecewise_breakpoints as _shared_validate_breakpoints,
)

_TWO_PI: float = 2.0 * np.pi


def _pad_coefficients(
    coeffs: list[npt.NDArray[np.float64]],
    target_degree: int,
) -> list[npt.NDArray[np.float64]]:
    """Right-pad every coefficient vector to ``target_degree + 1`` entries."""

    padded: list[npt.NDArray[np.float64]] = []
    for arr in coeffs:
        current = arr.size - 1
        if current == target_degree:
            padded.append(np.asarray(arr.copy(), dtype=np.float64))
            continue
        extra = target_degree - current
        pad = np.zeros(extra, dtype=np.float64)
        padded.append(np.concatenate([np.asarray(arr, dtype=np.float64), pad]))
    return padded


def _compute_piecewise_polynomial_alpha(
    num_state_qubits: int,
    breakpoints: list[int],
    coeffs: list[npt.NDArray[np.float64]],
) -> tuple[npt.NDArray[np.float64], bool]:
    """Compute the per-control folded angle table for piecewise polynomials.

    The original (non-incremental) per-segment coefficients are evaluated
    directly with :func:`numpy.polyval`; the caller retains responsibility
    for any ``UserWarning`` when ``oversized`` is ``True``.
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
        c_j = coeffs[j]
        raw[mask] = np.polyval(c_j[::-1], ks[mask])
    oversized = bool(np.any(np.abs(raw) > _TWO_PI))
    folded = np.mod(raw, _TWO_PI)
    return np.asarray(folded, dtype=np.float64), oversized


class PiecewisePolynomialPauliRotation:
    r"""Piecewise-polynomial Pauli rotation on an auxiliary qubit.

    The unsigned integer ``k`` held by the control register (LSB-first
    bit numbering) selects one of ``J`` contiguous segments defined by
    strictly ascending breakpoints ``breakpoints = [b_0, ..., b_J]``
    with ``b_0 == 0`` and ``b_J == 2**num_state_qubits``.  When
    ``k in [b_j, b_{j+1})`` the total rotation applied to the target
    qubit is

    .. math::

        \theta_k \;=\; \sum_{m=0}^{d_j} \texttt{coeffs}[j][m] \cdot k^m
                       \pmod{2\pi},

    where ``d_j == len(coeffs[j]) - 1`` is the degree of the ``j``-th
    segment polynomial.  Segments may use different degrees; internally
    every coefficient vector is right-zero-padded to the global maximum
    degree ``d_max`` before the incremental accumulator coefficients
    ``Delta_coeffs[j]`` are produced.  Expansion into multi-controlled
    rotations then follows exactly the multi-subset / multinomial logic
    of :class:`PolynomialPauliRotation`; each segment ``j >= 1`` simply
    prepends an additional ``geq_{j-1}`` control on every rotation.

    When the global maximum degree ``d_max`` exceeds ``n`` a single
    :class:`UserWarning` is emitted mirroring the warning of
    :class:`PolynomialPauliRotation`.

    Local wire layout matches the linear piecewise class (total
    ``num_qubits = n + J`` wires, no helper qubits):

        ``[x_0, ..., x_{n-1}, geq_1, ..., geq_{J-1}, result]``

    Parameters
    ----------
    num_state_qubits : int
        Positive bit width ``n`` of the control register.  Must be a
        real :class:`int`; :class:`bool` is rejected.
    breakpoints : Sequence[int]
        Strictly ascending endpoints with ``b_0 == 0`` and
        ``b_J == 2**num_state_qubits``.
    coeffs : Sequence[Sequence[float] | numpy.ndarray]
        Length ``J``.  ``coeffs[j]`` is the ordered coefficient vector
        (constant term first) for the ``j``-th segment and must be
        non-empty with finite entries; :class:`bool` entries are
        rejected.
    axis : Literal["x", "y", "z"], keyword-only
        Pauli rotation axis.  Defaults to ``"y"``.
    name : str | None, keyword-only
        Optional human-readable label.  Defaults to
        ``"PiecewisePolynomialPauliRotation"``.

    Attributes
    ----------
    num_state_qubits : int
    num_qubits : int
        ``n + J``.
    num_segments : int
        ``J``.
    breakpoints : list[int]
    coeffs : list[numpy.ndarray]
        Defensive copy of the original per-segment coefficients (each
        padded *only* to the segment's own degree; identical to the
        validated inputs).
    max_degree : int
        Global maximum ``max_j (len(coeffs[j]) - 1)``.
    axis : Literal["x", "y", "z"]
    name : str
    control_register_indices / geq_register_indices / result_qubit_index
        Identical layout to :class:`PiecewiseLinearPauliRotation`.
    alpha : numpy.ndarray
        Folded per-control angle table, shape ``(2**n,)``.

    Raises
    ------
    TypeError
        Disallowed types including ``bool`` for scalar / sequence
        entries.
    ValueError
        Length / bound / finiteness violations matching the linear
        piecewise class, plus empty per-segment coefficient vectors.

    Examples
    --------
    >>> from cqlib_algorithms.operators.arithmetic import (
    ...     PiecewisePolynomialPauliRotation,
    ... )
    >>> import numpy as np
    >>> # n=2, J=2: linear on [0,2); quadratic on [2,4).
    >>> op = PiecewisePolynomialPauliRotation(
    ...     2,
    ...     breakpoints=[0, 2, 4],
    ...     coeffs=[[0.1, 0.5], [1.0, -0.3, 0.2]],
    ... )
    >>> circ = op.build_circuit()
    >>> circ.num_qubits  # n + J = 2 + 2
    4
    """

    _DEFAULT_NAME = "PiecewisePolynomialPauliRotation"

    __slots__ = (
        "_alpha",
        "_axis",
        "_breakpoints",
        "_coeffs",
        "_const_terms",
        "_gammas",
        "_max_degree",
        "_name",
        "_num_state_qubits",
    )

    _num_state_qubits: int
    _breakpoints: list[int]
    _coeffs: list[npt.NDArray[np.float64]]
    _max_degree: int
    _const_terms: list[float]
    _gammas: list[dict[frozenset[int], float]]
    _axis: Literal["x", "y", "z"]
    _name: str
    _alpha: npt.NDArray[np.float64]

    def __init__(
        self,
        num_state_qubits: int,
        breakpoints: Sequence[int],
        coeffs: Sequence[Sequence[float] | npt.NDArray[np.float64]],
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

        bp_list = _shared_validate_breakpoints(breakpoints, num_state_qubits)
        num_segments = len(bp_list) - 1
        validated_coeffs = _shared_validate_nested_coeffs(
            coeffs, expected_len=num_segments
        )

        self._axis = _shared_validate_axis(axis)

        if name is not None and not isinstance(name, str):
            raise TypeError(f"name must be a string or None, got {name!r}.")

        n = int(num_state_qubits)
        self._num_state_qubits = n
        self._breakpoints = bp_list
        self._coeffs = [
            np.asarray(c.copy(), dtype=np.float64) for c in validated_coeffs
        ]
        max_degree = max(arr.size - 1 for arr in validated_coeffs)
        self._max_degree = int(max_degree)
        self._name = name or self._DEFAULT_NAME

        padded = _pad_coefficients(validated_coeffs, max_degree)
        delta_coeffs: list[npt.NDArray[np.float64]] = [
            np.asarray(padded[0].copy(), dtype=np.float64)
        ]
        for j in range(1, num_segments):
            delta = padded[j] - padded[j - 1]
            delta_coeffs.append(np.asarray(delta, dtype=np.float64))

        const_terms: list[float] = []
        gammas_list: list[dict[frozenset[int], float]] = []
        any_degree_gt_n = False
        for j in range(num_segments):
            const_term, gammas, degree_gt_n = _shared_expand_gammas(n, delta_coeffs[j])
            const_terms.append(float(const_term))
            gammas_list.append(gammas)
            if degree_gt_n:
                any_degree_gt_n = True
        self._const_terms = const_terms
        self._gammas = gammas_list

        if any_degree_gt_n:
            warnings.warn(
                f"PiecewisePolynomialPauliRotation max_degree d_max="
                f"{self._max_degree} exceeds control register width n={n}.  "
                "The bit-wise expansion remains mathematically correct, "
                "but no gate uses more than n control qubits, so "
                "high-degree monomials are generally wasteful.  Consider "
                "reducing every segment's degree to <= n.",
                UserWarning,
                stacklevel=2,
            )

        alpha, oversized = _compute_piecewise_polynomial_alpha(n, bp_list, self._coeffs)
        if oversized:
            warnings.warn(
                "One or more raw piecewise-polynomial angles exceed "
                "2*pi in absolute value.  Stored angles are folded mod "
                "2*pi.  To avoid silent aliasing reduce the magnitude "
                "of the coefficients accordingly.",
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
    def coeffs(self) -> list[npt.NDArray[np.float64]]:
        """Return defensive copies of the original per-segment coefficients."""
        return [np.asarray(arr.copy(), dtype=np.float64) for arr in self._coeffs]

    @property
    def max_degree(self) -> int:
        """Return the global maximum segment polynomial degree."""
        return self._max_degree

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
        """Return a defensive copy of the folded angle table."""
        return np.asarray(self._alpha.copy(), dtype=np.float64)

    # ------------------------------------------------------------------
    # Circuit construction.
    # ------------------------------------------------------------------

    def _iter_segment_gamma_items(
        self,
        j: int,
    ) -> Iterator[tuple[tuple[int, ...], float]]:
        """Yield canonical sorted gamma items for segment ``j``."""
        yield from _shared_sorted_gamma_items(self._gammas[j])

    def build_circuit(self) -> Circuit:
        """Build a fresh circuit implementing the piecewise polynomial map.

        Gate order is fully deterministic:

        1. **Comparator layer** — identical to
           :meth:`PiecewiseLinearPauliRotation.build_circuit`.
        2. **Rotation layer**, segment by segment ``j = 0, ..., J-1``:

            * Constant term (with extra control ``geq_{j-1}`` for
              ``j >= 1``).
            * Multi-subset controlled rotations in canonical order
              ``(|S| ascending, tuple(sorted(S)) lexicographic)``.  For
              ``j >= 1`` each term's control list is *prepended* with
              ``geq_{j-1}`` so the additional geq qubit always occupies
              the leading position of the control wire tuple (the SDK
              then internally decides the decomposition order, but the
              emitted wire list is stable).

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
        bp_list = self._breakpoints

        circuit = Circuit(num_qubits)

        # --------------------------------------------------------------
        # 1. Comparator layer.
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

            const_angle = float(self._const_terms[j])
            _shared_append_segmented(
                circuit,
                list(extra_ctrl),
                target,
                const_angle,
                axis=axis,
            )

            for controls_tuple, raw_angle in self._iter_segment_gamma_items(j):
                ctrls = list(extra_ctrl) + list(controls_tuple)
                _shared_append_segmented(
                    circuit,
                    ctrls,
                    target,
                    float(raw_angle),
                    axis=axis,
                )

        return circuit
