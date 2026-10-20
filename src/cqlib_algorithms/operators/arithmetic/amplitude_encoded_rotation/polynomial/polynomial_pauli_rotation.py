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

"""Bit-wise polynomial Pauli rotation via multi-controlled gates.

For a control register holding the unsigned integer
``k in [0, 2**num_state_qubits - 1]`` (LSB-first bit numbering) and a
polynomial coefficient vector ``coeffs`` with ``coeffs[m]`` multiplying
``k**m``, the overall block-diagonal action on an auxiliary (result)
qubit is

.. math::

    |k\\rangle |0\\rangle
        \\mapsto
    |k\\rangle \\, R_{\\text{axis}}(\\theta_k) \\, |0\\rangle,\\qquad
    \\theta_k \\;=\\; \\sum_{m=0}^{d} c_m \\, k^m \\pmod{2\\pi},

where ``d == len(coeffs) - 1`` is the polynomial degree.  Exploiting the
angle-addition property of co-axial Pauli rotations and the identity
``x_i**p == x_i`` for 0/1 variables ``x_i``, each monomial ``k**m`` is
expanded into a sum over m-element *multi-subsets* of the ``n`` bit
positions.  Every such multi-subset contributes a single
``|S|``-fold controlled ``R_axis`` gate which is built by calling
``StandardGate.R{axis}(angle).control(|S|)`` — the same idiom used by
every other operator in the project.  See
``polynomial._mc_rotation._build_mc_pauli_rotation`` for details.

References
----------
[1] N. Stamatopoulos, D. J. Egger, Y. Sun, C. Zoufal, R. Iten, N. Shen,
    and S. Woerner, "Option Pricing using Quantum Computers," Quantum, 4,
    291, 2020.
    arXiv:1905.02666.
    Relevance: Origin of the bit-wise payoff decomposition for linear and
    piecewise-linear rotation functions.
[2] A. Gilliam, S. Woerner, and C. Gonciulea, "Quantum computing enhanced
    optimization," 2019.
    arXiv:1912.04088.
    Published version: Quantum, 5, 428, 2021.
    Relevance: General bit-wise decomposition of quadratic forms and
    higher-degree polynomials using multiple-controlled rotation gates.
[3] M. Möttönen, J. J. Vartiainen, V. Bergholm, and M. M. Salomaa,
    "Transformation of quantum states using uniformly controlled rotations,"
    Quantum Information and Computation, 5(6), 467–473, 2005.
    arXiv:quant-ph/0407010.
    Relevance: Appendix B of the project design document contrasts the method
    of this module with the uniformly controlled rotation table look-up
    strategy; :class:`PolynomialPauliRotation` is the canonical example
    demonstrating that ``O(n**d)`` gates suffice for polynomials.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
from cqlib.circuit.gates import MCGate

from cqlib import Circuit

from ._mc_rotation import (
    _append_axis_rotation as _shared_append_axis_rotation,
)
from ._mc_rotation import (
    _build_mc_pauli_rotation as _shared_build_mc_pauli_rotation,
)
from ._mc_rotation import (
    _compute_polynomial_alpha as _shared_compute_polynomial_alpha,
)
from ._mc_rotation import (
    _expand_polynomial_gammas as _shared_expand_polynomial_gammas,
)
from ._mc_rotation import (
    _sorted_gamma_items as _shared_sorted_gamma_items,
)
from ._mc_rotation import (
    _validate_axis as _shared_validate_axis,
)
from ._mc_rotation import (
    _validate_coefficients as _shared_validate_coefficients,
)

_TWO_PI: float = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Local helper wrappers that delegate to the shared ``_mc_rotation`` module.
#
# These local aliases exist so the public class body in this file remains
# identical to the pre-refactor implementation; a future public base
# class will unify them with the sibling LinearPauliRotation class.
# ---------------------------------------------------------------------------


def _validate_axis(axis: str) -> Literal["x", "y", "z"]:
    return _shared_validate_axis(axis)


def _validate_coefficients(
    coeffs: Sequence[float] | np.ndarray,
) -> npt.NDArray[np.float64]:
    return _shared_validate_coefficients(coeffs)


def _append_axis_rotation(
    circuit: Circuit,
    target: int,
    angle: float,
    *,
    axis: Literal["x", "y", "z"],
) -> None:
    _shared_append_axis_rotation(circuit, target, angle, axis=axis)


def _build_mc_pauli_rotation(
    axis: Literal["x", "y", "z"],
    angle: float,
    num_controls: int,
) -> MCGate:
    return _shared_build_mc_pauli_rotation(axis, angle, num_controls)


def _expand_polynomial_gammas(
    n: int,
    coeffs: npt.NDArray[np.float64],
) -> tuple[float, dict[frozenset[int], float], bool]:
    return _shared_expand_polynomial_gammas(n, coeffs)


def _sorted_gamma_items(
    gammas: dict[frozenset[int], float],
) -> list[tuple[tuple[int, ...], float]]:
    return list(_shared_sorted_gamma_items(gammas))


def _compute_polynomial_alpha(
    n: int,
    coeffs: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], bool]:
    return _shared_compute_polynomial_alpha(n, coeffs)


# ---------------------------------------------------------------------------
# Public class.
# ---------------------------------------------------------------------------


class PolynomialPauliRotation:
    r"""Bit-wise polynomial Pauli rotation on an auxiliary qubit.

    The unsigned integer ``k`` held by the control register (LSB-first
    bit numbering) drives the total rotation applied to the target qubit
    according to

    .. math::

        \theta_k \;=\; \sum_{m=0}^{d} c_m \, k^m \pmod{2\pi}

    where ``d == len(coeffs) - 1``, ``c_m == coeffs[m]``.  Using the
    angle-addition property of co-axial Pauli rotations and the 0/1
    identity ``x_i**p == x_i``, the polynomial is expanded into a sum
    over multi-subsets of the ``n`` bit positions.  Each such set
    ``S`` corresponds to a ``|S|``-fold controlled rotation on the
    target.

    Angles larger than ``2*pi`` in absolute value are folded into
    ``[0, 2*pi)`` after emitting a single :class:`UserWarning` at
    construction time.  A second, independent :class:`UserWarning` is
    emitted when the polynomial degree ``d`` exceeds the register bit
    width ``n``; higher terms never produce sets of size larger than
    ``n``, so the expansion remains correct but the configuration is
    resource inefficient.

    Local wire layout is ``[control_0, ..., control_{n-1}, result]``
    where wire ``n`` is the auxiliary (result) qubit.

    Parameters
    ----------
    num_state_qubits : int
        Positive bit width ``n`` of the control register.  Must be a
        real :class:`int`; :class:`bool` is rejected explicitly.
    coeffs : Sequence[float] | numpy.ndarray
        Coefficient vector ordered by ascending powers: ``coeffs[m]``
        multiplies ``k**m``.  The length must be at least ``1``.
        :class:`bool` elements are rejected.
    axis : Literal["x", "y", "z"], keyword-only
        Pauli rotation axis.  Case insensitive.  Defaults to ``"y"``.
    name : str | None, keyword-only
        Optional human-readable label.  Defaults to
        ``"PolynomialPauliRotation"``.

    Attributes
    ----------
    num_state_qubits : int
        Control register bit width.
    num_qubits : int
        Total circuit width: ``n + 1``.
    coeffs : numpy.ndarray
        1-D float64 array of configured coefficients (defensive copy).
    degree : int
        Highest monomial power represented: ``len(coeffs) - 1``.
    axis : Literal["x", "y", "z"]
        Selected rotation axis (lower cased).
    name : str
        Configured label.
    control_register_indices : list[int]
        Local control wire indices ``[0, ..., n - 1]`` (defensive copy).
    result_qubit_index : int
        Local target wire index ``n``.
    alpha : numpy.ndarray
        Length ``2**n`` rotation-angle table, folded into ``[0, 2*pi)``
        and returned as a defensive copy.

    Raises
    ------
    TypeError
        If any parameter has a disallowed type (including :class:`bool`
        for integer or scalar parameters, or :class:`bool` entries
        inside ``coeffs`` sequences).
    ValueError
        If ``num_state_qubits < 1``.

    Examples
    --------
    >>> from cqlib_algorithms.operators.arithmetic import (
    ...     PolynomialPauliRotation,
    ... )
    >>> import numpy as np
    >>> # Quadratic example: theta(k) = 0.3*k**2 + 0.5*k + 0.2
    >>> op = PolynomialPauliRotation(2, [0.2, 0.5, 0.3])
    >>> circ = op.build_circuit()
    >>> circ.num_qubits
    3
    >>> ks = np.arange(4, dtype=np.float64)
    >>> expected = (0.3 * ks**2 + 0.5 * ks + 0.2) % (2 * np.pi)
    >>> np.allclose(np.sin(op.alpha / 2), np.sin(expected / 2), atol=1e-12)
    True
    """

    _DEFAULT_NAME = "PolynomialPauliRotation"

    __slots__ = (
        "_alpha",
        "_axis",
        "_coeffs",
        "_const_term",
        "_degree",
        "_gammas",
        "_name",
        "_num_state_qubits",
    )

    _num_state_qubits: int
    _coeffs: npt.NDArray[np.float64]
    _degree: int
    _axis: Literal["x", "y", "z"]
    _name: str
    _const_term: float
    _gammas: dict[frozenset[int], float]
    _alpha: npt.NDArray[np.float64]

    def __init__(
        self,
        num_state_qubits: int,
        coeffs: Sequence[float] | np.ndarray,
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

        validated_coeffs = _validate_coefficients(coeffs)
        self._axis = _validate_axis(axis)

        if name is not None and not isinstance(name, str):
            raise TypeError(f"name must be a string or None, got {name!r}.")

        n = int(num_state_qubits)
        self._num_state_qubits = n
        self._coeffs = validated_coeffs
        self._degree = len(validated_coeffs) - 1
        self._name = name or self._DEFAULT_NAME

        const_term, gammas, degree_gt_n = _expand_polynomial_gammas(
            n, validated_coeffs
        )
        self._const_term = const_term
        self._gammas = gammas
        if degree_gt_n:
            warnings.warn(
                f"PolynomialPauliRotation degree d={self._degree} "
                f"exceeds control register width n={n}.  The bit-wise "
                "expansion remains mathematically correct, but no gate "
                "uses more than n control qubits, so high-degree "
                "monomials are generally wasteful.  Consider reducing "
                "degree to <= n.",
                UserWarning,
                stacklevel=2,
            )

        alpha, oversized = _compute_polynomial_alpha(n, validated_coeffs)
        if oversized:
            warnings.warn(
                "One or more raw polynomial angles exceed 2*pi in "
                "absolute value.  Stored angles are folded mod 2*pi.  "
                "To avoid silent aliasing reduce the magnitude of the "
                "coefficients accordingly.",
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
    def coeffs(self) -> npt.NDArray[np.float64]:
        """Return a defensive copy of the coefficient vector."""
        return np.asarray(self._coeffs.copy(), dtype=np.float64)

    @property
    def degree(self) -> int:
        """Return the polynomial degree ``len(coeffs) - 1``."""
        return self._degree

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
        """Return a defensive copy of the folded rotation angle table.

        ``alpha[k]`` is the net rotation angle for the ``k``-th control
        basis after Euclidean ``mod 2*pi`` folding.  Shape
        ``(2**num_state_qubits,)``, ``dtype=float64``.
        """
        return np.asarray(self._alpha.copy(), dtype=np.float64)

    # ------------------------------------------------------------------
    # Circuit construction.
    # ------------------------------------------------------------------

    def build_circuit(self) -> Circuit:
        """Build a fresh circuit implementing the polynomial rotation.

        The returned circuit has width ``num_state_qubits + 1``.  Gates
        are emitted in a fixed canonical order:

        1. ``R_axis(mod2pi(const_term))`` on the result wire.
        2. Controlled rotation gates sorted by
           ``(|S| ascending, tuple(sorted(S)) lexicographically)``.
           Each term ``S`` is implemented as a ``|S|``-fold controlled
           ``R_axis(mod2pi(gammas[S]))`` via
           ``StandardGate.R{axis}(angle).control(|S|)`` (see
           ``_mc_rotation._build_mc_pauli_rotation``); the angle is folded
           modulo ``2*pi`` before construction so that the SDK receives
           the minimal equivalent parameter.

        Returns
        -------
        Circuit
            Independent :class:`cqlib.Circuit` instance.
        """

        n = self._num_state_qubits
        target = n
        circuit = Circuit(n + 1)
        axis = self._axis

        # Unconditional constant term.
        const_angle = float(self._const_term)
        _append_axis_rotation(circuit, target, const_angle, axis=axis)

        # Controlled terms in canonical order.
        for controls_tuple, raw_angle in _sorted_gamma_items(self._gammas):
            controls_list = list(controls_tuple)
            angle = float(raw_angle)
            num_ctrl = len(controls_list)
            if num_ctrl == 0:
                # Should not happen because gammas has no empty-set key,
                # but stay defensive.
                _append_axis_rotation(circuit, target, angle, axis=axis)
                continue
            gate = _build_mc_pauli_rotation(axis, angle, num_ctrl)
            circuit.append_mc_gate(gate, controls_list + [target])

        return circuit
