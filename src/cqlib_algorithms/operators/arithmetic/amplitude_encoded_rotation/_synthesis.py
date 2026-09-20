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

"""Internal synthesis helpers and base classes for amplitude-encoded rotations.

This private module contains only **axis-generic** logic shared between the
three uniformly controlled rotation gates (X/Y/Z) and the four function-table
rotations built on top of them.  Alpha-table look-ups for the functional
families live inside their respective public modules.

Academic references
-------------------
1. Möttönen, Vartiainen, Bergholm, Salomaa (2005).  Transformation of
   quantum states using uniformly controlled rotations.  *Quantum
   Information & Computation*, 5(6), 467–473.  arXiv:quant-ph/0407010.
   — Original Gray-code / CX decomposition and M^k angle inversion.
2. Vartiainen, Möttönen, Salomaa (2004).  Efficient decomposition of
   quantum gates.  *Phys. Rev. Lett.* 92, 177902.  — Precursor to the
   uniformly controlled rotation framework.
3. Bergholm, Vartiainen, Möttönen, Salomaa (2005).  Quantum circuits
   with uniformly controlled one-qubit gates.  *Phys. Rev. A* 71, 052330.
   — Generalisation to arbitrary U(2) quantum multiplexors.
4. Harrow, Hassidim, Lloyd (2009).  Quantum algorithm for solving linear
   systems of equations.  *Phys. Rev. Lett.* 103, 150502.  — Origin of
   the amplitude-encoded reciprocal used in HHL eigenvalue inversion.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar, Literal

import numpy as np

from cqlib import Circuit

# ---------------------------------------------------------------------------
# Axis-generic numerical helpers
# ---------------------------------------------------------------------------


def _is_power_of_two(value: int) -> bool:
    """Return whether ``value`` is a positive power of two.

    Parameters
    ----------
    value : int
        Integer to validate.

    Returns
    -------
    bool
        ``True`` when ``value`` is strictly positive and a power of two.
    """

    return value > 0 and (value & (value - 1)) == 0


def gray_code(index: int) -> int:
    """Return the binary reflected Gray code of ``index``.

    Parameters
    ----------
    index : int
        Zero-based Gray code index.

    Returns
    -------
    int
        Gray code integer value.

    Raises
    ------
    ValueError
        If ``index`` is negative.
    """

    if index < 0:
        raise ValueError("Gray code index must be non-negative.")
    return index ^ (index >> 1)


def gray_code_sequence(num_controls: int) -> list[int]:
    """Return the Gray code sequence for ``num_controls`` controls.

    Parameters
    ----------
    num_controls : int
        Number of control qubits.

    Returns
    -------
    list[int]
        Gray code sequence of length ``2**num_controls``.

    Raises
    ------
    ValueError
        If ``num_controls`` is negative.
    """

    if isinstance(num_controls, bool) or not isinstance(num_controls, int):
        raise TypeError(
            f"num_controls must be an int, got {num_controls!r}."
        )
    if num_controls < 0:
        raise ValueError("num_controls must be non-negative.")
    return [gray_code(index) for index in range(2**num_controls)]


def gray_control_positions(num_controls: int) -> list[int]:
    """Return local CNOT control positions from the Gray-code cycle.

    Parameters
    ----------
    num_controls : int
        Number of control qubits.

    Returns
    -------
    list[int]
        Control positions for the alternating CNOT sequence, using the
        local numbering convention ``0..num_controls-1``.  The returned
        list has length ``2**num_controls``; the last entry corresponds
        to the wrap-around transition that closes the cycle and is
        intentionally preserved so that ``zip(theta, controls)`` in the
        standard interleaving always produces an equal-length pair.

    Raises
    ------
    ValueError
        If ``num_controls`` is negative.
    """

    if isinstance(num_controls, bool) or not isinstance(num_controls, int):
        raise TypeError(
            f"num_controls must be an int, got {num_controls!r}."
        )
    if num_controls < 0:
        raise ValueError("num_controls must be non-negative.")
    if num_controls == 0:
        return []

    sequence = gray_code_sequence(num_controls)
    controls: list[int] = []
    for index in range(len(sequence)):
        current = sequence[index]
        nxt = sequence[(index + 1) % len(sequence)]
        diff = current ^ nxt
        controls.append(diff.bit_length() - 1)
    return controls


def _binary_dot_parity(left: int, right: int) -> int:
    """Return the parity of the bitwise inner product modulo 2.

    Parameters
    ----------
    left : int
        First binary integer.
    right : int
        Second binary integer.

    Returns
    -------
    int
        ``0`` for even parity and ``1`` for odd parity.
    """

    return (left & right).bit_count() % 2


def gray_matrix(num_controls: int) -> np.ndarray[Any, Any]:
    """Construct the Gray-code matrix ``M^k``.

    ``M^k[row, col] = (-1) ** <row, gray(col)>_2`` where ``<·,·>_2`` is
    the binary dot-product parity.  The inverse of ``M^k`` is
    ``(M^k).T / 2**k`` which is used to recover the circuit-level
    angles ``theta`` from the block-diagonal targets ``alpha``.

    Parameters
    ----------
    num_controls : int
        Number of control qubits ``k``.

    Returns
    -------
    np.ndarray
        Real matrix with shape ``(2**k, 2**k)`` and ``dtype=float64``.

    Raises
    ------
    ValueError
        If ``num_controls`` is negative.
    """

    if isinstance(num_controls, bool) or not isinstance(num_controls, int):
        raise TypeError(
            f"num_controls must be an int, got {num_controls!r}."
        )
    if num_controls < 0:
        raise ValueError("num_controls must be non-negative.")

    dim = 2**num_controls
    codes = gray_code_sequence(num_controls)
    matrix = np.empty((dim, dim), dtype=np.float64)
    for row in range(dim):
        for col, gray in enumerate(codes):
            matrix[row, col] = (-1.0) ** _binary_dot_parity(row, gray)
    return matrix


def solve_theta_from_alpha(
    alpha: Sequence[float] | np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Solve the Gray-code linear system ``M^k theta = alpha``.

    Parameters
    ----------
    alpha : Sequence[float] | np.ndarray
        Target block-diagonal rotation angles, length ``2**k``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The solved circuit rotation angles ``theta`` of shape
        ``(2**k,)`` and the Gray matrix ``M^k`` of shape ``(2**k, 2**k)``
        used in the solve.

    Raises
    ------
    TypeError
        If ``alpha`` is not a sequence of finite floats.
    ValueError
        If ``alpha`` is not one-dimensional or its length is not a power
        of two.
    """

    if isinstance(alpha, (str, bytes)) or not isinstance(alpha, (Sequence, np.ndarray)):
        raise TypeError(
            "alpha must be a sequence of floats, got "
            f"{type(alpha).__name__}."
        )
    if isinstance(alpha, np.ndarray):
        alpha_arr = np.asarray(alpha, dtype=np.float64)
    else:
        alpha_arr = np.asarray(list(alpha), dtype=np.float64)
    if not np.all(np.isfinite(alpha_arr)):
        raise ValueError("alpha must contain only finite floats.")
    if alpha_arr.ndim != 1:
        raise ValueError("alpha must be a one-dimensional array.")
    if not _is_power_of_two(alpha_arr.shape[0]):
        raise ValueError("alpha length must be a power of two.")

    num_controls = int(math.log2(alpha_arr.shape[0]))
    m_matrix = gray_matrix(num_controls)
    theta = (m_matrix.T @ alpha_arr) / (2**num_controls)
    return theta, m_matrix


# ---------------------------------------------------------------------------
# Axis-generic circuit synthesis
# ---------------------------------------------------------------------------

_VALID_UCR_AXES: frozenset[str] = frozenset({"x", "y", "z"})


def _validate_ucr_axis(axis: str) -> Literal["x", "y", "z"]:
    """Validate that ``axis`` is one of ``"x"``, ``"y"`` or ``"z"``."""

    if not isinstance(axis, str):
        raise TypeError(f"axis must be a string, got {axis!r}.")
    axis_lower = axis.lower()
    if axis_lower not in _VALID_UCR_AXES:
        raise ValueError(
            f"Unsupported UCR axis {axis!r}.  Expected one of "
            f"{' '.join(sorted(_VALID_UCR_AXES))}."
        )
    return axis_lower  # type: ignore[return-value]


def _append_axis_rotation(
    circuit: Circuit,
    target: int,
    angle: float,
    *,
    axis: Literal["x", "y", "z"],
) -> None:
    """Append ``R_{axis}(angle)`` to ``circuit`` on ``target``."""

    if axis == "x":
        circuit.rx(target, angle)
    elif axis == "y":
        circuit.ry(target, angle)
    else:  # "z"
        circuit.rz(target, angle)


def _synthesize_ucr_circuit_impl(
    theta: Sequence[float] | np.ndarray[Any, Any],
    controls: Sequence[int],
    target: int,
    *,
    axis: Literal["x", "y", "z"],
) -> Circuit:
    """Synthesize a local-numbering UCR circuit from Gray-expanded angles.

    The output circuit has width ``num_controls + 1`` under the local
    convention that wires ``0..num_controls-1`` are the controls and wire
    ``num_controls`` is the target.  The circuit is the standard
    Möttönen interleaving: ``R_axis(theta_0) CX R_axis(theta_1) CX ...``.

    Parameters
    ----------
    theta : Sequence[float] | np.ndarray
        Circuit rotation angles with length ``2**k``.
    controls : Sequence[int]
        Local control positions as produced by
        :func:`gray_control_positions`.  Must have length ``2**k``.
    target : int
        Local target qubit index.  Under the current convention this
        must equal ``k = log2(len(theta))``.
    axis : Literal["x", "y", "z"]
        Pauli rotation axis.

    Returns
    -------
    Circuit
        A ``cqlib.Circuit`` of width ``k + 1`` consisting of alternating
        ``R_{axis}`` and ``CX`` gates.

    Raises
    ------
    TypeError / ValueError
        For mismatched lengths, non-continuous local numbering, bad
        axis value, or non-finite angles.
    """

    axis = _validate_ucr_axis(axis)

    theta_arr = np.asarray(list(theta), dtype=np.float64)
    if not np.all(np.isfinite(theta_arr)):
        raise ValueError("theta must contain only finite floats.")
    if theta_arr.ndim != 1:
        raise ValueError("theta must be a one-dimensional array.")
    if theta_arr.size == 0:
        raise ValueError("theta must contain at least one rotation angle.")
    if not _is_power_of_two(theta_arr.size):
        raise ValueError("theta length must be a power of two.")

    num_controls = int(math.log2(theta_arr.size))

    controls_list = list(controls)
    target_list = target
    if isinstance(target_list, bool) or not isinstance(target_list, int):
        raise TypeError(f"target must be an int, got {target!r}.")

    if num_controls == 0:
        if controls_list:
            raise ValueError("For zero controls, controls must be empty.")
        if target_list != 0:
            raise ValueError("For zero controls, the local target must be 0.")
        circuit = Circuit(1)
        _append_axis_rotation(circuit, 0, float(theta_arr[0]), axis=axis)
        return circuit

    if len(controls_list) != theta_arr.size:
        raise ValueError("controls and theta must have the same length.")
    expected_controls = gray_control_positions(num_controls)
    if controls_list != expected_controls:
        raise ValueError("controls must follow the local Gray-code convention.")
    if target_list != num_controls:
        raise ValueError("Under local numbering, target must equal num_controls.")

    circuit = Circuit(num_controls + 1)
    if axis == "x":
        # Rx(θ) = H · Rz(θ) · H,  block-diagonally we need
        #   UCRX(α₀,…,α_{N-1}) = H_t · UCRZ(α₀,…,α_{N-1}) · H_t
        # where t is the target wire.  This avoids re-deriving the X-axis
        # Gray expansion (which does NOT share the Y/Z CX interleaving
        # structure because [X, CX-control] behaves differently from
        # [Y, CX-control] or [Z, CX-control]).
        circuit.h(target_list)
        for angle, control in zip(theta_arr, controls_list):
            _append_axis_rotation(
                circuit, target_list, float(angle), axis="z"
            )
            circuit.cx(control, target_list)
        circuit.h(target_list)
    else:
        for angle, control in zip(theta_arr, controls_list):
            _append_axis_rotation(
                circuit, target_list, float(angle), axis=axis
            )
            circuit.cx(control, target_list)
    return circuit


# ---------------------------------------------------------------------------
# Internal base class — uniformly controlled rotation
# ---------------------------------------------------------------------------


class _UCRBase:
    """Uniformly controlled rotation around a fixed Pauli axis.

    Private base for the public ``UCRX``, ``UCRY`` and ``UCRZ`` classes.
    Concrete subclasses bind ``_ROTATION_AXIS`` and ``_DEFAULT_NAME`` at
    class-definition time; this class remains axis-agnostic otherwise.

    The decomposition follows Möttönen et al. (2005): the block-diagonal
    target angles ``alpha`` are expanded via the Gray matrix ``M^k`` into
    circuit-level angles ``theta``, which are then interleaved with CX
    gates whose control-wire positions follow the binary-reflected Gray
    code transitions.
    """

    _ROTATION_AXIS: ClassVar[Literal["x", "y", "z"]]
    _DEFAULT_NAME: ClassVar[str] = "UCR"

    def __init__(
        self,
        num_controls: int,
        alpha: Sequence[float] | np.ndarray[Any, Any],
        *,
        name: str | None = None,
    ) -> None:
        if isinstance(num_controls, bool) or not isinstance(num_controls, int):
            raise TypeError(
                f"num_controls must be an int, got {num_controls!r}."
            )
        if num_controls < 0:
            raise ValueError(
                f"num_controls must be >= 0, got {num_controls!r}."
            )

        if isinstance(alpha, (str, bytes)) or not isinstance(alpha, (Sequence, np.ndarray)):
            raise TypeError(
                "alpha must be a sequence of floats, got "
                f"{type(alpha).__name__}."
            )
        if isinstance(alpha, np.ndarray):
            alpha_snapshot = np.asarray(alpha, dtype=np.float64)
        else:
            alpha_snapshot = np.asarray(list(alpha), dtype=np.float64)
        if not np.all(np.isfinite(alpha_snapshot)):
            raise ValueError("alpha must contain only finite floats.")
        if alpha_snapshot.ndim != 1:
            raise ValueError("alpha must be a one-dimensional array.")
        expected_len = 2**num_controls
        if alpha_snapshot.size != expected_len:
            raise ValueError(
                "alpha length must equal 2**num_controls "
                f"(expected {expected_len}, got {alpha_snapshot.size})."
            )

        if not hasattr(self, "_ROTATION_AXIS"):
            raise TypeError(
                f"{type(self).__name__} must define a _ROTATION_AXIS class "
                "attribute."
            )
        self._axis: Literal["x", "y", "z"] = _validate_ucr_axis(
            self._ROTATION_AXIS
        )

        if name is not None and not isinstance(name, str):
            raise TypeError(f"name must be a string or None, got {name!r}.")

        self._num_controls = num_controls
        self._alpha = alpha_snapshot
        self._name: str = name or self._DEFAULT_NAME

        self._theta, self._gray_matrix = solve_theta_from_alpha(self._alpha)
        self._gray_codes = gray_code_sequence(self._num_controls)
        self._controls = gray_control_positions(self._num_controls)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_controls(self) -> int:
        """Return the number of control qubits ``k``."""

        return self._num_controls

    @property
    def alpha(self) -> np.ndarray[Any, Any]:
        """Return a defensive copy of the target angles ``alpha``."""

        return self._alpha.copy()

    @property
    def theta(self) -> np.ndarray[Any, Any]:
        """Return a defensive copy of the circuit angles ``theta``."""

        return self._theta.copy()

    @property
    def axis(self) -> Literal["x", "y", "z"]:
        """Return the fixed Pauli rotation axis."""

        return self._axis

    @property
    def gray_codes(self) -> list[int]:
        """Return a defensive copy of the Gray code sequence."""

        return list(self._gray_codes)

    @property
    def controls(self) -> list[int]:
        """Return a defensive copy of the local CX control positions."""

        return list(self._controls)

    @property
    def gray_matrix(self) -> np.ndarray[Any, Any]:
        """Return a defensive copy of the Gray matrix ``M^k``."""

        return self._gray_matrix.copy()

    @property
    def num_state_qubits(self) -> int:
        """Return ``num_controls`` (the control-register width)."""

        return self._num_controls

    @property
    def num_qubits(self) -> int:
        """Return the total circuit width: ``num_controls + 1``."""

        return self._num_controls + 1

    @property
    def control_register_indices(self) -> list[int]:
        """Return the local wire indices of the control qubits."""

        return list(range(self._num_controls))

    @property
    def result_qubit_index(self) -> int:
        """Return the local target-wire index (``num_controls``)."""

        return self._num_controls

    @property
    def name(self) -> str:
        """Return the configured human-readable name."""

        return self._name

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------

    def build_circuit(self) -> Circuit:
        """Return a fresh ``Circuit`` implementing this UCR gate.

        The returned circuit uses the local layout
        ``control_register_indices + [result_qubit_index]``; callers
        compose into larger circuits and remap wires as needed.
        """

        return _synthesize_ucr_circuit_impl(
            self._theta,
            self._controls,
            self.result_qubit_index,
            axis=self._axis,
        )


# ---------------------------------------------------------------------------
# Internal base class — function-table controlled rotations
# ---------------------------------------------------------------------------


class _FunctionalPauliRotation:
    """Private base for amplitude-encoded function rotations.

    Concrete subclasses implement :meth:`_compute_alpha` to evaluate
    the target function for each integer control value
    ``k in [0, 2**num_state_qubits - 1]`` (binary integer convention)
    and return the corresponding rotation-angle table.  The base class
    provides the common register-index properties and composes the
    final circuit via the matching axis-specific UCR class.
    """

    _DEFAULT_NAME: ClassVar[str] = "FunctionalPauliRotation"

    def __init__(
        self,
        num_state_qubits: int,
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

        self._axis: Literal["x", "y", "z"] = _validate_ucr_axis(axis)

        if name is not None and not isinstance(name, str):
            raise TypeError(f"name must be a string or None, got {name!r}.")

        self._num_state_qubits = num_state_qubits
        self._name: str = name or self._DEFAULT_NAME

    # ------------------------------------------------------------------
    # Abstract hook
    # ------------------------------------------------------------------

    def _compute_alpha(self) -> np.ndarray[Any, Any]:  # pragma: no cover
        raise NotImplementedError(
            f"{type(self).__name__} must implement _compute_alpha."
        )

    # ------------------------------------------------------------------
    # Common properties
    # ------------------------------------------------------------------

    @property
    def num_state_qubits(self) -> int:
        """Return the control-register bit width ``n``."""

        return self._num_state_qubits

    @property
    def num_qubits(self) -> int:
        """Return ``num_state_qubits + 1`` (controls + result qubit)."""

        return self._num_state_qubits + 1

    @property
    def control_register_indices(self) -> list[int]:
        """Return the local control wire indices ``[0, ..., n-1]``."""

        return list(range(self._num_state_qubits))

    @property
    def result_qubit_index(self) -> int:
        """Return the local target-wire index ``n``."""

        return self._num_state_qubits

    @property
    def axis(self) -> Literal["x", "y", "z"]:
        """Return the selected Pauli rotation axis."""

        return self._axis

    @property
    def name(self) -> str:
        """Return the configured human-readable name."""

        return self._name

    @property
    def alpha(self) -> np.ndarray[Any, Any]:
        """Return a defensive copy of the evaluated angle table."""

        return self._compute_alpha().copy()

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------

    def build_circuit(self) -> Circuit:
        """Return a fresh ``Circuit`` for this functional rotation.

        Internally instantiates the axis-matching ``UCRX/Y/Z`` class
        using the alpha table produced by :meth:`_compute_alpha` and
        returns the resulting local-numbering circuit.
        """

        alpha_table = self._compute_alpha()
        gate_cls: type[_UCRBase]
        if self._axis == "x":
            from .ucrx import UCRX as gate_cls
        elif self._axis == "y":
            from .ucry import UCRY as gate_cls
        else:
            from .ucrz import UCRZ as gate_cls
        return gate_cls(self._num_state_qubits, alpha_table).build_circuit()
