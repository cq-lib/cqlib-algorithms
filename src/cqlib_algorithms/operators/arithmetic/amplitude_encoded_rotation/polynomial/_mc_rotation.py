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

"""Multi-controlled Pauli rotation helpers shared by the polynomial subpackage.

The helpers centralise the construction of controlled rotations on axes X,
Y and Z, together with small validation utilities that every class in this
subpackage (LinearPauliRotation, PolynomialPauliRotation, ...) needs.  A
multi-controlled gate is always produced by calling
``StandardGate.R{axis}(angle).control(num_controls)`` — the same idiom used
by every other operator in the project (QFT, comparators, multipliers,
oracles, …) so the SDK-level path for controlled standard gates is uniform.

Every helper in this module is deliberately left module-private.  The three
public classes in neighbouring modules import exactly the helpers they need
via ``from ._mc_rotation import <name>``, which allows a thin but
consistent code style across the subpackage.
"""

from __future__ import annotations

import math
from collections import Counter
from collections import Counter as CounterT
from collections.abc import Iterable, Iterator, Sequence
from itertools import combinations_with_replacement
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
from cqlib.circuit.gates import MCGate, StandardGate

from cqlib import Circuit

# ---------------------------------------------------------------------------
# Axis & coefficient validation ------------------------------------------------
# ---------------------------------------------------------------------------


def _validate_axis(axis: str) -> Literal["x", "y", "z"]:
    """Validate the ``axis`` argument, rejecting ``bool`` / non-string.

    Raises
    ------
    TypeError
        If ``axis`` is not a :class:`str`.
    ValueError
        If ``axis`` lower-cased is not in ``{"x", "y", "z"}``.
    """

    if isinstance(axis, bool) or not isinstance(axis, str):
        raise TypeError(
            f"axis must be a string ('x'/'y'/'z'), got {axis!r}."
        )
    axis = axis.lower()
    if axis not in {"x", "y", "z"}:
        raise ValueError(
            f"axis must be one of 'x', 'y', 'z', got {axis!r}."
        )
    return axis  # type: ignore[return-value]


def _validate_coefficients(
    coefficients: Iterable[float] | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Validate the coefficient list ``[c_0, c_1, ..., c_d]``.

    Raises
    ------
    TypeError
        If any element is a :class:`bool` or the container is not 1-D.
    ValueError
        If coefficients are empty, non-finite, or multi-dimensional.
    """

    if isinstance(coefficients, np.ndarray):
        if coefficients.ndim != 1:
            raise ValueError(
                "coefficients must be a 1-D iterable, got array of shape "
                f"{coefficients.shape!r}."
            )
        arr = np.asarray(coefficients, dtype=np.float64, order="C")
    else:
        if not isinstance(coefficients, (list, tuple)):
            coefficients = list(coefficients)
        for element in coefficients:
            if isinstance(element, bool):
                raise TypeError(
                    "coefficients must not contain bool values; got "
                    f"{element!r}."
                )
        arr = np.asarray(coefficients, dtype=np.float64, order="C")
        if arr.ndim != 1:
            raise ValueError(
                "coefficients must be a 1-D iterable, got array of shape "
                f"{arr.shape!r}."
            )
    if arr.size == 0:
        raise ValueError(
            "coefficients must not be empty; at least one constant term "
            "is required."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("coefficients must contain only finite floats.")
    return arr


# ---------------------------------------------------------------------------
# Single- and multi-controlled rotation gate builders ----------------------
# ---------------------------------------------------------------------------


def _append_axis_rotation(
    circuit: Circuit,
    target: int,
    angle: float,
    *,
    axis: Literal["x", "y", "z"],
) -> None:
    """Append a plain single-qubit Pauli rotation.

    Parameters
    ----------
    circuit : Circuit
        Destination circuit.
    target : int
        Target wire index.
    angle : float
        Finite rotation angle.
    axis : Literal["x", "y", "z"]
        Pauli rotation axis.
    """

    if axis == "x":
        circuit.rx(target, angle)
    elif axis == "y":
        circuit.ry(target, angle)
    else:  # "z"
        circuit.rz(target, angle)


def _build_mc_pauli_rotation(
    axis: Literal["x", "y", "z"],
    angle: float,
    num_controls: int,
) -> MCGate:
    """Build a multi-controlled Pauli rotation gate.

    The returned object is a :class:`StandardGate` (possibly wrapped with
    controls) produced by chaining ``.control(num_controls)`` on the
    single-qubit rotation gate.  This matches the idiom used everywhere
    else in the project (QFT, comparators, multipliers, oracles) so the
    SDK path for controlled gates is uniform.

    Parameters
    ----------
    axis : Literal["x", "y", "z"]
        Lower-cased Pauli axis.
    angle : float
        Finite rotation angle.
    num_controls : int
        Non-negative control count.  ``bool`` is explicitly rejected.

    Returns
    -------
    MCGate
        A cqlib gate object consumable by :meth:`Circuit.append_mc_gate`
        together with a wire list of length ``num_controls + 1``.

    Raises
    ------
    TypeError
        If ``num_controls`` is a ``bool`` or not an :class:`int`.
    ValueError
        If ``num_controls`` is strictly negative.
    """

    if isinstance(num_controls, bool) or not isinstance(num_controls, int):
        raise TypeError(
            f"num_controls must be an int, got {num_controls!r}."
        )
    if num_controls < 0:
        raise ValueError(
            f"num_controls must be >= 0, got {num_controls!r}."
        )
    if axis == "x":
        base = StandardGate.RX(angle)
    elif axis == "y":
        base = StandardGate.RY(angle)
    else:  # "z"
        base = StandardGate.RZ(angle)
    if num_controls == 0:
        return cast(MCGate, base)
    return base.control(num_controls)


# ---------------------------------------------------------------------------
# Polynomial gamma expansion ------------------------------------------------
# ---------------------------------------------------------------------------


def _multinomial_multiplicity(counts: CounterT[int]) -> int:
    """Return ``multinomial(m; k_0, k_1, ...)`` for a multiset Counter.

    Helper for :func:`_expand_polynomial_gammas`; it computes how many
    permutations correspond to a specific multiset of multiplicities.
    """

    total = sum(counts.values())
    denominator = 1
    for multiplicity in counts.values():
        denominator *= math.factorial(multiplicity)
    return math.factorial(total) // denominator


def _expand_polynomial_gammas(
    n: int,
    coefficients: npt.NDArray[np.float64],
) -> tuple[float, dict[frozenset[int], float], bool]:
    """Expand ``Σ_m c_m * k^m`` into per-subset controlled-rotation angles.

    The substitution rule (Appendix B.2 in the design document) is::

        k^m = Σ_{(i_1,...,i_m)} 2^{i_1 + ... + i_m} * k_{i_1} ... k_{i_m},

    where each ``(i_1,...,i_m)`` tuple represents a multiset pick; the
    overall algebraic coefficient for the subset associated with the
    distinct indices in that multiset is the number of permutations
    ``2**(sum indices)`` times the multinomial factor
    ``multinomial(m, multiplicities)``.

    Returns
    -------
    const_term : float
        Constant (zero-control) rotation = ``c_0``.
    gammas : dict[frozenset[int], float]
        Mapping ``S -> rotation angle`` for each non-empty control subset
        ``S ⊆ {0,...,n-1}``.  Each key is a ``frozenset``; insertions are
        additive.
    degree_gt_n : bool
        ``True`` if ``d >= n``, meaning the expansion has "saturated" the
        available state qubits.  Higher-degree terms still contribute via
        subsets of cardinality ``≤ n``, but the user usually wants to be
        warned that ``k^m`` is no longer algebraically "full rank".
    """

    const_term: float = float(coefficients[0])
    gammas: dict[frozenset[int], float] = {}
    degree = coefficients.size - 1
    degree_gt_n = degree > n
    for m in range(1, degree + 1):
        cm = float(coefficients[m])
        if cm == 0.0:
            continue
        for indices in combinations_with_replacement(range(n), r=m):
            counts: CounterT[int] = Counter(indices)
            multiplicity = _multinomial_multiplicity(counts)
            bit_set = frozenset(indices)
            contribution = cm * multiplicity * (1 << sum(indices))
            if bit_set in gammas:
                gammas[bit_set] += contribution
            else:
                gammas[bit_set] = contribution
    return const_term, gammas, degree_gt_n


def _sorted_gamma_items(
    gammas: dict[frozenset[int], float],
) -> Iterator[tuple[tuple[int, ...], float]]:
    """Yield ``(sorted_tuple, angle)`` pairs in canonical order.

    The order is:
        1. Ascending subset size (degree of the multi-controlled rotation).
        2. Lexicographic ordering of the sorted tuple of control indices.
    """

    items = sorted(
        gammas.items(),
        key=lambda pair: (len(pair[0]), tuple(sorted(pair[0]))),
    )
    for bit_set, angle in items:
        yield tuple(sorted(bit_set)), angle


# ---------------------------------------------------------------------------
# Point-wise alpha computation ----------------------------------------------
# ---------------------------------------------------------------------------


def _compute_polynomial_alpha(
    num_state_qubits: int,
    coefficients: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], bool]:
    """Return (``alpha``, ``oversized``) via Horner evaluation on k=0..2^n-1.

    ``alpha[k] = f(k) mod 2*pi``; ``oversized`` is ``True`` when any
    ``|f(k)| >= 2*pi``.  The caller is responsible for issuing a
    ``UserWarning`` when ``oversized`` is set.
    """

    ks = np.arange(1 << num_state_qubits, dtype=np.float64)
    f_k = np.polyval(coefficients[::-1], ks)
    if np.any(np.abs(f_k) >= 2.0 * np.pi):
        oversized = True
    else:
        oversized = False
    alpha = np.mod(f_k, 2.0 * np.pi)
    return alpha, oversized


# ---------------------------------------------------------------------------
# Shared piecewise helpers (Linear + Polynomial) ----------------------------
# ---------------------------------------------------------------------------


def _append_segmented_axis_rotation(
    circuit: Circuit,
    controls: list[int],
    target: int,
    angle: float,
    *,
    axis: Literal["x", "y", "z"],
) -> None:
    """Append a multi-controlled rotation (or plain single-qubit when empty).

    Thin dispatcher over :func:`_append_axis_rotation` and
    :func:`_build_mc_pauli_rotation` so the piecewise classes can apply the
    same rotation template whether or not an extra comparator-flag control is
    present.
    """

    num_ctrl = len(controls)
    if num_ctrl == 0:
        _append_axis_rotation(circuit, target, float(angle), axis=axis)
        return
    gate = _build_mc_pauli_rotation(axis, float(angle), num_ctrl)
    circuit.append_mc_gate(gate, list(controls) + [target])


def _validate_piecewise_breakpoints(
    breakpoints: Sequence[int],
    num_state_qubits: int,
) -> list[int]:
    """Validate breakpoints and return a plain ``list[int]`` copy.

    Accepts any ``Sequence`` of ints (rejects ``str``/``bytes`` and
    ``bool`` entries).  The returned list is a defensive copy and the
    caller owns it.
    """

    if isinstance(breakpoints, bool) or not isinstance(
        breakpoints, Sequence
    ) or isinstance(breakpoints, (str, bytes)):
        raise TypeError(
            "breakpoints must be a sequence of ints, got "
            f"{breakpoints!r}."
        )
    bp_list = list(breakpoints)
    if len(bp_list) < 2:
        raise ValueError(
            "breakpoints must contain at least two endpoints "
            "(J >= 1 segment requires J+1 breakpoints), got "
            f"{len(bp_list)!r}."
        )
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
    if bp_list[0] != 0:
        raise ValueError(
            "breakpoints[0] must equal 0 (lower bound of the unsigned "
            f"register domain), got {bp_list[0]!r}."
        )
    upper_bound = 1 << int(num_state_qubits)
    if bp_list[-1] != upper_bound:
        raise ValueError(
            f"breakpoints[-1] must equal 2**num_state_qubits={upper_bound!r} "
            f"(upper bound of the register domain), got {bp_list[-1]!r}."
        )
    return bp_list


def _validate_float_sequence(
    values: Sequence[float],
    name: str,
    *,
    expected_len: int | None = None,
) -> list[float]:
    """Validate a finite float sequence and return a plain ``list[float]``.

    Rejects ``str``/``bytes`` as well as ``bool`` entries.  When
    ``expected_len`` is not ``None`` a length mismatch raises
    :class:`ValueError`.
    """

    if isinstance(values, bool) or not isinstance(values, Sequence) or isinstance(
        values, (str, bytes)
    ):
        raise TypeError(
            f"{name} must be a sequence of floats, got {values!r}."
        )
    val_list = list(values)
    if expected_len is not None and len(val_list) != expected_len:
        raise ValueError(
            f"{name} must have length {expected_len!r} (one per segment), "
            f"got {len(val_list)!r}."
        )
    for idx, value in enumerate(val_list):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"{name}[{idx}] must be a float, got {value!r}."
            )
        if not math.isfinite(float(value)):
            raise ValueError(
                f"{name}[{idx}] must be finite, got {value!r}."
            )
    return [float(v) for v in val_list]


def _validate_nested_coefficients(
    coeffs: Sequence[Sequence[float] | npt.NDArray[np.float64]],
    *,
    expected_len: int,
) -> list[npt.NDArray[np.float64]]:
    """Validate nested per-segment polynomial coefficients.

    Returns a list of 1-D ``float64`` arrays (one per segment), each with
    at least one element.
    """

    if isinstance(coeffs, bool) or not isinstance(coeffs, Sequence) or isinstance(
        coeffs, (str, bytes)
    ):
        raise TypeError(
            "coeffs must be a sequence of coefficient sequences/arrays, "
            f"got {coeffs!r}."
        )
    outer = list(coeffs)
    if len(outer) != expected_len:
        raise ValueError(
            "coeffs must have length equal to the number of segments "
            f"({expected_len!r}), got {len(outer)!r}."
        )
    validated: list[npt.NDArray[np.float64]] = []
    for idx, segment in enumerate(outer):
        if isinstance(segment, np.ndarray):
            arr = np.asarray(segment, dtype=np.float64, order="C")
            if arr.ndim != 1:
                raise ValueError(
                    f"coeffs[{idx}] must be 1-D, got array of shape "
                    f"{arr.shape!r}."
                )
        else:
            if isinstance(segment, bool) or not isinstance(
                segment, Sequence
            ) or isinstance(segment, (str, bytes)):
                raise TypeError(
                    f"coeffs[{idx}] must be a sequence of floats, got "
                    f"{segment!r}."
                )
            seg_list = list(segment)
            for j, c in enumerate(seg_list):
                if isinstance(c, bool) or not isinstance(c, (int, float)):
                    raise TypeError(
                        f"coeffs[{idx}][{j}] must be a float, got {c!r}."
                    )
                if not math.isfinite(float(c)):
                    raise ValueError(
                        f"coeffs[{idx}][{j}] must be finite, got {c!r}."
                    )
            arr = np.asarray(seg_list, dtype=np.float64, order="C")
        if arr.size == 0:
            raise ValueError(
                f"coeffs[{idx}] must not be empty (at least the constant "
                "term is required)."
            )
        validated.append(arr)
    return validated
