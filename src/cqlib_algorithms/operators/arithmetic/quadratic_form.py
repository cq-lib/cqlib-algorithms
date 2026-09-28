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
"""Quadratic form operator via QFT phase kickback (Gilliam 2019).

This module implements the QuadraticForm quantum arithmetic operator

.. math::

    Q(x) = x^T A x + b^T x + c,

where :math:`x \\in \\{0,1\\}^n` is an ``n``-qubit binary input register,
:math:`A \\in \\mathbb{R}^{n \\times n}` is the quadratic coefficient matrix,
:math:`b \\in \\mathbb{R}^n` is the linear coefficient vector and :math:`c \\in
\\mathbb{R}` is a scalar constant.  The overall circuit performs

.. math::

    |x\\rangle_n |0\\rangle_m
        \\mapsto
    |x\\rangle_n \\bigl| (Q(x) + 2^m) \\bmod 2^m \\bigr\\rangle_m,

where the ``m``-qubit result register stores :math:`Q(x)` in two's-complement
representation, enabling exact (probability-1) recovery whenever :math:`Q(x)`
is an integer.

The construction mirrors the QFT adder family (Draper, Ruiz-Perez) and
comprises three stages:

1. Forward QFT on the ``m``-qubit result accumulator (without final swaps).
2. Phase-kickback schedule that encodes each monomial coefficient via a
   controlled phase gate on each result qubit using the scaling factor
   :math:`\\kappa = \\pi \\cdot 2^{1-m}`:

   * constant term         :math:`c`               -> P  gate
   * linear + diagonal     :math:`b_j + A_{jj}`    -> CP gate
   * off-diagonal pair     :math:`A_{jk} + A_{kj}` -> MCP gate (2 controls)

3. Inverse QFT on the accumulator, mapping the encoded phases back into the
   computational basis.

References
----------
.. [1] Gilliam, A., et al. (2019). Optimizing Quantum Arithmetic for
       Shor's Algorithm on a Quantum Computer.  arXiv:1912.04088.
.. [2] Draper, T. G. (2000). Addition on a quantum computer.
       arXiv:quant-ph/0008033.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from cqlib.circuit.gates import StandardGate

from cqlib import Circuit
from cqlib_algorithms.transforms import QFT as TransformQFT

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

_PI: float = 3.141592653589793
_TWO_PI: float = 2.0 * _PI
_EPS: float = 1e-14


# -- validation helpers ------------------------------------------------------


def _validate_register_width(value: int, field: str) -> int:
    """Validate a positive integer register width.

    Parameters
    ----------
    value :
        Candidate register width.
    field :
        Human readable field name used in error messages.

    Returns
    -------
    int
        Normalised integer value.

    Raises
    ------
    TypeError
        If ``value`` is a ``bool`` or not an ``int``.
    ValueError
        If ``value`` is not strictly positive.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value <= 0:
        raise ValueError(f"{field} must be greater than zero")
    return int(value)


def _validate_A(A: ArrayLike, n: int) -> np.ndarray:
    """Validate and coerce the quadratic coefficient matrix.

    Parameters
    ----------
    A :
        Candidate matrix coercible to a float ``(n, n)`` array.
    n :
        Required side-length.

    Returns
    -------
    np.ndarray
        Float ``(n, n)`` matrix.

    Raises
    ------
    TypeError
        If ``A`` is not coercible or not 2-D.
    ValueError
        If the shape is not ``(n, n)``.
    """
    try:
        A_arr = np.asarray(A, dtype=float)
    except (TypeError, ValueError) as exc:
        msg = "A must be coercible to a 2-D numeric array"
        raise TypeError(msg) from exc
    if A_arr.ndim != 2:
        msg = f"A must be 2-D, got {A_arr.ndim}-D"
        raise TypeError(msg)
    if A_arr.shape != (n, n):
        msg = f"A must have shape ({n}, {n}), got {A_arr.shape}"
        raise ValueError(msg)
    return A_arr


def _validate_b(b: ArrayLike, n: int) -> np.ndarray:
    """Validate and coerce the linear coefficient vector.

    Parameters
    ----------
    b :
        Candidate vector coercible to a float ``(n,)`` array.
    n :
        Required length.

    Returns
    -------
    np.ndarray
        Float ``(n,)`` vector.

    Raises
    ------
    TypeError
        If ``b`` is not coercible or not 1-D.
    ValueError
        If the length is not ``n``.
    """
    try:
        b_arr = np.asarray(b, dtype=float)
    except (TypeError, ValueError) as exc:
        msg = "b must be coercible to a 1-D numeric array"
        raise TypeError(msg) from exc
    if b_arr.ndim != 1:
        msg = f"b must be 1-D, got {b_arr.ndim}-D"
        raise TypeError(msg)
    if b_arr.shape[0] != n:
        msg = f"b must have length {n}, got {b_arr.shape[0]}"
        raise ValueError(msg)
    return b_arr


def _validate_c(c: float) -> float:
    """Validate and coerce the constant term.

    Parameters
    ----------
    c :
        Candidate scalar coercible to float.

    Returns
    -------
    float
        Normalised float.

    Raises
    ------
    TypeError
        If ``c`` is not coercible to float.
    """
    try:
        return float(c)
    except (TypeError, ValueError) as exc:
        msg = "c must be coercible to float"
        raise TypeError(msg) from exc


def _validate_approximation_degree(
    approximation_degree: int | None,
    m: int,
) -> int:
    """Validate and convert the QFT approximation degree into the internal
    ``qft_degree`` form used by ``TransformQFT``.

    The mapping matches the Draper adder / QFT weighted-sum family:

        qft_degree = max(0, m - 1 - approximation_degree),

    clamped to ``[0, m-1]``.

    Parameters
    ----------
    approximation_degree :
        User-facing threshold or ``None``.
    m :
        Result register width.

    Returns
    -------
    int
        Internal ``qft_degree`` suitable for ``TransformQFT``.

    Raises
    ------
    TypeError
        If non-``None`` and not a non-``bool`` ``int``.
    ValueError
        If non-``None`` and strictly negative.
    """
    if approximation_degree is None:
        return 0
    if isinstance(approximation_degree, bool) or not isinstance(
        approximation_degree, int
    ):
        raise TypeError("approximation_degree must be an integer or None")
    if approximation_degree < 0:
        raise ValueError(
            "approximation_degree must be None or a non-negative integer"
        )
    max_kept = int(approximation_degree)
    if m < 1:
        return 0
    qft_degree = max(0, m - 1 - max_kept)
    if qft_degree >= m:
        qft_degree = m - 1
    return qft_degree


def _validate_name(name: str | None) -> str:
    """Validate the optional human-readable circuit name.

    Parameters
    ----------
    name :
        User provided string or ``None``.

    Returns
    -------
    str
        Normalised name, defaulting to ``"QuadraticForm"`` when ``None``.

    Raises
    ------
    TypeError
        If non-``None`` and not a ``str``.
    """
    if name is None:
        return "QuadraticForm"
    if not isinstance(name, str):
        raise TypeError("name must be a string or None")
    return name


# -- gate helpers ------------------------------------------------------------


def _append_k_phase(
    circuit: Circuit,
    controls: list[int],
    target: int,
    lam: float,
) -> None:
    """Append a ``len(controls)``-controlled Phase(lam) onto ``circuit``.

    Supports 0 controls (single-qubit P), 1 control (CP) and 2 controls
    (MCP) — the only gate arities required by the QuadraticForm schedule.

    Parameters
    ----------
    circuit :
        Target circuit to append the gate onto.
    controls :
        Control qubit indices.  Length 0, 1 or 2.
    target :
        Target qubit index.
    lam :
        Phase angle in radians.  Values of magnitude below ``_EPS`` are
        skipped as a no-op.
    """
    if abs(lam) < _EPS:
        return
    num_controls = len(controls)
    gate = StandardGate.Phase(lam).control(num_controls)
    circuit.append_mc_gate(gate, controls + [target])


# -- synthesis ---------------------------------------------------------------


def _build_quadratic_form_schedule(
    circuit: Circuit,
    A_arr: np.ndarray,
    b_arr: np.ndarray,
    c_float: float,
    n: int,
    m: int,
    input_indices: list[int],
    result_indices: list[int],
    qft_degree: int,
) -> None:
    """Append the QFT + phase-kickback + IQFT schedule into ``circuit``.

    The caller is responsible for providing a ``Circuit`` of the correct
    width and for mapping ``input_indices`` / ``result_indices`` into its
    local wire numbering.  This decomposition mirrors exactly the schedule
    described in the module-level docstring.

    Parameters
    ----------
    circuit :
        Target circuit to modify in-place.
    A_arr :
        Validated ``(n, n)`` float quadratic matrix.
    b_arr :
        Validated ``(n,)`` float linear vector.
    c_float :
        Validated float constant term.
    n :
        Input register width.
    m :
        Result register width.
    input_indices :
        ``n`` local wire indices for the read-only input register (LSB first).
    result_indices :
        ``m`` local wire indices for the result accumulator (LSB first).
    qft_degree :
        Internal ``TransformQFT`` approximation degree (already converted
        from the user-facing ``approximation_degree`` parameter).
    """
    scaling = _PI * (2 ** (1 - m))

    # ── 1. Forward QFT on result register ────────────────────────────
    fwd_qft_circ: Circuit = TransformQFT(
        m,
        approximation_degree=qft_degree,
        do_swaps=True,
        inverse=False,
    ).build_circuit()
    circuit.compose(fwd_qft_circ, result_indices)

    # ── 2. Phase kickback schedule ───────────────────────────────────

    # (a) Constant term c -> P gates on result qubits
    if abs(c_float) >= _EPS:
        for i in range(m):
            lam = scaling * (2**i) * c_float
            _append_k_phase(circuit, [], result_indices[i], lam)

    # (b) Linear + diagonal: (b_j + A_{jj}) -> CP gates
    for j in range(n):
        val = float(b_arr[j] + A_arr[j, j])
        if abs(val) < _EPS:
            continue
        ctrl_j = input_indices[j]
        for i in range(m):
            lam = scaling * (2**i) * val
            _append_k_phase(circuit, [ctrl_j], result_indices[i], lam)

    # (c) Off-diagonal quadratic: (A_{jk} + A_{kj}) -> MCP gates (2 ctrls)
    for j in range(n):
        for k in range(j + 1, n):
            val = float(A_arr[j, k] + A_arr[k, j])
            if abs(val) < _EPS:
                continue
            ctrls = [input_indices[j], input_indices[k]]
            for i in range(m):
                lam = scaling * (2**i) * val
                _append_k_phase(circuit, ctrls, result_indices[i], lam)

    # ── 3. Inverse QFT on result register ────────────────────────────
    inv_qft_circ: Circuit = TransformQFT(
        m,
        approximation_degree=qft_degree,
        do_swaps=True,
        inverse=True,
    ).build_circuit()
    circuit.compose(inv_qft_circ, result_indices)


# -- public class ------------------------------------------------------------


class QuadraticForm:
    """Quadratic form arithmetic operator with a fixed local qubit layout.

    The operator implements the unitary

    .. math::

        |x\\rangle_n |0\\rangle_m
            \\mapsto
        |x\\rangle_n \\bigl| (Q(x) + 2^m) \\bmod 2^m \\bigr\\rangle_m,

    with :math:`Q(x) = x^T A x + b^T x + c`.  The local wire layout is
    fixed to

    * input register   : qubits ``[m, m+n-1]``  (LSB-first, read-only)
    * result register  : qubits ``[0, m-1]``    (LSB-first, initialise ``|0>``)

    matching the conventions used by the Draper adder and QFT-weighted-sum
    operators in this package.

    Parameters
    ----------
    A :
        Quadratic coefficient matrix of shape ``(n, n)``.  Non-symmetric
        matrices are accepted: the contribution of pair ``(j, k)`` is
        ``A[j][k] + A[k][j]`` so that ``x^T A x`` is evaluated correctly
        regardless of symmetry.
    b :
        Linear coefficient vector of length ``n``.
    c :
        Constant (scalar) offset.
    num_input_qubits :
        Number ``n`` of input-register qubits.  Must satisfy
        ``isinstance(n, int) and not isinstance(n, bool) and n > 0``.
    num_result_qubits :
        Number ``m`` of result-register qubits.  Same type and sign
        constraints as ``num_input_qubits``.
    approximation_degree :
        Optional QFT approximation threshold forwarded to the forward and
        inverse Fourier transforms.  When ``None`` (default) the exact QFT
        is used.  Otherwise the value is clamped to the valid range
        ``[0, m-1]`` using the same clamping rule as the Draper adder
        family.
    name :
        Optional human readable circuit name.  Defaults to
        ``"QuadraticForm"``.

    Attributes
    ----------
    A :
        Stored quadratic coefficient matrix (defensive copy).
    b :
        Stored linear coefficient vector (defensive copy).
    c :
        Stored constant offset.
    num_input_qubits :
        Input register width ``n``.
    num_result_qubits :
        Result register width ``m``.
    approximation_degree :
        User-facing approximation threshold as provided.
    name :
        Human readable circuit name.
    num_qubits :
        Total local circuit width ``n + m``.
    input_register_indices :
        Local wire indices of the LSB-first input register.
    result_register_indices :
        Local wire indices of the LSB-first result register.
    """

    def __init__(
        self,
        A: ArrayLike,
        b: ArrayLike,
        c: float,
        *,
        num_input_qubits: int,
        num_result_qubits: int,
        approximation_degree: int | None = None,
        name: str | None = None,
    ) -> None:
        n = _validate_register_width(num_input_qubits, "num_input_qubits")
        m = _validate_register_width(num_result_qubits, "num_result_qubits")
        A_arr = _validate_A(A, n)
        b_arr = _validate_b(b, n)
        c_float = _validate_c(c)
        qft_degree = _validate_approximation_degree(approximation_degree, m)
        name_str = _validate_name(name)

        self._A = A_arr.copy()
        self._b = b_arr.copy()
        self._c = c_float
        self._num_input_qubits = n
        self._num_result_qubits = m
        self._approximation_degree = approximation_degree
        self._qft_degree = qft_degree
        self._name = name_str
        self._num_qubits = n + m
        self._input_register_indices = list(range(m, m + n))
        self._result_register_indices = list(range(m))

    # -- properties -----------------------------------------------------

    @property
    def A(self) -> NDArray[np.float64]:
        """Return the stored quadratic coefficient matrix."""
        return cast("NDArray[np.float64]", self._A.copy())

    @property
    def b(self) -> NDArray[np.float64]:
        """Return the stored linear coefficient vector."""
        return cast("NDArray[np.float64]", self._b.copy())

    @property
    def c(self) -> float:
        """Return the stored constant offset."""
        return self._c

    @property
    def num_input_qubits(self) -> int:
        """Return the input register width ``n``."""
        return self._num_input_qubits

    @property
    def num_result_qubits(self) -> int:
        """Return the result register width ``m``."""
        return self._num_result_qubits

    @property
    def approximation_degree(self) -> int | None:
        """Return the user-facing approximation threshold."""
        return self._approximation_degree

    @property
    def name(self) -> str:
        """Return the configured human readable circuit name."""
        return self._name

    @property
    def num_qubits(self) -> int:
        """Return the total local circuit width ``n + m``."""
        return self._num_qubits

    @property
    def input_register_indices(self) -> list[int]:
        """Return the local wire indices of the input register."""
        return list(self._input_register_indices)

    @property
    def result_register_indices(self) -> list[int]:
        """Return the local wire indices of the result register."""
        return list(self._result_register_indices)

    @property
    def phase_gate_counts(self) -> dict[str, int]:
        """Return the number of P/CP/MCP phase gates applied by the operator.

        The returned mapping has keys ``"P"``, ``"CP"`` and ``"MCP"``.  A
        phase gate is counted as applied when its magnitude is at least
        :data:`_EPS` (matching the circuit construction), so the counts are
        an exact predictor of the gates that will be emitted by
        :meth:`build_circuit`.
        """
        n = self._num_input_qubits
        m = self._num_result_qubits
        A = self._A
        b = self._b
        c = self._c
        scaling = _PI * (2 ** (1 - m))
        n_p = 0
        n_cp = 0
        n_mcp = 0

        if abs(c) >= _EPS:
            for i in range(m):
                if abs(scaling * (2**i) * c) >= _EPS:
                    n_p += 1

        for j in range(n):
            val = float(b[j] + A[j, j])
            if abs(val) < _EPS:
                continue
            for i in range(m):
                if abs(scaling * (2**i) * val) >= _EPS:
                    n_cp += 1

        for j in range(n):
            for k in range(j + 1, n):
                val = float(A[j, k] + A[k, j])
                if abs(val) < _EPS:
                    continue
                for i in range(m):
                    if abs(scaling * (2**i) * val) >= _EPS:
                        n_mcp += 1

        return {"P": n_p, "CP": n_cp, "MCP": n_mcp}

    # -- circuit construction -----------------------------------------

    def build_circuit(self) -> Circuit:
        """Return a fresh Circuit implementing this QuadraticForm instance.

        Returns
        -------
        Circuit
            A new cqlib ``Circuit`` instance of width ``num_qubits``
            preloaded with the full QFT / phase-kickback / IQFT schedule.
        """
        circuit = Circuit(self._num_qubits)
        _build_quadratic_form_schedule(
            circuit,
            self._A,
            self._b,
            self._c,
            self._num_input_qubits,
            self._num_result_qubits,
            self._input_register_indices,
            self._result_register_indices,
            self._qft_degree,
        )
        return circuit


# -- convenience function ----------------------------------------------------


def build_quadratic_form(
    A: ArrayLike,
    b: ArrayLike,
    c: float,
    *,
    num_input_qubits: int,
    num_result_qubits: int,
    approximation_degree: int | None = None,
) -> Circuit:
    """Thin wrapper that constructs a :class:`QuadraticForm` and returns its
    circuit immediately.

    See Also
    --------
    QuadraticForm : Full configuration class with layout properties.
    """
    return QuadraticForm(
        A,
        b,
        c,
        num_input_qubits=num_input_qubits,
        num_result_qubits=num_result_qubits,
        approximation_degree=approximation_degree,
    ).build_circuit()

