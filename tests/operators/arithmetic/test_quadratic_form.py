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
"""Test suite for the QuadraticForm operator.

Scope of this module:
    The public :class:`cqlib_algorithms.operators.arithmetic.QuadraticForm`
    class (and its thin functional wrapper
    :func:`cqlib_algorithms.operators.arithmetic.build_quadratic_form`) is
    exercised against the QuadraticForm test design document located under
    docs/design_md/test_design.  All tests rely on cqlib primitives
    (unitary simulation via :class:`cqlib.qis.Statevector` with
    :meth:`sample_shots`) and NumPy; no external quantum computing framework
    is imported.

Coverage map (case IDs match the design document):
    QF-VAL-001 .. QF-VAL-015   -> Construction parameter validation
    QF-ATTR-001 .. QF-ATTR-009 -> Property / layout interface
    QF-FUNC-001 .. QF-FUNC-014 -> Functional correctness across inputs
    QF-APPR-001 .. QF-APPR-005 -> approximation_degree clamping & influence
    QF-STRUCT-001 .. QF-STRUCT-007 -> Static gate-count & structural check
    QF-WRAP-001 .. QF-WRAP-002 -> build_quadratic_form thin wrapper
    QF-ROBUST-001 .. QF-ROBUST-005 -> Robustness & immutability guarantees
"""

from __future__ import annotations

import numpy as np
import pytest
from cqlib.qis import Statevector

from cqlib import Circuit
from cqlib_algorithms.operators.arithmetic import (
    QuadraticForm,
    build_quadratic_form,
)

RANDOM_SEED = 20260923
RANDOM_SAMPLE_SIZE = 20
LARGE_N = 6
LARGE_M = 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _fixed_random_seed() -> None:
    """Fix the global NumPy seed for every random-sampling test."""
    np.random.seed(RANDOM_SEED)
    yield


# ---------------------------------------------------------------------------
# Classical reference helpers (independent of the implementation)
# ---------------------------------------------------------------------------


def _classical_Q(
    A: np.ndarray,
    b: np.ndarray,
    c: float,
    x_int: int,
    n: int,
) -> float:
    """Return the classical quadratic form Q(x) = xᵀ A x + bᵀ x + c.

    The bit-string ``x_int`` is expanded as an LSB-first binary vector of
    length ``n`` so the conventions match :func:`_apply_x_to_value`.
    """
    x = np.array([(x_int >> k) & 1 for k in range(n)], dtype=np.float64)
    return float(x @ A @ x + b @ x + c)


def _encode_twos_complement(value: float, m: int) -> int:
    """Return the two's-complement encoding of ``value`` in an ``m``-bit reg.

    ``value`` is rounded to the nearest integer before encoding.  For any
    real scalar ``v`` the encoding is

        (⌊v⌉ + 2ᵐ) mod 2ᵐ,

    which is the document convention used in QuadraticForm (Gilliam et al.).
    """
    rounded = int(np.rint(value))
    modulus = 1 << m
    return (rounded + modulus) % modulus


def _apply_x_to_value(
    circuit: Circuit,
    value: int,
    bit_indices: list[int],
) -> None:
    """Flip qubits at ``bit_indices`` to encode integer value LSB-first."""
    width = len(bit_indices)
    if not 0 <= value < (1 << width):
        raise ValueError(
            f"Value {value} does not fit into a {width}-bit register."
        )
    for local_index, qubit in enumerate(bit_indices):
        if (value >> local_index) & 1:
            circuit.x(qubit)


def _decode_register_from_outcome(
    outcome,
    bit_indices: list[int],
) -> int:
    """Decode an LSB-first integer from selected qubit positions."""
    result = 0
    for shift, qubit in enumerate(bit_indices):
        if outcome.is_one(qubit):
            result |= 1 << shift
    return result


def _run_quadratic_form_single_x(
    qf: QuadraticForm,
    x_int: int,
) -> dict[str, int]:
    """Encode ``x_int``, run QF, return dict ``{input, result}`` (int form)."""
    n = qf.num_input_qubits
    m = qf.num_result_qubits
    num_qubits = n + m

    circuit = Circuit(num_qubits)
    _apply_x_to_value(circuit, x_int, qf.input_register_indices)
    circuit.compose(qf.build_circuit(), list(range(num_qubits)))

    statevector = Statevector.from_circuit(circuit)
    shots = statevector.sample_shots(1)
    outcome = shots[0]

    return {
        "input": _decode_register_from_outcome(
            outcome, qf.input_register_indices
        ),
        "result": _decode_register_from_outcome(
            outcome, qf.result_register_indices
        ),
    }


def _independent_fft_probe(
    A: np.ndarray,
    b: np.ndarray,
    c: float,
    n: int,
    m: int,
) -> np.ndarray:
    """Independent reference for the result-register probability distribution.

    For each computational-basis input ``x`` the ideal QuadraticForm
    circuit applies the sequence QFT → diagonal phase kickback (by ``κ Q(x)
    · Σ_i z_i 2^i``) → IQFT on the ``m``-qubit result register.  This
    helper computes the resulting ``|ψ|²`` distribution *directly* using
    the Fourier matrix convention that matches the operator implementation,
    producing an array shaped ``(2**n, 2**m)`` that is
    independent of the cqlib gate schedule.  The output is normalised per
    input row so that ``Σ_r ref[x, r] = 1``.
    """
    dim_x = 1 << n
    dim_z = 1 << m
    kappa = np.pi * 2 ** (1 - m)
    z = np.arange(dim_z, dtype=np.int64)
    amplitudes_sq = np.zeros((dim_x, dim_z), dtype=np.float64)

    for x_int in range(dim_x):
        qx = _classical_Q(A, b, c, x_int, n)
        phases = np.exp(1j * kappa * qx * z)
        psi = np.fft.fft(phases, norm="ortho")
        amplitudes_sq[x_int, :] = np.abs(psi) ** 2

    row_totals = amplitudes_sq.sum(axis=1, keepdims=True)
    amplitudes_sq /= np.where(row_totals == 0, 1, row_totals)
    return amplitudes_sq


def _collect_phase_gate_counts(qf: QuadraticForm) -> dict[str, int]:
    """Return the P/CP/MCP phase-gate counts via the operator property.

    Since ``cqlib.circuit.Circuit`` does not expose a public gate list, the
    counts are obtained from the :attr:`QuadraticForm.phase_gate_counts`
    read-only property on the operator itself.  That property mirrors the
    exact ``_EPS``-threshold logic used by the circuit builder, so it is a
    faithful predictor of the gates actually appended.
    """
    return dict(qf.phase_gate_counts)


def _circuits_unitary_equal(
    circuit_a: Circuit,
    circuit_b: Circuit,
    *,
    atol: float = 1e-12,
    rtol: float = 0.0,
) -> bool:
    """Return True iff the two circuits implement the same unitary up to a
    global phase.

    Unlike the functional tests above, this helper compares the full unitary
    matrices directly.  The comparison is therefore exact up to a single
    global phase factor and does not depend on basis-state sampling or
    probability-only heuristics.
    """
    if circuit_a.num_qubits != circuit_b.num_qubits:
        return False
    matrix_a = np.asarray(circuit_a.to_matrix())
    matrix_b = np.asarray(circuit_b.to_matrix())
    if matrix_a.shape != matrix_b.shape:
        return False
    nonzero = np.flatnonzero(np.abs(matrix_b) > atol)
    if nonzero.size == 0:
        return np.allclose(matrix_a, matrix_b, atol=atol, rtol=rtol)
    pivot = int(nonzero[0])
    phase = matrix_a.flat[pivot] / matrix_b.flat[pivot]
    return np.allclose(matrix_a, phase * matrix_b, atol=atol, rtol=rtol)


def _build_non_symmetric_A(n: int) -> np.ndarray:
    """Return a deterministic non-symmetric integer matrix for use in
    STRUCT-006 / FUNC-006 / FUNC-007 equivalence tests."""
    A = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        for k in range(n):
            A[j, k] = 1.0 + ((j + 1) * (k + 2)) % 5
    # Force asymmetry.
    if n >= 2:
        A[0, 1] += 2.0
        A[1, 0] -= 1.0
    return A


def _symmetrise(A: np.ndarray) -> np.ndarray:
    """Return the (unique) symmetric matrix equivalent to A under xᵀ A x."""
    return 0.5 * (A + A.T)


def _assert_clean_register_layout(qf: QuadraticForm) -> None:
    """Assert input/result registers do not overlap and cover no shared idx."""
    groups = [
        list(qf.input_register_indices),
        list(qf.result_register_indices),
    ]
    flat = [idx for group in groups for idx in group]
    assert len(flat) == len(set(flat)), (
        f"QuadraticForm register indices overlap: {flat}"
    )
    assert all(0 <= idx < (qf.num_input_qubits + qf.num_result_qubits)
               for idx in flat)


# ===========================================================================
# Section QF-VAL  --  Construction parameter validation (15 cases)
# ===========================================================================


class TestQuadraticFormValidation:
    """Invalid construction inputs raise the documented TypeError / ValueError.

    The test methods in this class are named after the corresponding
    ``QF-VAL-NNN`` design-document cases and additionally assert on the
    presence of keyword fragments in the raised exception messages.
    """

    # ------------------------------------------------------------------
    # QF-VAL-001  num_input_qubits is bool  ->  TypeError
    # ------------------------------------------------------------------

    def test_val_001_num_input_qubits_bool_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match="num_input_qubits"):
            QuadraticForm(
                [[0.0]], [0.0], 0.0,
                num_input_qubits=True,
                num_result_qubits=1,
            )

    # ------------------------------------------------------------------
    # QF-VAL-002  num_input_qubits is non-int -> TypeError
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("bad", [1.5, "2", None, [1], (1,)])
    def test_val_002_num_input_qubits_non_int_raises_typeerror(
        self, bad
    ) -> None:
        with pytest.raises(TypeError, match="num_input_qubits"):
            QuadraticForm(
                [[0.0]], [0.0], 0.0,
                num_input_qubits=bad,
                num_result_qubits=1,
            )

    # ------------------------------------------------------------------
    # QF-VAL-003  num_result_qubits is bool  ->  TypeError
    # ------------------------------------------------------------------

    def test_val_003_num_result_qubits_bool_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match="num_result_qubits"):
            QuadraticForm(
                [[0.0]], [0.0], 0.0,
                num_input_qubits=1,
                num_result_qubits=False,
            )

    # ------------------------------------------------------------------
    # QF-VAL-004  num_result_qubits is non-int -> TypeError
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("bad", [1.5, "2", None, [1], (1,)])
    def test_val_004_num_result_qubits_non_int_raises_typeerror(
        self, bad
    ) -> None:
        with pytest.raises(TypeError, match="num_result_qubits"):
            QuadraticForm(
                [[0.0]], [0.0], 0.0,
                num_input_qubits=1,
                num_result_qubits=bad,
            )

    # ------------------------------------------------------------------
    # QF-VAL-005  A not coercible to 2-D -> TypeError
    # ------------------------------------------------------------------

    def test_val_005_A_non_numeric_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match="2-D"):
            QuadraticForm(
                [["a", "b"], ["c", "d"]], [0.0, 0.0], 0.0,
                num_input_qubits=2,
                num_result_qubits=2,
            )

    # ------------------------------------------------------------------
    # QF-VAL-006  A wrong ndim -> TypeError
    # ------------------------------------------------------------------

    def test_val_006_A_1d_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match="2-D"):
            QuadraticForm(
                [1.0, 2.0], [0.0, 0.0], 0.0,
                num_input_qubits=2,
                num_result_qubits=2,
            )

    def test_val_006_A_3d_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match="2-D"):
            QuadraticForm(
                [[[1.0]]], [0.0], 0.0,
                num_input_qubits=1,
                num_result_qubits=2,
            )

    # ------------------------------------------------------------------
    # QF-VAL-007  A shape != (n,n) -> ValueError
    # ------------------------------------------------------------------

    def test_val_007_A_shape_mismatch_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match=r"shape"):
            QuadraticForm(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [0.0, 0.0],
                0.0,
                num_input_qubits=2,
                num_result_qubits=2,
            )

    # ------------------------------------------------------------------
    # QF-VAL-008  b not coercible to 1-D -> TypeError
    # ------------------------------------------------------------------

    def test_val_008_b_non_numeric_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match="1-D"):
            QuadraticForm(
                [[1.0]], ["bad"], 0.0,
                num_input_qubits=1,
                num_result_qubits=2,
            )

    # ------------------------------------------------------------------
    # QF-VAL-009  b is 2-D -> TypeError
    # ------------------------------------------------------------------

    def test_val_009_b_2d_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match="1-D"):
            QuadraticForm(
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0], [2.0]],
                0.0,
                num_input_qubits=2,
                num_result_qubits=2,
            )

    # ------------------------------------------------------------------
    # QF-VAL-010  len(b) != n -> ValueError
    # ------------------------------------------------------------------

    def test_val_010_b_length_mismatch_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="length"):
            QuadraticForm(
                [[1.0, 0.0], [0.0, 1.0]],
                [1.0],
                0.0,
                num_input_qubits=2,
                num_result_qubits=2,
            )

    # ------------------------------------------------------------------
    # QF-VAL-011  c not coercible to float -> TypeError
    # ------------------------------------------------------------------

    def test_val_011_c_non_numeric_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match="float"):
            QuadraticForm(
                [[0.0]], [0.0], "not a float",
                num_input_qubits=1,
                num_result_qubits=2,
            )

    # ------------------------------------------------------------------
    # QF-VAL-012  approximation_degree bool / float -> TypeError
    # ------------------------------------------------------------------

    def test_val_012_approximation_degree_bool_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match="approximation_degree"):
            QuadraticForm(
                [[0.0]], [0.0], 0.0,
                num_input_qubits=1,
                num_result_qubits=3,
                approximation_degree=True,
            )

    def test_val_012_approximation_degree_float_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match="approximation_degree"):
            QuadraticForm(
                [[0.0]], [0.0], 0.0,
                num_input_qubits=1,
                num_result_qubits=3,
                approximation_degree=1.5,
            )

    # ------------------------------------------------------------------
    # QF-VAL-013  approximation_degree negative -> ValueError
    # ------------------------------------------------------------------

    def test_val_013_approximation_degree_negative_raises_valueerror(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            QuadraticForm(
                [[0.0]], [0.0], 0.0,
                num_input_qubits=1,
                num_result_qubits=3,
                approximation_degree=-1,
            )

    # ------------------------------------------------------------------
    # QF-VAL-014  name wrong type -> TypeError
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("bad_name", [123, True, 1.0])
    def test_val_014_name_non_string_or_none_raises_typeerror(
        self, bad_name
    ) -> None:
        with pytest.raises(TypeError, match="name"):
            QuadraticForm(
                [[0.0]], [0.0], 0.0,
                num_input_qubits=1,
                num_result_qubits=2,
                name=bad_name,
            )

    # ------------------------------------------------------------------
    # QF-VAL-015  name=None resolves to default "QuadraticForm"
    # ------------------------------------------------------------------

    def test_val_015_name_none_defaults_to_class_name(self) -> None:
        op = QuadraticForm(
            [[0.0]], [0.0], 0.0,
            num_input_qubits=1,
            num_result_qubits=2,
            name=None,
        )
        assert op.name == "QuadraticForm"


# ===========================================================================
# Section QF-ATTR  --  Property / layout interface (9 cases)
# ===========================================================================


class TestQuadraticFormAttributes:
    """Read-only property behaviour matches the public contract."""

    def _make_example(self, n: int = 3, m: int = 4) -> QuadraticForm:
        A = np.diag([1.0, 2.0, 3.0]).tolist()
        b = [0.5, -1.0, 2.0]
        c = 1.5
        return QuadraticForm(
            A, b, c,
            num_input_qubits=n,
            num_result_qubits=m,
            approximation_degree=1,
            name="QFExample",
        )

    # ------------------------------------------------------------------
    # QF-ATTR-001  A returns defensive copy (mutating returned -> no change)
    # ------------------------------------------------------------------

    def test_attr_001_A_returns_defensive_copy(self) -> None:
        qf = self._make_example()
        A1 = qf.A
        A2 = qf.A
        assert A1 is not A2
        A1[0, 0] = 9999.0
        assert qf.A[0, 0] == pytest.approx(1.0)

    # ------------------------------------------------------------------
    # QF-ATTR-002  b returns defensive copy
    # ------------------------------------------------------------------

    def test_attr_002_b_returns_defensive_copy(self) -> None:
        qf = self._make_example()
        b1 = qf.b
        b2 = qf.b
        assert b1 is not b2
        b1[0] = 9999.0
        assert qf.b[0] == pytest.approx(0.5)

    # ------------------------------------------------------------------
    # QF-ATTR-003  c / num_input_qubits / num_result_qubits
    # ------------------------------------------------------------------

    def test_attr_003_c_stored_correctly(self) -> None:
        qf = self._make_example()
        assert qf.c == pytest.approx(1.5)

    def test_attr_003_num_input_qubits_n(self) -> None:
        qf = self._make_example(n=3, m=4)
        assert qf.num_input_qubits == 3

    def test_attr_003_num_result_qubits_m(self) -> None:
        qf = self._make_example(n=3, m=4)
        assert qf.num_result_qubits == 4

    # ------------------------------------------------------------------
    # QF-ATTR-006  approximation_degree returns user value (not clamped)
    # ------------------------------------------------------------------

    def test_attr_006_approximation_degree_returns_user_value(self) -> None:
        qf = QuadraticForm(
            [[0.0]], [0.0], 0.0,
            num_input_qubits=1,
            num_result_qubits=3,
            approximation_degree=5,
        )
        assert qf.approximation_degree == 5
        qf_none = QuadraticForm(
            [[0.0]], [0.0], 0.0,
            num_input_qubits=1,
            num_result_qubits=3,
            approximation_degree=None,
        )
        assert qf_none.approximation_degree is None

    # ------------------------------------------------------------------
    # QF-ATTR-007,008  layout contract
    # ------------------------------------------------------------------

    def test_attr_007_input_register_layout(self) -> None:
        n, m = 3, 4
        qf = self._make_example(n=n, m=m)
        assert qf.input_register_indices == list(range(m, m + n))

    def test_attr_008_result_register_layout(self) -> None:
        n, m = 3, 4
        qf = self._make_example(n=n, m=m)
        assert qf.result_register_indices == list(range(m))

    # ------------------------------------------------------------------
    # QF-ATTR-009  input_register_indices returns a fresh list
    # ------------------------------------------------------------------

    def test_attr_009_input_register_indices_returns_copy(self) -> None:
        qf = self._make_example()
        r1 = qf.input_register_indices
        r2 = qf.input_register_indices
        assert r1 == r2
        assert r1 is not r2


# ===========================================================================
# Section QF-FUNC  --  Functional correctness (14 cases)
# ===========================================================================


class TestQuadraticFormFunctional:
    """Evaluate QF across inputs and compare with the classical reference."""

    # ------------------------------------------------------------------
    # QF-FUNC-001  Zero coefficients (A=b=c=0) -> result register always 0
    # ------------------------------------------------------------------

    def test_func_001_all_zero_coefficients_yield_zero(self) -> None:
        n, m = 2, 3
        qf = QuadraticForm(
            [[0.0] * n for _ in range(n)], [0.0] * n, 0.0,
            num_input_qubits=n, num_result_qubits=m,
        )
        for x in range(1 << n):
            out = _run_quadratic_form_single_x(qf, x)
            assert out["input"] == x
            assert out["result"] == 0

    # ------------------------------------------------------------------
    # QF-FUNC-002  Constant only (c) -> result reg = (c + 2^m) mod 2^m
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("c", [0, 1, 5, -3, 7])
    def test_func_002_constant_only(self, c: int) -> None:
        n, m = 2, 4
        A = [[0.0] * n for _ in range(n)]
        b = [0.0] * n
        qf = QuadraticForm(
            A, b, c,
            num_input_qubits=n, num_result_qubits=m,
        )
        expected_encoding = _encode_twos_complement(c, m)
        for x in range(1 << n):
            out = _run_quadratic_form_single_x(qf, x)
            assert out["input"] == x
            assert out["result"] == expected_encoding

    # ------------------------------------------------------------------
    # QF-FUNC-003  Linear only (pure b)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "b",
        [
            [1, 0],   # sum x_0
            [0, 3],   # 3 x_1
            [2, -1],  # mixed
            [0, 0],   # zero
        ],
    )
    def test_func_003_linear_only(self, b: list[int]) -> None:
        n, m = 2, 4
        A = [[0.0] * n for _ in range(n)]
        qf = QuadraticForm(
            A, list(map(float, b)), 0.0,
            num_input_qubits=n, num_result_qubits=m,
        )
        for x in range(1 << n):
            out = _run_quadratic_form_single_x(qf, x)
            expected = _classical_Q(
                np.asarray(A), np.asarray(b, dtype=float), 0.0, x, n
            )
            assert out["input"] == x
            assert out["result"] == _encode_twos_complement(expected, m)

    # ------------------------------------------------------------------
    # QF-FUNC-004  Diagonal quadratic only
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "diag",
        [
            [1, 0],
            [0, 5],
            [-1, 2],
            [2, 3],
        ],
    )
    def test_func_004_diagonal_quadratic_only(self, diag: list[int]) -> None:
        n, m = 2, 4
        A = np.diag(list(map(float, diag)))
        b = [0.0] * n
        qf = QuadraticForm(A.tolist(), b, 0.0,
                           num_input_qubits=n, num_result_qubits=m)
        for x in range(1 << n):
            out = _run_quadratic_form_single_x(qf, x)
            expected = _classical_Q(A, np.asarray(b), 0.0, x, n)
            assert out["result"] == _encode_twos_complement(expected, m)

    # ------------------------------------------------------------------
    # QF-FUNC-005  Pure off-diagonal  A_{01}=1 (symmetric)
    # ------------------------------------------------------------------

    def test_func_005_off_diagonal_only(self) -> None:
        n, m = 2, 4
        A = np.array([[0.0, 1.0], [1.0, 0.0]])
        b = [0.0, 0.0]
        qf = QuadraticForm(A, b, 0.0,
                           num_input_qubits=n, num_result_qubits=m)
        for x in range(1 << n):
            out = _run_quadratic_form_single_x(qf, x)
            expected = _classical_Q(A, np.asarray(b), 0.0, x, n)
            assert out["result"] == _encode_twos_complement(expected, m)

    # ------------------------------------------------------------------
    # QF-FUNC-006  Non-symmetric A matches symmetrised equivalent
    # ------------------------------------------------------------------

    def test_func_006_non_symmetric_matches_symmetrised(self) -> None:
        n, m = 3, 5
        A_nonsym = _build_non_symmetric_A(n)
        A_sym = _symmetrise(A_nonsym)
        b = np.array([1.0, -2.0, 3.0])
        c = 2.0
        qf1 = QuadraticForm(A_nonsym, b, c,
                            num_input_qubits=n, num_result_qubits=m)
        qf2 = QuadraticForm(A_sym, b, c,
                            num_input_qubits=n, num_result_qubits=m)
        for x in range(1 << n):
            out1 = _run_quadratic_form_single_x(qf1, x)
            out2 = _run_quadratic_form_single_x(qf2, x)
            assert out1["input"] == out2["input"] == x
            assert out1["result"] == out2["result"]

    # ------------------------------------------------------------------
    # QF-FUNC-007  Combined: Q(x) = xᵀ A x + bᵀ x + c  (integer coeffs)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("n,m", [(2, 4), (3, 5), (1, 3)])
    def test_func_007_combined_integer_coefficients(
        self, n: int, m: int
    ) -> None:
        A = np.zeros((n, n), dtype=np.float64)
        for j in range(n):
            for k in range(j, n):
                v = 1.0 + (j + 2 * k) % 3
                A[j, k] = v
                A[k, j] = v
        b = np.array([(i + 1) % 4 - 1 for i in range(n)], dtype=np.float64)
        c = float((-1) ** n)
        qf = QuadraticForm(A, b, c,
                           num_input_qubits=n, num_result_qubits=m)
        for x in range(1 << n):
            out = _run_quadratic_form_single_x(qf, x)
            expected = _classical_Q(A, b, c, x, n)
            assert out["result"] == _encode_twos_complement(expected, m)

    # ------------------------------------------------------------------
    # QF-FUNC-008  Input register is never modified
    # ------------------------------------------------------------------

    def test_func_008_input_register_preserved(self) -> None:
        n, m = 3, 4
        A = np.ones((n, n), dtype=np.float64)
        b = np.arange(n, dtype=np.float64)
        qf = QuadraticForm(A, b, 1.0,
                           num_input_qubits=n, num_result_qubits=m)
        for x in range(1 << n):
            out = _run_quadratic_form_single_x(qf, x)
            assert out["input"] == x

    # ------------------------------------------------------------------
    # QF-FUNC-009  Two's-complement behaviour when Q(x) negative
    # ------------------------------------------------------------------

    def test_func_009_negative_result_twos_complement(self) -> None:
        n, m = 2, 3
        A = [[0.0, 0.0], [0.0, 0.0]]
        b = [0.0, 0.0]
        c = -3
        qf = QuadraticForm(A, b, c,
                           num_input_qubits=n, num_result_qubits=m)
        expected = _encode_twos_complement(-3, m)
        for x in range(1 << n):
            out = _run_quadratic_form_single_x(qf, x)
            assert out["result"] == expected

    # ------------------------------------------------------------------
    # QF-FUNC-010  Large combined value (Q(x) > 2^(m-1)) wraps modulo 2^m
    # ------------------------------------------------------------------

    def test_func_010_overflow_wraps_modulo(self) -> None:
        n, m = 2, 3
        A = [[0.0, 0.0], [0.0, 0.0]]
        b = [0.0, 0.0]
        c = 9  # 9 in twos-complement 3-bit = (9+8) mod 8 = 1
        qf = QuadraticForm(A, b, float(c),
                           num_input_qubits=n, num_result_qubits=m)
        expected = _encode_twos_complement(c, m)
        for x in range(1 << n):
            out = _run_quadratic_form_single_x(qf, x)
            assert out["result"] == expected

    # ------------------------------------------------------------------
    # QF-FUNC-011  Minimal dimensions n=1, m=1
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "A,b,c",
        [
            ([[0.0]], [0.0], 0.0),
            ([[1.0]], [1.0], 1.0),
        ],
    )
    def test_func_011_minimal_dimensions_n1_m1(
        self, A, b, c
    ) -> None:
        n, m = 1, 1
        qf = QuadraticForm(A, b, c,
                           num_input_qubits=n, num_result_qubits=m)
        for x in range(1 << n):
            out = _run_quadratic_form_single_x(qf, x)
            expected = _classical_Q(
                np.asarray(A), np.asarray(b), float(c), x, n
            )
            assert out["result"] == _encode_twos_complement(expected, m)

    # ------------------------------------------------------------------
    # QF-FUNC-012  Larger n=4, m=6 exhaustive
    # ------------------------------------------------------------------

    def test_func_012_exhaustive_larger_instance_n4_m6(self) -> None:
        n, m = 4, 6
        A = np.zeros((n, n), dtype=np.float64)
        for j in range(n):
            for k in range(j, n):
                v = ((j + 1) * (k + 1)) % 5
                A[j, k] = float(v)
                A[k, j] = float(v)
        b = np.array([1, -2, 3, -1], dtype=np.float64)
        c = -1.0
        qf = QuadraticForm(A, b, c,
                           num_input_qubits=n, num_result_qubits=m)
        for x in range(1 << n):
            out = _run_quadratic_form_single_x(qf, x)
            expected = _classical_Q(A, b, c, x, n)
            assert out["result"] == _encode_twos_complement(expected, m)

    # ------------------------------------------------------------------
    # QF-FUNC-013  Random sampling for dimensions too big for exhaustive
    # ------------------------------------------------------------------

    def test_func_013_large_random_sample(self) -> None:
        n, m = LARGE_N, LARGE_M
        A = np.zeros((n, n), dtype=np.float64)
        for j in range(n):
            for k in range(j, n):
                v = (j * 3 + k * 2 + 1) % 7
                A[j, k] = float(v)
                A[k, j] = float(v)
        b = (np.arange(n) % 5 - 2).astype(np.float64)
        c = float(((n + 1) % 3) - 1)
        qf = QuadraticForm(A, b, c,
                           num_input_qubits=n, num_result_qubits=m)
        for _ in range(RANDOM_SAMPLE_SIZE):
            x = int(np.random.randint(0, 1 << n))
            out = _run_quadratic_form_single_x(qf, x)
            expected = _classical_Q(A, b, c, x, n)
            assert out["input"] == x
            assert out["result"] == _encode_twos_complement(expected, m)

    # ------------------------------------------------------------------
    # QF-FUNC-014  Fractional coefficients: compare amplitude distribution
    #                against independent FFT reference (|ψ|², atol<1e-10)
    # ------------------------------------------------------------------

    def test_func_014_fractional_coefficients_fft_reference(self) -> None:
        n, m = 2, 3
        A = np.array([
            [0.25, 0.5],
            [0.5, -0.75],
        ])
        b = np.array([0.1, -0.3])
        c = 0.4
        qf = QuadraticForm(A, b, c,
                           num_input_qubits=n, num_result_qubits=m)
        reference = _independent_fft_probe(A, b, c, n, m)

        for x in range(1 << n):
            circuit = Circuit(n + m)
            _apply_x_to_value(circuit, x, qf.input_register_indices)
            circuit.compose(qf.build_circuit(), list(range(n + m)))
            sv = np.asarray(Statevector.from_circuit(circuit).data)

            observed = np.zeros(1 << m, dtype=np.float64)
            for idx in range(1 << (n + m)):
                input_part = 0
                for p, qubit in enumerate(qf.input_register_indices):
                    if (idx >> qubit) & 1:
                        input_part |= 1 << p
                if input_part != x:
                    continue
                result_part = 0
                for p, qubit in enumerate(qf.result_register_indices):
                    if (idx >> qubit) & 1:
                        result_part |= 1 << p
                observed[result_part] += abs(sv[idx]) ** 2
            total = observed.sum()
            if total < 1e-30:
                continue
            observed /= total
            assert np.allclose(observed, reference[x, :], atol=1e-8, rtol=1e-8)


# ===========================================================================
# Section QF-APPR  --  approximation_degree clamping / effect (5 cases)
# ===========================================================================


class TestQuadraticFormApproximation:
    """approximation_degree clamping / monotonic behaviour."""

    # ------------------------------------------------------------------
    # QF-APPR-001  None vs >=m-1 -> produce identical (exact) circuits
    # ------------------------------------------------------------------

    def test_appr_001_none_and_ge_m1_produce_identical_circuits(self) -> None:
        n, m = 2, 4
        A = [[1.0, 0.5], [0.5, 2.0]]
        b = [1.0, -1.0]
        c = 0.5
        op_none = QuadraticForm(A, b, c, num_input_qubits=n,
                                num_result_qubits=m, approximation_degree=None)
        op_big = QuadraticForm(A, b, c, num_input_qubits=n,
                               num_result_qubits=m,
                               approximation_degree=m - 1 + 3)
        assert _circuits_unitary_equal(
            op_none.build_circuit(), op_big.build_circuit()
        )

    # ------------------------------------------------------------------
    # QF-APPR-002  d=0 produces a distinct, maximum-approximation circuit
    # ------------------------------------------------------------------

    def test_appr_002_d0_is_different_from_exact(self) -> None:
        n, m = 2, 4
        A = [[1.0, 2.0], [3.0, 4.0]]
        b = [-1.0, 5.0]
        c = 0.25
        op_exact = QuadraticForm(A, b, c, num_input_qubits=n,
                                 num_result_qubits=m,
                                 approximation_degree=None)
        op_d0 = QuadraticForm(A, b, c, num_input_qubits=n,
                              num_result_qubits=m, approximation_degree=0)
        assert not _circuits_unitary_equal(
            op_exact.build_circuit(), op_d0.build_circuit()
        )

    # ------------------------------------------------------------------
    # QF-APPR-003  d stored as user-provided, not clamped
    # ------------------------------------------------------------------

    def test_appr_003_stored_value_not_clamped(self) -> None:
        op = QuadraticForm([[0.0]], [0.0], 0.0,
                           num_input_qubits=1, num_result_qubits=2,
                           approximation_degree=99)
        assert op.approximation_degree == 99

    # ------------------------------------------------------------------
    # QF-APPR-004  Negative raises ValueError (VAL-013 covered, here at API)
    # ------------------------------------------------------------------

    def test_appr_004_negative_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            QuadraticForm([[0.0]], [0.0], 0.0,
                          num_input_qubits=1, num_result_qubits=2,
                          approximation_degree=-1)

    # ------------------------------------------------------------------
    # QF-APPR-005  Frob distance to exact (compared to exact) is monotonic
    #               decreasing in d.
    # ------------------------------------------------------------------

    @staticmethod
    def _frob_distance(ca: Circuit, cb: Circuit) -> float:
        n_q = ca.num_qubits
        diff_sq = 0.0
        for basis in range(1 << min(12, n_q)):
            pa = Circuit(n_q)
            pb = Circuit(n_q)
            for k in range(n_q):
                if (basis >> k) & 1:
                    pa.x(k)
                    pb.x(k)
            pa.compose(ca, list(range(n_q)))
            pb.compose(cb, list(range(n_q)))
            sa = np.asarray(Statevector.from_circuit(pa).data)
            sb = np.asarray(Statevector.from_circuit(pb).data)
            diff_sq += float(np.sum(np.abs(sa - sb) ** 2))
        return float(np.sqrt(diff_sq))

    def test_appr_005_distance_monotonic_decreasing_in_d(self) -> None:
        n, m = 2, 4
        A = [[1.0, 2.0], [2.0, -1.0]]
        b = [1.0, -3.0]
        c = 0.5
        ops = {
            d: QuadraticForm(A, b, c, num_input_qubits=n,
                             num_result_qubits=m, approximation_degree=d)
            for d in range(m)
        }
        op_exact = QuadraticForm(A, b, c, num_input_qubits=n,
                                 num_result_qubits=m,
                                 approximation_degree=None).build_circuit()
        distances = [
            self._frob_distance(ops[d].build_circuit(), op_exact)
            for d in range(m)
        ]
        for prev, nxt in zip(distances, distances[1:]):
            assert nxt <= prev + 1e-12


# ===========================================================================
# Section QF-STRUCT  --  Gate-count / structural checks (7 cases)
# ===========================================================================


class TestQuadraticFormStructure:
    """Static structural properties independent of simulation outcome."""

    # ------------------------------------------------------------------
    # QF-STRUCT-001  Gate counts match theoretical (P: m, CP: n·m,
    #                MCP: n(n-1)/2·m) when every non-zero contribution has
    #                magnitude >= 1e-14.
    # ------------------------------------------------------------------

    def test_struct_001_gate_counts_with_all_nonzero(self) -> None:
        n, m = 3, 4
        A = np.ones((n, n), dtype=np.float64)
        b = np.ones(n, dtype=np.float64)
        c = 1.0
        qf = QuadraticForm(A, b, c, num_input_qubits=n, num_result_qubits=m)
        counts = _collect_phase_gate_counts(qf)
        assert counts["P"] == m
        assert counts["CP"] == n * m
        assert counts["MCP"] == n * (n - 1) // 2 * m

    # ------------------------------------------------------------------
    # QF-STRUCT-002  c == 0  -> no P gates
    # ------------------------------------------------------------------

    def test_struct_002_c_zero_no_constant_p_gates(self) -> None:
        n, m = 3, 4
        A = np.ones((n, n), dtype=np.float64)
        b = np.ones(n, dtype=np.float64)
        c = 0.0
        qf = QuadraticForm(A, b, c, num_input_qubits=n, num_result_qubits=m)
        counts = _collect_phase_gate_counts(qf)
        assert counts["P"] == 0

    # ------------------------------------------------------------------
    # QF-STRUCT-003  All b_j + A_jj == 0  -> no CP gates
    # ------------------------------------------------------------------

    def test_struct_003_no_cp_when_b_plus_diag_A_zero(self) -> None:
        n, m = 3, 4
        A = -np.eye(n, dtype=np.float64)
        b = np.ones(n, dtype=np.float64)
        A_off = np.tril(np.ones((n, n), dtype=np.float64), -1) + \
            np.triu(np.ones((n, n), dtype=np.float64), 1)
        A = A + A_off
        c = 1.0
        qf = QuadraticForm(A, b, c, num_input_qubits=n, num_result_qubits=m)
        counts = _collect_phase_gate_counts(qf)
        assert counts["CP"] == 0

    # ------------------------------------------------------------------
    # QF-STRUCT-004  All A_jk + A_kj == 0 for j<k  -> no MCP gates
    # ------------------------------------------------------------------

    def test_struct_004_no_mcp_when_off_diag_zero(self) -> None:
        n, m = 3, 4
        A = np.diag([1.0, 2.0, 3.0])
        b = np.ones(n, dtype=np.float64)
        c = 1.0
        qf = QuadraticForm(A, b, c, num_input_qubits=n, num_result_qubits=m)
        counts = _collect_phase_gate_counts(qf)
        assert counts["MCP"] == 0

    # ------------------------------------------------------------------
    # QF-STRUCT-005  Result and input registers never overlap
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("n,m", [(1, 1), (3, 5), (6, 2)])
    def test_struct_005_registers_no_overlap(self, n: int, m: int) -> None:
        qf = QuadraticForm(
            [[0.0] * n for _ in range(n)], [0.0] * n, 0.0,
            num_input_qubits=n, num_result_qubits=m,
        )
        _assert_clean_register_layout(qf)

    # ------------------------------------------------------------------
    # QF-STRUCT-006  Non-symmetric A -> same unitary as symmetrised A
    # ------------------------------------------------------------------

    def test_struct_006_nonsym_vs_symm_unitary_equal(self) -> None:
        n, m = 3, 5
        A_nonsym = _build_non_symmetric_A(n)
        A_sym = _symmetrise(A_nonsym)
        b = np.array([1.0, -2.0, 3.0])
        c = 2.0
        circ1 = QuadraticForm(A_nonsym, b, c, num_input_qubits=n,
                              num_result_qubits=m).build_circuit()
        circ2 = QuadraticForm(A_sym, b, c, num_input_qubits=n,
                              num_result_qubits=m).build_circuit()
        assert _circuits_unitary_equal(circ1, circ2)

    # ------------------------------------------------------------------
    # QF-STRUCT-007  Diagonal-only A -> zero MCP count (same as -004)
    # ------------------------------------------------------------------

    def test_struct_007_diagonal_A_zero_mcp(self) -> None:
        n, m = 4, 3
        A = np.diag([1.0, -1.0, 2.0, 3.0])
        b = np.ones(n, dtype=np.float64)
        c = 0.5
        qf = QuadraticForm(A, b, c, num_input_qubits=n, num_result_qubits=m)
        counts = _collect_phase_gate_counts(qf)
        assert counts["MCP"] == 0


# ===========================================================================
# Section QF-WRAP  --  build_quadratic_form thin wrapper (2 cases)
# ===========================================================================


class TestQuadraticFormWrapper:
    """Thin functional wrapper behaves exactly like the class API."""

    # ------------------------------------------------------------------
    # QF-WRAP-001  Same input types accepted (numpy / list)
    # ------------------------------------------------------------------

    def test_wrap_001_accepts_list_and_numpy_inputs(self) -> None:
        A_list = [[1.0, 0.5], [0.5, 2.0]]
        b_list = [1.0, -1.0]
        A_np = np.array(A_list)
        b_np = np.array(b_list)
        c = 0.5
        circ_list = build_quadratic_form(
            A_list, b_list, c, num_input_qubits=2, num_result_qubits=4
        )
        circ_np = build_quadratic_form(
            A_np, b_np, c, num_input_qubits=2, num_result_qubits=4
        )
        assert isinstance(circ_list, Circuit)
        assert isinstance(circ_np, Circuit)
        assert circ_list.num_qubits == circ_np.num_qubits == 6

    # ------------------------------------------------------------------
    # QF-WRAP-002  Circuit matches QuadraticForm(...).build_circuit()
    # ------------------------------------------------------------------

    def test_wrap_002_unitary_matches_class_build_circuit(self) -> None:
        A = [[1.0, 2.0], [2.0, -1.0]]
        b = [1.0, -3.0]
        c = 0.25
        n, m = 2, 4
        circ_wrap = build_quadratic_form(
            A, b, c, num_input_qubits=n, num_result_qubits=m,
            approximation_degree=1,
        )
        circ_class = QuadraticForm(
            A, b, c, num_input_qubits=n, num_result_qubits=m,
            approximation_degree=1,
        ).build_circuit()
        assert _circuits_unitary_equal(circ_wrap, circ_class)


# ===========================================================================
# Section QF-ROBUST  --  Robustness & immutability (5 cases)
# ===========================================================================


class TestQuadraticFormRobustness:
    """Build immutability and repeated-build independence."""

    # ------------------------------------------------------------------
    # QF-ROBUST-001  Two successive build_circuit() calls are independent
    #                new objects (not the same, but unitary equal).
    # ------------------------------------------------------------------

    def test_robust_001_repeated_builds_independent_objects(self) -> None:
        qf = QuadraticForm(
            [[1.0, 2.0], [2.0, 3.0]], [0.0, 0.0], 1.0,
            num_input_qubits=2, num_result_qubits=4,
        )
        c1 = qf.build_circuit()
        c2 = qf.build_circuit()
        assert c1 is not c2
        assert _circuits_unitary_equal(c1, c2)

    # ------------------------------------------------------------------
    # QF-ROBUST-002  list / numpy / tuple inputs all produce identical
    #                unitaries.
    # ------------------------------------------------------------------

    def test_robust_002_diverse_container_types_equivalent(self) -> None:
        A_list = [[1.0, 0.5], [0.5, -1.0]]
        A_tuple = tuple(tuple(row) for row in A_list)
        A_np = np.array(A_list)
        b_list = [1.0, -2.0]
        b_tuple = (1.0, -2.0)
        b_np = np.array([1.0, -2.0])
        c_int = 2
        c_float = 2.0
        n, m = 2, 4

        def _build(A_in, b_in, c_in) -> Circuit:
            return QuadraticForm(
                A_in, b_in, c_in,
                num_input_qubits=n, num_result_qubits=m,
            ).build_circuit()

        circs = [
            _build(A_list, b_list, c_int),
            _build(A_tuple, b_tuple, c_float),
            _build(A_np, b_np, c_int),
        ]
        for i in range(1, len(circs)):
            assert _circuits_unitary_equal(circs[0], circs[i])

    # ------------------------------------------------------------------
    # QF-ROBUST-003  Mutating external A/b containers AFTER construction
    #                does not affect the stored values.
    # ------------------------------------------------------------------

    def test_robust_003_external_mutation_after_construction_irrelevant(
        self,
    ) -> None:
        A = [[1.0, 0.0], [0.0, 2.0]]
        b = [1.0, -1.0]
        qf = QuadraticForm(
            A, b, 0.0, num_input_qubits=2, num_result_qubits=4,
        )
        A[0][0] = 9999.0
        A[0][1] = -9999.0
        b[0] = 9999.0
        assert qf.A[0, 0] == pytest.approx(1.0)
        assert qf.A[0, 1] == pytest.approx(0.0)
        assert qf.b[0] == pytest.approx(1.0)

    # ------------------------------------------------------------------
    # QF-ROBUST-004  Integer c / integer A / integer b elements all
    #                accepted (implicit float coercion).
    # ------------------------------------------------------------------

    def test_robust_004_integer_parameters_accepted(self) -> None:
        n, m = 2, 4
        A_int = [[1, 0], [0, -2]]
        b_int = [-3, 4]
        c_int = 2
        qf = QuadraticForm(
            A_int, b_int, c_int,
            num_input_qubits=n, num_result_qubits=m,
        )
        assert np.asarray(qf.A).dtype.kind == "f"
        assert np.asarray(qf.b).dtype.kind == "f"
        assert isinstance(qf.c, float)

    # ------------------------------------------------------------------
    # QF-ROBUST-005  With random integer coefficients (A, b, c), every
    #                randomly picked input x yields the classically
    #                expected two's-complement encoding.
    # ------------------------------------------------------------------

    def test_robust_005_random_integer_coefficients_random_inputs(
        self,
    ) -> None:
        n, m = 4, 6
        A = np.random.randint(-4, 5, size=(n, n)).astype(np.float64)
        b = np.random.randint(-4, 5, size=n).astype(np.float64)
        c = float(np.random.randint(-4, 5))
        qf = QuadraticForm(A, b, c, num_input_qubits=n, num_result_qubits=m)
        for _ in range(RANDOM_SAMPLE_SIZE):
            x = int(np.random.randint(0, 1 << n))
            out = _run_quadratic_form_single_x(qf, x)
            expected = _classical_Q(A, b, c, x, n)
            assert out["result"] == _encode_twos_complement(expected, m)
