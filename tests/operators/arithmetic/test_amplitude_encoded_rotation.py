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

"""Test suite for the amplitude-encoded rotation operator family.

Scope of this module:
    The uniformly controlled Pauli rotations (``UCRX``, ``UCRY``, ``UCRZ``)
    and the ``ExactReciprocal`` function-table rotation class are tested
    against the design document
    ``docs/design_md/test_design/aef_test_cases_1.md``.  The
    ``LinearPauliRotation`` and ``PolynomialPauliRotation`` classes are
    intentionally omitted from this suite because their synthesis strategy
    is *not* a pure alpha table look-up; they will be rewritten via
    controlled-monomial construction and re-added once ready.

Coverage map (each tag references a section / case ID in the doc):

    * Section 0   Helper infrastructure for statevector / gate counting
    * Section 1   Level 0 — ``UCRY`` Gate (UCRY-001 .. UCRY-005)
    * Section 2   Level 1 — Exact reciprocal (UCR look-up table)
                      (REC-001 .. REC-004, plus validation tests)
    * Section 3   Level 2 — Negative-value support
                      (NEG-REC-001, NEG-002 / NEG-003)
    * Section 4   Level 3 — Bounds & optimisations
                      (OPT-001, OPT-002, OPT-004)
    * Section 5   Level 4 — Integration & invariants
                      (INT-001 unitarity,
                       INT-003 superposition linearity,
                       INT-005 gate-count scaling curve;
                       INT-004 Qiskit comparison is skipped if Qiskit absent)
    * Section 6   UCRX / UCRZ — symmetric sanity checks vs UCRY
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest
from cqlib.qis.state import Statevector

from cqlib import Circuit
from cqlib_algorithms.operators.arithmetic import (
    UCRX,
    UCRY,
    UCRZ,
    ExactReciprocal,
)

AMPLITUDE_ATOL = 1e-10
AMPLITUDE_RTOL = 1e-10
UNITARITY_ATOL = 1e-10

# External-library dependency detection is centralised in
# tests/operators/arithmetic/conftest.py.  The conftest also registers a
# ``requires_qiskit`` pytest marker which does the equivalent skipif
# declaratively; the small inline probe is kept here too so that
# ``skipif(not HAS_QISKIT, ...)`` can be used inline without traversing the
# conftest's ``pytest``-only mechanism.
# NOTE: ``tests`` is deliberately left as a plain directory (no
# ``__init__.py``), so ``from tests.operators.arithmetic.conftest import ...``
# would fail at collection time.




# ---------------------------------------------------------------------------
# Section 0.  Simulation / assertion helpers -------------------------------
# ---------------------------------------------------------------------------


def _state_index(x: int, n: int, ancilla: int) -> int:
    """Return the statevector index for ``|x> |ancilla>``.

    Bit layout (LSB-first matches cqlib + downstream design doc):
      * qubits ``[0, n-1]`` -> control ``x`` (LSB = qubit 0)
      * qubit ``n`` -> ancilla ``|0>`` or ``|1>``.

    The design document gives the same formula:
      ``|x,0>`` index = ``x``; ``|x,1>`` index = ``x + 2**n``.
    """

    return x + (ancilla << n)


def _statevector_data(sv: Statevector) -> np.ndarray:
    """Return the complex amplitude vector from a ``cqlib`` ``Statevector``.

    Cqlib exposes amplitudes either via ``sv.data`` (attribute ndarray) or
    ``sv.data()`` (callable).  This helper tries both so tests remain
    portable across versions.
    """

    raw = sv.data
    if callable(raw):
        raw = raw()
    return np.asarray(raw, dtype=np.complex128)


def _assert_branch_amplitudes(
    circuit: Circuit,
    n: int,
    x: int,
    *,
    amp_0: complex,
    amp_1: complex,
) -> None:
    """Assert the two non-zero components produced when preparing ``|x>|0>``.

    Applies ``U`` to a product basis state ``|x> |0>`` and checks that only
    amplitudes ``<x, 0| psi>`` and ``<x, 1| psi>`` survive (input
    invariance on the control register, i.e.  the ``REC-004`` / §1.4
    condition).
    """

    full = Circuit(n + 1)
    for bit in range(n):
        if (x >> bit) & 1:
            full.x(bit)
    full.compose(circuit, list(range(n + 1)))

    sv = _statevector_data(Statevector.from_circuit(full))

    # Explicit zero for every other branch (input-invariance / UCR structure).
    expected = np.zeros_like(sv)
    idx_0 = _state_index(x, n, 0)
    idx_1 = _state_index(x, n, 1)
    expected[idx_0] = amp_0
    expected[idx_1] = amp_1
    np.testing.assert_allclose(
        sv,
        expected,
        atol=AMPLITUDE_ATOL,
        rtol=AMPLITUDE_RTOL,
        err_msg=(
            f"Branch mismatch for control basis state x={x} "
            f"(n={n}).  Expected amplitudes |{x}0>={amp_0:.4e}, "
            f"|{x}1>={amp_1:.4e}."
        ),
    )


def _count_ops(circuit: Circuit) -> Counter[str]:
    """Return a lowercase gate-name counter from a ``cqlib.Circuit``.

    Cqlib exposes the gate list via ``circuit.operations`` (each element has
    a ``.name`` attribute).  The helper also tolerates a Qiskit-style
    ``.instructions`` fallback or ``.count_ops()`` so tests remain portable
    across simulator versions.
    """

    if hasattr(circuit, "count_ops") and callable(circuit.count_ops):
        counts_raw = circuit.count_ops()
        return Counter({str(k).lower(): int(v) for k, v in counts_raw.items()})
    if hasattr(circuit, "operations"):
        iterable = circuit.operations
    elif hasattr(circuit, "instructions"):
        iterable = circuit.instructions  # pragma: no cover - portability
    else:
        return Counter()
    counts: Counter[str] = Counter()
    for instr in iterable:
        name = ""
        if hasattr(instr, "name"):
            name = str(instr.name).lower()
        elif hasattr(instr, "__getitem__") and len(instr) > 0:
            name = str(instr[0]).lower()
        if name:
            counts[name] += 1
    return counts


def _matrix_from_circuit(circuit: Circuit) -> np.ndarray:
    """Return the dense unitary matrix of ``circuit`` via Statevector probing.

    Uses the ``2^m`` basis-state probes; fine for tests with ``m <= 10`` which
    covers every curve-point in ``INT-005`` (n up to 4).
    """

    m = circuit.num_qubits
    dim = 1 << m
    mat = np.zeros((dim, dim), dtype=np.complex128)
    for basis in range(dim):
        probe = Circuit(m)
        for bit in range(m):
            if (basis >> bit) & 1:
                probe.x(bit)
        probe.compose(circuit, list(range(m)))
        sv = _statevector_data(Statevector.from_circuit(probe))
        mat[:, basis] = sv
    return mat


def _rot_2x2(axis: str, theta: float) -> np.ndarray:
    """Return the 2x2 single-qubit Pauli rotation matrix (global phase free).

    ::

        Rx(θ) = [[cos θ/2,    -i sin θ/2],
                 [-i sin θ/2,  cos θ/2]]

        Ry(θ) = [[cos θ/2,    -sin θ/2],
                 [ sin θ/2,    cos θ/2]]

        Rz(θ) = [[e^{-i θ/2},          0],
                 [          0,  e^{i θ/2}]]
    """

    c = complex(math.cos(theta / 2.0))
    s = complex(math.sin(theta / 2.0))
    if axis == "x":
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
    if axis == "y":
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    if axis == "z":
        return np.array(
            [[complex(math.cos(theta / 2.0) - 1j * math.sin(theta / 2.0)), 0j],
             [0j, complex(math.cos(theta / 2.0) + 1j * math.sin(theta / 2.0))]],
            dtype=np.complex128,
        )
    raise ValueError(f"Unknown axis: {axis!r}")


def _exact_ucr_matrix(axis: str, alpha: np.ndarray) -> np.ndarray:
    """Exact block-diagonal UCR matrix ``∑_x |x><x| ⊗ R_axis(α_x)``.

    The resulting matrix has shape ``(2*N, 2*N)`` where ``N == len(alpha)``,
    i.e. ``2^{num_controls+1}`` rows/cols, with the LSB-first control
    convention used by cqlib's UCR family: qubit ``q_i`` encodes the
    ``i``-th bit of the control integer ``x``.
    """

    n_branches = int(alpha.shape[0])
    dim = 2 * n_branches
    mat = np.zeros((dim, dim), dtype=np.complex128)
    for x, a in enumerate(alpha):
        r = _rot_2x2(axis, float(a))
        for i in range(2):
            for j in range(2):
                row = x + (i * n_branches)
                col = x + (j * n_branches)
                mat[row, col] = r[i, j]
    return mat


# ---------------------------------------------------------------------------
# Section 1.  Level 0 — UCRY gate (UCRY-001 ... UCRY-005) -----------------
# ---------------------------------------------------------------------------


class TestUCRYGateLevel0:
    """Validate the low-level uniformly controlled Y rotation.

    Design section 2 defines the ground-truth numeric tables which every
    subsequent class builds on.  These tests are the numerical anchor.
    """

    @staticmethod
    def test_ucry_001_two_angles_n1() -> None:
        """UCRY-001: n=1, angles = [pi/2, pi/4]."""

        angles = np.array([np.pi / 2.0, np.pi / 4.0])
        circ = UCRY(1, angles).build_circuit()
        expected = [
            (0, math.cos(np.pi / 4.0), math.sin(np.pi / 4.0)),
            (1, math.cos(np.pi / 8.0), math.sin(np.pi / 8.0)),
        ]
        for x, a0, a1 in expected:
            _assert_branch_amplitudes(circ, 1, x, amp_0=a0, amp_1=a1)

    @staticmethod
    def test_ucry_002_four_angles_n2() -> None:
        """UCRY-002: n=2, angles = [pi/2, pi/3, pi/4, pi/6]."""

        angles = np.array([np.pi / 2.0, np.pi / 3.0, np.pi / 4.0, np.pi / 6.0])
        circ = UCRY(2, angles).build_circuit()
        for x, theta in enumerate(angles):
            a0 = math.cos(theta / 2.0)
            a1 = math.sin(theta / 2.0)
            _assert_branch_amplitudes(circ, 2, x, amp_0=a0, amp_1=a1)

    @staticmethod
    def test_ucry_003_equal_angles_is_global_ry() -> None:
        """UCRY-003: uniform angles pi/3 for all n=3 inputs."""

        theta = np.pi / 3.0
        angles = np.full(8, theta)
        circ = UCRY(3, angles).build_circuit()
        expected_a0 = math.cos(theta / 2.0)
        expected_a1 = math.sin(theta / 2.0)
        for x in range(8):
            _assert_branch_amplitudes(
                circ, 3, x, amp_0=expected_a0, amp_1=expected_a1
            )

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_ucry_004_cnot_count(self, n: int) -> None:
        """UCRY-004: #CNOT = 2**n for every control width up to 4."""

        angles = np.ones(1 << n, dtype=float) * 0.123
        circ = UCRY(n, angles).build_circuit()
        counts = _count_ops(circ)
        cnots = counts.get("cx", 0) + counts.get("cnot", 0)
        assert cnots == 1 << n, (
            f"UCRY-004 violation for n={n}: expected 2^{n} CX gates, "
            f"observed {cnots}."
        )

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_ucry_005_ry_count(self, n: int) -> None:
        """UCRY-005: #RY = 2**n for every control width up to 4."""

        angles = np.ones(1 << n, dtype=float) * 0.123
        circ = UCRY(n, angles).build_circuit()
        counts = _count_ops(circ)
        rys = counts.get("ry", 0)
        assert rys == 1 << n, (
            f"UCRY-005 violation for n={n}: expected 2^{n} RY gates, "
            f"observed {rys}."
        )


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Section 2.  Level 1 — Exact reciprocal (REC-001 .. REC-004) -------------
# ---------------------------------------------------------------------------



class TestExactReciprocalLevel2:
    """Amplitude-mode only (design §4 preamble: reciprocal is always arcsin
    encoded with `scaling` global multiplier)."""

    @staticmethod
    def test_rec001_n1_scaling_half() -> None:
        """REC-001: n=1, s=1/2.  x=1 => sf = (1/2)*2^1 / 1 = 1."""

        n = 1
        s = 1.0 / 2.0
        circ = ExactReciprocal(n, s).build_circuit()
        # x = 0: alpha = 0 (identity rotation)
        _assert_branch_amplitudes(circ, n, 0, amp_0=1.0 + 0j, amp_1=0.0 + 0j)
        # x = 1: sf=1 => alpha=pi
        _assert_branch_amplitudes(circ, n, 1, amp_0=0.0 + 0j, amp_1=1.0 + 0j)

    @staticmethod
    def test_rec002_n2_scaling_quarter() -> None:
        """REC-002: n=2, s=1/4.  sf = 1 / k."""

        n = 2
        s = 1.0 / 4.0
        circ = ExactReciprocal(n, s).build_circuit()
        # x=0: identity
        _assert_branch_amplitudes(circ, n, 0, amp_0=1.0, amp_1=0.0)
        # x=1: sf=1
        _assert_branch_amplitudes(circ, n, 1, amp_0=0.0, amp_1=1.0)
        # x=2: sf=0.5
        f2 = 0.5
        _assert_branch_amplitudes(
            circ, n, 2,
            amp_0=math.sqrt(1.0 - f2 * f2), amp_1=f2,
        )
        # x=3: sf=1/3
        f3 = 1.0 / 3.0
        _assert_branch_amplitudes(
            circ, n, 3,
            amp_0=math.sqrt(1.0 - f3 * f3), amp_1=f3,
        )

    @staticmethod
    def test_rec003_n3_full_table() -> None:
        """REC-003: n=3, s=1/8, every basis state's |1> amp = 1/k for k>0."""

        n = 3
        s = 1.0 / 8.0
        circ = ExactReciprocal(n, s).build_circuit()
        for k in range(1 << n):
            if k == 0:
                f_k = 0.0
            else:
                f_k = s * (1 << n) / k
            a1 = f_k
            a0 = math.sqrt(1.0 - a1 * a1) if abs(a1) <= 1.0 else 0.0
            _assert_branch_amplitudes(circ, n, k, amp_0=a0, amp_1=a1)

    @staticmethod
    def test_rec004_input_invariance() -> None:
        """REC-004: The control register never toggles (every y!=x branch 0).

        The helper ``_assert_branch_amplitudes`` already enforces this by
        constructing an ``expected`` vector that is zero everywhere except
        ``(x, 0)`` and ``(x, 1)``.  Running the helper for every basis on
        an n=3 reciprocal instance therefore fully covers REC-004.
        """

        n = 3
        circ = ExactReciprocal(n, 1.0 / 8.0).build_circuit()
        for k in range(1 << n):
            if k == 0:
                f_k = 0.0
            else:
                f_k = (1.0 / 8.0) * (1 << n) / k
            _assert_branch_amplitudes(
                circ, n, k,
                amp_0=math.sqrt(1.0 - f_k * f_k), amp_1=f_k,
            )

    # -- Validation tests (beyond the numeric doc cases) -------------------

    @staticmethod
    @pytest.mark.parametrize("bad", [True, False, "2", 3.5])
    def test_rec_validation_bad_n_type(bad: object) -> None:
        """num_state_qubits must be a non-bool int."""

        with pytest.raises((TypeError, ValueError)):
            ExactReciprocal(bad, 0.1)  # type: ignore[arg-type]

    @staticmethod
    def test_rec_validation_negative_scaling() -> None:
        with pytest.raises(ValueError, match="scaling must be >= 0"):
            ExactReciprocal(2, -0.1)

    @staticmethod
    def test_rec_validation_nonfinite_scaling() -> None:
        with pytest.raises(ValueError, match="finite"):
            ExactReciprocal(2, float("nan"))

    @staticmethod
    def test_rec_validation_scaling_overshoot() -> None:
        """s=1.0 with n=2 violates s <= 1/4."""

        with pytest.raises(ValueError, match="scaling must satisfy"):
            ExactReciprocal(2, 1.0)

    @staticmethod
    def test_rec_validation_neg_vals_requires_n_ge_2() -> None:
        with pytest.raises(ValueError, match="neg_vals=True"):
            ExactReciprocal(1, 0.5, neg_vals=True)

# ---------------------------------------------------------------------------
# Section 3.  Level 2 — Negative value support (neg_vals) ----------------
# ---------------------------------------------------------------------------




class TestNegativeValuesLevel5:
    """Only ``ExactReciprocal`` implements ``neg_vals``."""

    @staticmethod
    def test_neg_rec_001_n2_scaling_quarter() -> None:
        """NEG-REC-001: n=2 (1 sign + 1 magnitude), s=1/4, neg_vals=True.

        The neg_vals=True instance uses:
          * sign bit: highest index of the state register (qubit n-1)
          * magnitude bits: qubits 0..n-2
        which matches doc §7 bullet qubit layout.
        """

        n = 2
        s = 1.0 / 4.0
        circ = ExactReciprocal(n, s, neg_vals=True).build_circuit()
        cases = [
            # (state k, sign, magnitude, f_1_amp)
            (0b00, +1, 0, 0.0),
            (0b01, +1, 1, +0.5),
            (0b10, -1, 0, 0.0),
            (0b11, -1, 1, -0.5),
        ]
        for k, _sign, _mag, amp1 in cases:
            if amp1 == 0:
                a0 = 1.0
                a1 = 0.0
            else:
                a0 = math.sqrt(1.0 - amp1 * amp1)
                a1 = amp1
            _assert_branch_amplitudes(circ, n, k, amp_0=a0, amp_1=a1)

    @staticmethod
    def test_neg_002_xzero_positive_equals_negative() -> None:
        """NEG-002: With x=0 the +/- branches produce identical final states.

        We re-use the NEG-REC-001 fixture: state 0b00 (positive, magnitude
        0) and state 0b10 (negative, magnitude 0) must both end up as
        |...00>|0>.
        """

        circ = ExactReciprocal(2, 1.0 / 4.0, neg_vals=True).build_circuit()
        for k in (0b00, 0b10):
            _assert_branch_amplitudes(circ, 2, k, amp_0=1.0, amp_1=0.0)

    @staticmethod
    def test_neg_003_antisymmetry_property() -> None:
        """NEG-003: amp_1(-x) = -amp_1(x); amp_0(-x) = +amp_0(x).

        Probe on (x=+magnitude, sign=+) vs (x=+magnitude, sign=-).
        """

        circ = ExactReciprocal(3, 1.0 / 4.0, neg_vals=True).build_circuit()
        for mag in [1, 2, 3]:
            x_pos = mag            # qubit layout: sign on bit 2, magnitude low
            x_neg = mag | 0b100
            sv_pos = _probed_statevector(circ, 3, x_pos)
            sv_neg = _probed_statevector(circ, 3, x_neg)
            amp0_pos = sv_pos[_state_index(x_pos, 3, 0)]
            amp1_pos = sv_pos[_state_index(x_pos, 3, 1)]
            amp0_neg = sv_neg[_state_index(x_neg, 3, 0)]
            amp1_neg = sv_neg[_state_index(x_neg, 3, 1)]
            # cos (amp0) is even under sign flip; sin (amp1) is odd.
            np.testing.assert_allclose(
                amp0_pos, amp0_neg, atol=AMPLITUDE_ATOL, rtol=AMPLITUDE_RTOL
            )
            np.testing.assert_allclose(
                amp1_pos, -amp1_neg, atol=AMPLITUDE_ATOL, rtol=AMPLITUDE_RTOL
            )


def _probed_statevector(circuit: Circuit, n: int, x: int) -> np.ndarray:
    """Apply ``circuit`` to the basis state ``|x> |0>`` and return sv data."""

    full = Circuit(n + 1)
    for bit in range(n):
        if (x >> bit) & 1:
            full.x(bit)
    full.compose(circuit, list(range(n + 1)))
    return _statevector_data(Statevector.from_circuit(full))


# ---------------------------------------------------------------------------
# Section 4.  Level 3 — Bounds & optimisations -----------------------------
# ---------------------------------------------------------------------------


class TestBoundsLevel6:
    @staticmethod
    def test_opt001_all_zero_angles_is_identity() -> None:
        """OPT-001: zero-angle UCRY = identity on 3 control qubits."""

        circ = UCRY(3, np.zeros(8, dtype=float)).build_circuit()
        for x in range(8):
            _assert_branch_amplitudes(circ, 3, x, amp_0=1.0, amp_1=0.0)

    @staticmethod
    def test_opt002_amplitude_mode_on_boundary_exactly_one() -> None:
        """OPT-002: theta = pi -> |1> = 1.0 (arcsin boundary)."""

        circ = UCRY(2, np.full(4, np.pi)).build_circuit()
        for x in range(4):
            _assert_branch_amplitudes(circ, 2, x, amp_0=0.0, amp_1=1.0)

    @staticmethod
    def test_opt004_angle_mode_has_no_domain_bound() -> None:
        """OPT-004: angle mode accepts large rotations, still 2pi-periodic.

        theta = 100*pi -> RY(100 pi) has sin(50 pi)=0 for all x (identity).
        RY(theta) = RY(theta mod 4 pi); even multiples of 2pi give identity
        up to global phase.
        """

        big = 100.0 * np.pi
        circ = UCRY(2, np.full(4, big)).build_circuit()
        for x in range(4):
            # The identity rotation on |0> -> |0> (+ potential global phase
            # that cancels out when we assert on the two targeted entries of
            # the statevector).
            sv = _probed_statevector(circ, 2, x)
            # |x,0> amplitude should have modulus 1, |x,1> should be 0.
            idx0 = _state_index(x, 2, 0)
            idx1 = _state_index(x, 2, 1)
            assert abs(abs(sv[idx0]) - 1.0) < AMPLITUDE_ATOL
            assert abs(sv[idx1]) < AMPLITUDE_ATOL


# ---------------------------------------------------------------------------
# Section 5.  Level 4 — Integration & invariants ---------------------------
# ---------------------------------------------------------------------------


class TestIntegrationLevel7:
    """Full-circuit invariant tests that combine many of the above checks."""

    # -- INT-001: Unitarity -------------------------------------------------

    @pytest.mark.parametrize(
        "name,circuit_builder",
        [
            ("UCRY n=3", lambda: UCRY(3, np.linspace(0, np.pi, 8)).build_circuit()),
            (
                "ExactReciprocal n=2",
                lambda: ExactReciprocal(2, 1.0 / 4.0).build_circuit(),
            ),
        ],
    )
    def test_int001_unitarity(
        self, name: str, circuit_builder: object
    ) -> None:
        """U†U = I within ``UNITARITY_ATOL`` for each representative circuit."""

        del name
        u = _matrix_from_circuit(circuit_builder())  # type: ignore[misc]
        dim = u.shape[0]
        product = u.conj().T @ u
        np.testing.assert_allclose(
            product,
            np.eye(dim, dtype=np.complex128),
            atol=UNITARITY_ATOL,
            rtol=AMPLITUDE_RTOL,
        )

    # -- INT-003: Superposition linearity ----------------------------------

    @staticmethod
    def test_int003_superposition_linearity() -> None:
        """Feeding a uniform superposition |+...+> yields coherent sum.

        Prepares input  (|0> + |1> + |2> + |3>) / 2 x |0>, applies U,
        asserts the final state equals sum_{x} [c_x0 |x0> + c_x1 |x1>] / 2.

        Uses the ExactReciprocal table (s=1/4, n=2) — sf(k=1..3) =
        [1, 0.5, 0.25] and k=0 is identity rotation (alpha=0).
        """

        n = 2
        s = 1.0 / 4.0
        ks = np.arange(1 << n, dtype=np.float64)
        target = np.zeros_like(ks)
        mask = ks > 0
        target[mask] = s * (1 << n) / ks[mask]
        target = np.clip(target, -1.0, 1.0)
        alpha = 2.0 * np.arcsin(target)
        circ = ExactReciprocal(n, s).build_circuit()
        expected = np.zeros(1 << (n + 1), dtype=np.complex128)
        for x, theta in enumerate(alpha):
            c0 = math.cos(theta / 2.0) + 0j
            c1 = math.sin(theta / 2.0) + 0j
            expected[_state_index(x, n, 0)] = c0 / 2.0
            expected[_state_index(x, n, 1)] = c1 / 2.0

        full = Circuit(n + 1)
        for bit in range(n):
            full.h(bit)
        full.compose(circ, list(range(n + 1)))
        sv = _statevector_data(Statevector.from_circuit(full))
        np.testing.assert_allclose(
            sv, expected, atol=AMPLITUDE_ATOL, rtol=AMPLITUDE_RTOL
        )

    # -- INT-004: Frobenius matrix-equivalence against exact block-diagonal

    @pytest.mark.parametrize(
        "kind,axis,n,scaling",
        [
            pytest.param("ucr", axis, n, None, id=f"UCR{axis.upper()}_n{n}")
            for axis in ("x", "y", "z")
            for n in (1, 2, 3)
        ]
        + [
            pytest.param(
                "exact_reciprocal", "y", 2, 1 / 4.0, id="ExactReciprocal_n2_amp"
            ),
            pytest.param(
                "exact_reciprocal", "y", 3, 1 / 8.0, id="ExactReciprocal_n3_amp"
            ),
        ],
    )
    def test_int004_block_diagonal_matrix_equivalent(
        self,
        kind: str,
        axis: str,
        n: int,
        scaling: float | None,
    ) -> None:
        """Frobenius-equivalent to the exact block-diagonal definition.

        For any UCR-type circuit the unitary must equal::

            U = ∑_x |x><x| ⊗ R_axis(α_x)

        where ``α`` is either directly the user-supplied ``alpha`` table
        (``UCRX/UCRY/UCRZ``) or the preprocessed amplitude-encoded table
        exposed via the operator's ``.alpha`` attribute
        (``ExactReciprocal``).  No Qiskit gate dependency is involved: both
        sides are reduced to their ``2^{n+1} × 2^{n+1}`` matrix and compared
        with the normalised Frobenius norm.
        """

        if kind == "ucr":
            gate_cls = {"x": UCRX, "y": UCRY, "z": UCRZ}[axis]
            alpha = np.round(
                (np.arange(1 << n, dtype=np.float64) + 1) * 0.1, 4
            )
            circ = gate_cls(n, alpha).build_circuit()
        elif kind == "exact_reciprocal":
            assert scaling is not None
            er = ExactReciprocal(n, scaling=scaling)
            alpha = er.alpha
            circ = er.build_circuit()
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unknown case kind {kind!r}")

        u_cqlib = _matrix_from_circuit(circ)
        u_exact = _exact_ucr_matrix(axis, alpha)

        diff = float(np.linalg.norm(u_cqlib - u_exact, "fro"))
        norm = float(np.linalg.norm(u_exact, "fro")) or 1.0
        rel = diff / norm
        assert rel <= max(AMPLITUDE_ATOL * 1e3, 1e-8), (
            f"Frobenius rel. diff {rel:.3e} exceeds tolerance (abs {diff:.3e})"
        )

    # -- INT-005: Gate-count growth curve ----------------------------------

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_int005_gate_count_curve(self, n: int) -> None:
        """RY == CNOT == 2^n.  Total gates = 2^(n+1) (excluding barriers)."""

        angles = np.linspace(0.1, 0.9, 1 << n)
        circ = UCRY(n, angles).build_circuit()
        counts = _count_ops(circ)
        cnots = counts.get("cx", 0) + counts.get("cnot", 0)
        rys = counts.get("ry", 0)
        assert cnots == 1 << n
        assert rys == 1 << n


# ---------------------------------------------------------------------------
# Section 6.  UCRX & UCRZ — symmetry sanity vs UCRY ------------------------
# ---------------------------------------------------------------------------


class TestUCRXUCRZSymmetry:
    """UCRX / UCRY / UCRZ share a block-diagonal structure ``∑_x |x><x| ⊗ R_axis(α_x)``.

    Implementation detail:

    * ``UCRY`` and ``UCRZ`` are built directly via the Möttönen et al.
      Gray-code / CX interleaving pattern, using ``RY`` / ``RZ`` on the
      target wire.
    * ``UCRX`` uses an **H-sandwich** reduction onto ``UCRZ``::

          UCRX(α₀,…,α_{N-1}) = H_t · UCRZ(α₀,…,α_{N-1}) · H_t

      because ``Rx(θ) = H · Rz(θ) · H`` holds block-diagonally.  The direct
      ``RX + CX`` pattern does *not* share the Y/Z angle-expansion
      formula, so it is avoided.

    Each axis is cross-checked against the explicit 2x2 matrix convention
    so phase-convention bugs are caught early::

        Rx(θ) = [[cos θ/2,    -i sin θ/2],
                 [-i sin θ/2,  cos θ/2]]

        Ry(θ) = [[cos θ/2,    -sin θ/2],
                 [ sin θ/2,    cos θ/2]]

        Rz(θ) = [[e^{-i θ/2},          0],
                 [          0,  e^{i θ/2}]]
    """

    @pytest.mark.parametrize("n", [1, 2, 3])
    @pytest.mark.parametrize(
        "axis,gate_cls,counts_factory,amp_0_of,amp_1_of",
        [
            (
                "x",
                UCRX,
                lambda n: {"h": 2, "rz": 1 << n, "cx": 1 << n},
                lambda t: complex(math.cos(t / 2.0)),
                lambda t: -1j * math.sin(t / 2.0),
            ),
            (
                "y",
                UCRY,
                lambda n: {"ry": 1 << n, "cx": 1 << n},
                lambda t: complex(math.cos(t / 2.0)),
                lambda t: complex(math.sin(t / 2.0)),
            ),
            (
                "z",
                UCRZ,
                lambda n: {"rz": 1 << n, "cx": 1 << n},
                lambda t: complex(math.cos(t / 2.0) - 1j * math.sin(t / 2.0)),
                lambda _t: 0j,
            ),
        ],
    )
    def test_axis_matrix_element_and_gate_count(
        self,
        n: int,
        axis: str,
        gate_cls: type,
        counts_factory: object,
        amp_0_of: object,
        amp_1_of: object,
    ) -> None:
        """Gate-count shape + per-branch matrix-element convention, n=1..3."""

        del axis
        alpha = np.round(
            (np.arange(1 << n, dtype=np.float64) + 1) * 0.1, 4
        )
        circ = gate_cls(n, alpha).build_circuit()

        counts = _count_ops(circ)
        expected_counts: dict[str, int] = counts_factory(n)  # type: ignore[misc]
        for g, want in expected_counts.items():
            c = counts.get(g, 0)
            if g in ("cx", "cnot"):
                c = counts.get("cx", 0) + counts.get("cnot", 0)
            assert c == want, f"{g}: got {c}, want {want}"

        for x, a in enumerate(alpha):
            amp_0 = amp_0_of(a)  # type: ignore[misc]
            amp_1 = amp_1_of(a)  # type: ignore[misc]
            _assert_branch_amplitudes(
                circ, n, x, amp_0=amp_0, amp_1=amp_1
            )


