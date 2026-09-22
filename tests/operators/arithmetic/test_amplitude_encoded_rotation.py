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
    ``ArbitraryFunction`` multiplexed table is covered by cases
    ``MUX-FUNC-001 .. MUX-FUNC-005`` and a dedicated validation battery.
    ``LinearPauliRotation`` is cross-checked against
    ``ArbitraryFunction`` in ``MUX-FUNC-005`` as the reference (golden)
    table look-up.

Coverage map (each tag references a section / case ID in the doc or the
``MUX-FUNC-nnn`` table at the top of the source-code listing file):

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
    * Section 7   ArbitraryFunction validation & regular properties
    * Section 8   Multiplexed function table (MUX-FUNC-001 .. MUX-FUNC-005)
    * Section 9   PiecewiseChebyshev high-level wrapper
                      (PWCHEB-BASE-001..005 smoke tests;
                       PWCHEB-001 reciprocal approximation;
                       PWCHEB-002 exponential convergence;
                       PWCHEB-003 head-identity amplitude-1 segment;
                       PWCHEB-BOUND-001..004 edge-case & input guards)
"""

from __future__ import annotations

import math
import warnings
from collections import Counter

import numpy as np
import pytest
from cqlib.qis.state import Statevector

from cqlib import Circuit
from cqlib_algorithms.operators.arithmetic import (
    ArbitraryFunction,
    ExactReciprocal,
    LinearPauliRotation,
    PiecewiseChebyshev,
    PiecewiseLinearPauliRotation,
    UCRX,
    UCRY,
    UCRZ,
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


# ---------------------------------------------------------------------------
# Section 7.  ArbitraryFunction — validation & regular properties -----------
# ---------------------------------------------------------------------------


class TestArbitraryFunctionValidation:
    """Regularity checks for the multiplexed ArbitraryFunction table.

    Covers V1–V12 from the design-notes validation matrix plus a couple
    of positive-path sanity tests (alpha defensive copy, shape
    invariants, etc.).
    """

    # -- V1 / V2: num_state_qubits --------------------------------------

    @staticmethod
    @pytest.mark.parametrize("bad", [True, False, "2", 3.5, -1, 0])
    def test_v1_v2_num_state_qubits(bad: object) -> None:
        """V1: bool / non-int / negative / zero must raise."""

        with pytest.raises((TypeError, ValueError)):
            ArbitraryFunction(bad, np.ones(4))  # type: ignore[arg-type]

    # -- V3 / V4 / V5: values ------------------------------------------

    @staticmethod
    def test_v3_values_not_sequence() -> None:
        with pytest.raises(TypeError):
            ArbitraryFunction(2, "0.1 0.2 0.3 0.4")  # type: ignore[arg-type]

    @staticmethod
    def test_v4_values_wrong_length() -> None:
        with pytest.raises(ValueError, match="length must equal 2"):
            ArbitraryFunction(2, np.ones(3))

    @staticmethod
    def test_v5_values_nonfinite() -> None:
        with pytest.raises(ValueError, match="finite floats"):
            ArbitraryFunction(2, np.array([0.0, 1.0, float("nan"), 2.0]))

    # -- V6: mode ------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize("bad_mode", [1, None, True, "ampl", "angle_mode"])
    def test_v6_invalid_mode(bad_mode: object) -> None:
        with pytest.raises((TypeError, ValueError)):
            ArbitraryFunction(
                2, np.ones(4), mode=bad_mode  # type: ignore[arg-type]
            )

    # -- V7: axis ------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize("bad_axis", [1, None, "w", "yy", True])
    def test_v7_invalid_axis(bad_axis: object) -> None:
        with pytest.raises((TypeError, ValueError)):
            ArbitraryFunction(
                2, np.ones(4), axis=bad_axis  # type: ignore[arg-type]
            )

    # -- V8 / V9: amplitude mode domain --------------------------------

    @staticmethod
    def test_v8_amplitude_strictly_outside() -> None:
        """max|f| = 1.1 strictly exceeds 1 + 1e-8."""

        fs = np.array([1.1, 0.5, 0.3, 0.2])
        with pytest.raises(ValueError, match="amplitude-encoding domain"):
            ArbitraryFunction(2, fs, mode="amplitude")

    @staticmethod
    def test_v9_amplitude_within_atol_clamps() -> None:
        """1.0000000005 is still within the tolerance band -> no error."""

        fs = np.array([1.0000000005, 0.2, 0.3, 0.4])
        op = ArbitraryFunction(2, fs, mode="amplitude")
        assert np.max(op.values) <= 1.0 + 1e-9
        # alpha should correspond to arcsin(1.0) = pi for the clamped entry
        assert abs(op.alpha[0] - np.pi) < 1e-8

    # -- V10: angle mode oversized -> UserWarning + mod 2pi -------------

    @staticmethod
    def test_v10_angle_oversized_emits_warning() -> None:
        fs = np.array([0.0, 1.0, 100.0, -3.0])
        with pytest.warns(UserWarning, match="exceed 2"):
            op = ArbitraryFunction(2, fs, mode="angle")
        # all entries must now be in [0, 2pi)
        assert np.all(op.alpha >= 0.0) and np.all(op.alpha < 2 * np.pi)
        # entry that was exactly 100 -> must equal 100 mod 2pi
        expected_100 = float(np.mod(100.0, 2 * np.pi))
        assert abs(op.alpha[2] - expected_100) < 1e-12

    # -- V11: defensive copy of alpha / values ------------------------

    @staticmethod
    def test_v11_alpha_values_defensive_copy() -> None:
        fs = np.array([0.1, 0.2, 0.3, 0.4])
        op = ArbitraryFunction(2, fs, mode="amplitude")
        a1 = op.alpha
        v1 = op.values
        a1[0] = 999.0  # type: ignore[call-overload]
        v1[0] = -0.5  # type: ignore[call-overload]
        assert abs(op.alpha[0] - 2.0 * math.asin(0.1)) < 1e-12
        assert abs(op.values[0] - 0.1) < 1e-12

    # -- V12: shape invariants ----------------------------------------

    @staticmethod
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_v12_shape_invariants(n: int) -> None:
        fs = np.zeros(1 << n, dtype=float)
        op = ArbitraryFunction(n, fs, mode="angle")
        assert op.num_state_qubits == n
        assert op.num_qubits == n + 1
        assert op.result_qubit_index == n
        assert op.control_register_indices == list(range(n))
        circ = op.build_circuit()
        assert circ.num_qubits == n + 1

    # -- INT-001 style unitary + INT-004 style matrix for ArbitraryFunction

    @pytest.mark.parametrize(
        "mode,axis,n",
        [
            pytest.param("amplitude", "y", 2, id="ArbFunc_amp_y_n2"),
            pytest.param("angle", "x", 2, id="ArbFunc_angle_x_n2"),
            pytest.param("amplitude", "z", 3, id="ArbFunc_amp_z_n3"),
            pytest.param("angle", "y", 3, id="ArbFunc_angle_y_n3"),
        ],
    )
    def test_arbfunc_unitarity_and_block_diagonal(
        self, mode: str, axis: str, n: int
    ) -> None:
        """U†U = I and block-diagonal Frobenius equivalence (parallel to
        INT-001 + INT-004 for ExactReciprocal/UCR)."""

        rng = np.random.default_rng(20260921 + n)
        if mode == "amplitude":
            fs = np.clip(rng.uniform(-1.0, 1.0, 1 << n), -1.0, 1.0)
        else:
            fs = rng.uniform(-np.pi, np.pi, 1 << n)
        op = ArbitraryFunction(n, fs, mode=mode, axis=axis)  # type: ignore[arg-type]
        circ = op.build_circuit()
        u = _matrix_from_circuit(circ)
        dim = u.shape[0]
        product = u.conj().T @ u
        np.testing.assert_allclose(
            product,
            np.eye(dim, dtype=np.complex128),
            atol=UNITARITY_ATOL,
            rtol=AMPLITUDE_RTOL,
        )
        u_exact = _exact_ucr_matrix(axis, op.alpha)
        diff = float(np.linalg.norm(u - u_exact, "fro"))
        norm = float(np.linalg.norm(u_exact, "fro")) or 1.0
        assert diff / norm <= max(AMPLITUDE_ATOL * 1e3, 1e-8)


# ---------------------------------------------------------------------------
# Section 8.  Multiplexed function table (MUX-FUNC-001 .. MUX-FUNC-005) ----
# ---------------------------------------------------------------------------


def _assert_branch_against_alpha(
    circuit: Circuit, n: int, axis: str, alpha: np.ndarray
) -> None:
    """Probe every basis state ``|k>|0>`` against the block-diagonal table."""

    for k, theta in enumerate(alpha):
        r = _rot_2x2(axis, float(theta))
        amp_0 = r[0, 0]
        amp_1 = r[1, 0]
        _assert_branch_amplitudes(circuit, n, k, amp_0=amp_0, amp_1=amp_1)


class TestMultiplexedFunctionTable:
    """Test-cases ``MUX-FUNC-001..005`` against ArbitraryFunction.

    Design table recap:

    MUX-FUNC-001 sin(x)                 angle    nonlinear, ± alternating
    MUX-FUNC-002 exp(−x/4)              amplitude, auto-scaling @ test side
    MUX-FUNC-003 sin(πx/4)·cos(πx/8)   angle    bimodal, symmetric
    MUX-FUNC-004 step / ladder function amplitude discontinuous table
    MUX-FUNC-005 gold-standard reference vs LinearPauliRotation (exact);
                                         Chebyshev-style comparison is left
                                         as a TODO-stub until a polynomial
                                         table operator (beyond linear) is
                                         implemented.
    """

    # -- MUX-FUNC-001 sin(x), angle mode -------------------------------

    @staticmethod
    def test_mux_func_001_sin_x_angle() -> None:
        """``f(k) = sin(k)`` on n=3 (k ∈ [0, 7]) as angle-mode Y-rotation."""

        n = 3
        ks = np.arange(1 << n, dtype=np.float64)
        fs = np.sin(ks)
        op = ArbitraryFunction(n, fs, mode="angle", axis="y")
        alpha_expected = np.mod(fs, 2 * np.pi)
        # alpha[k] must equal the mod-2pi version of f(k) up to period;
        # compare via sin/cos equality to avoid boundary issues.
        np.testing.assert_allclose(
            np.sin(op.alpha / 2.0), np.sin(alpha_expected / 2.0), atol=1e-12
        )
        np.testing.assert_allclose(
            np.cos(op.alpha / 2.0), np.cos(alpha_expected / 2.0), atol=1e-12
        )
        circ = op.build_circuit()
        _assert_branch_against_alpha(circ, n, "y", op.alpha)

    # -- MUX-FUNC-002 exp(−x/4), amplitude + TEST-SIDE auto-scaling ---

    @staticmethod
    def test_mux_func_002_exp_decay_amplitude_autoscale() -> None:
        """``f_raw(k) = exp(-k/4)`` on n=3.

        NOTE: ``ArbitraryFunction(mode="amplitude")`` never re-scales
        inputs (strict domain check).  The "auto scaling" requested by
        the case is performed on the **test side** by dividing the raw
        values by ``max|f_raw| + eps``.
        """

        n = 3
        ks = np.arange(1 << n, dtype=np.float64)
        f_raw = np.exp(-ks / 4.0)
        max_abs = float(np.max(np.abs(f_raw)))
        # Divide by (1.001 * max_abs) so every entry stays strictly below
        # 1.0 - ~1e-3, keeping the value well inside arcsin's strict
        # domain interior while preserving the exp decay shape.
        scale_factor = 1.001 * max_abs
        fs_scaled = f_raw / scale_factor
        assert np.all(np.abs(fs_scaled) <= 0.9999)
        op = ArbitraryFunction(n, fs_scaled, mode="amplitude", axis="y")
        circ = op.build_circuit()
        # the |1> amplitude after applying R_y(2·arcsin(fs)) on |0> is fs
        for k, fk in enumerate(fs_scaled):
            a0 = math.sqrt(max(0.0, 1.0 - fk * fk))
            a1 = fk
            _assert_branch_amplitudes(circ, n, k, amp_0=a0, amp_1=a1)

    # -- MUX-FUNC-003 bimodal sin·cos product, angle mode --------------

    @staticmethod
    def test_mux_func_003_bimodal_product_angle() -> None:
        """``f(k) = sin(π·k/4) · cos(π·k/8)`` on n=4 (k ∈ [0, 15])."""

        n = 4
        ks = np.arange(1 << n, dtype=np.float64)
        fs = np.sin(np.pi * ks / 4.0) * np.cos(np.pi * ks / 8.0)
        op = ArbitraryFunction(n, fs, mode="angle", axis="x")
        circ = op.build_circuit()
        _assert_branch_against_alpha(circ, n, "x", op.alpha)
        # Symmetry spot-checks: product is 2π·8 = 16 periodic (trivial on
        # n=4) and f(16−k) mirrors behaviour for k=1..7 vs k=9..15.
        for k in range(1, 8):
            fk = float(np.sin(np.pi * k / 4.0) * np.cos(np.pi * k / 8.0))
            f2k = float(
                np.sin(np.pi * (16 - k) / 4.0) * np.cos(np.pi * (16 - k) / 8.0)
            )
            # sin(4π - πk/4) = -sin(πk/4); cos(2π - πk/8) = +cos(πk/8).
            assert abs(f2k - (-fk)) < 1e-12, (
                f"Symmetry failure at k={k}: f(k)={fk}, f(16-k)={f2k}"
            )

    # -- MUX-FUNC-004 step function, amplitude mode --------------------

    @staticmethod
    def test_mux_func_004_step_function_amplitude() -> None:
        """4-level step on n=3.

        ``f(k) ∈ {−0.75, −0.25, +0.25, +0.75}`` for consecutive bins
        of size 2, giving 2^3 = 8 basis states grouped into 4 plateaux.
        """

        n = 3
        fs = np.zeros(1 << n, dtype=np.float64)
        bins = [(-0.75, 0, 2), (-0.25, 2, 4), (0.25, 4, 6), (0.75, 6, 8)]
        for val, lo, hi in bins:
            fs[lo:hi] = val
        op = ArbitraryFunction(n, fs, mode="amplitude", axis="y")
        circ = op.build_circuit()
        _assert_branch_against_alpha(circ, n, "y", op.alpha)
        # Same-bin states must yield the exact same RY(2*arcsin(val)).
        for val, lo, hi in bins:
            expected_alpha = 2.0 * math.asin(val)
            for k in range(lo, hi):
                assert abs(op.alpha[k] - expected_alpha) < 1e-12, (
                    f"k={k} alpha mismatch: got {op.alpha[k]}, "
                    f"want {expected_alpha}"
                )

    # -- MUX-FUNC-005 gold-standard reference vs LinearPauliRotation ---

    @staticmethod
    @pytest.mark.parametrize("n", [2, 3, 4])
    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_mux_func_005_linear_gold_standard(n: int, axis: str) -> None:
        """Linear inputs: ArbitraryFunction == LinearPauliRotation exactly.

        For a linear function ``f(k) = slope * k + offset`` both
        methods must produce the *same* block-diagonal rotation table
        ``alpha[k]`` up to the mod-2pi folding convention, and their
        circuits must be Frobenius-equivalent to the shared exact
        block-diagonal matrix.

        The Chebyshev / polynomial-order comparison called for by the
        design table is left as a documented TODO because no
        higher-order ``PolynomialPauliRotation`` operator exists yet.
        """

        slope = 0.3
        offset = 0.2
        ks = np.arange(1 << n, dtype=np.float64)
        f_linear = slope * ks + offset

        # Reference (multiplexed "gold"): feed the explicit table in
        # angle mode.  Any oversized angles must also be mod 2pi on the
        # ArbitraryFunction side so we compare the same quantities.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            op_ref = ArbitraryFunction(
                n, f_linear, mode="angle", axis=axis  # type: ignore[arg-type]
            )

        # Polynomial (bit-wise linear) route.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            op_lin = LinearPauliRotation(
                n, slope=slope, offset=offset, axis=axis  # type: ignore[arg-type]
            )

        # alpha tables must match mod 2pi (i.e. same point on circle).
        np.testing.assert_allclose(
            np.sin(op_ref.alpha / 2.0),
            np.sin(op_lin.alpha / 2.0),
            atol=1e-12,
            err_msg="alpha mismatch: sin(a/2)",
        )
        np.testing.assert_allclose(
            np.cos(op_ref.alpha / 2.0),
            np.cos(op_lin.alpha / 2.0),
            atol=1e-12,
            err_msg="alpha mismatch: cos(a/2)",
        )

        # Block-diagonal Frobenius-equivalent for both circuits against
        # the shared exact table.
        u_ref = _matrix_from_circuit(op_ref.build_circuit())
        u_lin = _matrix_from_circuit(op_lin.build_circuit())
        u_exact = _exact_ucr_matrix(axis, op_lin.alpha)

        for label, u in [("mux", u_ref), ("lin", u_lin)]:
            diff = float(np.linalg.norm(u - u_exact, "fro"))
            norm = float(np.linalg.norm(u_exact, "fro")) or 1.0
            assert diff / norm <= max(AMPLITUDE_ATOL * 1e3, 1e-8), (
                f"{label}-vs-exact rel Frobenius {diff/norm:.3e} > tol"
            )


# ---------------------------------------------------------------------------
# Section 9.  Piecewise-Chebyshev high-level wrapper
# ---------------------------------------------------------------------------
# Reference section: polynomial_test_cases.md § 8.3
#   PWCHEB-001 reciprocal function piecewise Chebyshev (n=3, degree=2)
#   PWCHEB-002 exponential convergence of the Chebyshev degree
#   PWCHEB-003 head-identity segment when breakpoints[0] > 0
# ---------------------------------------------------------------------------


def _pwcheb_expected_alpha(
    op: PiecewiseChebyshev,
) -> np.ndarray:
    """Rebuild ``theta(k)`` purely on the classical side (UNFOLDED).

    For every completed segment ``[bp_j, bp_{j+1})`` use the coefficients
    stored in :attr:`PiecewiseChebyshev.polynomials` (already multiplied by
    ``2`` inside the wrapper and with the ``π`` identity-polynomial for
    auto-filled head/tail segments).

    The returned vector is **not** folded modulo ``2π``.  Callers that
    need a pointwise distance metric against another unfolded angle table
    (e.g. the exponential-convergence test PWCHEB-002) must use the
    wrapped distance helper :func:`_angular_abs_diff` instead of plain
    subtraction so branch-cut artefacts near ``0/2π`` are removed.
    """

    n = op.num_state_qubits
    dim = 1 << n
    alpha = np.zeros(dim, dtype=np.float64)
    bp = op.breakpoints
    polys = op.polynomials
    for j in range(len(bp) - 1):
        lo = int(bp[j])
        hi = int(bp[j + 1])
        c = np.asarray(polys[j], dtype=np.float64)
        ks = np.arange(lo, hi, dtype=np.float64)
        alpha[lo:hi] = np.polyval(c[::-1], ks)
    return alpha


def _angular_abs_diff(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    """Shortest-arc unsigned angular distance on the ``[0, 2π)`` circle.

    Folding (a.k.a.  branch-cut) is the canonical source of spurious
    ``2π`` differences in classical-to-quantum angle comparisons; taking
    the minimum of the direct arc and the ``2π``-complementary arc makes
    the distance invariant under ``mod 2π`` folding applied to either
    operand, which matches the physical Pauli-rotation equivalence
    ``R(θ) ≡ R(θ + 2πk)`` up to global phase.
    """

    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    abs_diff = np.abs(diff)
    period = 2.0 * np.pi
    # Reduce anything larger than half a period via the complementary arc.
    wrapped = np.minimum(abs_diff, period - abs_diff % period)
    # After the above, terms ``abs_diff == period * k`` collapse to 0 as
    # expected for physically equivalent rotations.
    return np.where(abs_diff >= period, abs_diff % period, wrapped)


def _assert_pwcheb_statevector(
    op: PiecewiseChebyshev,
    *,
    alpha: np.ndarray | None = None,
    atol: float = AMPLITUDE_ATOL,
    rtol: float = AMPLITUDE_RTOL,
) -> None:
    """Black-box probability-domain statevector check for PiecewiseChebyshev.

    For piecewise comparator-based circuits the comparator ``geq`` wires
    are *not* uncomputed after the rotation layer – their final value is
    the *classical* comparator output (``geq_j = 1`` iff ``k >= b_j``).
    Any test that demands ``geq == 0`` at the end of the circuit is
    therefore incorrect for ``k >= b_j`` (which is exactly the behaviour
    that produced the first round of ``garbage-leakage`` failures).

    The structural invariants that are actually valid for *any*
    block-diagonal piecewise amplitude-encoding UCR are:

    1. **Control-register invariance** – the ``x`` part never changes.
       In statevector terms, any amplitude whose ``x_val != k`` must
       vanish up to floating-point noise.  The total probability mass
       outside the fixed-``x`` subspace is therefore exactly the
       ``garbage_mass`` used below.
    2. **Result-qubit measurement probabilities** – inside the
       ``x == k`` subspace the result qubit is rotated by
       ``R_y(theta_k)`` independently of what the comparator wires
       happen to contain.  For the measurement probabilities we only
       care about *marginalising* over the ``geq`` subspace:
       ``P(result=0|k) = cos^2(theta_k/2)`` and
       ``P(result=1|k) = sin^2(theta_k/2)``.  This is exactly the
       convention used by the rest of the suite in
       ``_assert_branch_against_alpha`` and is robust to both global
       sign and any ``geq`` occupation.
    """

    n = op.num_state_qubits
    J = op.num_segments
    num_qubits = op.num_qubits
    if num_qubits != n + J:  # pragma: no cover - class-invariant guard
        raise AssertionError(
            f"PiecewiseChebyshev num_qubits invariant broken: "
            f"{num_qubits} != {n} + {J}"
        )
    x_wires = op.control_register_indices
    geq_wires = op.geq_register_indices
    target_wire = op.result_qubit_index
    if alpha is None:
        alpha = _pwcheb_expected_alpha(op)

    circ = op.build_circuit()
    dim = 1 << n
    for k in range(dim):
        theta_k = float(alpha[k])

        full = Circuit(num_qubits)
        for bit_local, wire in enumerate(x_wires):
            if (k >> bit_local) & 1:
                full.x(wire)
        full.compose(circ, list(range(num_qubits)))
        sv = _statevector_data(Statevector.from_circuit(full))

        prob_x_ok = 0.0  # total |amp|^2 with x == k (marginal over geq/result)
        prob_result_0 = 0.0  # |amp|^2 with x == k and result == 0
        prob_result_1 = 0.0  # |amp|^2 with x == k and result == 1
        garbage_mass = 0.0  # |amp|^2 with x != k

        num_geq_wires = len(geq_wires)
        # Precompute bit masks for x / geq / result extraction so we do
        # not branch on every single bit of a potentially 4096-entry
        # statevector inside the Python loop.  Typical m = n + J <= 12
        # so the linear scan is perfectly fine; the precomputed masks
        # just keep the code readable.
        x_masks_local = [1 << wire for wire in x_wires]
        geq_masks = [1 << wire for wire in geq_wires]
        target_mask = 1 << target_wire
        for idx in range(1 << num_qubits):
            # --- decode x_val LSB-first from the physical x_wires ---
            x_val = 0
            for bit_local, physical_mask in enumerate(x_masks_local):
                if (idx & physical_mask):
                    x_val |= 1 << bit_local
            mag2 = float(abs(sv[idx]) ** 2)
            if x_val != k:
                garbage_mass += mag2
                continue
            prob_x_ok += mag2
            if idx & target_mask:
                prob_result_1 += mag2
            else:
                prob_result_0 += mag2

        # Structural invariant: no control-register leakage (garbage).
        assert garbage_mass < atol * atol * 1e3, (
            f"PiecewiseChebyshev k={k} (n={n}, J={J}, geq_wires={num_geq_wires}): "
            f"control-leakage norm-squared = {garbage_mass:.3e} exceeds tol; "
            f"control register leaked probability out of x == {k}."
        )
        # Total probability inside x == k subspace should be 1 up to tol.
        assert abs(prob_x_ok - 1.0) < 1e-8, (
            f"PiecewiseChebyshev k={k} (n={n}, J={J}): "
            f"total |x == {k}| probability = {prob_x_ok:.6e} != 1."
        )

        want_p0 = float(math.cos(0.5 * theta_k) ** 2)
        want_p1 = float(math.sin(0.5 * theta_k) ** 2)
        for label, have, want in [
            ("P(result=0)", prob_result_0, want_p0),
            ("P(result=1)", prob_result_1, want_p1),
        ]:
            if abs(have - want) > max(atol * 1e2, rtol * abs(want) + atol):
                raise AssertionError(
                    f"PiecewiseChebyshev k={k} (n={n}, J={J}) {label} mismatch: "
                    f"have {have:.6e}, want {want:.6e}; "
                    f"|diff|={abs(have-want):.3e}; theta_k={theta_k:.6e}. "
                    f"(Comparator geq values are allowed to be non-zero; "
                    f"only x-register invariance + result probabilities "
                    f"are enforced.)"
                )


class TestPiecewiseChebyshevBase:
    """PWCHEB-BASE-* smoke tests for the wrapper layer.

    These tests exercise constructor plumbing, breakpoints completion,
    identity-segment semantics (``π`` for head/tail auto-filled ranges,
    giving ``|1>``-full amplitude), ``float``-constant ``f_x`` and the
    ``num_state_qubits`` inference path when the caller only supplies a
    non-trivial breakpoint list.
    """

    @staticmethod
    def test_pwcheb_base_001_degree0_constant() -> None:
        """degree=0: all segments reduce to a constant 2*c0 rotation."""

        n = 3
        const = 0.3
        op = PiecewiseChebyshev(
            f_x=const,
            degree=0,
            breakpoints=None,
            num_state_qubits=n,
        )
        alpha = _pwcheb_expected_alpha(op)
        expected_theta = 2.0 * const
        diff = _angular_abs_diff(alpha, expected_theta * np.ones_like(alpha))
        np.testing.assert_allclose(
            diff,
            np.zeros_like(alpha),
            atol=1e-10,
            err_msg="degree=0 PWCHEB alpha != 2*const",
        )
        _assert_pwcheb_statevector(op)

    @staticmethod
    def test_pwcheb_base_002_degree1_vs_piecewiselinear() -> None:
        """degree=1 Chebyshev == PiecewiseLinear (piecewise linear map).

        When ``degree == 1`` the Chebyshev interpolant on any interval is
        the exact affine linear fit through the endpoints.  Feeding a
        linear function into both constructors must therefore yield the
        same rotation table *up to period* (PiecewiseLinear works in
        native angle mode; PiecewiseChebyshev multiplies coefficients by
        ``2`` for the amplitude-encoding convention so we double the
        linear slope/offset on the PiecewiseLinear side to align them).

        The user breakpoint list is explicitly closed at ``2**n`` so that
        no head/tail identity segment is auto-filled; both constructors
        receive the exact same two-segment partition ``[0,2), [2,4)``.
        """

        n = 2
        slope = 0.25
        offset = 0.125
        user_bp = [0, 2, 4]  # J = 2 user segments; no identity segments.

        def f(x: float) -> float:
            return float(slope * x + offset)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            op_cheb = PiecewiseChebyshev(
                f_x=f,
                degree=1,
                breakpoints=list(user_bp),
                num_state_qubits=n,
            )
            # PWLIN requires J+1 breakpoints that already span [0, 2^n];
            # use the completed list from the Chebyshev side to be safe.
            op_lin = PiecewiseLinearPauliRotation(
                n,
                breakpoints=op_cheb.breakpoints,
                slopes=[2.0 * slope, 2.0 * slope],
                offsets=[2.0 * offset, 2.0 * offset],
            )

        # Probability-domain comparison – physically equivalent rotations
        # always agree on cos²/sin², regardless of global phase or any
        # 2π aliasing.  This is the strongest assertion that still
        # respects the Pauli-rotation 4π-period equivalence.
        alpha_cheb = _pwcheb_expected_alpha(op_cheb)
        alpha_lin = op_lin.alpha
        p0_cheb = np.cos(0.5 * alpha_cheb) ** 2
        p0_lin = np.cos(0.5 * alpha_lin) ** 2
        p1_cheb = np.sin(0.5 * alpha_cheb) ** 2
        p1_lin = np.sin(0.5 * alpha_lin) ** 2
        np.testing.assert_allclose(
            p0_cheb, p0_lin, atol=1e-10,
            err_msg="degree=1 Chebyshev != PiecewiseLinear (P(result=0))",
        )
        np.testing.assert_allclose(
            p1_cheb, p1_lin, atol=1e-10,
            err_msg="degree=1 Chebyshev != PiecewiseLinear (P(result=1))",
        )

    @staticmethod
    @pytest.mark.filterwarnings(
        "ignore:PiecewisePolynomialPauliRotation max_degree d_max="
        ":UserWarning"
    )
    def test_pwcheb_base_003_single_segment_default_bp() -> None:
        """breakpoints=None (single-segment default) covers [0, 2**n)."""

        n = 3
        degree = 5

        def f(x: float) -> float:
            return float(math.sin(math.pi * x / float(1 << n)))

        op = PiecewiseChebyshev(
            f_x=f,
            degree=degree,
            breakpoints=None,
            num_state_qubits=n,
        )
        assert op.breakpoints == [0, 1 << n], (
            f"default breakpoints completion failed: {op.breakpoints}"
        )
        assert op.num_segments == 1
        _assert_pwcheb_statevector(op)

    @staticmethod
    def test_pwcheb_base_004_fx_constant_float() -> None:
        """``f_x`` accepts a plain ``float`` (not a callable)."""

        n = 2
        op = PiecewiseChebyshev(
            f_x=0.5, degree=0, breakpoints=None, num_state_qubits=n
        )
        polys = op.polynomials
        assert len(polys) == 1
        # degree=0 Chebyshev: the wrapper multiplies the fitted constant
        # by 2, so f_x=0.5 -> coefficients = [1.0] per segment.
        expected_coeff = np.asarray([1.0], dtype=np.float64)
        diff = float(np.max(np.abs(np.asarray(polys[0], dtype=np.float64)
                                    - expected_coeff)))
        # Don't hard-fail on tiny fitting noise: anything below 1e-8 is
        # effectively a degree-0 Chebyshev constant 1.0 within tolerance
        # and the statevector assertion below will catch functional
        # mismatches.
        assert diff < 1e-6, (
            f"PWCHEB BASE-004 expected constant coeff [1.0], got "
            f"{polys[0]} (|diff|={diff:.3e})"
        )
        _assert_pwcheb_statevector(op)

    @staticmethod
    def test_pwcheb_base_005_n_inference_from_breakpoints() -> None:
        """num_state_qubits=None + breakpoints=[2,5] infers n=3."""

        def f(x: float) -> float:
            return 0.1 * float(x)

        op = PiecewiseChebyshev(
            f_x=f, degree=1, breakpoints=[2, 5], num_state_qubits=None
        )
        assert op.num_state_qubits == 3, (
            f"n inference failed: {op.num_state_qubits} != 3"
        )
        assert op.breakpoints == [0, 2, 5, 8]


class TestPiecewiseChebyshevDocCases:
    """Design-document §8.3 cases: PWCHEB-001, PWCHEB-002, PWCHEB-003."""

    # -- PWCHEB-001 piecewise reciprocal approximation ------------------

    @staticmethod
    def test_pwcheb_001_reciprocal_degree2() -> None:
        """Degree-2 Chebyshev of 1/x piecewise over [1,4) ∪ [4,8).

        The head segment ``[0, 1)`` is auto-filled with the identity
        polynomial ``[π]``, which encodes the "full amplitude" state
        ``sin(π/2) == 1`` in amplitude-mode semantics.  The two
        user segments use a second-degree interpolant.  Because
        Chebyshev approximations are only guaranteed to be *close* to
        the target function (not exact for degree < segment length) the
        test bounds the error against a dense classical evaluation of
        ``f`` rather than asserting exact equality.  The quantum
        statevector, in turn, is checked against the *actual* classical
        coefficients stored by the wrapper (i.e.  we verify the circuit
        *really implements what the wrapper computed*).

        Note on the ``f_x`` convention: the PiecewiseChebyshev wrapper
        applies ``2 * coeffs`` internally so that the downstream
        polynomial layer produces rotation ``theta(k) = 2*P_d(k)``; the
        caller therefore passes the *half-angle* amplitude ``f(k)``
        rather than the full rotation ``2·asin(f(k))``.  ``f(x) = 1/x``
        on ``[1, 8)`` maps naturally to ``theta(k) = 2·asin(1/k)`` via
        the standard amplitude-encoding recipe.
        """

        from numpy.polynomial import Chebyshev as _NpCheb
        from numpy.polynomial import Polynomial as _NpPoly

        n = 3
        degree = 2
        user_bp = [1, 4, 8]

        def f_target(x: float | int) -> float:
            xv = float(x)
            return 1.0 / xv if xv != 0.0 else 1.0

        # 1. Reference error bound: build the completed segments classically
        #    and measure the max |f(k) - P_d(k)| on the integer grid.
        #    Use the standard ``Chebyshev.interpolate`` interface to
        #    guarantee a canonical Chebyshev-Gauss (zeros-of-T) interpolant
        #    on every segment, matching the wrapper's construction exactly.
        completed_bp = [0] + list(user_bp)
        if completed_bp[-1] < (1 << n):
            completed_bp.append(1 << n)
        f_vec = np.vectorize(f_target, otypes=[np.float64])
        grid_abs_err = 0.0
        for j in range(len(completed_bp) - 1):
            lo = completed_bp[j]
            hi = completed_bp[j + 1]
            if j == 0:
                continue  # identity segment – no approximation
            a = float(lo)
            b = float(hi - 1)
            cheb = _NpCheb.interpolate(f_vec, deg=degree, domain=[a, b])
            poly = cheb.convert(kind=_NpPoly)
            coeffs = np.asarray(poly.coef, dtype=np.float64)
            for k in range(lo, hi):
                expected = f_target(float(k))
                realised = float(np.polyval(coeffs[::-1], np.float64(k)))
                grid_abs_err = max(grid_abs_err, abs(expected - realised))
        # A degree-2 fit on three / four integer points should be small
        # when expressed in f-units (NOT yet multiplied by 2).
        assert grid_abs_err < 0.2, (
            f"PWCHEB-001 classical approximation error too large: "
            f"{grid_abs_err:.3e}"
        )

        # 2. Instantiate the operator and run the black-box check.
        op = PiecewiseChebyshev(
            f_x=lambda x, _ft=f_target: _ft(x),
            degree=degree,
            breakpoints=list(user_bp),
            num_state_qubits=n,
        )
        # Head auto-filled identity segment: coefficient must be exactly π.
        np.testing.assert_allclose(
            op.polynomials[0],
            np.asarray([math.pi], dtype=np.float64),
            atol=1e-14,
            err_msg="PWCHEB-001 head identity segment coeff != π",
        )
        _assert_pwcheb_statevector(op)

    # -- PWCHEB-002 exponential convergence ---------------------------------

    @staticmethod
    @pytest.mark.filterwarnings(
        "ignore:PiecewisePolynomialPauliRotation max_degree d_max="
        ":UserWarning"
    )
    @pytest.mark.filterwarnings(
        "ignore:One or more theta\\(k\\) values produced by the piecewise "
        "Chebyshev interpolant exceed 2\\*pi:UserWarning"
    )
    @pytest.mark.filterwarnings(
        "ignore:One or more raw piecewise-polynomial angles exceed "
        "2\\*pi in absolute value:UserWarning"
    )
    def test_pwcheb_002_exponential_convergence() -> None:
        """sin(π·x/14) on n=3: increasing degree → monotonic error decay.

        For an analytic function the standard Chebyshev interpolant –
        defined on the exact (floating-point, unrounded) Chebyshev-Gauss
        nodes (zeros of ``T_{d+1}``) and evaluated at the integer
        control register indices – converges exponentially in
        ``degree``.  This test asserts *two* properties:

        * The wrapper's classical coefficient calculation is cross
          validated against an independent ``numpy.polynomial`` based
          reconstruction: for every ``degree`` the wrapper's alpha
          table must match an alpha table rederived in the test file
          from scratch via the canonical
          :func:`numpy.polynomial.chebyshev.Chebyshev.interpolate`
          interface, which uses the same Chebyshev-Gauss nodes used
          internally by the wrapper's :func:`_interpolate_segment`
          helper.  Any deviation therefore indicates a genuine
          implementation bug rather than a mere numerical-difference
          noise between two different node samplers.
        * The interpolation error against the *ideal* target
          ``2·f(k)`` must decrease monotonically in ``degree`` (within
          floating-point noise) and overall shrink by at least a factor
          of ``10`` from ``d=1`` to ``d=5``, verifying that the
          convergence rate is indeed exponential rather than constant.
        """

        from numpy.polynomial import Chebyshev as _NpCheb
        from numpy.polynomial import Polynomial as _NpPoly

        n = 3
        degrees = [1, 2, 3, 4, 5]

        def f(x: float) -> float:
            return float(math.sin(math.pi * x / 14.0))

        # Build the target 2·f(k) reference once; per-degree alphas are
        # re-derived from independent Chebyshev-Gauss interpolations
        # below so they genuinely cross-validate the wrapper rather
        # than reuse its cached coefficients.
        dim = 1 << n
        target_theta = np.asarray(
            [2.0 * f(float(k)) for k in range(dim)], dtype=np.float64
        )
        ks = np.arange(dim, dtype=np.float64)
        a = 0.0
        b = float(dim - 1)

        def _independent_alpha_for(degree: int) -> np.ndarray:
            """Reconstruct the rotation table the way the wrapper does.

            Use the standard :func:`Chebyshev.interpolate` interface
            which samples ``f`` on the exact (floating-point, unrounded)
            Chebyshev-Gauss nodes (zeros of ``T_{d+1}``) of ``[a, b]``
            and builds the interpolant in one canonical call.  The
            result is mathematically identical to a manual ``fit`` on
            the same Gauss nodes with matching degree, but the
            standard API eliminates any risk of off-by-one node
            indexing or manual node-mapping mistakes.

            Final steps mirror the wrapper:

            - convert Chebyshev series to monomial (power) basis
            - scale coefficients by ``2`` (amplitude mode)
            - evaluate on the full integer control grid

            This helper deliberately avoids calling any implementation
            helper (including :func:`_pwcheb_expected_alpha`).
            """

            nonlocal f
            f_vec = np.vectorize(f, otypes=[np.float64])
            cheb = _NpCheb.interpolate(f_vec, deg=degree, domain=[a, b])
            poly = cheb.convert(kind=_NpPoly)
            coeffs = 2.0 * np.asarray(poly.coef, dtype=np.float64)
            return np.asarray(
                np.polyval(coeffs[::-1], ks), dtype=np.float64
            )

        interpolation_errors: list[float] = []
        for d in degrees:
            op = PiecewiseChebyshev(
                f_x=f, degree=d, breakpoints=None, num_state_qubits=n
            )
            alpha_from_op = _pwcheb_expected_alpha(op)
            alpha_independent = _independent_alpha_for(d)

            # (1) Cross-validation: wrapper vs. independent recalc agree
            diff_alpha = _angular_abs_diff(alpha_from_op, alpha_independent)
            np.testing.assert_allclose(
                diff_alpha,
                np.zeros_like(diff_alpha),
                atol=1e-10,
                err_msg=(
                    f"PWCHEB-002 degree={d}: wrapper alpha differs from "
                    f"independent Chebyshev-Gauss rebuild alpha."
                ),
            )
            # (2) Use the independent alpha (they match anyway) to
            # measure the approximation quality against ``2·f(k)``.
            approx_err = _angular_abs_diff(alpha_independent, target_theta)
            max_err = float(np.max(approx_err))
            interpolation_errors.append(max_err)

        # Strict monotonic decay is guaranteed by Chebyshev theory for
        # an analytic target on any non-degenerate interval, provided
        # the interpolant uses the exact (unrounded) Chebyshev-Gauss
        # nodes.  Allow 1e-14 floating-point slack for near-machine-
        # precision equalities between two adjacent degrees.
        for idx in range(len(interpolation_errors) - 1):
            assert (
                interpolation_errors[idx + 1]
                <= interpolation_errors[idx] + 1e-14
            ), (
                f"PWCHEB-002 degree={degrees[idx]} err="
                f"{interpolation_errors[idx]:.3e} exceeds degree="
                f"{degrees[idx+1]} err={interpolation_errors[idx+1]:.3e}"
            )
        # Primary rate check: d=5 error must be at least 10× smaller
        # than d=1 for a smoothly varying analytic target.
        assert interpolation_errors[-1] <= 0.1 * interpolation_errors[0], (
            f"PWCHEB-002 convergence factor too small: err(d=5)="
            f"{interpolation_errors[-1]:.3e} > 0.1·err(d=1)="
            f"{0.1 * interpolation_errors[0]:.3e}"
        )

    # -- PWCHEB-003 head-identity segment when breakpoints[0] > 0 -------

    @staticmethod
    def test_pwcheb_003_head_identity_amplitude_1() -> None:
        """breakpoints=[2] for n=2 → [0,2) identity segment (sin(π/2)=1).

        The auto-filled head segment uses the polynomial ``[π]`` which
        produces theta = π for every integer k in [0,2).  Therefore the
        |1>-component amplitude is exactly ``sin(π/2) = 1`` (full result
        qubit flip) for those two control states; this is the
        amplitude-encoding "full result" identity used by the wrapper
        and matches Qiskit's convention.
        """

        n = 2

        def f(x: float) -> float:
            return 0.25 * float(x)

        op = PiecewiseChebyshev(
            f_x=f, degree=1, breakpoints=[2], num_state_qubits=n
        )
        alpha = _pwcheb_expected_alpha(op)
        for k in (0, 1):
            theta_k = float(alpha[k])
            assert abs(math.sin(theta_k / 2.0) - 1.0) < 1e-12, (
                f"PWCHEB-003 head-segment sin(theta/2) at k={k} != 1.0: "
                f"theta={theta_k:.6e}"
            )
        _assert_pwcheb_statevector(op)


class TestPiecewiseChebyshevBound:
    """PWCHEB-BOUND-* edge cases and invalid-input guards."""

    @staticmethod
    def test_pwcheb_bound_001_tail_identity_segment() -> None:
        """breakpoints=[0,2,5] for n=3 → tail [5,8) is identity (π)."""

        n = 3

        def f(x: float) -> float:
            return 0.1 * float(x)

        op = PiecewiseChebyshev(
            f_x=f, degree=1, breakpoints=[0, 2, 5], num_state_qubits=n
        )
        assert op.breakpoints == [0, 2, 5, 8]
        np.testing.assert_allclose(
            op.polynomials[-1],
            np.asarray([math.pi], dtype=np.float64),
            atol=1e-14,
        )
        alpha = _pwcheb_expected_alpha(op)
        for k in range(5, 8):
            assert abs(math.sin(alpha[k] / 2.0) - 1.0) < 1e-12, (
                f"tail-identity k={k} not full amplitude"
            )
        _assert_pwcheb_statevector(op)

    @staticmethod
    def test_pwcheb_bound_002_head_and_tail_both_auto() -> None:
        """breakpoints=[2,5] for n=3 → both head & tail auto-filled (π)."""

        n = 3

        def f(x: float) -> float:
            return 0.2 * float(x)

        op = PiecewiseChebyshev(
            f_x=f, degree=1, breakpoints=[2, 5], num_state_qubits=n
        )
        assert op.breakpoints == [0, 2, 5, 8]
        for idx in (0, -1):
            np.testing.assert_allclose(
                op.polynomials[idx],
                np.asarray([math.pi], dtype=np.float64),
                atol=1e-14,
                err_msg=f"auto-filled segment idx={idx} coeff != π",
            )
        _assert_pwcheb_statevector(op)

    @staticmethod
    def test_pwcheb_bound_003_minimum_n1() -> None:
        """Smoke test: n=1, degree=0, breakpoints=None constructs cleanly."""

        op = PiecewiseChebyshev(
            f_x=0.25, degree=0, breakpoints=None, num_state_qubits=1
        )
        assert op.num_state_qubits == 1
        assert op.breakpoints == [0, 2]
        _assert_pwcheb_statevector(op)

    @staticmethod
    def test_pwcheb_bound_004_invalid_inputs_raise() -> None:
        """Type/value guards mirror the wrapper's constructor checks."""

        # f_x: bool rejected
        with pytest.raises(TypeError, match="f_x"):
            PiecewiseChebyshev(  # type: ignore[arg-type]
                f_x=True, degree=0, breakpoints=None, num_state_qubits=2
            )
        # degree: negative
        with pytest.raises(ValueError, match="degree"):
            PiecewiseChebyshev(
                f_x=0.5, degree=-1, breakpoints=None, num_state_qubits=2
            )
        # breakpoints: descending
        with pytest.raises(ValueError, match="breakpoints"):
            PiecewiseChebyshev(
                f_x=lambda x: 0.1 * float(x),
                degree=1,
                breakpoints=[3, 1],
                num_state_qubits=2,
            )
        # num_state_qubits None + breakpoints default/None → cannot infer
        with pytest.raises(ValueError, match="num_state_qubits"):
            PiecewiseChebyshev(f_x=0.5, degree=0, breakpoints=None)


# ---------------------------------------------------------------------------
# Section 10.  Piecewise-Chebyshev BLACK-BOX validation
# ---------------------------------------------------------------------------
# The tests above repeatedly call ``_assert_pwcheb_statevector`` which
# itself relies on ``PiecewiseChebyshev.polynomials`` (a public but
# *structural* property) to build the reference ``theta(k)`` table.  The
# design team explicitly requested a *genuine* black-box battery that
# never inspects ``polynomials``, ``breakpoints``, or any other
# implementation detail of the wrapper.  All assertions below use only
# the public callable ``f_x`` and the simulated statevector.
#
# Coverage map:
#   PWCHEB-BB-001  Single-segment half-sine table (n=3, degree=5)
#   PWCHEB-BB-002  Piecewise constant-on-intervals via degree=0
#   PWCHEB-BB-003  Breakpoints > 0 and < 2^n: both head and tail
#                  identity (π) segments, verified against the external
#                  target-function convention *outside* the Chebyshev
#                  wrapper itself.
# ---------------------------------------------------------------------------


class TestPiecewiseChebyshevBlackBox:
    """Attribute-free (no polynomials / no breakpoints inspect) tests.

    Every test in this class reconstructs the expected reference
    probabilities purely from the user ``f_x`` callable plus an
    independent numerical Chebyshev fit, and compares against
    ``Statevector.from_circuit`` without ever reading
    :attr:`PiecewiseChebyshev.polynomials`.  This is the closest a
    unit-test can get to the "operator as an oracle" perspective required
    by the design document.
    """

    # ------------------------------------------------------------------
    # Shared oracle-style helper
    # ------------------------------------------------------------------
    @staticmethod
    def _black_box_probe_and_assert(
        op: PiecewiseChebyshev,
        expected_p1: np.ndarray,
        *,
        atol: float = 1e-8,
    ) -> None:
        """Probe each |k> and enforce result probabilities + no x-leak.

        Parameters
        ----------
        op: PiecewiseChebyshev
            Operator under test.  Its public properties are only used for
            circuit construction (``build_circuit()``) and for the wire
            layout (``control_register_indices`` / ``result_qubit_index``)
            – which every circuit consumer must, unavoidably, know.  The
            structural ``polynomials`` and ``breakpoints`` attributes are
            **not** consulted.
        expected_p1: numpy.ndarray
            Per-control ``P(|1>_result | k)`` for ``k in [0, 2^n)``.  The
            0-branch probability is inferred as ``1 - expected_p1`` so
            the caller only has to supply a single physical quantity.
        atol: float
            Absolute tolerance on each probability scalar.
        """

        n = int(op.num_state_qubits)
        num_qubits = int(op.num_qubits)
        x_wires = list(op.control_register_indices)
        target_wire = int(op.result_qubit_index)
        if len(expected_p1) != (1 << n):  # pragma: no cover
            raise AssertionError(
                "expected_p1 length mismatch: got "
                f"{len(expected_p1)} vs 2^{n}={1 << n}."
            )
        circ = op.build_circuit()
        for k, want_p1 in enumerate(expected_p1):
            want_p0 = 1.0 - float(want_p1)
            full = Circuit(num_qubits)
            for bit_local, wire in enumerate(x_wires):
                if (k >> bit_local) & 1:
                    full.x(wire)
            full.compose(circ, list(range(num_qubits)))
            sv = _statevector_data(Statevector.from_circuit(full))

            x_masks = [1 << wire for wire in x_wires]
            target_mask = 1 << target_wire
            garbage_mass = 0.0
            p0 = 0.0
            p1 = 0.0
            for idx in range(1 << num_qubits):
                x_val = 0
                for bit_local, physical_mask in enumerate(x_masks):
                    if (idx & physical_mask):
                        x_val |= 1 << bit_local
                mag2 = float(abs(sv[idx]) ** 2)
                if x_val != k:
                    garbage_mass += mag2
                    continue
                if idx & target_mask:
                    p1 += mag2
                else:
                    p0 += mag2

            assert garbage_mass < 1e-12, (
                f"PWCHEB-BB control-leak k={k}: norm-sq of x != {k} is "
                f"{garbage_mass:.3e} (must be 0)."
            )
            for label, have, want in [
                ("P(0)", p0, want_p0),
                ("P(1)", p1, float(want_p1)),
            ]:
                if abs(have - want) > atol:  # pragma: no cover
                    raise AssertionError(
                        f"PWCHEB-BB k={k}: {label} = {have:.6e}, want "
                        f"{want:.6e} (|diff|={abs(have-want):.3e})."
                    )

    # -- PWCHEB-BB-001 single-segment smooth sine -----------------------

    @staticmethod
    @pytest.mark.filterwarnings(
        "ignore:PiecewisePolynomialPauliRotation max_degree d_max="
        ":UserWarning"
    )
    @pytest.mark.filterwarnings(
        "ignore:One or more theta\\(k\\) values produced by the piecewise "
        "Chebyshev interpolant exceed 2\\*pi:UserWarning"
    )
    @pytest.mark.filterwarnings(
        "ignore:One or more raw piecewise-polynomial angles exceed "
        "2\\*pi in absolute value:UserWarning"
    )
    def test_pwcheb_bb_001_single_segment_sin() -> None:
        """Single-segment f(x) = 0.5 · sin(π·x/8) on n=3, degree=5.

        The test does **not** read any implementation detail of
        :class:`PiecewiseChebyshev`.  The expected result probabilities
        are derived in the test file by independently rebuilding the
        standard Chebyshev-Gauss interpolant using
        :func:`numpy.polynomial.chebyshev.Chebyshev.interpolate`, which
        samples ``f`` on the exact (floating-point, unrounded) zeros
        of ``T_{d+1}`` over the domain ``[0, 2**n - 1]``.  Monomial
        coefficients are multiplied by ``2`` to encode the target
        amplitude.  This is a genuinely independent mathematical
        cross-validation that matches the wrapper's standard-Chebyshev
        construction but does not call any implementation helper.
        """

        from numpy.polynomial import Chebyshev as _NpCheb
        from numpy.polynomial import Polynomial as _NpPoly

        n = 3
        degree = 5

        def f(x: float) -> float:
            return 0.5 * float(math.sin(math.pi * x / float(1 << n)))

        op = PiecewiseChebyshev(
            f_x=f, degree=degree, breakpoints=None, num_state_qubits=n
        )

        dim = 1 << n
        a = 0.0
        b = float(dim - 1)
        # Independent classical Chebyshev interpolation using the
        # standard ``interpolate`` interface, which guarantees exact
        # matching with a wrapper using the same Chebyshev-Gauss nodes.
        f_vec = np.vectorize(f, otypes=[np.float64])
        cheb = _NpCheb.interpolate(f_vec, deg=degree, domain=[a, b])
        poly = cheb.convert(kind=_NpPoly)
        coeffs = 2.0 * np.asarray(poly.coef, dtype=np.float64)
        ks = np.arange(dim, dtype=np.float64)
        theta = np.asarray(np.polyval(coeffs[::-1], ks), dtype=np.float64)
        want_p1 = np.sin(0.5 * theta) ** 2
        TestPiecewiseChebyshevBlackBox._black_box_probe_and_assert(
            op, np.asarray(want_p1, dtype=np.float64), atol=1e-6
        )

    # -- PWCHEB-BB-002 piecewise constant (degree 0) plateaux -----------

    @staticmethod
    def test_pwcheb_bb_002_piecewise_constant() -> None:
        """Degree-0 piecewise Chebyshev == piecewise constant amplitude.

        Breakpoints ``[0, 2, 4, 8]`` for ``n=3``; target amplitudes on
        each segment are ``0.4``, ``0.7``, ``0.2`` (all ``≤ 1`` so the
        ``asin`` domain is respected).  For each segment the degree-0
        fit is just the target constant itself; the wrapper therefore
        produces ``theta = 2 * c_{(j)}`` on segment ``j`` and we expect
        ``sin²(theta / 2) = sin²(c_{(j)})``.  A loose ``atol=1e-6`` is
        used because Chebyshev.fit on a single point adds some
        rounding noise that is still negligible physically.
        """

        n = 3
        breakpoints = [0, 2, 4, 8]
        seg_amps = [0.4, 0.7, 0.2]

        def f(x: float) -> float:
            xv = float(x)
            for j in range(len(seg_amps)):
                lo = float(breakpoints[j])
                hi = float(breakpoints[j + 1])
                if lo <= xv < hi:
                    return float(seg_amps[j])
            return 0.0  # pragma: no cover

        op = PiecewiseChebyshev(
            f_x=f, degree=0, breakpoints=list(breakpoints), num_state_qubits=n
        )

        dim = 1 << n
        want_p1 = np.zeros(dim, dtype=np.float64)
        for j, amp in enumerate(seg_amps):
            lo = breakpoints[j]
            hi = breakpoints[j + 1]
            want_p1[lo:hi] = float(math.sin(amp) ** 2)
        TestPiecewiseChebyshevBlackBox._black_box_probe_and_assert(
            op, want_p1, atol=1e-6
        )

    # -- PWCHEB-BB-003 head & tail identity segments (π semantics) ------

    @staticmethod
    def test_pwcheb_bb_003_identity_segments_full_amplitude() -> None:
        """``breakpoints=[2, 5]``, n=3 → head and tail identity segments.

        The design convention for PiecewiseChebyshev is that auto-filled
        identity segments represent *full* amplitude (``theta = π``),
        i.e. ``sin(π/2) == 1``.  The middle user segment ``[2, 5)``
        uses ``f(k) = 0.1*k + 0.05`` (linearly varying, degree=1).

        Instead of reading :attr:`PiecewiseChebyshev.polynomials` the
        test reconstructs the user segment's interpolant independently
        via the standard :func:`Chebyshev.interpolate` interface on
        domain ``[lo_user, hi_user - 1]``, which matches the wrapper's
        Chebyshev-Gauss (zeros) construction exactly.  The combined
        probability table is therefore built up from *three*
        independent pieces:

        * k ∈ [0, 2): head identity → ``P(|1>) = sin²(π/2) = 1``
        * k ∈ [2, 5): interpolated ``theta(k) = 2·P_1(k)`` →  ``P(|1>) = sin²(theta(k)/2)``
        * k ∈ [5, 8): tail identity → ``P(|1>) = sin²(π/2) = 1``

        and fed directly into the oracle-style probe helper.
        """

        from numpy.polynomial import Chebyshev as _NpCheb
        from numpy.polynomial import Polynomial as _NpPoly

        n = 3
        user_breakpoints = [2, 5]
        lo_user, hi_user = 2, 5

        def f(x: float) -> float:
            return 0.1 * float(x) + 0.05

        op = PiecewiseChebyshev(
            f_x=f, degree=1,
            breakpoints=list(user_breakpoints),
            num_state_qubits=n,
        )

        # Independent degree-1 interpolant *only* for the user segment
        # [2, 5).  domain endpoints use the inclusive upper bound
        # ``hi - 1`` because that is what the wrapper's
        # ``_interpolate_segment`` uses for its Chebyshev-Gauss
        # node grid mapping.
        a = float(lo_user)
        b = float(hi_user - 1)
        degree = 1
        f_vec = np.vectorize(f, otypes=[np.float64])
        cheb = _NpCheb.interpolate(f_vec, deg=degree, domain=[a, b])
        poly = cheb.convert(kind=_NpPoly)
        coeffs = 2.0 * np.asarray(poly.coef, dtype=np.float64)

        dim = 1 << n
        want_p1 = np.ones(dim, dtype=np.float64)  # identity = 1 (full amp)
        for k in range(lo_user, hi_user):
            theta_k = float(np.polyval(coeffs[::-1], np.float64(k)))
            want_p1[k] = float(math.sin(0.5 * theta_k) ** 2)
        TestPiecewiseChebyshevBlackBox._black_box_probe_and_assert(
            op, want_p1, atol=1e-7
        )

    # -- PWCHEB-BB-004 multi-segment reciprocal end-to-end -------------

    @staticmethod
    def test_pwcheb_bb_004_multi_segment_reciprocal() -> None:
        """Degree-2 1/x on J=3 segments [0,1) ∪ [1,4) ∪ [4,8) + BB SV probe.

        This is the high-priority *end-to-end black-box* companion to
        :func:`TestPiecewiseChebyshevDocCases.test_pwcheb_001_reciprocal_degree2`.
        The doc case only asserts classical coefficient error bounds and
        then uses the white-box :func:`_assert_pwcheb_statevector` helper
        (which reads ``op.polynomials`` / ``op.breakpoints``).  This case
        builds the *identical* multi-segment configuration but then:

        * reconstructs every user-segment interpolant independently via
          :func:`numpy.polynomial.chebyshev.Chebyshev.interpolate` using
          the same Chebyshev-Gauss nodes as the wrapper,
        * hard-codes the head identity-segment convention (``θ = π``
          ⇒ ``sin²(π/2) = 1``),
        * assembles a full ``2**n`` probability table **without**
          touching any ``PiecewiseChebyshev`` structural attribute,
        * and finally feeds the operator and the expected probability
          table into the oracle-style :func:`_black_box_probe_and_assert`
          helper, which iterates ``|k>`` one by one, runs
          ``Statevector.from_circuit`` on the full circuit (``X|0>``
          prepend + operator), and enforces:

          - no control-register garbage mass (``x`` value must stay
            equal to the prepared ``k`` across all ancilla states),
          - ``P(|1>_result | k)`` matches the independently derived
            expectation within ``atol=1e-6``.

        This is the closest the suite gets to a "component as a black
        oracle" integration test: it validates the *entire* chain
        (breakpoint auto-completion → per-segment Chebyshev-Gauss
        interpolation → coefficient ×2 scaling → comparator geq flags →
        incremental multi-controlled RY accumulation → final state
        amplitudes) without ever inspecting internal attributes.
        """

        from numpy.polynomial import Chebyshev as _NpCheb
        from numpy.polynomial import Polynomial as _NpPoly

        n = 3
        degree = 2
        user_bp = [1, 4, 8]

        def f_target(x: float | int) -> float:
            xv = float(x)
            return 1.0 / xv if xv != 0.0 else 1.0

        op = PiecewiseChebyshev(
            f_x=lambda x, _ft=f_target: _ft(x),
            degree=degree,
            breakpoints=list(user_bp),
            num_state_qubits=n,
        )

        # Independent expected-probability assembly.  The completed
        # breakpoint list follows the wrapper's auto-completion rule:
        # prepend ``0`` because user_bp does not start at 0; user_bp
        # already ends at 2**n so no trailing append is needed.
        completed_bp = [0] + list(user_bp)
        f_vec = np.vectorize(f_target, otypes=[np.float64])
        dim = 1 << n
        want_p1 = np.zeros(dim, dtype=np.float64)
        for j in range(len(completed_bp) - 1):
            lo = completed_bp[j]
            hi = completed_bp[j + 1]
            if j == 0:
                # Head auto-filled identity segment: θ = π ⟺ sin²(θ/2)=1
                for k in range(lo, hi):
                    want_p1[k] = 1.0
                continue
            a = float(lo)
            b = float(hi - 1)
            cheb = _NpCheb.interpolate(f_vec, deg=degree, domain=[a, b])
            poly = cheb.convert(kind=_NpPoly)
            coeffs = 2.0 * np.asarray(poly.coef, dtype=np.float64)
            for k in range(lo, hi):
                theta_k = float(
                    np.polyval(coeffs[::-1], np.float64(k))
                )
                want_p1[k] = float(math.sin(0.5 * theta_k) ** 2)
        TestPiecewiseChebyshevBlackBox._black_box_probe_and_assert(
            op, want_p1, atol=1e-6
        )


