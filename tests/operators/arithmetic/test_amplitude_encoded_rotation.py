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


