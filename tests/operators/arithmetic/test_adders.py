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
"""Test suite for the quantum arithmetic adder classes.

Scope of this module:
    The following public adder classes from the arithmetic package are
    exercised here against the design specification located under
    docs/design_md/test_design. All tests rely only on cqlib primitives
    (unitary simulation via Statevector) and NumPy; no external quantum
    computing framework is imported.

Coverage map (case IDs match the design document):
    ADD-H-01 .. ADD-H-13  -> HalfAdder
    ADD-F-01 .. ADD-F-14  -> FullAdder
    ADD-M-01 .. ADD-M-11  -> ModularAdder
    ADD-C-01 .. ADD-C-05  -> DraperConstantAdder
    GEN-01 .. GEN-05      -> Generic parameterized tests run against the
                             three two-register adder classes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from cqlib import Circuit
from cqlib.qis.state import Statevector

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
TEST_DIR = Path(__file__).resolve().parent
for _path in (str(SRC), str(TEST_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from _adder_utils import (  # noqa: E402  (sys.path adjustment above)
    _assert_clean_register_layout,
    _expected_num_qubits,
    _run_adder,
)

from cqlib_algorithms.operators.arithmetic import (  # noqa: E402
    DraperConstantAdder,
    FullAdder,
    HalfAdder,
    ModularAdder,
)

RANDOM_SEED = 42
LARGE_WIDTH = 8
RANDOM_SAMPLE_SIZE = 50


@pytest.fixture(scope="module", autouse=True)
def _fixed_random_seed() -> None:
    """Fix the global NumPy seed for every random-sampling test."""
    np.random.seed(RANDOM_SEED)
    yield


def _run_constant_adder(
    adder: DraperConstantAdder,
    b: int,
) -> int:
    """Encode b, simulate DraperConstantAdder and decode the result."""
    circuit = Circuit(adder.num_qubits)
    for bit, qubit in enumerate(adder.b_register_indices):
        if (b >> bit) & 1:
            circuit.x(qubit)

    circuit.compose(adder.build_circuit(), list(range(adder.num_qubits)))
    statevector = Statevector.from_circuit(circuit)
    probs = np.asarray(statevector.probabilities(), dtype=np.float64)
    if not np.isclose(probs.max(), 1.0, atol=1e-8):
        raise AssertionError(
            f"Expected deterministic state with max prob 1, got {probs.max()}"
        )

    outcome = int(np.argmax(probs))
    result = 0
    for bit, qubit in enumerate(adder.b_register_indices):
        if (outcome >> qubit) & 1:
            result |= 1 << bit
    return result


# ---------------------------------------------------------------------------
# Section 1.  HalfAdder -----------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3])
def test_half_adder_exhaustive_small_widths(n: int) -> None:
    """ADD-H-01, ADD-H-07, ADD-H-08. Exhaustive check for widths n <= 3."""
    adder = HalfAdder(n)
    modulus = 1 << n
    for a in range(modulus):
        for b in range(modulus):
            result = _run_adder(adder, a, b)
            total = a + b
            assert result["a"] == a
            assert result["b_sum"] == total % modulus
            assert result["carry_out"] == (total >> n) & 1


@pytest.mark.parametrize("n", [4, 5, 6])
def test_half_adder_carry_chain_longest_propagation(n: int) -> None:
    """ADD-H-02. (2**n - 1) + 1 overflows exactly into the carry bit."""
    adder = HalfAdder(n)
    a = (1 << n) - 1
    b = 1
    result = _run_adder(adder, a, b)
    assert result["a"] == a
    assert result["b_sum"] == 0
    assert result["carry_out"] == 1


@pytest.mark.parametrize("n", [4, 5, 6])
def test_half_adder_carry_chain_termination(n: int) -> None:
    """ADD-H-03. A short carry chain that dies out after one step."""
    adder = HalfAdder(n)
    a = (1 << n) - 2
    b = 2
    result = _run_adder(adder, a, b)
    total = a + b
    assert result["a"] == a
    assert result["b_sum"] == total % (1 << n)
    assert result["carry_out"] == (total >> n) & 1


@pytest.mark.parametrize("n", [4, 8])
def test_half_adder_alternating_bit_pattern(n: int) -> None:
    """ADD-H-04. 0xAA... plus 0x55... fills every position without carry."""
    adder = HalfAdder(n)
    a = sum((1 << i) for i in range(1, n, 2))
    b = sum((1 << i) for i in range(0, n, 2))
    result = _run_adder(adder, a, b)
    assert result["a"] == a
    assert result["b_sum"] == (1 << n) - 1
    assert result["carry_out"] == 0


@pytest.mark.parametrize("n", [3, 7])
def test_half_adder_all_ones_plus_all_ones(n: int) -> None:
    """ADD-H-05. Maximum value plus itself equals max minus two plus carry."""
    adder = HalfAdder(n)
    value = (1 << n) - 1
    result = _run_adder(adder, value, value)
    assert result["a"] == value
    assert result["b_sum"] == value - 1
    assert result["carry_out"] == 1


def test_half_adder_plus_zero_is_identity() -> None:
    """ADD-H-06. Adding zero leaves the other operand untouched."""
    n = 3
    adder = HalfAdder(n)
    for a in range(1 << n):
        result = _run_adder(adder, a, 0)
        assert result["a"] == a
        assert result["b_sum"] == a
        assert result["carry_out"] == 0


def test_half_adder_large_width_random_sample() -> None:
    """ADD-H-09. Random sampling for a width that is too large to exhaust."""
    n = LARGE_WIDTH
    modulus = 1 << n
    adder = HalfAdder(n)
    for _ in range(RANDOM_SAMPLE_SIZE):
        a, b = np.random.randint(0, modulus, size=2).tolist()
        result = _run_adder(adder, a, b)
        total = a + b
        assert result["a"] == a
        assert result["b_sum"] == total % modulus
        assert result["carry_out"] == (total >> n) & 1


def test_half_adder_a_register_preserved_exhaustive() -> None:
    """ADD-H-10. Operand register a is never modified by the schedule."""
    n = 3
    adder = HalfAdder(n)
    modulus = 1 << n
    for a in range(modulus):
        for b in range(modulus):
            assert _run_adder(adder, a, b)["a"] == a


def test_half_adder_properties_interface() -> None:
    """ADD-H-11. Attribute values match the HalfAdder layout contract."""
    n = 4
    adder = HalfAdder(n)
    assert adder.num_state_qubits == n
    assert adder.num_qubits == 2 * n + 2
    assert len(adder.a_register_indices) == n
    assert len(adder.b_register_indices) == n
    assert 0 <= adder.carry_out_index < adder.num_qubits
    assert 0 <= adder.helper_index < adder.num_qubits
    assert adder.carry_in_index is None
    assert min(adder.a_register_indices) == 0
    assert max(adder.a_register_indices) == n - 1
    assert min(adder.b_register_indices) == n
    assert max(adder.b_register_indices) == 2 * n - 1
    assert adder.carry_out_index == 2 * n
    assert adder.helper_index == 2 * n + 1
    _assert_clean_register_layout(adder)


def test_half_adder_invalid_inputs_raise_valueerror() -> None:
    """ADD-H-12. Illegal construction inputs raise ValueError."""
    with pytest.raises(ValueError):
        HalfAdder(0)
    with pytest.raises(ValueError):
        HalfAdder(-1)
    with pytest.raises(ValueError):
        HalfAdder(1.5)
    with pytest.raises(ValueError):
        HalfAdder(4, synthesis="nonexistent")


@pytest.mark.parametrize("n", [1, 2, 3])
def test_half_adder_synthesis_consistency(n: int) -> None:
    """ADD-H-13. ripple_carry and draper_qft must agree on every input."""
    adder_rc = HalfAdder(n, synthesis="ripple_carry")
    adder_qft = HalfAdder(n, synthesis="draper_qft")
    modulus = 1 << n
    for a in range(modulus):
        for b in range(modulus):
            result_rc = _run_adder(adder_rc, a, b)
            result_qft = _run_adder(adder_qft, a, b)
            assert result_rc["a"] == result_qft["a"]
            assert result_rc["b_sum"] == result_qft["b_sum"]
            assert result_rc["carry_out"] == result_qft["carry_out"]


# ---------------------------------------------------------------------------
# Section 2.  FullAdder -----------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2])
def test_full_adder_exhaustive_small_widths(n: int) -> None:
    """ADD-F-01, ADD-F-09. Exhaustive search for n <= 2 including c_in."""
    adder = FullAdder(n)
    modulus = 1 << n
    for c_in in (0, 1):
        for a in range(1 << n):
            for b in range(1 << n):
                result = _run_adder(adder, a, b, carry_in=c_in)
                total = c_in + a + b
                assert result["a"] == a
                assert result["b_sum"] == total % modulus
                assert result["carry_out"] == (total >> n) & 1
                assert result["carry_in"] == c_in


@pytest.mark.parametrize("n", [3, 5, 7])
def test_full_adder_carry_chain_penetrates_from_cin(n: int) -> None:
    """ADD-F-02. c_in plus all-ones propagates cleanly to the carry out."""
    adder = FullAdder(n)
    result = _run_adder(adder, (1 << n) - 1, 0, carry_in=1)
    assert result["a"] == (1 << n) - 1
    assert result["b_sum"] == 0
    assert result["carry_out"] == 1
    assert result["carry_in"] == 1


@pytest.mark.parametrize("n", [4, 6])
def test_full_adder_all_ones_plus_cin_one_overflows(n: int) -> None:
    """ADD-F-03. Maximum value plus itself plus one saturates the output."""
    adder = FullAdder(n)
    max_value = (1 << n) - 1
    result = _run_adder(adder, max_value, max_value, carry_in=1)
    expected_total = 2 * ((1 << n) - 1) + 1
    assert result["a"] == max_value
    assert result["b_sum"] == expected_total % (1 << n)
    assert result["carry_out"] == (expected_total >> n) & 1
    assert result["carry_in"] == 1


@pytest.mark.parametrize("n", [4])
def test_full_adder_carry_chain_commutative_paths(n: int) -> None:
    """ADD-F-04. Swapping operands yields identical deterministic outcomes."""
    adder = FullAdder(n)
    result_a = _run_adder(adder, (1 << n) - 1, 0, carry_in=1)
    result_b = _run_adder(adder, 0, (1 << n) - 1, carry_in=1)
    assert result_a["b_sum"] == result_b["b_sum"]
    assert result_a["carry_out"] == result_b["carry_out"]
    assert result_a["carry_in"] == result_b["carry_in"] == 1


def test_full_adder_zero_plus_zero_no_cin() -> None:
    """ADD-F-05. Identity corner case with no incoming carry."""
    result = _run_adder(FullAdder(5), 0, 0, carry_in=0)
    assert result["a"] == 0
    assert result["b_sum"] == 0
    assert result["carry_out"] == 0
    assert result["carry_in"] == 0


def test_full_adder_zero_plus_zero_with_cin() -> None:
    """ADD-F-06. Identity corner case with a single incoming carry."""
    result = _run_adder(FullAdder(5), 0, 0, carry_in=1)
    assert result["a"] == 0
    assert result["b_sum"] == 1
    assert result["carry_out"] == 0
    assert result["carry_in"] == 1


def test_full_adder_plus_zero_identity_without_cin() -> None:
    """ADD-F-07. Operand b is zero, so output equals a register."""
    n = 3
    adder = FullAdder(n)
    for a in range(1 << n):
        result = _run_adder(adder, a, 0, carry_in=0)
        assert result["a"] == a
        assert result["b_sum"] == a
        assert result["carry_out"] == 0
        assert result["carry_in"] == 0


def test_full_adder_cin0_matches_half_adder_output() -> None:
    """ADD-F-08. FullAdder(c_in=0) reproduces HalfAdder output format."""
    n = 3
    half = HalfAdder(n)
    full = FullAdder(n)
    modulus = 1 << n
    for a in range(modulus):
        for b in range(modulus):
            half_result = _run_adder(half, a, b)
            full_result = _run_adder(full, a, b, carry_in=0)
            assert full_result["a"] == half_result["a"]
            assert full_result["b_sum"] == half_result["b_sum"]
            assert full_result["carry_out"] == half_result["carry_out"]
            assert full_result["carry_in"] == 0


def test_full_adder_large_width_random_sample() -> None:
    """ADD-F-10. Large width random sample covering (c_in, a, b) triples."""
    n = LARGE_WIDTH
    modulus = 1 << n
    adder = FullAdder(n)
    for _ in range(RANDOM_SAMPLE_SIZE):
        a, b = np.random.randint(0, modulus, size=2).tolist()
        c_in = int(np.random.randint(0, 2))
        result = _run_adder(adder, a, b, carry_in=c_in)
        total = c_in + a + b
        assert result["a"] == a
        assert result["b_sum"] == total % modulus
        assert result["carry_out"] == (total >> n) & 1
        assert result["carry_in"] == c_in


def test_full_adder_a_register_preserved_exhaustive() -> None:
    """ADD-F-11. Operand a never changes for any full input combination."""
    n = 2
    adder = FullAdder(n)
    for c_in in (0, 1):
        for a in range(1 << n):
            for b in range(1 << n):
                assert _run_adder(adder, a, b, carry_in=c_in)["a"] == a


def test_full_adder_properties_interface() -> None:
    """ADD-F-12. Static attributes match the FullAdder layout contract."""
    n = 4
    adder = FullAdder(n)
    assert adder.num_state_qubits == n
    assert adder.num_qubits == 2 * n + 2
    assert len(adder.a_register_indices) == n
    assert len(adder.b_register_indices) == n
    assert adder.carry_in_index == 0
    assert 0 <= adder.carry_out_index < adder.num_qubits
    assert min(adder.a_register_indices) == 1
    assert max(adder.a_register_indices) == n
    assert min(adder.b_register_indices) == n + 1
    assert max(adder.b_register_indices) == 2 * n
    assert adder.carry_out_index == 2 * n + 1
    _assert_clean_register_layout(adder)


def test_full_adder_invalid_inputs_raise_valueerror() -> None:
    """ADD-F-13. Illegal construction inputs raise ValueError."""
    with pytest.raises(ValueError):
        FullAdder(0)
    with pytest.raises(ValueError):
        FullAdder(-3)
    with pytest.raises(ValueError):
        FullAdder(2, synthesis="does_not_exist")
    with pytest.raises(ValueError):
        FullAdder(2, synthesis="draper_qft")


def test_full_adder_rejects_draper_qft_synthesis() -> None:
    """ADD-F-14. FullAdder exposes only ripple_carry synthesis."""
    with pytest.raises(ValueError):
        FullAdder(3, synthesis="draper_qft")


# ---------------------------------------------------------------------------
# Section 3.  ModularAdder --------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3])
def test_modular_adder_exhaustive_small_widths(n: int) -> None:
    """ADD-M-01, ADD-M-07. Exhaustive check for small widths."""
    adder = ModularAdder(n)
    modulus = 1 << n
    for a in range(modulus):
        for b in range(modulus):
            result = _run_adder(adder, a, b)
            assert result["a"] == a
            assert result["b_sum"] == (a + b) % modulus


@pytest.mark.parametrize("n", [3, 5, 7])
def test_modular_adder_max_plus_one_wraps_to_zero(n: int) -> None:
    """ADD-M-02. (2**n - 1) + 1 reduces to zero in modular arithmetic."""
    adder = ModularAdder(n)
    result = _run_adder(adder, (1 << n) - 1, 1)
    assert result["a"] == (1 << n) - 1
    assert result["b_sum"] == 0


@pytest.mark.parametrize("n", [3, 5, 7])
def test_modular_adder_max_plus_max_reduces_to_n_minus_two(n: int) -> None:
    """ADD-M-03. Maximum plus maximum equals two modulo 2**n minus two."""
    adder = ModularAdder(n)
    value = (1 << n) - 1
    result = _run_adder(adder, value, value)
    assert result["a"] == value
    assert result["b_sum"] == (value + value) % (1 << n)


def test_modular_adder_zero_plus_zero() -> None:
    """ADD-M-04. Identity case: zero plus zero stays zero."""
    result = _run_adder(ModularAdder(5), 0, 0)
    assert result["a"] == 0
    assert result["b_sum"] == 0


def test_modular_adder_plus_zero_is_identity() -> None:
    """ADD-M-05. Adding zero leaves the other operand unchanged."""
    n = 3
    adder = ModularAdder(n)
    for a in range(1 << n):
        result = _run_adder(adder, a, 0)
        assert result["a"] == a
        assert result["b_sum"] == a


def test_modular_adder_matches_full_adder_stripping_carry() -> None:
    """ADD-M-06. FullAdder lower n bits agree with ModularAdder output."""
    n = 3
    modular = ModularAdder(n)
    full = FullAdder(n)
    modulus = 1 << n
    for a in range(modulus):
        for b in range(modulus):
            m_result = _run_adder(modular, a, b)
            f_result = _run_adder(full, a, b, carry_in=0)
            assert m_result["a"] == f_result["a"]
            assert m_result["b_sum"] == f_result["b_sum"] % modulus


def test_modular_adder_large_width_random_sample() -> None:
    """ADD-M-08. Large width random sampling."""
    n = LARGE_WIDTH
    modulus = 1 << n
    adder = ModularAdder(n)
    for _ in range(RANDOM_SAMPLE_SIZE):
        a, b = np.random.randint(0, modulus, size=2).tolist()
        result = _run_adder(adder, a, b)
        assert result["a"] == a
        assert result["b_sum"] == (a + b) % modulus


def test_modular_adder_properties_interface() -> None:
    """ADD-M-09. Static attributes match the ModularAdder layout contract."""
    n = 4
    adder = ModularAdder(n)
    assert adder.num_state_qubits == n
    assert adder.num_qubits == 2 * n + 1
    assert len(adder.a_register_indices) == n
    assert len(adder.b_register_indices) == n
    assert adder.carry_in_index is None
    assert adder.carry_out_index is None
    assert 0 <= adder.helper_index < adder.num_qubits
    assert min(adder.a_register_indices) == 0
    assert max(adder.a_register_indices) == n - 1
    assert min(adder.b_register_indices) == n
    assert max(adder.b_register_indices) == 2 * n - 1
    assert adder.helper_index == 2 * n
    _assert_clean_register_layout(adder)


def test_modular_adder_invalid_inputs_raise_valueerror() -> None:
    """ADD-M-10. Illegal construction inputs raise ValueError."""
    with pytest.raises(ValueError):
        ModularAdder(0)
    with pytest.raises(ValueError):
        ModularAdder(-2)
    with pytest.raises(ValueError):
        ModularAdder(3, synthesis="bogus")


@pytest.mark.parametrize("n", [1, 2, 3])
def test_modular_adder_synthesis_consistency(n: int) -> None:
    """ADD-M-11. ripple_carry and draper_qft must agree on every input."""
    adder_rc = ModularAdder(n, synthesis="ripple_carry")
    adder_qft = ModularAdder(n, synthesis="draper_qft")
    modulus = 1 << n
    for a in range(modulus):
        for b in range(modulus):
            rc_result = _run_adder(adder_rc, a, b)
            qft_result = _run_adder(adder_qft, a, b)
            assert rc_result["a"] == qft_result["a"]
            assert rc_result["b_sum"] == qft_result["b_sum"]


# ---------------------------------------------------------------------------
# Section 4.  DraperConstantAdder -------------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3])
def test_constant_adder_exhaustive_small_widths(n: int) -> None:
    """ADD-C-01. Exhaustive correctness for small widths and constants."""
    modulus = 1 << n
    for value in range(modulus):
        adder = DraperConstantAdder(n, value)
        for b in range(modulus):
            assert _run_constant_adder(adder, b) == (b + value) % modulus


def test_constant_adder_plus_zero_is_identity() -> None:
    """ADD-C-02. Adding the zero constant preserves every basis state."""
    n = 4
    adder = DraperConstantAdder(n, 0)
    for b in range(1 << n):
        assert _run_constant_adder(adder, b) == b


@pytest.mark.parametrize("n", [3, 5, 7])
def test_constant_adder_max_constant_plus_one_wraps_to_zero(n: int) -> None:
    """ADD-C-03. Adding the largest constant wraps modulo 2**n."""
    adder = DraperConstantAdder(n, (1 << n) - 1)
    assert _run_constant_adder(adder, 1) == 0


def test_constant_adder_properties_interface() -> None:
    """ADD-C-04. Static attributes match the one-register layout contract."""
    n = 4
    value = 6
    adder = DraperConstantAdder(n, value)
    assert adder.num_state_qubits == n
    assert adder.num_qubits == n
    assert adder.value == value
    assert adder.b_register_indices == list(range(n))


def test_constant_adder_invalid_inputs_raise_valueerror() -> None:
    """ADD-C-05. Illegal construction inputs raise ValueError."""
    with pytest.raises(ValueError):
        DraperConstantAdder(0, 0)
    with pytest.raises(ValueError):
        DraperConstantAdder(-1, 0)
    with pytest.raises(ValueError):
        DraperConstantAdder(3, -1)
    with pytest.raises(ValueError):
        DraperConstantAdder(3, 8)
    with pytest.raises(ValueError):
        DraperConstantAdder(3, 1.5)


# ---------------------------------------------------------------------------
# Section 5.  Generic tests (GEN-01 .. GEN-05) ------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "adder_cls", [HalfAdder, FullAdder, ModularAdder]
)
def test_adder_general_constructible_and_qubit_count(
    adder_cls: type[HalfAdder | FullAdder | ModularAdder],
) -> None:
    """GEN-01 and GEN-02. Constructor succeeds and num_qubits is exact."""
    n = 4
    adder = adder_cls(n)
    assert adder.num_qubits == _expected_num_qubits(adder_cls, n)


@pytest.mark.parametrize(
    "adder_cls", [HalfAdder, FullAdder, ModularAdder]
)
def test_adder_general_registers_clean_layout(
    adder_cls: type[HalfAdder | FullAdder | ModularAdder],
) -> None:
    """GEN-03 and GEN-04. No overlap; every index inside [0, num_qubits)."""
    n = 5
    adder = adder_cls(n)
    _assert_clean_register_layout(adder)


@pytest.mark.parametrize(
    "adder_cls", [HalfAdder, FullAdder, ModularAdder]
)
def test_adder_general_zero_width_raises_valueerror(
    adder_cls: type[HalfAdder | FullAdder | ModularAdder],
) -> None:
    """GEN-05. Zero bit-width is always illegal."""
    with pytest.raises(ValueError):
        adder_cls(0)

