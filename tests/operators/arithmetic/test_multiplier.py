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
"""Test suite for the quantum arithmetic multiplier class.

Scope of this module:
    The public Multiplier class is exercised here against the compact
    arithmetic test design document located under docs/design_md/test_design.
    Both available synthesis strategies (hrs_cumulative and draper_qft) are
    covered along with the optional approximation_degree configuration of the
    draper_qft branch. All tests rely only on cqlib primitives (unitary
    simulation via Statevector with sample_shots) and NumPy; no external
    quantum computing framework is imported.

Coverage map (case IDs match the design document):
    MUL-01 .. MUL-06  -> Multiplier across both synthesis strategies
    MUL-07            -> Cross-synthesis consistency check
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from cqlib import Circuit
from cqlib.qis import Statevector

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
TEST_DIR = Path(__file__).resolve().parent
for _path in (str(SRC), str(TEST_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from cqlib_algorithms.operators.arithmetic import (  # noqa: E402
    Multiplier,
)

RANDOM_SEED = 42
RANDOM_SAMPLE_SIZE = 20
LARGE_WIDTH = 4

_SYNTHESIS_STRATEGIES = ["hrs_cumulative", "draper_qft"]


@pytest.fixture(scope="module", autouse=True)
def _fixed_random_seed() -> None:
    """Fix the global NumPy seed for every random-sampling test."""
    np.random.seed(RANDOM_SEED)
    yield


def _apply_x_to_value(
    circuit: Circuit,
    value: int,
    bit_indices: list[int],
) -> None:
    """Flip qubits at bit_indices on circuit to encode integer value LSB first."""
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
    """Decode an LSB-first integer from selected qubit positions using Outcome."""
    result = 0
    for shift, qubit in enumerate(bit_indices):
        if outcome.is_one(qubit):
            result |= 1 << shift
    return result


def _run_multiplier(
    multiplier: Multiplier,
    a: int,
    b: int,
) -> dict[str, int]:
    """Encode a, b, simulate the multiplier and decode every register.

    The hrs_cumulative synthesis treats the product register as an
    accumulator, so for a plain multiplication it must be left in the
    all-zero ground state when the multiplier circuit is composed. The
    draper_qft synthesis also requires an all-zero product register. This
    helper therefore never flips any qubit inside
    multiplier.product_register_indices prior to composition.
    """
    n = multiplier.num_state_qubits
    modulus = 1 << n
    if not 0 <= a < modulus or not 0 <= b < modulus:
        raise ValueError(
            f"Operands a={a}, b={b} out of range for {n}-bit multiplier."
        )

    circuit = Circuit(multiplier.num_qubits)
    _apply_x_to_value(circuit, a, multiplier.a_register_indices)
    _apply_x_to_value(circuit, b, multiplier.b_register_indices)
    circuit.compose(
        multiplier.build_circuit(),
        list(range(multiplier.num_qubits)),
    )

    statevector = Statevector.from_circuit(circuit)
    shots = statevector.sample_shots(1)
    outcome = shots[0]

    decoded: dict[str, int] = {
        "a": _decode_register_from_outcome(
            outcome, multiplier.a_register_indices
        ),
        "b": _decode_register_from_outcome(
            outcome, multiplier.b_register_indices
        ),
        "product": _decode_register_from_outcome(
            outcome, multiplier.product_register_indices
        ),
    }
    helper_index = multiplier.helper_index
    if helper_index is not None:
        decoded["helper"] = 1 if outcome.is_one(helper_index) else 0
    return decoded


def _collect_register_index_lists(multiplier: Multiplier) -> list[list[int]]:
    """Return every index list exposed by multiplier for sanity checks."""
    groups: list[list[int]] = [
        list(multiplier.a_register_indices),
        list(multiplier.b_register_indices),
        list(multiplier.product_register_indices),
    ]
    helper_index = multiplier.helper_index
    if helper_index is not None:
        groups.append([helper_index])
    return groups


def _assert_clean_register_layout(multiplier: Multiplier) -> None:
    """Assert no overlap and every index inside [0, num_qubits)."""
    groups = _collect_register_index_lists(multiplier)
    flat: list[int] = [idx for group in groups for idx in group]
    assert len(flat) == len(set(flat)), (
        f"Register indices overlap for {multiplier.synthesis}: {flat}"
    )
    assert all(0 <= idx < multiplier.num_qubits for idx in flat), (
        f"Index out of range for {multiplier.synthesis}: {flat}"
    )


# ---------------------------------------------------------------------------
# Section 1.  MUL-01 Basic correctness --------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("synthesis", _SYNTHESIS_STRATEGIES)
def test_multiplier_basic_correctness_random_sample(synthesis: str) -> None:
    """MUL-01. Random operands match the classical a * b product."""
    n = 3
    modulus = 1 << n
    product_modulus = 1 << (2 * n)
    multiplier = Multiplier(n, synthesis=synthesis)
    for _ in range(RANDOM_SAMPLE_SIZE):
        a, b = np.random.randint(0, modulus, size=2).tolist()
        result = _run_multiplier(multiplier, a, b)
        assert result["a"] == a
        assert result["b"] == b
        assert result["product"] == (a * b) % product_modulus


# ---------------------------------------------------------------------------
# Section 2.  MUL-02 Multiplication by zero ---------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("synthesis", _SYNTHESIS_STRATEGIES)
def test_multiplier_multiply_by_zero(synthesis: str) -> None:
    """MUL-02. b = 0 forces product to 0 for every input a."""
    n = 3
    modulus = 1 << n
    multiplier = Multiplier(n, synthesis=synthesis)
    for a in range(modulus):
        result = _run_multiplier(multiplier, a, 0)
        assert result["a"] == a
        assert result["b"] == 0
        assert result["product"] == 0
        result2 = _run_multiplier(multiplier, 0, a)
        assert result2["a"] == 0
        assert result2["b"] == a
        assert result2["product"] == 0


# ---------------------------------------------------------------------------
# Section 3.  MUL-03 Maximum value times maximum value ----------------------
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("synthesis", _SYNTHESIS_STRATEGIES)
def test_multiplier_max_times_max(synthesis: str) -> None:
    """MUL-03. (2**n - 1) * (2**n - 1) = 2**(2n) - 2**(n+1) + 1."""
    n = 3
    max_value = (1 << n) - 1
    expected_product = max_value * max_value
    multiplier = Multiplier(n, synthesis=synthesis)
    result = _run_multiplier(multiplier, max_value, max_value)
    assert result["a"] == max_value
    assert result["b"] == max_value
    assert result["product"] == expected_product


# ---------------------------------------------------------------------------
# Section 4.  MUL-04 Minimal width n = 1 ------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("synthesis", _SYNTHESIS_STRATEGIES)
def test_multiplier_minimal_width_n1(synthesis: str) -> None:
    """MUL-04. The n=1 boundary case covers every 1-bit operand pair."""
    multiplier = Multiplier(1, synthesis=synthesis)
    for a in (0, 1):
        for b in (0, 1):
            result = _run_multiplier(multiplier, a, b)
            assert result["a"] == a
            assert result["b"] == b
            assert result["product"] == a * b


# ---------------------------------------------------------------------------
# Section 5.  MUL-05 Properties interface -----------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("synthesis", _SYNTHESIS_STRATEGIES)
def test_multiplier_properties_interface(synthesis: str) -> None:
    """MUL-05. Static attributes match the Multiplier layout contract."""
    n = 4
    multiplier = Multiplier(n, synthesis=synthesis)
    assert multiplier.num_state_qubits == n
    assert multiplier.synthesis == synthesis
    assert len(multiplier.a_register_indices) == n
    assert len(multiplier.b_register_indices) == n
    assert len(multiplier.product_register_indices) == 2 * n
    assert min(multiplier.a_register_indices) == 0
    assert max(multiplier.a_register_indices) == n - 1
    assert min(multiplier.b_register_indices) == n
    assert max(multiplier.b_register_indices) == 2 * n - 1
    assert min(multiplier.product_register_indices) == 2 * n
    assert max(multiplier.product_register_indices) == 4 * n - 1

    if synthesis == "hrs_cumulative":
        assert multiplier.num_qubits == 4 * n + 1
        assert multiplier.helper_index == 4 * n
    else:
        assert multiplier.num_qubits == 4 * n
        assert multiplier.helper_index is None

    _assert_clean_register_layout(multiplier)


def test_multiplier_draper_qft_approximation_degree_property() -> None:
    """MUL-05 extension. approximation_degree is stored only for draper_qft."""
    multiplier_qft = Multiplier(
        4,
        synthesis="draper_qft",
        approximation_degree=1,
    )
    assert multiplier_qft.approximation_degree == 1

    multiplier_hrs = Multiplier(4, synthesis="hrs_cumulative")
    assert multiplier_hrs.approximation_degree is None


def test_multiplier_name_default_and_override() -> None:
    """MUL-05 extension. Default name is Multiplier and can be overridden."""
    multiplier_default = Multiplier(3)
    assert multiplier_default.name == "Multiplier"

    multiplier_named = Multiplier(3, name="Mul")
    assert multiplier_named.name == "Mul"


# ---------------------------------------------------------------------------
# Section 6.  MUL-06 Invalid construction inputs ----------------------------
# ---------------------------------------------------------------------------


def test_multiplier_invalid_num_state_qubits_raise_valueerror() -> None:
    """MUL-06. Zero, negative, and non-integer widths raise ValueError."""
    with pytest.raises(ValueError):
        Multiplier(0)
    with pytest.raises(ValueError):
        Multiplier(-1)
    with pytest.raises(ValueError):
        Multiplier(2.5)


def test_multiplier_unknown_synthesis_raises_valueerror() -> None:
    """MUL-06. Unsupported synthesis strings raise ValueError."""
    with pytest.raises(ValueError):
        Multiplier(3, synthesis="bogus")


def test_multiplier_hrs_cumulative_rejects_approximation_degree() -> None:
    """MUL-06. hrs_cumulative does not accept approximation_degree."""
    with pytest.raises(ValueError):
        Multiplier(
            3,
            synthesis="hrs_cumulative",
            approximation_degree=1,
        )


def test_multiplier_approximation_degree_must_be_nonnegative_int() -> None:
    """MUL-06. Negative or non-integer approximation_degree is rejected."""
    with pytest.raises(ValueError):
        Multiplier(
            3,
            synthesis="draper_qft",
            approximation_degree=-1,
        )
    with pytest.raises(ValueError):
        Multiplier(
            3,
            synthesis="draper_qft",
            approximation_degree=1.5,
        )


def test_multiplier_name_must_be_string_or_none() -> None:
    """MUL-06. Non-string name values raise ValueError."""
    with pytest.raises(ValueError):
        Multiplier(3, name=123)
    with pytest.raises(ValueError):
        Multiplier(3, name=True)


# ---------------------------------------------------------------------------
# Section 7.  MUL-07 Cross-synthesis consistency + extras -------------------
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3])
def test_multiplier_synthesis_consistency(n: int) -> None:
    """MUL-07. hrs_cumulative and draper_qft agree on every input pair."""
    hrs = Multiplier(n, synthesis="hrs_cumulative")
    qft = Multiplier(n, synthesis="draper_qft")
    modulus = 1 << n
    for a in range(modulus):
        for b in range(modulus):
            hrs_result = _run_multiplier(hrs, a, b)
            qft_result = _run_multiplier(qft, a, b)
            assert hrs_result["a"] == qft_result["a"] == a
            assert hrs_result["b"] == qft_result["b"] == b
            assert hrs_result["product"] == qft_result["product"]


@pytest.mark.parametrize("synthesis", _SYNTHESIS_STRATEGIES)
def test_multiplier_a_and_b_registers_preserved(synthesis: str) -> None:
    """Extra. Operand registers a and b are never modified."""
    n = 3
    modulus = 1 << n
    multiplier = Multiplier(n, synthesis=synthesis)
    for a in range(modulus):
        for b in range(modulus):
            result = _run_multiplier(multiplier, a, b)
            assert result["a"] == a
            assert result["b"] == b


@pytest.mark.parametrize("synthesis", _SYNTHESIS_STRATEGIES)
def test_multiplier_hrs_helper_returns_to_zero_when_present(synthesis: str) -> None:
    """Extra. The hrs_cumulative helper qubit returns to |0> after the circuit."""
    if synthesis != "hrs_cumulative":
        pytest.skip("Helper qubit exists only for hrs_cumulative synthesis.")
    n = 3
    modulus = 1 << n
    multiplier = Multiplier(n, synthesis=synthesis)
    for a in range(modulus):
        for b in range(modulus):
            result = _run_multiplier(multiplier, a, b)
            assert result["helper"] == 0


@pytest.mark.parametrize("synthesis", _SYNTHESIS_STRATEGIES)
def test_multiplier_large_width_random_sample(synthesis: str) -> None:
    """Extra. Large width sampling for a size too costly to exhaust."""
    n = LARGE_WIDTH
    modulus = 1 << n
    product_modulus = 1 << (2 * n)
    multiplier = Multiplier(n, synthesis=synthesis)
    for _ in range(RANDOM_SAMPLE_SIZE):
        a, b = np.random.randint(0, modulus, size=2).tolist()
        result = _run_multiplier(multiplier, a, b)
        assert result["a"] == a
        assert result["b"] == b
        assert result["product"] == (a * b) % product_modulus
