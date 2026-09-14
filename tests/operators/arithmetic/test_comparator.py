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
"""Test suite for the arithmetic integer comparator.

Scope of this module:
    The public IntegerComparator class is exercised here against the compact
    arithmetic test design document. Tests focus on MVP coverage only:
    correctness, interface attributes and invalid-construction handling.
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

from cqlib_algorithms.operators.arithmetic import (  # noqa: E402
    IntegerComparator,
)

RANDOM_SEED = 42
RANDOM_SAMPLE_SIZE = 20


@pytest.fixture(scope="module", autouse=True)
def _fixed_random_seed() -> None:
    """Fix the global NumPy seed for every random-sampling test."""
    np.random.seed(RANDOM_SEED)
    yield


def _decode_subregister_bits(
    outcome_int: int,
    bit_indices: list[int],
) -> int:
    """Decode an LSB-first integer from selected qubit positions."""
    result = 0
    for shift, qubit in enumerate(bit_indices):
        if (outcome_int >> qubit) & 1:
            result |= 1 << shift
    return result


def _run_comparator(
    comparator: IntegerComparator,
    x: int,
    flag: int = 0,
) -> dict[str, int]:
    """Encode, simulate and decode a single deterministic comparator run."""
    circuit = Circuit(comparator.num_qubits)
    for bit, qubit in enumerate(comparator.x_register_indices):
        if (x >> bit) & 1:
            circuit.x(qubit)
    if flag:
        circuit.x(comparator.flag_index)

    circuit.compose(
        comparator.build_circuit(),
        list(range(comparator.num_qubits)),
    )

    statevector = Statevector.from_circuit(circuit)
    probs = np.asarray(statevector.probabilities(), dtype=np.float64)
    if not np.isclose(probs.max(), 1.0, atol=1e-8):
        raise AssertionError(
            f"Expected deterministic state with max prob 1, got {probs.max()}"
        )

    outcome = int(np.argmax(probs))
    return {
        "x": _decode_subregister_bits(outcome, comparator.x_register_indices),
        "flag": (outcome >> comparator.flag_index) & 1,
        "helper": (outcome >> comparator.helper_index) & 1,
    }


@pytest.mark.parametrize("synthesis", ["ripple_carry", "draper_qft"])
def test_integer_comparator_geq_basic_correctness(synthesis: str) -> None:
    """CMP-01. Random inputs match the classical x >= value predicate."""
    n = 4
    value = 8
    comparator = IntegerComparator(
        n,
        value,
        relation="geq",
        synthesis=synthesis,
    )
    for _ in range(RANDOM_SAMPLE_SIZE):
        x = int(np.random.randint(0, 1 << n))
        result = _run_comparator(comparator, x)
        assert result["x"] == x
        assert result["flag"] == int(x >= value)


@pytest.mark.parametrize("synthesis", ["ripple_carry", "draper_qft"])
def test_integer_comparator_lt_basic_correctness(synthesis: str) -> None:
    """CMP-02. Random inputs match the classical x < value predicate."""
    n = 4
    value = 8
    comparator = IntegerComparator(
        n,
        value,
        relation="lt",
        synthesis=synthesis,
    )
    for _ in range(RANDOM_SAMPLE_SIZE):
        x = int(np.random.randint(0, 1 << n))
        result = _run_comparator(comparator, x)
        assert result["x"] == x
        assert result["flag"] == int(x < value)


@pytest.mark.parametrize("synthesis", ["ripple_carry", "draper_qft"])
def test_integer_comparator_x_equal_value_boundary(synthesis: str) -> None:
    """CMP-03. Equality must satisfy geq and reject lt."""
    n = 4
    value = 8
    geq = IntegerComparator(n, value, relation="geq", synthesis=synthesis)
    lt = IntegerComparator(n, value, relation="lt", synthesis=synthesis)
    assert _run_comparator(geq, value)["flag"] == 1
    assert _run_comparator(lt, value)["flag"] == 0


@pytest.mark.parametrize("synthesis", ["ripple_carry", "draper_qft"])
def test_integer_comparator_value_zero_boundary(synthesis: str) -> None:
    """CMP-04. Every x must satisfy x >= 0 and fail x < 0."""
    n = 4
    geq = IntegerComparator(n, 0, relation="geq", synthesis=synthesis)
    lt = IntegerComparator(n, 0, relation="lt", synthesis=synthesis)
    for x in range(1 << n):
        assert _run_comparator(geq, x)["flag"] == 1
        assert _run_comparator(lt, x)["flag"] == 0


@pytest.mark.parametrize("synthesis", ["ripple_carry", "draper_qft"])
def test_integer_comparator_minimal_width(synthesis: str) -> None:
    """CMP-05. The n=1, value=0 boundary case is constructible and correct."""
    geq = IntegerComparator(1, 0, relation="geq", synthesis=synthesis)
    lt = IntegerComparator(1, 0, relation="lt", synthesis=synthesis)
    assert _run_comparator(geq, 0)["flag"] == 1
    assert _run_comparator(geq, 1)["flag"] == 1
    assert _run_comparator(lt, 0)["flag"] == 0
    assert _run_comparator(lt, 1)["flag"] == 0


def test_integer_comparator_properties_interface() -> None:
    """CMP-06. Static attributes match the comparator layout contract."""
    n = 4
    value = 8
    comparator = IntegerComparator(
        n,
        value,
        relation="geq",
        synthesis="ripple_carry",
    )
    assert comparator.num_state_qubits == n
    assert comparator.num_qubits == n + 2
    assert comparator.value == value
    assert comparator.relation == "geq"
    assert comparator.synthesis == "ripple_carry"
    assert comparator.x_register_indices == list(range(n))
    assert comparator.flag_index == n
    assert comparator.helper_index == n + 1


def test_integer_comparator_invalid_inputs_raise_valueerror() -> None:
    """CMP-07. Illegal construction inputs raise ValueError."""
    with pytest.raises(ValueError):
        IntegerComparator(0, 0)
    with pytest.raises(ValueError):
        IntegerComparator(-1, 0)
    with pytest.raises(ValueError):
        IntegerComparator(4, -1)
    with pytest.raises(ValueError):
        IntegerComparator(4, 16)
    with pytest.raises(ValueError):
        IntegerComparator(4, 1.5)
    with pytest.raises(ValueError):
        IntegerComparator(4, 3, relation="bad")
    with pytest.raises(ValueError):
        IntegerComparator(4, 3, synthesis="bad")

