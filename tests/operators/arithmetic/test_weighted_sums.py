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
"""Test suite for the WeightedSum arithmetic operator.

Scope of this module:
    The :class:`~cqlib_algorithms.operators.arithmetic.WeightedSum` class is
    exercised here against the design specification located under
    ``docs/design_md/test_design`` ("arithmetic test.md").  Tests cover the
    two available synthesis strategies (``qft`` and ``carry``) and share the
    generic layout / exception tests required of every arithmetic component.

Coverage map (case IDs match the design document):
    WS-01 .. WS-05   -> WeightedSum functionality and basic interface
    GEN-01 .. GEN-05 -> Generic parametric tests run against WeightedSum
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from cqlib.qis.state import Statevector

from cqlib import Circuit
from cqlib_algorithms.operators.arithmetic import WeightedSum

RANDOM_SEED = 20260917
BASIC_NUM_INPUTS = [2, 2]
BASIC_WEIGHTS = [2, 3]
RANDOM_SAMPLE_SIZE = 30
CARRY_MIN_RANDOM_WIDTH = 2
CARRY_MAX_RANDOM_WIDTH = 3
QFT_MIN_RANDOM_WIDTH = 2
QFT_MAX_RANDOM_WIDTH = 4
MAX_RANDOM_WEIGHT = 4


@pytest.fixture(scope="module", autouse=True)
def _fixed_random_seed() -> None:
    """Fix the global NumPy seed for every random-sampling test."""

    np.random.seed(RANDOM_SEED)
    yield


# ---------------------------------------------------------------------------
# Section 0.  Simulation helpers -------------------------------------------
# ---------------------------------------------------------------------------


def _encode_value(circuit: Circuit, value: int, qubits: list[int]) -> None:
    """Flip the qubits corresponding to the LSB-first binary of ``value``."""

    for bit, qubit in enumerate(qubits):
        if (value >> bit) & 1:
            circuit.x(qubit)


def _decode_value(bitstring: str, qubits: list[int]) -> int:
    """Decode an LSB-first integer from ``qubits`` using ``bitstring``."""

    total = 0
    num_qubits = len(bitstring)
    for bit, qubit in enumerate(qubits):
        if bitstring[num_qubits - 1 - qubit] == "1":
            total |= 1 << bit
    return total


def _sample_deterministic_bitstring(circuit: Circuit) -> str:
    """Simulate ``circuit`` and return its unique computational bitstring."""

    statevector = Statevector.from_circuit(circuit)
    probs = np.asarray(statevector.probabilities(), dtype=np.float64)
    if not np.isclose(float(probs.max()), 1.0, atol=1e-8):
        raise AssertionError(
            "WeightedSum must leave the state in a single computational "
            "basis state; got max probability "
            f"{float(probs.max()):.6f}."
        )
    outcome = int(np.argmax(probs))
    return format(outcome, f"0{circuit.num_qubits}b")


def _run_weighted_sum(
    operator: WeightedSum,
    input_values: list[int],
) -> dict[str, int | list[int]]:
    """Simulate WeightedSum on concrete classical integers.

    Parameters
        operator A constructed WeightedSum wrapper.
        input_values Classical integer values (one per input register) used
            to initialise the corresponding state qubits.  Each value must
            fit inside ``num_inputs[k]`` bits.

    Returns
        A dictionary with keys ``"inputs"`` (decoded back from the quantum
        state to assert invariance), ``"sum"`` (the computed weighted sum
        decoded from the sum register) and ``"expected"`` (the classical
        reference value ``sum(weights[k] * input_values[k])``).
    """

    if len(input_values) != len(operator.num_inputs):
        raise AssertionError(
            "input_values length must match num_inputs length."
        )

    circuit = Circuit(operator.num_qubits)
    for value, qubits in zip(
        input_values, operator.input_register_indices, strict=True
    ):
        _encode_value(circuit, value, qubits)

    circuit.compose(operator.build_circuit(), list(range(operator.num_qubits)))
    bitstring = _sample_deterministic_bitstring(circuit)

    inputs_out: list[int] = []
    for qubits in operator.input_register_indices:
        inputs_out.append(_decode_value(bitstring, qubits))
    sum_out = _decode_value(bitstring, operator.sum_register_indices)

    expected = 0
    for weight, value in zip(
        operator.weights, input_values, strict=True
    ):
        expected += weight * value

    return {
        "inputs": inputs_out,
        "sum": sum_out,
        "expected": expected,
    }


def _assert_clean_layout(operator: WeightedSum) -> None:
    """Assert that all exposed registers are disjoint and inside bounds."""

    all_indices: set[int] = set()
    for register in operator.input_register_indices:
        for idx in register:
            assert idx not in all_indices, f"Duplicated index {idx}."
            all_indices.add(idx)
    for idx in operator.sum_register_indices:
        assert idx not in all_indices, f"Duplicated index {idx}."
        all_indices.add(idx)
    if operator.carry_register_indices is not None:
        for idx in operator.carry_register_indices:
            assert idx not in all_indices, f"Duplicated index {idx}."
            all_indices.add(idx)
    if operator.helper_index is not None:
        assert operator.helper_index not in all_indices, (
            f"Duplicated helper index {operator.helper_index}."
        )
        all_indices.add(operator.helper_index)

    for idx in all_indices:
        assert 0 <= idx < operator.num_qubits, (
            f"Index {idx} is out of range [0, {operator.num_qubits})."
        )


# ---------------------------------------------------------------------------
# Section 1.  WS-01 / WS-02 / WS-03  Functional tests (both routes) --------
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("synthesis", ["qft", "carry"])
def test_weighted_sum_ws01_basic_correctness_hand_picked(
    synthesis: str,
) -> None:
    """WS-01. Hand-picked case: 2*2 + 3*3 = 13 (two 2-bit inputs)."""

    operator = WeightedSum(
        BASIC_WEIGHTS, BASIC_NUM_INPUTS, synthesis=synthesis
    )
    result = _run_weighted_sum(operator, [2, 3])
    assert result["inputs"] == [2, 3]
    assert result["sum"] == result["expected"]
    assert result["sum"] == 13


@pytest.mark.parametrize("synthesis", ["qft", "carry"])
def test_weighted_sum_ws01_exhaustive_small_widths(synthesis: str) -> None:
    """WS-01. Exhaustive enumeration for tiny num_inputs widths & weight sweeps."""

    num_inputs = [2, 2]
    weight_cases = [
        [2, 3],
        [1, 4],
        [5, 2],
        [3, 7],
    ]
    input_ranges = [range(1 << width) for width in num_inputs]
    for weights in weight_cases:
        operator = WeightedSum(weights, num_inputs, synthesis=synthesis)
        for values in itertools.product(*input_ranges):
            result = _run_weighted_sum(operator, list(values))
            assert result["inputs"] == list(values)
            assert result["sum"] == result["expected"], (
                f"Mismatch for weights={weights}, values={list(values)}: "
                f"expected {result['expected']}, got {result['sum']}."
            )


@pytest.mark.parametrize("synthesis", ["qft", "carry"])
def test_weighted_sum_ws02_all_zero_inputs(synthesis: str) -> None:
    """WS-02. When every input is zero the sum register must stay zero."""

    weights = [5, 7, 2]
    num_inputs = [3, 2, 3]
    operator = WeightedSum(weights, num_inputs, synthesis=synthesis)
    zeros = [0 for _ in num_inputs]
    result = _run_weighted_sum(operator, zeros)
    assert result["inputs"] == zeros
    assert result["sum"] == 0
    assert result["expected"] == 0


@pytest.mark.parametrize("synthesis", ["qft", "carry"])
def test_weighted_sum_ws03_weights_all_one(synthesis: str) -> None:
    """WS-03. All weights equal to 1 -> sum equals hamming popcount."""

    num_inputs = [1, 1, 1, 1]
    weights = [1, 1, 1, 1]
    operator = WeightedSum(weights, num_inputs, synthesis=synthesis)

    for bits in itertools.product([0, 1], repeat=len(num_inputs)):
        result = _run_weighted_sum(operator, list(bits))
        popcount = sum(bits)
        assert result["inputs"] == list(bits)
        assert result["sum"] == popcount
        assert result["expected"] == popcount


@pytest.mark.parametrize("synthesis", ["qft", "carry"])
def test_weighted_sum_ws01_random_sampling(synthesis: str) -> None:
    """WS-01. Random (weights, values) sampling."""

    if synthesis == "carry":
        min_width, max_width = (
            CARRY_MIN_RANDOM_WIDTH,
            CARRY_MAX_RANDOM_WIDTH,
        )
    else:
        min_width, max_width = (
            QFT_MIN_RANDOM_WIDTH,
            QFT_MAX_RANDOM_WIDTH,
        )

    for _ in range(RANDOM_SAMPLE_SIZE):
        num_registers = int(np.random.randint(2, 4))
        num_inputs: list[int] = [
            int(np.random.randint(min_width, max_width + 1))
            for _ in range(num_registers)
        ]
        weights: list[int] = [
            int(np.random.randint(0, MAX_RANDOM_WEIGHT + 1))
            for _ in range(num_registers)
        ]
        if all(w == 0 for w in weights):
            weights[0] = 1
        input_values = [
            int(np.random.randint(0, 1 << width))
            for width in num_inputs
        ]

        operator = WeightedSum(
            weights, num_inputs, synthesis=synthesis
        )
        result = _run_weighted_sum(operator, input_values)
        assert result["inputs"] == input_values
        assert result["sum"] == result["expected"], (
            f"Mismatch for weights={weights}, num_inputs={num_inputs}, "
            f"values={input_values}: expected {result['expected']}, "
            f"got {result['sum']}."
        )


@pytest.mark.parametrize("synthesis", ["qft", "carry"])
def test_weighted_sum_input_registers_invariant_single_large(
    synthesis: str,
) -> None:
    """WS-01. Larger hand-picked example validating input invariance."""

    weights = [1, 2]
    num_inputs = [4, 3]
    operator = WeightedSum(weights, num_inputs, synthesis=synthesis)
    values = [13, 6]
    result = _run_weighted_sum(operator, values)
    assert result["inputs"] == values
    assert result["sum"] == result["expected"]


# ---------------------------------------------------------------------------
# Section 2.  WS-04 / GEN-01 .. GEN-04  Property / layout tests ------------
# ---------------------------------------------------------------------------


def _expected_num_qubits(
    weights: list[int],
    num_inputs: list[int],
    synthesis: str,
) -> int:
    """Compute the expected total qubit count from first principles."""

    total_state = sum(num_inputs)
    total_weight = 0
    for weight, width in zip(weights, num_inputs, strict=True):
        total_weight += weight * ((1 << width) - 1)
    s = 1 if total_weight == 0 else total_weight.bit_length()
    if synthesis == "carry":
        carries = max(s - 1, 0)
        helper = 1 if s > 2 else 0
        return total_state + s + carries + helper
    return total_state + s


@pytest.mark.parametrize(
    "case",
    [
        (
            "simple",
            [1, 1],
            [3, 3],
            "qft",
        ),
        (
            "simple_carry",
            [1, 1],
            [3, 3],
            "carry",
        ),
        (
            "single_register",
            [5],
            [4],
            "qft",
        ),
    ],
)
def test_weighted_sum_ws04_gen01_gen02_properties(
    case: tuple[str, list[int], list[int], str],
) -> None:
    """WS-04 + GEN-01 + GEN-02. Circuit builds; num_qubits match expected."""

    name, weights, num_inputs, synthesis = case
    operator = WeightedSum(
        weights,
        num_inputs,
        synthesis=synthesis,
        name=name,
    )
    exp_state = sum(num_inputs)
    exp_total = _expected_num_qubits(weights, num_inputs, synthesis)
    assert operator.name == name
    assert operator.num_state_qubits == exp_state
    assert operator.num_qubits == exp_total
    _assert_clean_layout(operator)

    circuit = operator.build_circuit()
    assert circuit.num_qubits == exp_total


def test_weighted_sum_gen03_gen04_carry_registers_are_clean() -> None:
    """GEN-03, GEN-04. Carry layout is disjoint and inside range."""

    operator = WeightedSum([3, 4], [4, 4], synthesis="carry")
    _assert_clean_layout(operator)
    assert operator.carry_register_indices is not None
    assert len(operator.carry_register_indices) == operator.num_sum_qubits - 1
    assert operator.helper_index is not None or operator.num_sum_qubits <= 2


def test_weighted_sum_gen05_zero_width_illegal() -> None:
    """GEN-05. Zero bit widths always raise ValueError."""

    with pytest.raises(ValueError):
        WeightedSum([1], [0])
    with pytest.raises(ValueError):
        WeightedSum([1, 2], [3, 0])


# ---------------------------------------------------------------------------
# Section 3.  WS-05 ValueError tests ---------------------------------------
# ---------------------------------------------------------------------------


def test_weighted_sum_ws05_invalid_inputs_raise_valueerror() -> None:
    """WS-05. Illegal construction *values* always raise ValueError."""

    with pytest.raises(ValueError):
        WeightedSum([], [])
    with pytest.raises(ValueError):
        WeightedSum([1, 2], [1])
    with pytest.raises(ValueError):
        WeightedSum([1], [1], synthesis="unknown")
    with pytest.raises(ValueError):
        WeightedSum([-1], [1])
    with pytest.raises(ValueError):
        WeightedSum([1], [-2])
    with pytest.raises(ValueError):
        WeightedSum(
            [1], [1], synthesis="carry", approximation_degree=0
        )
    with pytest.raises(ValueError):
        WeightedSum(
            [1], [1], synthesis="qft", approximation_degree=-1
        )


# ---------------------------------------------------------------------------
# Section 4.  TypeError tests (strictly separated from ValueError) ---------
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case,callable_wrapper",
    [
        (
            "weights_tuple",
            lambda: WeightedSum((1, 2), [1, 1]),
        ),
        (
            "num_inputs_tuple",
            lambda: WeightedSum([1, 2], (1, 1)),
        ),
        (
            "weights_element_float",
            lambda: WeightedSum([1.5, 2], [1, 1]),
        ),
        (
            "weights_element_bool",
            lambda: WeightedSum([True, 2], [1, 1]),
        ),
        (
            "num_inputs_element_float",
            lambda: WeightedSum([1, 2], [1.5, 1]),
        ),
        (
            "num_inputs_element_bool",
            lambda: WeightedSum([1, 2], [True, 1]),
        ),
        (
            "name_integer",
            lambda: WeightedSum([1], [1], name=42),
        ),
        (
            "approximation_degree_float_qft",
            lambda: WeightedSum(
                [1], [1], synthesis="qft", approximation_degree=1.5
            ),
        ),
        (
            "approximation_degree_bool_qft",
            lambda: WeightedSum(
                [1], [1], synthesis="qft", approximation_degree=True
            ),
        ),
    ],
)
def test_weighted_sum_invalid_types_raise_typeerror(
    case: str, callable_wrapper
) -> None:
    """Invalid construction *types* always raise TypeError (never ValueError)."""

    with pytest.raises(TypeError):
        callable_wrapper()
