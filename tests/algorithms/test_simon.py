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

"""Tests for Simon's hidden-period algorithm."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from cqlib.device import ExecutionResult

from cqlib_algorithms.algorithms import Simon, SimonRequest, SimonResult
from cqlib_algorithms.operators.oracle import (
    BooleanFunction,
    BooleanVectorFunction,
)
from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.protocols import Sampler


def _vector_simon_function(num_qubits: int, period: int) -> BooleanVectorFunction:
    """Return a promised F with one distinct output per XOR pair."""
    assert num_qubits >= 2
    representatives = [min(value, value ^ period) for value in range(1 << num_qubits)]
    return BooleanVectorFunction.from_truth_table(
        [
            tuple(bool((representative >> output) & 1) for output in range(num_qubits))
            for representative in representatives
        ]
    )


def _scalar_simon_function() -> BooleanFunction:
    """Return the n=2, m=1 promised function with period 0b11."""
    return BooleanFunction.from_truth_table([False, True, True, False])


def _walsh_hadamard(num_qubits: int) -> np.ndarray:
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    matrix = np.array([[1.0]])
    for _ in range(num_qubits):
        matrix = np.kron(hadamard, matrix)
    return matrix


def _independent_simon_matrix(
    function: BooleanFunction | BooleanVectorFunction,
) -> np.ndarray:
    """Construct ``(H on input) U_F (H on input)`` as a permutation product."""
    num_inputs = function.num_variables
    table = (
        tuple((value,) for value in function.truth_table)
        if isinstance(function, BooleanFunction)
        else function.truth_table
    )
    num_outputs = len(table[0])
    dimension = 1 << (num_inputs + num_outputs)
    oracle = np.zeros((dimension, dimension), dtype=np.complex128)
    input_mask = (1 << num_inputs) - 1
    for column in range(dimension):
        input_value = column & input_mask
        output_value = sum(
            int(bit) << index for index, bit in enumerate(table[input_value])
        )
        row = column ^ (output_value << num_inputs)
        oracle[row, column] = 1.0
    input_hadamards = np.kron(np.eye(1 << num_outputs), _walsh_hadamard(num_inputs))
    return input_hadamards @ oracle @ input_hadamards


def _request(num_qubits: int, period: int, shots: int = 32) -> SimonRequest:
    return SimonRequest(_vector_simon_function(num_qubits, period), shots)


class _ScriptedSampler(Sampler):
    """Sampler returning caller-controlled counts for solver tests."""

    def __init__(self, scripts):
        self._scripts = scripts
        self.calls = 0

    def run(self, requests):
        self.calls += 1
        return [
            ExecutionResult.from_counts(
                "scripted",
                list(request.qubits),
                request.shots,
                len(request.qubits),
                counts,
                "fake-backend",
            )
            for request, counts in zip(requests, self._scripts)
        ]


@pytest.mark.parametrize(
    ("num_qubits", "period"),
    [(2, 1), (2, 3), (3, 1), (3, 5), (3, 7), (4, 10)],
)
def test_circuit_matches_independent_xor_oracle_matrix(num_qubits, period):
    function = _vector_simon_function(num_qubits, period)
    circuit = Simon.build_circuit(function)

    np.testing.assert_allclose(
        circuit.to_matrix(),
        _independent_simon_matrix(function),
        rtol=0.0,
        atol=1e-12,
    )


def test_scalar_function_circuit_matches_independent_reference():
    function = _scalar_simon_function()
    np.testing.assert_allclose(
        Simon.build_circuit(function).to_matrix(),
        _independent_simon_matrix(function),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(("num_qubits", "period"), [(2, 3), (3, 5), (4, 10)])
def test_ideal_samples_are_orthogonal_to_the_hidden_period(num_qubits, period):
    result = Simon(StatevectorSampler()).run([_request(num_qubits, period, shots=128)])[
        0
    ]

    for equation in result.independent_equations:
        equation_integer = sum(int(bit) << q for q, bit in enumerate(equation))
        assert (equation_integer & period).bit_count() % 2 == 0


def test_complete_equations_recover_period_with_little_endian_bits():
    sampler = _ScriptedSampler([{"010": 3, "101": 3, "111": 2}])
    result = Simon(sampler).run([_request(3, 0b101, shots=8)])[0]

    assert result.rank == 2
    assert result.is_complete
    assert result.period == (True, False, True)
    assert result.period_integer == 0b101
    assert result.nullspace_basis == ((True, False, True),)


def test_repeated_and_dependent_equations_do_not_inflate_rank():
    sampler = _ScriptedSampler([{"010": 4, "101": 2, "111": 2}])
    result = Simon(sampler).run([_request(3, 0b101, shots=8)])[0]

    assert result.rank == 2
    assert len(result.independent_equations) == 2


def test_insufficient_rank_returns_full_nullspace_without_period():
    sampler = _ScriptedSampler([{"000": 4, "010": 4}])
    result = Simon(sampler).run([_request(3, 0b101, shots=8)])[0]

    assert result.rank == 1
    assert not result.is_complete
    assert result.period is None
    assert result.period_integer is None
    assert result.nullspace_basis == (
        (True, False, False),
        (False, False, True),
    )


def test_full_rank_formally_resolves_to_zero_even_for_noisy_equations():
    sampler = _ScriptedSampler([{"001": 2, "010": 2, "100": 2}])
    result = Simon(sampler).run([_request(3, 0b101, shots=6)])[0]

    assert result.rank == 3
    assert result.is_complete
    assert result.nullspace_basis == ()
    assert result.period == (False, False, False)
    assert result.period_integer == 0


def test_identity_vector_function_recovers_zero_period_from_full_rank():
    function = BooleanVectorFunction.from_truth_table(
        [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ]
    )
    np.testing.assert_allclose(
        Simon.build_circuit(function).to_matrix(),
        _independent_simon_matrix(function),
        rtol=0.0,
        atol=1e-12,
    )
    sampler = _ScriptedSampler([{"01": 2, "10": 2}])
    result = Simon(sampler).run([SimonRequest(function, 4)])[0]

    assert result.rank == 2
    assert result.is_complete
    assert result.period == (False, False)
    assert result.period_integer == 0


def test_one_bit_injective_function_recovers_zero_period():
    function = BooleanFunction.from_truth_table([False, True])
    np.testing.assert_allclose(
        Simon.build_circuit(function).to_matrix(),
        _independent_simon_matrix(function),
        rtol=0.0,
        atol=1e-12,
    )
    sampler = _ScriptedSampler([{"0": 1, "1": 3}])
    result = Simon(sampler).run([SimonRequest(function, 4)])[0]

    assert result.rank == 1
    assert result.is_complete
    assert result.period == (False,)
    assert result.period_integer == 0


def test_one_bit_constant_function_recovers_nonzero_period():
    function = BooleanFunction.from_truth_table([False, False])
    sampler = _ScriptedSampler([{"0": 4}])
    result = Simon(sampler).run([SimonRequest(function, 4)])[0]

    assert result.rank == 0
    assert result.is_complete
    assert result.period == (True,)
    assert result.period_integer == 1


def test_batch_preserves_order_and_uses_one_sampler_call():
    sampler = _ScriptedSampler([{"10": 4}, {"01": 4}])
    algorithm = Simon(sampler)
    results = algorithm.run([_request(2, 1, shots=4), _request(2, 2, shots=4)])

    assert sampler.calls == 1
    assert [result.period_integer for result in results] == [1, 2]


def test_empty_batch_does_not_call_sampler():
    sampler = _ScriptedSampler([])
    assert Simon(sampler).run([]) == []
    assert sampler.calls == 0


def test_complete_batch_is_validated_before_sampler_call():
    sampler = _ScriptedSampler([{"00": 4}])
    with pytest.raises(TypeError, match="requests must contain SimonRequest"):
        Simon(sampler).run(
            [_request(2, 1, shots=4), "invalid"]  # type: ignore[list-item]
        )
    assert sampler.calls == 0


def test_sampler_measures_only_the_input_register():
    class _InspectingSampler(Sampler):
        def run(self, requests):
            request = requests[0]
            assert len(request.circuit.qubits) == 6
            assert len(request.qubits) == 3
            assert request.qubits == tuple(request.circuit.qubits[:3])
            return [
                ExecutionResult.from_counts(
                    "inspect",
                    list(request.qubits),
                    request.shots,
                    3,
                    {"000": request.shots},
                )
            ]

    Simon(_InspectingSampler()).run([_request(3, 5, shots=4)])


def test_filtered_counts_are_rejected_by_public_sampler_contract():
    sampler = _ScriptedSampler([{"00": 3}])
    with pytest.raises(ValueError, match="counts must sum"):
        Simon(sampler).run([_request(2, 1, shots=4)])


def test_wrong_result_count_is_rejected():
    sampler = _ScriptedSampler([{"00": 4}])
    with pytest.raises(ValueError, match="one result per request"):
        Simon(sampler).run([_request(2, 1, 4), _request(2, 2, 4)])


def test_wrong_result_qubit_order_is_rejected():
    class _ReversedSampler(Sampler):
        def run(self, requests):
            request = requests[0]
            return [
                ExecutionResult.from_counts(
                    "reversed",
                    list(reversed(request.qubits)),
                    request.shots,
                    len(request.qubits),
                    {"00": request.shots},
                )
            ]

    with pytest.raises(ValueError, match="qubits must preserve request order"):
        Simon(_ReversedSampler()).run([_request(2, 1, 4)])


@pytest.mark.parametrize("shots", [True, 1.5, "8"])
def test_request_rejects_non_integer_shots(shots):
    with pytest.raises(TypeError, match="shots must be an integer"):
        SimonRequest(_scalar_simon_function(), shots)  # type: ignore[arg-type]


@pytest.mark.parametrize("shots", [0, -1])
def test_request_rejects_non_positive_shots(shots):
    with pytest.raises(ValueError, match="shots must be greater than zero"):
        SimonRequest(_scalar_simon_function(), shots)


def test_request_revalidates_mutated_function_field():
    request = SimonRequest(_scalar_simon_function(), 4)
    object.__setattr__(request, "function", "invalid")
    with pytest.raises(TypeError, match="BooleanFunction or BooleanVectorFunction"):
        request.validate()


def test_algorithm_rejects_non_sampler():
    with pytest.raises(TypeError, match="sampler must be a Sampler instance"):
        Simon(StatevectorSampler().run)  # type: ignore[arg-type]


def test_request_and_result_are_frozen():
    request = _request(2, 1, 4)
    result = SimonResult(
        num_qubits=2,
        independent_equations=((False, True),),
        nullspace_basis=((True, False),),
        shots=4,
    )
    with pytest.raises(FrozenInstanceError):
        request.shots = 8  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.shots = 8  # type: ignore[misc]


def test_result_rejects_noncanonical_equations():
    with pytest.raises(ValueError, match="canonical RREF"):
        SimonResult(
            num_qubits=3,
            independent_equations=((False, True, False), (True, False, True)),
            nullspace_basis=((True, False, True),),
            shots=4,
        )


def test_result_rejects_incorrect_nullspace():
    with pytest.raises(ValueError, match="canonical basis"):
        SimonResult(
            num_qubits=2,
            independent_equations=((False, True),),
            nullspace_basis=((False, True),),
            shots=4,
        )


def test_build_circuit_returns_independent_circuits():
    function = _vector_simon_function(3, 5)
    first = Simon.build_circuit(function)
    second = Simon.build_circuit(function)

    first.x(0)
    assert len(first.operations) == len(second.operations) + 1


def test_single_oracle_component_recovers_xor_period():
    from cqlib_algorithms.operators import BitFlipOracle

    function = _scalar_simon_function()
    result = Simon(StatevectorSampler()).run(BitFlipOracle(function), shots=128)
    assert result.period == (True, True)
    assert result.num_qubits == 2


def test_single_query_uses_injected_bit_flip_oracle():
    from cqlib import Circuit

    from cqlib_algorithms.operators import BitFlipOracle

    class CompactParity(BitFlipOracle):
        calls = 0

        def build_circuit(self):
            type(self).calls += 1
            circuit = Circuit(3)
            circuit.cx(0, 2)
            circuit.cx(1, 2)
            return circuit

    oracle = CompactParity(_scalar_simon_function())
    algorithm = Simon(StatevectorSampler())
    result = algorithm.run(oracle, shots=128)
    assert result.period == (True, True)
    assert CompactParity.calls == 1
    with pytest.raises(TypeError, match="shots"):
        algorithm.run(oracle)
    with pytest.raises(TypeError, match="batch"):
        algorithm.run([], shots=32)


@pytest.mark.parametrize("kind", ["type", "width", "symbol", "measure"])
def test_invalid_single_oracle_fails_before_sampling(kind):
    from cqlib import Circuit, Parameter

    from cqlib_algorithms.operators import BitFlipOracle

    class InvalidOracle(BitFlipOracle):
        def build_circuit(self):
            if kind == "type":
                return None
            circuit = Circuit(2 if kind == "width" else 3)
            if kind == "symbol":
                circuit.rx(0, Parameter("x"))
            elif kind == "measure":
                circuit.measure(0)
            return circuit

    sampler = _ScriptedSampler([])
    with pytest.raises((TypeError, ValueError)):
        Simon(sampler).run(InvalidOracle(_scalar_simon_function()), shots=32)
    assert sampler.calls == 0
