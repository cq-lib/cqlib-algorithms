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

"""Tests for the Deutsch-Jozsa algorithm component."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from cqlib.device import ExecutionResult

from cqlib_algorithms.algorithms import (
    DeutschJozsa,
    DeutschJozsaRequest,
    DeutschJozsaResult,
)
from cqlib_algorithms.operators.oracle import (
    BooleanFunction,
    BooleanVectorFunction,
)
from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.protocols import (
    Sampler,
    SamplerRequest,
    validate_sampler_result,
)


def _truth_table(num_qubits, marked_states):
    values = [False] * (1 << num_qubits)
    for state in marked_states:
        values[state] = True
    return values


def _function(num_qubits, marked_states):
    return BooleanFunction.from_truth_table(_truth_table(num_qubits, marked_states))


def _hadamard_matrix(num_qubits):
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2.0)
    matrix = hadamard
    for _ in range(num_qubits - 1):
        matrix = np.kron(hadamard, matrix)
    return matrix


def _dj_matrix(num_qubits, marked_states):
    """Independent ``H S_f H`` reference from the truth-table signs."""
    signs = np.ones(1 << num_qubits, dtype=np.complex128)
    for state in marked_states:
        signs[state] = -1.0
    hadamard = _hadamard_matrix(num_qubits)
    return hadamard @ np.diag(signs) @ hadamard


def _parity_function(num_qubits):
    return _function(
        num_qubits,
        tuple(state for state in range(1 << num_qubits) if state.bit_count() % 2),
    )


def _request(num_qubits, marked_states, shots=16, **kwargs):
    return DeutschJozsaRequest(_function(num_qubits, marked_states), shots, **kwargs)


@pytest.mark.parametrize(
    ("num_qubits", "marked_states"),
    [
        (1, (0,)),
        (2, (0, 3)),
        (3, (1, 4)),
        (3, (0, 2, 5, 7)),
        (3, (0, 1, 2)),
    ],
)
def test_circuit_matches_independent_h_oracle_h_matrix(num_qubits, marked_states):
    circuit = DeutschJozsa.build_circuit(_function(num_qubits, marked_states))

    np.testing.assert_allclose(
        circuit.to_matrix(),
        _dj_matrix(num_qubits, marked_states),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("num_qubits", [1, 2, 3, 4])
def test_constant_functions_leave_all_amplitude_on_all_zero(num_qubits):
    for marked_states in ((), tuple(range(1 << num_qubits))):
        circuit = DeutschJozsa.build_circuit(_function(num_qubits, marked_states))
        amplitude = circuit.to_matrix()[0, 0]
        assert abs(amplitude) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
def test_balanced_functions_leave_no_amplitude_on_all_zero(num_qubits):
    for function in (
        _parity_function(num_qubits),
        _function(num_qubits, tuple(range(0, 1 << num_qubits, 2))),
    ):
        circuit = DeutschJozsa.build_circuit(function)
        assert abs(circuit.to_matrix()[0, 0]) < 1e-12


def test_constant_zero_function_verdict_is_constant():
    result = DeutschJozsa(StatevectorSampler()).run([_request(3, (), shots=16)])[0]

    assert result.is_constant
    assert not result.is_balanced
    assert result.zero_count == 16
    assert result.shots == 16


def test_constant_one_function_verdict_is_constant_despite_global_phase():
    result = DeutschJozsa(StatevectorSampler()).run(
        [_request(2, (0, 1, 2, 3), shots=8)]
    )[0]

    assert result.is_constant
    assert result.zero_count == 8


@pytest.mark.parametrize(
    ("marked_states", "is_constant"),
    [
        ((), True),
        ((0, 2), False),
    ],
)
def test_one_shot_is_deterministic_for_promised_ideal_functions(
    marked_states, is_constant
):
    result = DeutschJozsa(StatevectorSampler()).run(
        [_request(2, marked_states, shots=1)]
    )[0]

    assert result.is_constant is is_constant
    assert result.zero_count == int(is_constant)


@pytest.mark.parametrize(
    ("num_qubits", "marked_states"),
    [
        (1, (1,)),
        (2, (0, 2)),
        (3, (0, 1, 4, 5)),
    ],
)
def test_balanced_functions_verdict_is_balanced(num_qubits, marked_states):
    # half of the 2**n inputs are marked in every parametrized case
    result = DeutschJozsa(StatevectorSampler()).run(
        [_request(num_qubits, marked_states, shots=16)]
    )[0]

    assert result.is_balanced
    assert not result.is_constant
    assert result.zero_count == 0


def test_parity_function_is_detected_as_balanced():
    result = DeutschJozsa(StatevectorSampler()).run(
        [DeutschJozsaRequest(_parity_function(4), 32)]
    )[0]

    assert result.is_balanced
    assert result.zero_count == 0


def test_vector_function_uses_output_mask_parity():
    function = BooleanVectorFunction.from_truth_table(
        [
            (False, False),
            (True, True),
            (False, False),
            (True, True),
        ]
    )
    # One selected output is balanced, while their parity is constant zero.
    single_output, both_outputs = DeutschJozsa(StatevectorSampler()).run(
        [
            DeutschJozsaRequest(function, 16, output_mask=0b01),
            DeutschJozsaRequest(function, 16, output_mask=0b11),
        ]
    )

    assert single_output.is_balanced
    assert single_output.zero_count == 0
    assert both_outputs.is_constant
    assert both_outputs.zero_count == 16


def test_batch_results_preserve_request_order():
    algorithm = DeutschJozsa(StatevectorSampler())
    results = algorithm.run(
        [
            _request(2, (), shots=8),
            _request(2, (0, 2), shots=8),
            _request(2, (0, 1, 2, 3), shots=8),
        ]
    )

    assert [result.is_constant for result in results] == [True, False, True]


def test_empty_batch_returns_empty_list():
    assert DeutschJozsa(StatevectorSampler()).run([]) == []


def test_promise_violating_function_still_returns_a_well_formed_result():
    result = DeutschJozsa(StatevectorSampler()).run([_request(3, (0, 1, 2), shots=16)])[
        0
    ]

    assert isinstance(result, DeutschJozsaResult)
    assert 0 <= result.zero_count <= 16


def test_build_circuit_returns_fresh_independent_circuits():
    function = _function(2, (1,))
    first = DeutschJozsa.build_circuit(function)
    second = DeutschJozsa.build_circuit(function)

    assert first is not second
    assert first == second
    second_size = len(second.operations)
    first.x(0)
    assert len(second.operations) == second_size


def test_algorithm_is_a_slotted_component():
    algorithm = DeutschJozsa(StatevectorSampler())

    assert not hasattr(algorithm, "__dict__")
    assert isinstance(algorithm.sampler, StatevectorSampler)
    with pytest.raises(AttributeError):
        algorithm.sampler = StatevectorSampler()  # type: ignore[misc]


def test_result_is_frozen_and_validated():
    result = DeutschJozsaResult(is_constant=True, zero_count=4, shots=4)

    assert result.is_balanced is False
    with pytest.raises(AttributeError):
        result.is_constant = False  # type: ignore[misc]
    with pytest.raises(TypeError, match="is_constant must be a bool"):
        DeutschJozsaResult(is_constant=1, zero_count=4, shots=4)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="zero_count must be an integer"):
        DeutschJozsaResult(is_constant=True, zero_count=True, shots=4)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="zero_count must be non-negative"):
        DeutschJozsaResult(is_constant=True, zero_count=-1, shots=4)
    with pytest.raises(ValueError, match="zero_count must not exceed shots"):
        DeutschJozsaResult(is_constant=True, zero_count=5, shots=4)
    with pytest.raises(ValueError, match="shots must be greater than zero"):
        DeutschJozsaResult(is_constant=True, zero_count=0, shots=0)
    with pytest.raises(ValueError, match="is_constant must match"):
        DeutschJozsaResult(is_constant=True, zero_count=3, shots=4)
    with pytest.raises(ValueError, match="is_constant must match"):
        DeutschJozsaResult(is_constant=False, zero_count=4, shots=4)


@pytest.mark.parametrize("shots", [True, 1.5, "8"])
def test_request_rejects_non_integer_shots(shots):
    with pytest.raises(TypeError, match="shots must be an integer"):
        DeutschJozsaRequest(_function(1, ()), shots)  # type: ignore[arg-type]


@pytest.mark.parametrize("shots", [0, -1])
def test_request_rejects_non_positive_shots(shots):
    with pytest.raises(ValueError, match="shots must be greater than zero"):
        DeutschJozsaRequest(_function(1, ()), shots)


@pytest.mark.parametrize("function", [None, "f", 2])
def test_request_rejects_non_boolean_functions(function):
    with pytest.raises(TypeError, match="BooleanFunction or BooleanVectorFunction"):
        DeutschJozsaRequest(function, 8)  # type: ignore[arg-type]


def test_request_rejects_missing_mask_for_vector_function():
    function = BooleanVectorFunction.from_truth_table([(False, False), (True, True)])
    with pytest.raises(
        ValueError, match="output_mask is required for a BooleanVectorFunction"
    ):
        DeutschJozsaRequest(function, 8)


def test_request_rejects_out_of_range_mask():
    function = BooleanVectorFunction.from_truth_table([(False, False), (True, True)])
    with pytest.raises(
        ValueError, match="output_mask must select one or more existing outputs"
    ):
        DeutschJozsaRequest(function, 8, output_mask=4)


def test_algorithm_rejects_non_sampler():
    with pytest.raises(TypeError, match="sampler must be a Sampler instance"):
        DeutschJozsa(StatevectorSampler().run)  # type: ignore[arg-type]


def test_run_rejects_non_request_entries():
    with pytest.raises(
        TypeError, match="requests must contain DeutschJozsaRequest instances"
    ):
        DeutschJozsa(StatevectorSampler()).run(["request"])  # type: ignore[list-item]


class _ScriptedSampler(Sampler):
    """Sampler returning caller-controlled counts for contract tests."""

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


def test_complete_valid_batch_uses_one_sampler_call():
    sampler = _ScriptedSampler([{"00": 4}, {"00": 4}])

    results = DeutschJozsa(sampler).run(
        [_request(2, (), shots=4), _request(2, (), shots=4)]
    )

    assert sampler.calls == 1
    assert len(results) == 2


def test_complete_batch_is_validated_before_sampler_call():
    sampler = _ScriptedSampler([{"00": 4}])

    with pytest.raises(
        TypeError, match="requests must contain DeutschJozsaRequest instances"
    ):
        DeutschJozsa(sampler).run(
            [_request(2, (), shots=4), "invalid"]  # type: ignore[list-item]
        )

    assert sampler.calls == 0


def test_run_rejects_filtered_sampler_counts():
    sampler = _ScriptedSampler([{"00": 3}])  # sums to 3, not the 4 shots
    with pytest.raises(
        ValueError, match="sampler counts must sum to the requested shots"
    ):
        DeutschJozsa(sampler).run([_request(2, (), shots=4)])


def test_run_rejects_non_execution_results():
    class _DictSampler(Sampler):
        def run(self, requests):
            return [{"00": 4} for _ in requests]

    with pytest.raises(TypeError, match="cqlib.device.ExecutionResult"):
        DeutschJozsa(_DictSampler()).run([_request(2, (), shots=4)])


def test_run_rejects_non_list_sampler_result_container():
    class _TupleSampler(Sampler):
        def run(self, requests):
            return tuple(
                ExecutionResult.from_counts(
                    "tuple",
                    list(request.qubits),
                    request.shots,
                    len(request.qubits),
                    {"0" * len(request.qubits): request.shots},
                    "fake-backend",
                )
                for request in requests
            )

    with pytest.raises(TypeError, match="sampler must return a list"):
        DeutschJozsa(_TupleSampler()).run([_request(2, (), shots=4)])


def test_run_rejects_mismatched_result_shots():
    class _ShiftedShotsSampler(Sampler):
        def run(self, requests):
            return [
                ExecutionResult.from_counts(
                    "shifted",
                    list(request.qubits),
                    request.shots + 1,
                    len(request.qubits),
                    {"0" * len(request.qubits): request.shots + 1},
                    "fake-backend",
                )
                for request in requests
            ]

    with pytest.raises(ValueError, match="sampler result shots must match"):
        DeutschJozsa(_ShiftedShotsSampler()).run([_request(2, (), shots=4)])


def test_run_rejects_wrong_result_count():
    sampler = _ScriptedSampler([{"00": 4}])  # one result for two requests
    with pytest.raises(ValueError, match="one result per request"):
        DeutschJozsa(sampler).run([_request(2, (), shots=4), _request(2, (), shots=4)])


class _InvalidResultSampler(Sampler):
    """Return one result with a selected Sampler-contract violation."""

    def __init__(self, mode):
        self._mode = mode

    def run(self, requests):
        request = requests[0]
        if self._mode == "queued":
            return [
                ExecutionResult(
                    "queued",
                    list(request.qubits),
                    request.shots,
                    len(request.qubits),
                )
            ]
        if self._mode == "failed":
            result = ExecutionResult(
                "failed",
                list(request.qubits),
                request.shots,
                len(request.qubits),
            )
            result.fail("deliberate failure", 1)
            return [result]
        if self._mode == "width":
            return [
                ExecutionResult.from_counts(
                    "width", [request.qubits[0]], request.shots, 1, {"0": 4}
                )
            ]
        if self._mode == "qubits":
            return [
                ExecutionResult.from_counts(
                    "qubits",
                    list(reversed(request.qubits)),
                    request.shots,
                    len(request.qubits),
                    {"00": 4},
                )
            ]
        raise AssertionError("unsupported invalid-result mode")


@pytest.mark.parametrize("mode", ["queued", "failed"])
def test_run_rejects_unsuccessful_or_nonterminal_results(mode):
    with pytest.raises(ValueError, match="successful terminal"):
        DeutschJozsa(_InvalidResultSampler(mode)).run([_request(2, (), shots=4)])


def test_run_rejects_result_width_mismatch():
    with pytest.raises(ValueError, match="width"):
        DeutschJozsa(_InvalidResultSampler("width")).run([_request(2, (), shots=4)])


def test_run_rejects_result_qubit_order_mismatch():
    with pytest.raises(ValueError, match="qubits"):
        DeutschJozsa(_InvalidResultSampler("qubits")).run([_request(2, (), shots=4)])


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        ({"0": 4}, "width"),
        ({"0x": 4}, "binary"),
        ({"00": True, "11": 3}, "non-negative integers"),
        ({"00": 1.5, "11": 2.5}, "non-negative integers"),
        ({"00": -1, "11": 5}, "non-negative integers"),
    ],
)
def test_run_rejects_malformed_count_entries(counts, message):
    circuit = DeutschJozsa.build_circuit(_function(2, ()))
    request = SamplerRequest(circuit, tuple(circuit.qubits), 4)
    valid = ExecutionResult.from_counts(
        "valid",
        list(request.qubits),
        request.shots,
        len(request.qubits),
        {"00": request.shots},
        "fake-backend",
    )
    malformed = SimpleNamespace(
        status=valid.status,
        shots=valid.shots,
        num_qubits=valid.num_qubits,
        qubits=valid.qubits,
        counts=counts,
    )

    with pytest.raises(ValueError, match=message):
        validate_sampler_result(malformed, request)  # type: ignore[arg-type]


def test_verdict_comes_from_counts_not_truth_table():
    # A balanced function reported with all-zero counts yields "constant":
    # the algorithm trusts the sampler, never the truth table.
    sampler = _ScriptedSampler([{"00": 8}])
    result = DeutschJozsa(sampler).run([_request(2, (0, 2), shots=8)])[0]

    assert result.is_constant
    assert result.zero_count == 8
