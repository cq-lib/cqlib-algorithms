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

"""Tests for the Bernstein-Vazirani algorithm component."""

from __future__ import annotations

import numpy as np
import pytest
from cqlib.device import ExecutionResult

from cqlib_algorithms.algorithms import (
    BernsteinVazirani,
    BernsteinVaziraniRequest,
    BernsteinVaziraniResult,
)
from cqlib_algorithms.operators.oracle import (
    BooleanFunction,
    BooleanVectorFunction,
)
from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.protocols import Sampler


def _linear_function(num_qubits, hidden_integer):
    """Build f(x) = s . x (mod 2) for the hidden integer ``s``."""
    return BooleanFunction.from_truth_table(
        [
            (state & hidden_integer).bit_count() % 2 == 1
            for state in range(1 << num_qubits)
        ]
    )


def _query_matrix(num_qubits, hidden_integer):
    """Independent ``H S_f H`` reference from the linear-function signs."""
    signs = np.array(
        [
            (-1.0) ** ((state & hidden_integer).bit_count() % 2)
            for state in range(1 << num_qubits)
        ],
        dtype=np.complex128,
    )
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    matrix = hadamard
    for _ in range(num_qubits - 1):
        matrix = np.kron(hadamard, matrix)
    return matrix @ np.diag(signs) @ matrix


def _request(num_qubits, hidden_integer, shots=16, **kwargs):
    return BernsteinVaziraniRequest(
        _linear_function(num_qubits, hidden_integer), shots, **kwargs
    )


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


@pytest.mark.parametrize(
    ("num_qubits", "hidden_integer"),
    [
        (1, 0),
        (1, 1),
        (2, 1),
        (2, 2),
        (3, 0),
        (3, 5),
        (3, 7),
        (4, 10),
    ],
)
def test_circuit_matches_independent_h_oracle_h_matrix(num_qubits, hidden_integer):
    circuit = BernsteinVazirani.build_circuit(
        _linear_function(num_qubits, hidden_integer)
    )

    np.testing.assert_allclose(
        circuit.to_matrix(),
        _query_matrix(num_qubits, hidden_integer),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("num_qubits", "hidden_integer"),
    [
        (1, 0),
        (1, 1),
        (3, 0),
        (3, 1),
        (3, 5),
        (3, 7),
        (4, 10),
    ],
)
def test_circuit_final_state_is_exactly_the_hidden_basis_state(
    num_qubits, hidden_integer
):
    circuit = BernsteinVazirani.build_circuit(
        _linear_function(num_qubits, hidden_integer)
    )
    expected = np.zeros(1 << num_qubits, dtype=np.complex128)
    expected[hidden_integer] = 1.0

    np.testing.assert_allclose(
        circuit.to_matrix()[:, 0], expected, rtol=0.0, atol=1e-12
    )


@pytest.mark.parametrize(
    ("num_qubits", "hidden_integer"),
    [
        (1, 0),
        (2, 1),
        (3, 5),
        (4, 15),
    ],
)
def test_recovery_end_to_end(num_qubits, hidden_integer):
    result = BernsteinVazirani(StatevectorSampler()).run(
        [_request(num_qubits, hidden_integer, shots=16)]
    )[0]

    expected_bits = tuple(bool((hidden_integer >> q) & 1) for q in range(num_qubits))
    assert result.hidden_string == expected_bits
    assert result.hidden_integer == hidden_integer
    assert result.modal_strings == (expected_bits,)
    assert not result.is_ambiguous
    assert result.support == 16
    assert result.shots == 16


def test_zero_hidden_string_is_recovered_as_all_false():
    result = BernsteinVazirani(StatevectorSampler()).run([_request(3, 0, shots=8)])[0]

    assert result.hidden_string == (False, False, False)
    assert result.hidden_integer == 0
    assert result.support == 8


def test_vector_function_uses_output_mask_parity():
    function = BooleanVectorFunction.from_truth_table(
        [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ]
    )
    # mask parity g(x) = x0 XOR x1 -> hidden integer 0b11
    result = BernsteinVazirani(StatevectorSampler()).run(
        [BernsteinVaziraniRequest(function, 16, output_mask=0b11)]
    )[0]

    assert result.hidden_string == (True, True)
    assert result.hidden_integer == 3


def test_batch_results_preserve_request_order():
    algorithm = BernsteinVazirani(StatevectorSampler())
    results = algorithm.run(
        [
            _request(2, 0, shots=8),
            _request(2, 1, shots=8),
            _request(2, 3, shots=8),
        ]
    )

    assert [result.hidden_integer for result in results] == [0, 1, 3]


def test_empty_batch_returns_empty_list():
    assert BernsteinVazirani(StatevectorSampler()).run([]) == []


def test_non_linear_function_still_returns_a_well_formed_result():
    # f(x) = x0 AND x1 violates the linearity promise; no promise check occurs
    function = BooleanFunction.from_truth_table([False, False, False, True])
    result = BernsteinVazirani(StatevectorSampler()).run(
        [BernsteinVaziraniRequest(function, 16)]
    )[0]

    assert isinstance(result, BernsteinVaziraniResult)
    assert result.modal_strings
    assert all(len(candidate) == 2 for candidate in result.modal_strings)
    assert 0 < result.support <= 16


def test_unique_empirical_mode_is_exposed_as_hidden_string():
    sampler = _ScriptedSampler([{"01": 5, "00": 3}])
    result = BernsteinVazirani(sampler).run([_request(2, 1, shots=8)])[0]

    # little-endian: rightmost character of "01" is qubit 0
    assert result.hidden_string == (True, False)
    assert result.hidden_integer == 1
    assert result.support == 5


def test_unique_mode_need_not_have_an_absolute_majority():
    sampler = _ScriptedSampler([{"00": 3, "01": 2, "10": 2, "11": 1}])
    result = BernsteinVazirani(sampler).run([_request(2, 0, shots=8)])[0]

    assert result.hidden_string == (False, False)
    assert result.hidden_integer == 0
    assert result.support == 3
    assert not result.is_ambiguous


def test_tied_outcomes_are_reported_as_ambiguous_modal_candidates():
    sampler = _ScriptedSampler([{"01": 4, "00": 4}])
    result = BernsteinVazirani(sampler).run([_request(2, 1, shots=8)])[0]

    assert result.modal_strings == ((False, False), (True, False))
    assert result.hidden_string is None
    assert result.hidden_integer is None
    assert result.is_ambiguous
    assert result.support == 4


def test_three_way_tie_preserves_all_candidates_in_integer_order():
    sampler = _ScriptedSampler([{"11": 2, "10": 2, "00": 2}])
    result = BernsteinVazirani(sampler).run([_request(2, 0, shots=6)])[0]

    assert result.modal_strings == (
        (False, False),
        (False, True),
        (True, True),
    )
    assert result.is_ambiguous


def test_build_circuit_returns_fresh_independent_circuits():
    function = _linear_function(2, 1)
    first = BernsteinVazirani.build_circuit(function)
    second = BernsteinVazirani.build_circuit(function)

    assert first is not second
    assert first == second
    second_size = len(second.operations)
    first.x(0)
    assert len(second.operations) == second_size


def test_algorithm_is_a_slotted_component():
    algorithm = BernsteinVazirani(StatevectorSampler())

    assert not hasattr(algorithm, "__dict__")
    assert isinstance(algorithm.sampler, StatevectorSampler)
    with pytest.raises(AttributeError):
        algorithm.sampler = StatevectorSampler()  # type: ignore[misc]


def test_result_is_frozen_and_validated():
    result = BernsteinVaziraniResult(modal_strings=((True, False),), support=4, shots=4)

    assert result.hidden_integer == 1
    with pytest.raises(AttributeError):
        result.support = 3  # type: ignore[misc]
    with pytest.raises(TypeError, match="modal_strings must be a tuple"):
        BernsteinVaziraniResult(
            modal_strings=[(True,)],
            support=1,
            shots=1,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="modal_strings entries must be tuples"):
        BernsteinVaziraniResult(
            modal_strings=([True],),
            support=1,
            shots=1,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="modal string entries must be bool"):
        BernsteinVaziraniResult(
            modal_strings=((1,),),
            support=1,
            shots=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="modal_strings must not be empty"):
        BernsteinVaziraniResult(modal_strings=(), support=1, shots=1)
    with pytest.raises(ValueError, match="modal strings must have equal width"):
        BernsteinVaziraniResult(
            modal_strings=((False,), (False, True)), support=1, shots=1
        )
    with pytest.raises(ValueError, match="unique and sorted"):
        BernsteinVaziraniResult(modal_strings=((True,), (False,)), support=1, shots=1)
    with pytest.raises(TypeError, match="support must be an integer"):
        BernsteinVaziraniResult(
            modal_strings=((True,),),
            support=True,
            shots=1,  # type: ignore[arg-type]
        )
    with pytest.raises(
        ValueError, match="support must be positive and not exceed shots"
    ):
        BernsteinVaziraniResult(modal_strings=((True,),), support=0, shots=1)
    with pytest.raises(
        ValueError, match="support must be positive and not exceed shots"
    ):
        BernsteinVaziraniResult(modal_strings=((True,),), support=2, shots=1)
    with pytest.raises(ValueError, match="shots must be greater than zero"):
        BernsteinVaziraniResult(modal_strings=((True,),), support=1, shots=0)
    with pytest.raises(ValueError, match="modal statistics are inconsistent"):
        BernsteinVaziraniResult(modal_strings=((False,), (True,)), support=4, shots=4)
    with pytest.raises(ValueError, match="modal statistics are inconsistent"):
        BernsteinVaziraniResult(modal_strings=((False,),), support=1, shots=2)


@pytest.mark.parametrize("shots", [True, 1.5, "8"])
def test_request_rejects_non_integer_shots(shots):
    with pytest.raises(TypeError, match="shots must be an integer"):
        BernsteinVaziraniRequest(_linear_function(1, 0), shots)  # type: ignore[arg-type]


@pytest.mark.parametrize("shots", [0, -1])
def test_request_rejects_non_positive_shots(shots):
    with pytest.raises(ValueError, match="shots must be greater than zero"):
        BernsteinVaziraniRequest(_linear_function(1, 0), shots)


@pytest.mark.parametrize("function", [None, "f", 2])
def test_request_rejects_non_boolean_functions(function):
    with pytest.raises(TypeError, match="BooleanFunction or BooleanVectorFunction"):
        BernsteinVaziraniRequest(function, 8)  # type: ignore[arg-type]


def test_request_rejects_missing_mask_for_vector_function():
    function = BooleanVectorFunction.from_truth_table([(False, False), (True, True)])
    with pytest.raises(
        ValueError, match="output_mask is required for a BooleanVectorFunction"
    ):
        BernsteinVaziraniRequest(function, 8)


def test_request_rejects_out_of_range_mask():
    function = BooleanVectorFunction.from_truth_table([(False, False), (True, True)])
    with pytest.raises(
        ValueError, match="output_mask must select one or more existing outputs"
    ):
        BernsteinVaziraniRequest(function, 8, output_mask=4)


def test_algorithm_rejects_non_sampler():
    with pytest.raises(TypeError, match="sampler must be a Sampler instance"):
        BernsteinVazirani(StatevectorSampler().run)  # type: ignore[arg-type]


def test_run_rejects_non_request_entries():
    with pytest.raises(
        TypeError, match="requests must contain BernsteinVaziraniRequest instances"
    ):
        BernsteinVazirani(StatevectorSampler()).run(["request"])  # type: ignore[list-item]


def test_complete_valid_batch_uses_one_sampler_call():
    sampler = _ScriptedSampler([{"00": 4}, {"01": 4}])

    results = BernsteinVazirani(sampler).run(
        [_request(2, 0, shots=4), _request(2, 1, shots=4)]
    )

    assert sampler.calls == 1
    assert [result.hidden_integer for result in results] == [0, 1]


def test_complete_batch_is_validated_before_sampler_call():
    sampler = _ScriptedSampler([{"00": 4}])

    with pytest.raises(
        TypeError, match="requests must contain BernsteinVaziraniRequest instances"
    ):
        BernsteinVazirani(sampler).run(
            [_request(2, 0, shots=4), "invalid"]  # type: ignore[list-item]
        )

    assert sampler.calls == 0


def test_run_rejects_filtered_sampler_counts():
    sampler = _ScriptedSampler([{"00": 3}])  # sums to 3, not the 4 shots
    with pytest.raises(
        ValueError, match="sampler counts must sum to the requested shots"
    ):
        BernsteinVazirani(sampler).run([_request(2, 1, shots=4)])


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
        BernsteinVazirani(_ShiftedShotsSampler()).run([_request(2, 1, shots=4)])


def test_run_rejects_wrong_result_count():
    sampler = _ScriptedSampler([{"01": 4}])  # one result for two requests
    with pytest.raises(ValueError, match="one result per request"):
        BernsteinVazirani(sampler).run(
            [_request(2, 1, shots=4), _request(2, 2, shots=4)]
        )


def test_run_rejects_non_execution_results():
    class _DictSampler(Sampler):
        def run(self, requests):
            return [{"01": 4} for _ in requests]

    with pytest.raises(TypeError, match="cqlib.device.ExecutionResult"):
        BernsteinVazirani(_DictSampler()).run([_request(2, 1, shots=4)])


def test_single_query_component_recovers_two_bit_hidden_string():
    from cqlib_algorithms.operators import HadamardPhaseQuery, PhaseOracle

    function = BooleanFunction.from_truth_table([False, True, True, False])
    query = HadamardPhaseQuery(PhaseOracle(function))
    algorithm = BernsteinVazirani(StatevectorSampler())
    result = algorithm.run(query, shots=32)
    assert result == algorithm.run([BernsteinVaziraniRequest(function, 32)])[0]
    assert result.modal_strings == ((True, True),)


def test_single_query_uses_compact_oracle_and_preserves_mask():
    from cqlib import Circuit

    from cqlib_algorithms.operators import HadamardPhaseQuery, PhaseOracle

    class CompactParity(PhaseOracle):
        calls = 0

        def build_circuit(self):
            type(self).calls += 1
            circuit = Circuit(2)
            circuit.z(1)
            return circuit

    vector = BooleanVectorFunction.from_truth_table(
        [(False, False), (True, False), (False, True), (True, True)]
    )
    query = HadamardPhaseQuery(CompactParity(vector, output_mask=2))
    algorithm = BernsteinVazirani(StatevectorSampler())
    result = algorithm.run(query, shots=32)
    assert result.modal_strings == ((False, True),)
    assert CompactParity.calls == 1
    assert (
        result
        == algorithm.run([BernsteinVaziraniRequest(vector, 32, output_mask=2)])[0]
    )
    with pytest.raises(TypeError, match="shots"):
        algorithm.run(query)
    with pytest.raises(TypeError, match="batch"):
        algorithm.run([], shots=32)


@pytest.mark.parametrize("kind", ["type", "width", "measure"])
def test_invalid_query_component_is_rejected_before_sampling(kind):
    from cqlib import Circuit

    from cqlib_algorithms.operators import HadamardPhaseQuery

    class InvalidQuery(HadamardPhaseQuery):
        def build_circuit(self):
            if kind == "type":
                return None
            circuit = Circuit(1 if kind == "width" else 2)
            if kind == "measure":
                circuit.measure(0)
            return circuit

    class UnusedSampler(Sampler):
        def run(self, requests):
            pytest.fail("invalid query must be rejected before execution")

    function = BooleanFunction.from_truth_table([False, True, True, False])
    with pytest.raises((TypeError, ValueError)):
        BernsteinVazirani(UnusedSampler()).run(InvalidQuery(function), shots=32)
