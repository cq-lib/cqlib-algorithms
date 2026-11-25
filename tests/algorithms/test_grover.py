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

"""Tests for the Grover search algorithm."""

from __future__ import annotations

import math

import numpy as np
import pytest
from cqlib import Circuit
from cqlib.device import ExecutionResult

from cqlib_algorithms.algorithms import Grover, GroverRequest, GroverResult
from cqlib_algorithms.operators.composite import GroverOperator
from cqlib_algorithms.operators.oracle import (
    BooleanFunction,
    BooleanVectorFunction,
    PhaseOracle,
)
from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.operators.state_preparation import (
    UniformSuperposition,
    UserCircuitPreparation,
)
from cqlib_algorithms.protocols import Sampler


def _function(num_qubits: int, marked_states: tuple[int, ...]) -> BooleanFunction:
    return BooleanFunction.from_marked_states(num_qubits, marked_states)


def test_standard_and_batch_search_pass_components_to_the_iteration(monkeypatch):
    original_init = GroverOperator.__init__
    inputs = []

    def record_inputs(self, oracle, *, state_preparation=None):
        inputs.append((oracle, state_preparation))
        original_init(self, oracle, state_preparation=state_preparation)

    monkeypatch.setattr(GroverOperator, "__init__", record_inputs)
    function = _function(2, (2,))
    circuit = Grover.build_circuit(function)
    (result,) = Grover(StatevectorSampler()).run([GroverRequest(function, shots=32)])

    assert abs(circuit.to_matrix()[2, 0]) ** 2 == pytest.approx(1.0)
    assert result.counts == (0, 0, 32, 0)
    assert len(inputs) == 2
    for oracle, preparation in inputs:
        assert isinstance(oracle, PhaseOracle)
        assert oracle.function is function
        assert isinstance(preparation, UniformSuperposition)


def test_run_with_preconfigured_iteration_accepts_components():
    iteration = GroverOperator(
        oracle=PhaseOracle(_function(2, (2,))),
        state_preparation=UniformSuperposition(2),
    )
    grover = Grover(iteration=iteration, sampler=StatevectorSampler())

    result = grover.run(iterations=1, shots=256, seed=7)

    assert result.counts == (0, 0, 256, 0)
    assert result.marked_states == (2,)
    assert result.iterations == 1
    assert result.success_probability == 1.0


def test_run_prepares_the_same_nonuniform_state_used_in_reflection():
    preparation = Circuit(2)
    # Only |00> and |10> have support, with initial marked probability 1/4.
    preparation.ry(1, math.pi / 3)
    iteration = GroverOperator(
        oracle=PhaseOracle(_function(2, (2,))),
        state_preparation=UserCircuitPreparation(preparation),
    )
    grover = Grover(iteration=iteration, sampler=StatevectorSampler())

    result = grover.run(iterations=1, shots=64, seed=7)

    assert result.counts == (0, 0, 64, 0)
    assert grover.run(iterations=1, shots=64, seed=7) == result


def test_run_uses_the_preconfigured_oracles_vector_output_mask():
    function = BooleanVectorFunction.from_truth_table(
        [(False, False), (False, False), (True, False), (True, True)]
    )
    iteration = GroverOperator(PhaseOracle(function, output_mask=3))

    result = Grover(StatevectorSampler(), iteration=iteration).run(
        iterations=1, shots=32, seed=7
    )

    assert result.marked_states == (2,)
    assert result.counts == (0, 0, 32, 0)


def test_preconfigured_search_requires_a_phase_oracle_predicate_and_full_reflection():
    oracle = PhaseOracle(_function(2, (2,)))
    sampler = StatevectorSampler()

    with pytest.raises(ValueError, match="PhaseOracle that marks"):
        Grover(sampler, iteration=GroverOperator(PhaseOracle(_function(2, ()))))
    with pytest.raises(ValueError, match="reflect all search qubits"):
        Grover(sampler, iteration=GroverOperator(oracle, reflection_qubits=(0,)))


def test_preconfigured_search_requires_explicit_iterations_and_separate_batch_mode():
    function = _function(2, (2,))
    iteration = GroverOperator(PhaseOracle(function))
    sampler = StatevectorSampler()
    grover = Grover(sampler, iteration=iteration)

    with pytest.raises(TypeError, match="requires iterations and shots"):
        grover.run(shots=32)  # type: ignore[call-overload]
    with pytest.raises(TypeError, match="batch requests cannot be combined"):
        grover.run([GroverRequest(function, shots=32)])
    with pytest.raises(TypeError, match="requires a preconfigured iteration"):
        Grover(sampler).run(iterations=1, shots=32)


def _hadamard_matrix(num_qubits: int) -> np.ndarray:
    single = np.array([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2.0)
    matrix = single
    for _ in range(num_qubits - 1):
        matrix = np.kron(single, matrix)
    return matrix


def _independent_search_matrix(
    num_qubits: int,
    marked_states: tuple[int, ...],
    iterations: int,
) -> np.ndarray:
    """Construct ``(D O)**iterations H**n`` from dense projectors."""
    dimension = 1 << num_qubits
    uniform = np.ones(dimension, dtype=np.complex128) / math.sqrt(dimension)
    diffusion = 2.0 * np.outer(uniform, uniform.conj()) - np.eye(dimension)
    signs = np.ones(dimension, dtype=np.complex128)
    signs[list(marked_states)] = -1.0
    iterate = diffusion @ np.diag(signs)
    return np.linalg.matrix_power(iterate, iterations) @ _hadamard_matrix(num_qubits)


class _ScriptedSampler(Sampler):
    """Return caller-controlled complete count distributions."""

    def __init__(self, scripts):
        self.scripts = tuple(dict(script) for script in scripts)
        self.calls = 0
        self.requests = ()

    def run(self, requests):
        self.calls += 1
        self.requests = tuple(requests)
        return [
            ExecutionResult.from_counts(
                "scripted",
                list(request.qubits),
                request.shots,
                len(request.qubits),
                counts,
                "fake-backend",
            )
            for request, counts in zip(requests, self.scripts)
        ]


@pytest.mark.parametrize(
    ("num_qubits", "marked_states", "iterations"),
    [
        (1, (0,), 0),
        (1, (1,), 1),
        (2, (2,), 1),
        (3, (1,), 2),
        (3, (1, 6), 1),
        (3, (0, 3, 5), 2),
    ],
)
def test_circuit_matches_independent_dense_amplification_matrix(
    num_qubits, marked_states, iterations
):
    circuit = Grover.build_circuit(
        _function(num_qubits, marked_states), iterations=iterations
    )

    np.testing.assert_allclose(
        circuit.to_matrix(),
        _independent_search_matrix(num_qubits, marked_states, iterations),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("num_qubits", "num_solutions", "expected"),
    [
        (1, 1, 0),
        (1, 2, 0),
        (2, 1, 1),
        (2, 2, 0),
        (3, 1, 2),
        (3, 2, 1),
        (4, 1, 3),
        (4, 4, 1),
        (4, 16, 0),
    ],
)
def test_optimal_iteration_count_reaches_first_discrete_maximum(
    num_qubits, num_solutions, expected
):
    assert Grover.optimal_iterations(num_qubits, num_solutions) == expected

    theta = math.asin(math.sqrt(num_solutions / (1 << num_qubits)))
    actual = math.sin((2 * expected + 1) * theta) ** 2
    if expected:
        previous = math.sin((2 * expected - 1) * theta) ** 2
        assert actual >= previous - 1e-15


@pytest.mark.parametrize(
    ("num_qubits", "marked_states"),
    [
        (2, (3,)),
        (3, (5,)),
        (3, (1, 6)),
        (4, (0, 3, 9, 14)),
    ],
)
def test_default_circuit_has_analytic_marked_probability(num_qubits, marked_states):
    circuit = Grover.build_circuit(_function(num_qubits, marked_states))
    state = circuit.to_matrix()[:, 0]
    observed = sum(abs(state[index]) ** 2 for index in marked_states)
    iterations = Grover.optimal_iterations(num_qubits, len(marked_states))
    theta = math.asin(math.sqrt(len(marked_states) / (1 << num_qubits)))
    expected = math.sin((2 * iterations + 1) * theta) ** 2
    assert observed == pytest.approx(expected, abs=1e-12)


def test_single_solution_two_qubits_is_recovered_with_unit_probability():
    result = Grover(StatevectorSampler()).run(
        [GroverRequest(_function(2, (2,)), shots=32, seed=7)]
    )[0]

    assert result.iterations == 1
    assert result.counts == (0, 0, 32, 0)
    assert result.modal_states == (2,)
    assert result.modal_bitstrings == ((False, True),)
    assert result.marked_modal_states == (2,)
    assert result.success_probability == 1.0


def test_two_solutions_three_qubits_are_amplified_with_unit_probability():
    result = Grover(StatevectorSampler()).run(
        [GroverRequest(_function(3, (1, 6)), shots=200, seed=19)]
    )[0]

    assert result.iterations == 1
    assert result.success_count == result.shots
    assert sum(result.counts[index] for index in (1, 6)) == 200
    assert all(
        count == 0 for index, count in enumerate(result.counts) if index not in {1, 6}
    )


def test_one_solution_three_qubits_matches_expected_high_success_probability():
    result = Grover(StatevectorSampler()).run(
        [GroverRequest(_function(3, (5,)), shots=4096, seed=12345)]
    )[0]

    theta = math.asin(math.sqrt(1.0 / 8.0))
    expected = math.sin(5.0 * theta) ** 2
    assert result.iterations == 2
    assert result.success_probability == pytest.approx(expected, abs=0.02)
    assert result.modal_states == (5,)


def test_all_states_marked_uses_zero_iterations_and_always_succeeds():
    result = Grover(StatevectorSampler()).run(
        [GroverRequest(_function(3, tuple(range(8))), shots=64, seed=4)]
    )[0]

    assert result.iterations == 0
    assert result.success_count == 64
    assert result.success_probability == 1.0


def test_explicit_zero_iterations_returns_uniform_baseline():
    result = Grover(StatevectorSampler()).run(
        [GroverRequest(_function(2, (3,)), shots=4096, iterations=0, seed=91)]
    )[0]

    assert result.iterations == 0
    assert result.success_probability == pytest.approx(0.25, abs=0.03)


def test_vector_output_mask_searches_selected_output_parity():
    function = BooleanVectorFunction.from_marked_state_sets(
        2,
        ((1, 3), (3,)),
    )

    result = Grover(StatevectorSampler()).run(
        [GroverRequest(function, shots=32, output_mask=3, seed=5)]
    )[0]

    # XOR of the two outputs is true only on state 1.
    assert result.marked_states == (1,)
    assert result.success_count == 32


def test_batch_uses_one_sampler_call_and_preserves_request_order():
    sampler = _ScriptedSampler([{"01": 8}, {"110": 6}])
    requests = [
        GroverRequest(_function(2, (1,)), shots=8),
        GroverRequest(_function(3, (6,)), shots=6),
    ]

    results = Grover(sampler).run(requests)

    assert sampler.calls == 1
    assert [result.modal_states for result in results] == [(1,), (6,)]
    assert [result.success_count for result in results] == [8, 6]


def test_empty_batch_does_not_call_sampler():
    sampler = _ScriptedSampler([])
    assert Grover(sampler).run([]) == []
    assert sampler.calls == 0


def test_complete_batch_is_validated_before_sampler_call():
    sampler = _ScriptedSampler([{"01": 8}])
    with pytest.raises(TypeError, match="GroverRequest instances"):
        Grover(sampler).run(
            [GroverRequest(_function(2, (1,)), shots=8), "bad"]  # type: ignore[list-item]
        )
    assert sampler.calls == 0


def test_seed_is_forwarded_to_sampler_request():
    sampler = _ScriptedSampler([{"10": 4}])
    Grover(sampler).run([GroverRequest(_function(2, (2,)), shots=4, seed=73)])
    assert sampler.requests[0].seed == 73


def test_scripted_counts_determine_result_instead_of_truth_table():
    sampler = _ScriptedSampler([{"00": 7, "11": 1}])
    result = Grover(sampler).run([GroverRequest(_function(2, (3,)), shots=8)])[0]

    assert result.counts == (7, 0, 0, 1)
    assert result.modal_states == (0,)
    assert result.marked_modal_states == ()
    assert result.success_probability == 0.125


@pytest.mark.parametrize("shots", [True, 1.5, "4"])
def test_request_rejects_non_integer_shots(shots):
    with pytest.raises(TypeError, match="shots must be an integer"):
        GroverRequest(_function(2, (1,)), shots)  # type: ignore[arg-type]


@pytest.mark.parametrize("shots", [0, -1])
def test_request_rejects_non_positive_shots(shots):
    with pytest.raises(ValueError, match="shots must be greater than zero"):
        GroverRequest(_function(2, (1,)), shots)


@pytest.mark.parametrize("iterations", [True, 1.5, "1"])
def test_request_rejects_non_integer_iterations(iterations):
    with pytest.raises(TypeError, match="iterations must be an integer or None"):
        GroverRequest(_function(2, (1,)), 4, iterations=iterations)  # type: ignore[arg-type]


def test_request_rejects_negative_iterations():
    with pytest.raises(ValueError, match="iterations must be non-negative"):
        GroverRequest(_function(2, (1,)), 4, iterations=-1)


@pytest.mark.parametrize("seed", [True, 1.5, "1"])
def test_request_rejects_invalid_seed_type(seed):
    with pytest.raises(TypeError, match="seed must be an integer or None"):
        GroverRequest(_function(2, (1,)), 4, seed=seed)  # type: ignore[arg-type]


def test_request_rejects_negative_seed():
    with pytest.raises(ValueError, match="seed must be non-negative"):
        GroverRequest(_function(2, (1,)), 4, seed=-1)


def test_request_and_builder_reject_functions_with_no_marked_state():
    function = _function(2, ())
    with pytest.raises(ValueError, match="must mark at least one state"):
        GroverRequest(function, 4)
    with pytest.raises(ValueError, match="must mark at least one state"):
        Grover.build_circuit(function)


@pytest.mark.parametrize(
    ("num_qubits", "num_solutions", "error"),
    [
        (True, 1, TypeError),
        (0, 1, ValueError),
        (2, True, TypeError),
        (2, 0, ValueError),
        (2, 5, ValueError),
    ],
)
def test_optimal_iterations_rejects_invalid_arguments(num_qubits, num_solutions, error):
    with pytest.raises(error):
        Grover.optimal_iterations(num_qubits, num_solutions)


def test_invalid_vector_mask_is_rejected_by_oracle_contract():
    function = BooleanVectorFunction.from_marked_state_sets(2, ((1,), (2,)))
    with pytest.raises(ValueError, match="output_mask is required"):
        GroverRequest(function, 4)


def test_algorithm_requires_sampler_instance():
    with pytest.raises(TypeError, match="sampler must be a Sampler instance"):
        Grover(StatevectorSampler().run)  # type: ignore[arg-type]


def test_sampler_counts_must_be_complete():
    sampler = _ScriptedSampler([{"11": 3}])
    with pytest.raises(ValueError, match="counts must sum"):
        Grover(sampler).run([GroverRequest(_function(2, (3,)), shots=4)])


def test_result_validates_and_exposes_immutable_complete_statistics():
    result = GroverResult(
        marked_states=(1, 3),
        counts=(1, 4, 1, 4),
        iterations=1,
        shots=10,
    )

    assert result.num_qubits == 2
    assert result.num_solutions == 2
    assert result.success_count == 8
    assert result.success_probability == 0.8
    assert result.modal_states == (1, 3)
    assert result.modal_bitstrings == ((True, False), (True, True))
    assert result.marked_modal_states == (1, 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"marked_states": (), "counts": (1, 1), "iterations": 0, "shots": 2},
        {"marked_states": (1, 1), "counts": (1, 1), "iterations": 0, "shots": 2},
        {"marked_states": (-1,), "counts": (1, 1), "iterations": 0, "shots": 2},
        {"marked_states": (2,), "counts": (1, 1), "iterations": 0, "shots": 2},
        {"marked_states": (1,), "counts": (2, 0, 0), "iterations": 0, "shots": 2},
        {"marked_states": (1,), "counts": (1, -1), "iterations": 0, "shots": 2},
        {"marked_states": (1,), "counts": (1, 1), "iterations": -1, "shots": 2},
        {"marked_states": (1,), "counts": (1, 1), "iterations": 0, "shots": 3},
    ],
)
def test_result_rejects_inconsistent_values(kwargs):
    with pytest.raises((TypeError, ValueError)):
        GroverResult(**kwargs)
