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

"""Tests for standard and iterative quantum phase estimation."""

from __future__ import annotations

import math

import numpy as np
import pytest
from cqlib import Circuit
from cqlib.device import ExecutionResult

from cqlib_algorithms import (
    IterativePhaseEstimation,
    IterativePhaseEstimationRequest,
    PhaseEstimation,
    PhaseEstimationProblem,
    PhaseEstimationRequest,
)
from cqlib_algorithms.operators import StatevectorSampler
from cqlib_algorithms.protocols import Sampler


def _phase_problem(
    phase: float, *, superposition: bool = False
) -> PhaseEstimationProblem:
    unitary = Circuit(1)
    unitary.phase(0, 2.0 * math.pi * phase)
    preparation = Circuit(1)
    preparation.h(0) if superposition else preparation.x(0)
    return PhaseEstimationProblem(unitary, preparation)


def _qpe_probability(phase: float, point: int, dimension: int) -> float:
    offset = phase - point / dimension
    terms = np.exp(2j * math.pi * offset * np.arange(dimension))
    return float(abs(np.sum(terms) / dimension) ** 2)


def _evaluation_probabilities(circuit: Circuit, width: int) -> np.ndarray:
    dimension = 1 << width
    state = circuit.to_matrix()[:, 0]
    return np.sum(np.abs(state.reshape(-1, dimension)) ** 2, axis=0)


class _ScriptedSampler(Sampler):
    def __init__(self, scripts: list[dict[str, int]]) -> None:
        self.scripts = scripts
        self.calls = 0
        self.requests = []

    def run(self, requests):
        self.calls += 1
        self.requests.extend(requests)
        counts = self.scripts.pop(0)
        return [
            ExecutionResult.from_counts(
                "scripted",
                list(request.qubits),
                request.shots,
                len(request.qubits),
                counts,
                "test",
            )
            for request in requests
        ]


@pytest.mark.reference
def test_standard_qpe_matches_independent_non_grid_distribution() -> None:
    width = 4
    dimension = 1 << width
    phase = 0.271
    circuit = PhaseEstimation.build_circuit(_phase_problem(phase), width)

    expected = np.asarray(
        [_qpe_probability(phase, point, dimension) for point in range(dimension)]
    )
    np.testing.assert_allclose(
        _evaluation_probabilities(circuit, width),
        expected,
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.reference
def test_standard_qpe_samples_eigenphases_by_overlap() -> None:
    width = 3
    probabilities = _evaluation_probabilities(
        PhaseEstimation.build_circuit(
            _phase_problem(3 / 8, superposition=True),
            width,
        ),
        width,
    )

    expected = np.zeros(1 << width)
    expected[0] = 0.5
    expected[3] = 0.5
    np.testing.assert_allclose(probabilities, expected, rtol=0.0, atol=1e-12)


@pytest.mark.integration
def test_standard_qpe_executes_with_statevector_sampler() -> None:
    result = PhaseEstimation(StatevectorSampler()).run(
        [PhaseEstimationRequest(_phase_problem(3 / 8), 3, 32)]
    )[0]

    assert result.phase == 3 / 8
    assert result.grid_point == 3
    assert dict(result.phase_probabilities) == {3: 1.0}
    assert result.num_unitary_applications == 7
    assert result.num_unitary_queries == 224


def test_standard_qpe_uses_stable_mode_and_sparse_distribution() -> None:
    sampler = _ScriptedSampler([{"000": 0, "011": 4, "101": 4}])
    result = PhaseEstimation(sampler).run(
        [PhaseEstimationRequest(_phase_problem(3 / 8), 3, 8)]
    )[0]

    assert result.grid_point == 3
    assert result.phase == 3 / 8
    assert dict(result.phase_probabilities) == {3: 0.5, 5: 0.5}
    assert sampler.calls == 1


@pytest.mark.parametrize("grid_point", range(16))
@pytest.mark.integration
def test_standard_and_iterative_qpe_recover_every_four_bit_phase(
    grid_point: int,
) -> None:
    phase = grid_point / 16
    problem = _phase_problem(phase)
    standard = PhaseEstimation(StatevectorSampler()).run(
        [PhaseEstimationRequest(problem, 4, 1)]
    )[0]
    iterative = IterativePhaseEstimation(StatevectorSampler()).run(
        [IterativePhaseEstimationRequest(problem, 4, 1)]
    )[0]

    assert standard.phase == phase
    assert standard.grid_point == grid_point
    assert iterative.phase == phase
    assert iterative.bitstring == f"{grid_point:04b}"
    assert iterative.num_unitary_applications == 15
    assert iterative.num_unitary_queries == 15


@pytest.mark.integration
def test_standard_and_iterative_qpe_preserve_request_order() -> None:
    phases = (1 / 4, 3 / 4)
    standard = PhaseEstimation(StatevectorSampler()).run(
        [PhaseEstimationRequest(_phase_problem(phase), 2, 1) for phase in phases]
    )
    iterative = IterativePhaseEstimation(StatevectorSampler()).run(
        [
            IterativePhaseEstimationRequest(_phase_problem(phase), 2, 1)
            for phase in phases
        ]
    )

    assert [result.phase for result in standard] == list(phases)
    assert [result.phase for result in iterative] == list(phases)


def test_iterative_qpe_applies_feedback_and_tie_selects_zero() -> None:
    sampler = _ScriptedSampler(
        [
            {"1": 4},
            {"0": 2, "1": 2},
            {"1": 4},
        ]
    )
    result = IterativePhaseEstimation(sampler).run(
        [IterativePhaseEstimationRequest(_phase_problem(5 / 8), 3, 4)]
    )[0]

    assert result.bitstring == "101"
    assert result.phase == 5 / 8
    assert result.one_probabilities == (1.0, 0.5, 1.0)
    assert sampler.calls == 3
    assert all(len(request.qubits) == 1 for request in sampler.requests)
    assert any(
        operation.name == "Phase"
        for operation in sampler.requests[1].circuit.operations
    )


def test_problem_snapshots_inputs_and_rejects_width_mismatch() -> None:
    unitary = Circuit(1)
    unitary.z(0)
    preparation = Circuit(1)
    preparation.x(0)
    problem = PhaseEstimationProblem(unitary, preparation)
    unitary.x(0)
    preparation.h(0)

    assert len(problem.unitary.operations) == 1
    assert len(problem.state_preparation.operations) == 1
    with pytest.raises(ValueError, match="same qubit count"):
        PhaseEstimationProblem(Circuit(1), Circuit(2))


def test_empty_batches_do_not_call_sampler() -> None:
    standard_sampler = _ScriptedSampler([])
    iterative_sampler = _ScriptedSampler([])

    assert PhaseEstimation(standard_sampler).run([]) == []
    assert IterativePhaseEstimation(iterative_sampler).run([]) == []
    assert standard_sampler.calls == 0
    assert iterative_sampler.calls == 0


def test_requests_reject_numeric_boundaries_before_sampling() -> None:
    problem = _phase_problem(0.25)
    with pytest.raises(ValueError, match="must not exceed 52"):
        PhaseEstimationRequest(problem, 53, 1)
    with pytest.raises(ValueError, match=r"must not exceed 2\*\*53 - 1"):
        PhaseEstimationRequest(problem, 3, 1 << 53)
    with pytest.raises(ValueError, match="must not exceed 52"):
        IterativePhaseEstimationRequest(problem, 53)
