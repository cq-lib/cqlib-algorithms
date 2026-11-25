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
from cqlib import Circuit, Parameter
from cqlib.device import ExecutionResult

from cqlib_algorithms import (
    AmplitudeEstimationProblem,
    IterativePhaseEstimation,
    IterativePhaseEstimationRequest,
    PhaseEstimation,
    PhaseEstimationProblem,
    PhaseEstimationRequest,
    StandardAmplitudeEstimation,
)
from cqlib_algorithms.operators import (
    ArbitraryStatePreparation,
    BasisStateInitialization,
    BooleanFunction,
    GroverOperator,
    PhaseOracle,
    StatevectorSampler,
    UniformSuperposition,
    UnitaryMatrixOracle,
    UserCircuitPreparation,
)
from cqlib_algorithms.operators.arithmetic import DraperConstantAdder
from cqlib_algorithms.protocols import Sampler


def _phase_problem(
    phase: float, *, superposition: bool = False
) -> PhaseEstimationProblem:
    unitary = Circuit(1)
    unitary.phase(0, 2.0 * math.pi * phase)
    preparation = Circuit(1)
    preparation.h(0) if superposition else preparation.x(0)
    return PhaseEstimationProblem(unitary, UserCircuitPreparation(preparation))


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
    problem = PhaseEstimationProblem(unitary, UserCircuitPreparation(preparation))
    unitary.x(0)
    preparation.h(0)

    assert len(problem.unitary.operations) == 1
    assert len(problem.preparation_circuit.operations) == 1
    with pytest.raises(ValueError, match="same qubit count"):
        PhaseEstimationProblem(Circuit(1), UserCircuitPreparation(Circuit(2)))


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


@pytest.mark.integration
@pytest.mark.parametrize(("index", "bits"), [(0, "001"), (1, "011")])
def test_shared_component_problem_matches_independent_circuit_batches(
    index, bits, monkeypatch
):
    phase = math.pi / 4
    unitary = UnitaryMatrixOracle(np.exp(1j * phase) * np.diag([1, 1j]))
    preparation = BasisStateInitialization(1, index)
    problem = PhaseEstimationProblem(unitary.build_circuit(), preparation)

    def unexpected_build(*args):
        pytest.fail("a snapshotted component was built again")

    monkeypatch.setattr(BasisStateInitialization, "build_circuit", unexpected_build)

    raw_u = Circuit(1)
    raw_u.set_global_phase(phase)
    raw_u.s(0)
    raw_a = Circuit(1)
    if index:
        raw_a.x(0)
    legacy = PhaseEstimationProblem(raw_u, UserCircuitPreparation(raw_a))
    qpe = PhaseEstimation(StatevectorSampler())
    iqpe = IterativePhaseEstimation(StatevectorSampler())
    standard = qpe.run(problem, num_evaluation_qubits=3, shots=32)
    iterative = iqpe.run(problem, num_iterations=3, shots_per_iteration=8)

    assert standard == qpe.run([PhaseEstimationRequest(legacy, 3, 32)])[0]
    assert iterative == iqpe.run([IterativePhaseEstimationRequest(legacy, 3, 8)])[0]
    assert standard.phase == iterative.phase == int(bits, 2) / 8
    assert iterative.bitstring == bits
    assert dict(standard.phase_probabilities) == {int(bits, 2): 1.0}


@pytest.mark.integration
def test_change_phase_target_from_marking_to_grover_iteration():
    marking = PhaseOracle(BooleanFunction.from_marked_states(1, [1]))
    # Reflection about |0> and the marking oracle both implement Z.
    iteration = GroverOperator(
        marking, state_preparation=BasisStateInitialization(1, 0)
    )
    eigenstate = BasisStateInitialization(1, 1)
    qpe = PhaseEstimation(StatevectorSampler())
    results = qpe.run(
        [
            PhaseEstimationRequest(
                PhaseEstimationProblem(target.build_circuit(), eigenstate), 2, 8
            )
            for target in (marking, iteration)
        ]
    )
    assert [r.phase for r in results] == [0.5, 0.0]


def test_phase_problem_rejects_incompatible_components():
    marking = PhaseOracle(BooleanFunction.from_marked_states(1, [1]))
    with pytest.raises(ValueError, match="same qubit count"):
        PhaseEstimationProblem(marking.build_circuit(), BasisStateInitialization(2, 0))
    with pytest.raises(TypeError, match="unitary"):
        PhaseEstimationProblem(object(), UserCircuitPreparation(Circuit(1)))
    with pytest.raises(TypeError, match="state_preparation"):
        PhaseEstimationProblem(marking.build_circuit(), marking)
    symbolic = Circuit(1)
    symbolic.ry(0, Parameter("angle"))
    with pytest.raises(ValueError, match="fully bound"):
        PhaseEstimationProblem(
            marking.build_circuit(), UserCircuitPreparation(symbolic)
        )


@pytest.mark.reference
def test_arithmetic_target_circuit_and_preparation_component_recover_eigenphase():
    unitary = DraperConstantAdder(1, 1).build_circuit()
    eigenstate = ArbitraryStatePreparation([1 / math.sqrt(2), -1 / math.sqrt(2)])
    problem = PhaseEstimationProblem(unitary, eigenstate)
    standard = PhaseEstimation(StatevectorSampler()).run(
        problem, num_evaluation_qubits=2, shots=32
    )
    iterative = IterativePhaseEstimation(StatevectorSampler()).run(
        problem, num_iterations=2, shots_per_iteration=8
    )
    assert standard.phase == iterative.phase == 0.5
    assert iterative.bitstring == "10"


def test_phase_problem_keeps_target_and_initial_state_roles_distinct():
    from cqlib_algorithms.operators import PhaseEstimationCircuit

    preparation = UniformSuperposition(1)
    with pytest.raises(TypeError, match="unitary"):
        PhaseEstimationProblem(preparation, preparation)
    with pytest.raises(TypeError, match="unitary"):
        PhaseEstimationCircuit(preparation, 2)


@pytest.mark.reference
def test_standard_ae_folds_the_two_qpe_phases_of_the_same_grover_component():
    preparation = UniformSuperposition(1)
    oracle = PhaseOracle(BooleanFunction.from_marked_states(1, [1]))
    iteration = GroverOperator(oracle, state_preparation=preparation)
    problem = PhaseEstimationProblem(iteration.build_circuit(), preparation)
    distribution = _evaluation_probabilities(
        PhaseEstimation.build_circuit(problem, 2), 2
    )
    np.testing.assert_allclose(distribution, [0.0, 0.5, 0.0, 0.5], rtol=0, atol=1e-12)
    amplitude_problem = AmplitudeEstimationProblem(
        preparation, iteration, (0,), lambda bits: bits == "1"
    )
    result = StandardAmplitudeEstimation(StatevectorSampler()).run(
        amplitude_problem, num_evaluation_qubits=2, shots=16
    )
    assert result.estimation == pytest.approx(0.5, abs=1e-12)


def test_phase_run_rejects_ambiguous_requests_before_sampling():
    problem = _phase_problem(0.25)
    sampler = _ScriptedSampler([])
    qpe, iqpe = PhaseEstimation(sampler), IterativePhaseEstimation(sampler)
    with pytest.raises(TypeError, match="requires"):
        qpe.run(problem, shots=1)
    with pytest.raises(TypeError, match="batch requests"):
        qpe.run([PhaseEstimationRequest(problem, 2, 1)], approximation_degree=0)
    with pytest.raises(ValueError, match="approximation_degree"):
        qpe.run(problem, num_evaluation_qubits=2, shots=1, approximation_degree=2)
    with pytest.raises(TypeError, match="requires"):
        iqpe.run(problem, shots_per_iteration=2)
    with pytest.raises(TypeError, match="batch requests"):
        iqpe.run([], num_iterations=2)
    with pytest.raises(ValueError, match="shots_per_iteration"):
        iqpe.run(problem, num_iterations=2, shots_per_iteration=0)
    with pytest.raises(TypeError, match="must contain"):
        iqpe.run([IterativePhaseEstimationRequest(problem, 2), object()])
    assert sampler.calls == 0
