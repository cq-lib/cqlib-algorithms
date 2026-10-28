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

"""Scientific, sampling and contract tests for Sampling VQE."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest
from cqlib import Circuit, Parameter
from cqlib.device import ExecutionResult
from cqlib.qis import Hamiltonian, PauliString

from cqlib_algorithms import SamplingVQE, SamplingVQEResult
from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.protocols import (
    Ansatz,
    Estimator,
    EstimatorRequest,
    Optimizer,
    OptimizerRequest,
    OptimizerResult,
    OptimizerStatus,
    Sampler,
    SamplerRequest,
)


class _OneParameterAnsatz(Ansatz):
    def validate(self) -> None:
        return None

    def build_circuit(self, prefix: str) -> Circuit:
        circuit = Circuit(2)
        circuit.rx(0, Parameter(f"{prefix}_0"))
        return circuit

    def num_parameters(self) -> int:
        return 1

    def num_qubits(self) -> int:
        return 2


class _StaticOneState(Ansatz):
    def validate(self) -> None:
        return None

    def build_circuit(self, prefix: str) -> Circuit:
        del prefix
        circuit = Circuit(1)
        circuit.x(0)
        return circuit

    def num_parameters(self) -> int:
        return 0

    def num_qubits(self) -> int:
        return 1


class _CountsSampler(Sampler):
    def __init__(self, counts: Sequence[Mapping[str, int]]) -> None:
        self._counts = tuple(dict(item) for item in counts)
        self.requests: list[SamplerRequest] = []

    def run(
        self,
        requests: Sequence[SamplerRequest],
        /,
    ) -> list[ExecutionResult]:
        results: list[ExecutionResult] = []
        for request in requests:
            request.validate()
            self.requests.append(request)
            counts = self._counts[len(self.requests) - 1]
            results.append(
                ExecutionResult.from_counts(
                    f"sample-{len(self.requests)}",
                    list(request.qubits),
                    request.shots,
                    len(request.qubits),
                    counts,
                    "fake",
                )
            )
        return results


class _BatchEvaluatingOptimizer(Optimizer):
    def __init__(self) -> None:
        self.values: list[float] = []
        self.variances: list[float | None] = []

    def run(
        self,
        estimator: Estimator,
        requests: Sequence[OptimizerRequest],
        /,
    ) -> list[OptimizerResult]:
        request = next(iter(requests))
        base = request.estimator_request

        estimates = estimator.run(
            [
                EstimatorRequest(
                    base.circuit,
                    base.observable,
                    {request.parameter_names[0]: value},
                    base.shots,
                )
                for value in (0.0, 1.0)
            ]
        )
        self.values = [result.expectation for result in estimates]
        self.variances = [result.variance for result in estimates]
        return [
            OptimizerResult(
                request.parameter_names,
                (1.0,),
                estimates[1].expectation,
                1,
                OptimizerStatus.CONVERGED,
            )
        ]


class _NoParameterOptimizer(Optimizer):
    def run(
        self,
        estimator: Estimator,
        requests: Sequence[OptimizerRequest],
        /,
    ) -> list[OptimizerResult]:
        request = next(iter(requests))
        result = estimator.run([request.estimator_request])[0]
        return [
            OptimizerResult(
                (), (), result.expectation, 0, OptimizerStatus.NO_PARAMETERS
            )
        ]


def _correlated_diagonal_operator() -> Hamiltonian:
    operator = Hamiltonian(2)
    operator.add_term(PauliString.from_str("II"), 0.25)
    operator.add_term(PauliString.from_str("IZ"), 1.0)
    operator.add_term(PauliString.from_str("ZI"), 2.0)
    operator.add_term(PauliString.from_str("ZZ"), -0.5)
    return operator


def _independent_value(bitstring: str) -> float:
    q0 = int(bitstring[-1])
    q1 = int(bitstring[-2])
    z0 = 1.0 if q0 == 0 else -1.0
    z1 = 1.0 if q1 == 0 else -1.0
    return 0.25 + z0 + 2.0 * z1 - 0.5 * z0 * z1


@pytest.mark.reference
def test_joint_counts_give_independent_mean_variance_and_little_endian_values() -> None:
    objective_counts = {"00": 2, "01": 3, "10": 1, "11": 2}
    final_counts = {"01": 4, "10": 4}
    sampler = _CountsSampler((objective_counts, objective_counts, final_counts))
    optimizer = _BatchEvaluatingOptimizer()
    result = SamplingVQE(
        sampler, _OneParameterAnsatz(), optimizer
    ).compute_minimum_eigenvalue(
        _correlated_diagonal_operator(), {"theta_0": 0.0}, shots=8, seed=19
    )

    expected_objective = (
        sum(
            count * _independent_value(bitstring)
            for bitstring, count in objective_counts.items()
        )
        / 8
    )
    expected_final = (
        sum(
            count * _independent_value(bitstring)
            for bitstring, count in final_counts.items()
        )
        / 8
    )
    expected_variance = (
        sum(
            count * _independent_value(bitstring) ** 2
            for bitstring, count in objective_counts.items()
        )
        / 8
        - expected_objective**2
    ) / 7
    assert optimizer.values == pytest.approx([expected_objective] * 2)
    assert optimizer.variances == pytest.approx([expected_variance] * 2)
    assert result.final_sample_value == pytest.approx(expected_final)
    assert result.final_distribution == {"01": 0.5, "10": 0.5}
    assert result.best_measurement.bitstring == "11"
    assert result.best_measurement.value == pytest.approx(_independent_value("11"))
    assert len(sampler.requests) == 3
    assert all(
        request.qubits == tuple(request.circuit.qubits) for request in sampler.requests
    )


def test_every_objective_point_uses_one_full_register_sampler_request() -> None:
    sampler = _CountsSampler(({"00": 4}, {"11": 4}, {"01": 4}))
    SamplingVQE(
        sampler, _OneParameterAnsatz(), _BatchEvaluatingOptimizer()
    ).compute_minimum_eigenvalue(
        _correlated_diagonal_operator(), {"theta_0": 0.0}, shots=4
    )

    assert [request.shots for request in sampler.requests] == [4, 4, 4]
    assert [len(request.qubits) for request in sampler.requests] == [2, 2, 2]


def test_optimization_best_and_independent_final_sample_are_distinct() -> None:
    sampler = _CountsSampler(({"00": 4}, {"00": 4}, {"11": 4}))
    result = SamplingVQE(
        sampler, _OneParameterAnsatz(), _BatchEvaluatingOptimizer()
    ).compute_minimum_eigenvalue(
        _correlated_diagonal_operator(), {"theta_0": 0.0}, shots=4
    )

    assert result.optimal_value == pytest.approx(_independent_value("00"))
    assert result.final_sample_value == pytest.approx(_independent_value("11"))
    assert result.optimal_value != result.final_sample_value
    assert result.best_measurement.bitstring == "00"
    assert result.best_measurement.bitstring not in result.final_distribution


def test_equal_values_prefer_higher_probability_across_objective_samples() -> None:
    identity = Hamiltonian(2)
    identity.add_term(PauliString.from_str("II"), 1.0)
    sampler = _CountsSampler(({"00": 3, "01": 1}, {"10": 4}, {"11": 4}))
    result = SamplingVQE(
        sampler, _OneParameterAnsatz(), _BatchEvaluatingOptimizer()
    ).compute_minimum_eigenvalue(identity, {"theta_0": 0.0}, shots=4)

    assert result.best_measurement.bitstring == "10"
    assert result.best_measurement.value == 1.0
    assert result.best_measurement.probability == 1.0


def test_complete_best_measurement_tie_uses_stable_bitstring_order() -> None:
    identity = Hamiltonian(2)
    identity.add_term(PauliString.from_str("II"), 1.0)
    sampler = _CountsSampler(({"01": 2, "00": 2}, {"11": 2, "10": 2}, {"11": 4}))
    result = SamplingVQE(
        sampler, _OneParameterAnsatz(), _BatchEvaluatingOptimizer()
    ).compute_minimum_eigenvalue(identity, {"theta_0": 0.0}, shots=4)

    assert result.best_measurement.bitstring == "00"
    assert result.best_measurement.probability == 0.5


def test_zero_count_keys_are_not_observed_outcomes() -> None:
    sampler = _CountsSampler(({"0": 0, "1": 4}, {"0": 0, "1": 4}))
    result = SamplingVQE(
        sampler, _StaticOneState(), _NoParameterOptimizer()
    ).compute_minimum_eigenvalue(PauliString.from_str("Z"), {}, shots=4)

    assert result.best_measurement.bitstring == "1"
    assert result.final_distribution == {"1": 1.0}


def test_master_seed_reproducibly_derives_distinct_child_seeds() -> None:
    def run(seed: int) -> list[int | None]:
        sampler = _CountsSampler(({"00": 2}, {"00": 2}, {"00": 2}))
        SamplingVQE(
            sampler, _OneParameterAnsatz(), _BatchEvaluatingOptimizer()
        ).compute_minimum_eigenvalue(
            _correlated_diagonal_operator(), {"theta_0": 0.0}, shots=2, seed=seed
        )
        return [request.seed for request in sampler.requests]

    first = run(7)
    assert first == run(7)
    assert first != run(8)
    assert len(set(first)) == len(first)


def test_real_sampler_integration_returns_exact_deterministic_state() -> None:
    result = SamplingVQE(
        StatevectorSampler(), _StaticOneState(), _NoParameterOptimizer()
    ).compute_minimum_eigenvalue(PauliString.from_str("Z"), {}, shots=100, seed=4)

    assert isinstance(result, SamplingVQEResult)
    assert result.optimal_value == -1.0
    assert result.final_sample_value == -1.0
    assert result.final_distribution == {"1": 1.0}
    assert result.best_measurement.bitstring == "1"
    assert result.best_measurement.probability == 1.0


def test_non_diagonal_nonhermitian_and_empty_objectives_are_rejected() -> None:
    algorithm = SamplingVQE(
        StatevectorSampler(), _StaticOneState(), _NoParameterOptimizer()
    )
    with pytest.raises(ValueError, match="diagonal"):
        algorithm.compute_minimum_eigenvalue(PauliString.from_str("X"), {}, shots=10)
    nonhermitian = Hamiltonian(1)
    nonhermitian.add_term(PauliString.from_str("Z"), 1.0j)
    with pytest.raises(ValueError, match="Hermitian"):
        algorithm.compute_minimum_eigenvalue(nonhermitian, {}, shots=10)
    with pytest.raises(ValueError, match="empty"):
        algorithm.compute_minimum_eigenvalue(Hamiltonian(1), {}, shots=10)
    zero_width = Hamiltonian(0)
    zero_width.add_term(PauliString(0), 1.0)
    with pytest.raises(ValueError, match="at least one qubit"):
        algorithm.compute_minimum_eigenvalue(zero_width, {}, shots=10)


@pytest.mark.parametrize("shots", [0, -1, True, 1.5])
def test_invalid_shots_are_rejected(shots: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        SamplingVQE(
            StatevectorSampler(), _StaticOneState(), _NoParameterOptimizer()
        ).compute_minimum_eigenvalue(
            PauliString.from_str("Z"),
            {},
            shots=shots,  # type: ignore[arg-type]
        )


def test_initial_point_and_operator_are_not_mutated_and_result_is_immutable() -> None:
    operator = _correlated_diagonal_operator()
    original = operator.to_matrix().copy()
    point = {"theta_0": 0.2}
    sampler = _CountsSampler(({"00": 2}, {"00": 2}, {"00": 2}))
    result = SamplingVQE(
        sampler, _OneParameterAnsatz(), _BatchEvaluatingOptimizer()
    ).compute_minimum_eigenvalue(operator, point, shots=2)

    assert point == {"theta_0": 0.2}
    assert (operator.to_matrix() == original).all()
    with pytest.raises(TypeError):
        result.final_distribution["11"] = 1.0  # type: ignore[index]
    with pytest.raises(TypeError):
        hash(result)


class _BadOptimizer(Optimizer):
    def __init__(self, response: object) -> None:
        self.response = response

    def run(
        self,
        estimator: Estimator,
        requests: Sequence[OptimizerRequest],
        /,
    ) -> list[OptimizerResult]:
        del estimator, requests
        return self.response  # type: ignore[return-value]


@pytest.mark.parametrize("response", [(), [], [object()]])
def test_invalid_optimizer_responses_are_rejected(response: object) -> None:
    algorithm = SamplingVQE(
        StatevectorSampler(), _StaticOneState(), _BadOptimizer(response)
    )
    with pytest.raises((TypeError, ValueError)):
        algorithm.compute_minimum_eigenvalue(PauliString.from_str("Z"), {}, shots=2)


class _BadSampler(Sampler):
    def run(
        self,
        requests: Sequence[SamplerRequest],
        /,
    ) -> list[ExecutionResult]:
        del requests
        return ()  # type: ignore[return-value]


def test_invalid_sampler_response_is_rejected() -> None:
    with pytest.raises(TypeError, match="sampler must return a list"):
        SamplingVQE(
            _BadSampler(), _StaticOneState(), _NoParameterOptimizer()
        ).compute_minimum_eigenvalue(PauliString.from_str("Z"), {}, shots=2)


def test_constructor_rejects_invalid_dependencies() -> None:
    with pytest.raises(TypeError, match="sampler"):
        SamplingVQE(object(), _StaticOneState(), _NoParameterOptimizer())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ansatz"):
        SamplingVQE(StatevectorSampler(), object(), _NoParameterOptimizer())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="optimizer"):
        SamplingVQE(StatevectorSampler(), _StaticOneState(), object())  # type: ignore[arg-type]
