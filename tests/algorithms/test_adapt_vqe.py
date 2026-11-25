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

"""Scientific, integration and contract tests for ADAPT-VQE."""

from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np
import pytest
from cqlib.circuit.ansatz import EvolutionStrategy
from cqlib.qis import Hamiltonian, JordanWignerMapper, PauliString, TrotterMode

from cqlib_algorithms import AdaptVQE, AdaptVQEResult, AdaptVQETermination
from cqlib_algorithms.algorithms.variational.adapt_vqe import (
    _commutator_observable,
)
from cqlib_algorithms.extension.electronic_structure import FermionicExcitation
from cqlib_algorithms.operators.estimator import StatevectorEstimator
from cqlib_algorithms.operators.optimizer import LBFGSBOptimizer
from cqlib_algorithms.operators.state_preparation import (
    BasisStateInitialization,
    UniformSuperposition,
)
from cqlib_algorithms.protocols import (
    Estimator,
    EstimatorRequest,
    EstimatorResult,
    Optimizer,
    OptimizerRequest,
    OptimizerResult,
    OptimizerStatus,
)


class _GradientSequenceEstimator(Estimator):
    def __init__(self, gradients: Sequence[Sequence[float]]) -> None:
        self._gradients = tuple(tuple(batch) for batch in gradients)
        self.batches: list[list[EstimatorRequest]] = []

    def run(self, requests: Sequence[EstimatorRequest], /) -> list[EstimatorResult]:
        batch = list(requests)
        self.batches.append(batch)
        index = min(len(self.batches) - 1, len(self._gradients) - 1)
        values = self._gradients[index]
        assert len(values) == len(batch)
        return [EstimatorResult(value, 0.0) for value in values]


class _WarmStartOptimizer(Optimizer):
    def __init__(self, *, energy_step: float = 1.0) -> None:
        self.energy_step = energy_step
        self.requests: list[OptimizerRequest] = []

    def run(
        self,
        estimator: Estimator,
        requests: Sequence[OptimizerRequest],
        /,
    ) -> list[OptimizerResult]:
        del estimator
        request = next(iter(requests))
        self.requests.append(request)
        values = tuple(
            request.estimator_request.parameter_bindings[name] + 0.25
            for name in request.parameter_names
        )
        return [
            OptimizerResult(
                request.parameter_names,
                values,
                -self.energy_step * len(request.parameter_names),
                1 if request.parameter_names else 0,
                OptimizerStatus.CONVERGED,
            )
        ]


class _SeedRecordingEstimator(Estimator):
    def __init__(self) -> None:
        self.seeds: list[int | None] = []
        self.calls = 0

    def run(self, requests: Sequence[EstimatorRequest], /) -> list[EstimatorResult]:
        results: list[EstimatorResult] = []
        for request in requests:
            request.validate()
            self.seeds.append(request.seed)
            assert request.seed is not None
            if self.calls == 1:
                value = 1.0
            elif self.calls == 3:
                value = 0.0
            else:
                value = float(np.random.default_rng(request.seed).uniform(-0.5, 0.5))
            self.calls += 1
            results.append(EstimatorResult(value, 0.0))
        return results


class _EvaluatingOptimizer(Optimizer):
    def run(
        self,
        estimator: Estimator,
        requests: Sequence[OptimizerRequest],
        /,
    ) -> list[OptimizerResult]:
        request = next(iter(requests))
        estimate = estimator.run([request.estimator_request])[0]
        parameters = tuple(
            request.estimator_request.parameter_bindings[name]
            for name in request.parameter_names
        )
        return [
            OptimizerResult(
                request.parameter_names,
                parameters,
                estimate.expectation,
                0,
                (
                    OptimizerStatus.NO_PARAMETERS
                    if not parameters
                    else OptimizerStatus.CONVERGED
                ),
            )
        ]


def _pool() -> list[PauliString]:
    return [PauliString.from_str("XI"), PauliString.from_str("IY")]


@pytest.mark.reference
def test_commutator_gradient_matches_independent_dense_matrix_reference() -> None:
    target = Hamiltonian(1)
    target.add_term(PauliString.from_str("Z"), 0.7)
    target.add_term(PauliString.from_str("X"), -0.2)
    generator = PauliString.from_str("Y")
    reference = UniformSuperposition(1).build_circuit("unused").to_matrix()[:, 0]
    dense_target = target.to_matrix()
    dense_generator = generator.to_matrix()
    np.testing.assert_allclose(
        _commutator_observable(Hamiltonian.from_pauli(generator), target).to_matrix(),
        1j * (dense_generator @ dense_target - dense_target @ dense_generator),
        atol=1e-12,
    )
    expected = float(
        np.real(
            1j
            * np.vdot(
                reference,
                (dense_generator @ dense_target - dense_target @ dense_generator)
                @ reference,
            )
        )
    )

    algorithm = AdaptVQE(
        StatevectorEstimator(),
        _WarmStartOptimizer(),
        [generator],
        initial_state=UniformSuperposition(1),
        gradient_threshold=abs(expected) + 1e-12,
    )
    result = algorithm.compute_minimum_eigenvalue(target)

    assert result.termination is AdaptVQETermination.GRADIENT_CONVERGED
    assert result.gradient_norms == pytest.approx((abs(expected),), abs=1e-12)


@pytest.mark.reference
@pytest.mark.parametrize("entry_point", ["run", "compute_minimum_eigenvalue"])
def test_real_components_reach_one_qubit_ground_state(entry_point: str) -> None:
    algorithm = AdaptVQE(
        StatevectorEstimator(),
        LBFGSBOptimizer(maxiter=100, gtol=1e-10),
        [PauliString.from_str("Z"), PauliString.from_str("Y")],
        initial_state=UniformSuperposition(1),
        gradient_threshold=1e-7,
        max_iterations=3,
    )
    result = getattr(algorithm, entry_point)(PauliString.from_str("Z"))

    dense_ground = float(np.linalg.eigvalsh(PauliString.from_str("Z").to_matrix())[0])
    state = result.optimal_circuit.to_matrix()[:, 0]
    assert isinstance(result, AdaptVQEResult)
    assert result.eigenvalue == pytest.approx(dense_ground, abs=1e-10)
    assert abs(state[1]) ** 2 == pytest.approx(1.0, abs=1e-10)
    assert result.selected_operator_indices == (1,)
    assert result.termination is AdaptVQETermination.GRADIENT_CONVERGED


@pytest.mark.reference
def test_fermionic_excitation_maps_to_a_working_adaptive_pool() -> None:
    excitation = FermionicExcitation(2, (0,), (1,))
    mapper = JordanWignerMapper()
    generator = mapper.map_operator(excitation.to_fermionic_generator())
    transition = excitation.to_fermionic_op()
    target = mapper.map_operator(transition + transition.adjoint())
    expected_ground = float(np.linalg.eigvalsh(target.to_matrix())[0])

    result = AdaptVQE(
        StatevectorEstimator(),
        LBFGSBOptimizer(maxiter=100, gtol=1e-10),
        [generator],
        initial_state=BasisStateInitialization(2, 1),
        gradient_threshold=1e-7,
        max_iterations=2,
    ).compute_minimum_eigenvalue(target)

    assert result.eigenvalue == pytest.approx(expected_ground, abs=1e-10)
    assert result.selected_operator_indices == (0,)
    assert result.termination is AdaptVQETermination.GRADIENT_CONVERGED


def test_full_pool_norm_stable_selection_and_warm_start() -> None:
    estimator = _GradientSequenceEstimator(((2.0, -1.0), (0.5, -3.0), (0.0, 0.0)))
    optimizer = _WarmStartOptimizer()
    result = AdaptVQE(
        estimator,
        optimizer,
        _pool(),
        gradient_threshold=0.1,
        max_iterations=4,
    ).compute_minimum_eigenvalue(PauliString.from_str("ZZ"))

    assert result.selected_operator_indices == (0, 1)
    assert result.gradient_norms == pytest.approx((np.sqrt(5.0), np.sqrt(9.25), 0.0))
    assert result.termination is AdaptVQETermination.GRADIENT_CONVERGED
    assert optimizer.requests[1].estimator_request.parameter_bindings == {
        "adapt_0": 0.0
    }
    assert optimizer.requests[2].estimator_request.parameter_bindings == {
        "adapt_0": 0.25,
        "adapt_1": 0.0,
    }


def test_equal_absolute_gradients_select_first_pool_entry_stably() -> None:
    result = AdaptVQE(
        _GradientSequenceEstimator(((1.0, -1.0), (0.0, 0.0))),
        _WarmStartOptimizer(),
        _pool(),
        gradient_threshold=0.1,
    ).compute_minimum_eigenvalue(PauliString.from_str("ZZ"))

    assert result.selected_operator_indices == (0,)


def test_initial_gradient_convergence_returns_valid_zero_parameter_result() -> None:
    result = AdaptVQE(
        StatevectorEstimator(),
        _WarmStartOptimizer(),
        [PauliString.from_str("X")],
        gradient_threshold=1e-12,
    ).compute_minimum_eigenvalue(PauliString.from_str("Z"))

    assert result.selected_operator_indices == ()
    assert result.optimal_parameters == {}
    assert result.optimizer_result.status is OptimizerStatus.CONVERGED
    assert result.termination is AdaptVQETermination.GRADIENT_CONVERGED
    assert result.eigenvalue_history == (0.0,)


def test_energy_threshold_rolls_back_unaccepted_candidate() -> None:
    result = AdaptVQE(
        _GradientSequenceEstimator(((1.0,),)),
        _WarmStartOptimizer(energy_step=1e-8),
        [PauliString.from_str("X")],
        gradient_threshold=0.1,
        eigenvalue_threshold=1e-6,
    ).compute_minimum_eigenvalue(PauliString.from_str("Z"))

    assert result.termination is AdaptVQETermination.ENERGY_CONVERGED
    assert result.selected_operator_indices == ()
    assert result.optimal_parameters == {}
    assert result.eigenvalue_history == (0.0,)


def test_repeated_suffix_and_iteration_limit_have_distinct_termination() -> None:
    cyclic = AdaptVQE(
        _GradientSequenceEstimator(((1.0,), (1.0,))),
        _WarmStartOptimizer(),
        [PauliString.from_str("X")],
        gradient_threshold=0.1,
        max_iterations=3,
    ).compute_minimum_eigenvalue(PauliString.from_str("Z"))
    limited = AdaptVQE(
        _GradientSequenceEstimator(((1.0,), (0.5,))),
        _WarmStartOptimizer(),
        [PauliString.from_str("X")],
        gradient_threshold=0.1,
        max_iterations=1,
    ).compute_minimum_eigenvalue(PauliString.from_str("Z"))

    assert cyclic.termination is AdaptVQETermination.CYCLIC_SEQUENCE
    assert limited.termination is AdaptVQETermination.MAX_ITERATIONS
    assert cyclic.selected_operator_indices == limited.selected_operator_indices == (0,)


def test_explicit_approximate_strategy_builds_and_runs_honestly() -> None:
    strategy = EvolutionStrategy.trotter(TrotterMode.first_order(), 2)
    result = AdaptVQE(
        _GradientSequenceEstimator(((1.0,), (0.0,))),
        _WarmStartOptimizer(),
        [PauliString.from_str("X")],
        gradient_threshold=0.1,
        max_iterations=2,
        evolution_strategy=strategy,
    ).compute_minimum_eigenvalue(PauliString.from_str("Z"))

    assert result.selected_operator_indices == (0,)
    assert result.optimal_circuit.used_symbols == []


def test_inputs_are_snapshotted_and_not_mutated() -> None:
    generator = Hamiltonian.from_pauli(PauliString.from_str("X"))
    target = Hamiltonian.from_pauli(PauliString.from_str("Z"))
    target_matrix = target.to_matrix().copy()
    pool = [generator]
    algorithm = AdaptVQE(
        _GradientSequenceEstimator(((0.0,),)),
        _WarmStartOptimizer(),
        pool,
    )
    pool.clear()
    generator.add_term(PauliString.from_str("I"), 1.0j)

    result = algorithm.compute_minimum_eigenvalue(target)

    assert result.termination is AdaptVQETermination.GRADIENT_CONVERGED
    np.testing.assert_allclose(target.to_matrix(), target_matrix)


def test_finite_shot_master_seed_reproduces_the_complete_request_stream() -> None:
    def run(
        seed: int, entry_point: str = "run"
    ) -> tuple[list[int | None], AdaptVQEResult]:
        estimator = _SeedRecordingEstimator()
        algorithm = AdaptVQE(
            estimator,
            _EvaluatingOptimizer(),
            [PauliString.from_str("X")],
            gradient_threshold=0.1,
            max_iterations=2,
        )
        result = getattr(algorithm, entry_point)(
            PauliString.from_str("Z"), shots=32, seed=seed
        )
        return estimator.seeds, result

    first_seeds, first = run(17)
    repeated_seeds, repeated = run(17, "compute_minimum_eigenvalue")
    other_seeds, other = run(18)

    assert first_seeds == repeated_seeds
    assert first_seeds != other_seeds
    assert len(first_seeds) == len(set(first_seeds)) == 4
    assert first.optimal_value == repeated.optimal_value
    assert first.gradient_norms == repeated.gradient_norms
    assert first.selected_operator_indices == repeated.selected_operator_indices == (0,)
    assert first.termination == repeated.termination
    assert first.eigenvalue_history == repeated.eigenvalue_history
    assert first.optimal_value != other.optimal_value


@pytest.mark.parametrize("seed", [True, -1, 1.5])
def test_invalid_master_seed_is_rejected(seed: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        AdaptVQE(
            _SeedRecordingEstimator(),
            _EvaluatingOptimizer(),
            [PauliString.from_str("X")],
        ).compute_minimum_eigenvalue(
            PauliString.from_str("Z"),
            shots=8,
            seed=seed,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "pool",
    [
        [],
        [PauliString.from_str("I")],
        [PauliString.from_str("X"), PauliString.from_str("ZZ")],
    ],
)
def test_invalid_pools_are_rejected(pool: list[PauliString]) -> None:
    with pytest.raises(ValueError):
        AdaptVQE(_GradientSequenceEstimator(((0.0,),)), _WarmStartOptimizer(), pool)


def test_nonhermitian_pool_and_wrong_target_width_are_rejected() -> None:
    nonhermitian = Hamiltonian(1)
    nonhermitian.add_term(PauliString.from_str("X"), 1.0j)
    with pytest.raises(ValueError, match="Hermitian"):
        AdaptVQE(
            _GradientSequenceEstimator(((0.0,),)),
            _WarmStartOptimizer(),
            [nonhermitian],
        )
    algorithm = AdaptVQE(
        _GradientSequenceEstimator(((0.0,),)),
        _WarmStartOptimizer(),
        [PauliString.from_str("X")],
    )
    with pytest.raises(ValueError, match="equal width"):
        algorithm.compute_minimum_eigenvalue(PauliString.from_str("ZZ"))


def test_result_metadata_is_immutable_and_strict() -> None:
    result = AdaptVQE(
        _GradientSequenceEstimator(((0.0,),)),
        _WarmStartOptimizer(),
        [PauliString.from_str("X")],
    ).compute_minimum_eigenvalue(PauliString.from_str("Z"))

    with pytest.raises((AttributeError, TypeError)):
        result.gradient_norms += (1.0,)  # type: ignore[misc]
    with pytest.raises(TypeError):
        hash(result)
    with pytest.raises(ValueError, match="reference value"):
        copy.copy(result).__class__(
            result.optimal_value,
            result.optimal_parameters,
            result.optimal_circuit,
            result.optimizer_result,
            selected_operator_indices=(0,),
            gradient_norms=result.gradient_norms,
            eigenvalue_history=result.eigenvalue_history,
            final_max_gradient=result.final_max_gradient,
            termination=result.termination,
        )


class _BadEstimator(Estimator):
    def run(self, requests: Sequence[EstimatorRequest], /) -> list[EstimatorResult]:
        del requests
        return ()  # type: ignore[return-value]


class _BadOptimizer(Optimizer):
    def run(
        self,
        estimator: Estimator,
        requests: Sequence[OptimizerRequest],
        /,
    ) -> list[OptimizerResult]:
        del estimator, requests
        return ()  # type: ignore[return-value]


def test_invalid_component_responses_are_rejected() -> None:
    with pytest.raises(TypeError, match="optimizer must return a list"):
        AdaptVQE(
            _GradientSequenceEstimator(((0.0,),)),
            _BadOptimizer(),
            [PauliString.from_str("X")],
        ).compute_minimum_eigenvalue(PauliString.from_str("Z"))
    with pytest.raises(TypeError, match="estimator must return a list"):
        AdaptVQE(
            _BadEstimator(),
            _WarmStartOptimizer(),
            [PauliString.from_str("X")],
            gradient_threshold=0.1,
        ).compute_minimum_eigenvalue(PauliString.from_str("Z"))


def test_constructor_rejects_invalid_dependencies_and_options() -> None:
    estimator = _GradientSequenceEstimator(((0.0,),))
    optimizer = _WarmStartOptimizer()
    pool = [PauliString.from_str("X")]
    with pytest.raises(TypeError, match="estimator"):
        AdaptVQE(object(), optimizer, pool)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="optimizer"):
        AdaptVQE(estimator, object(), pool)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="gradient_threshold"):
        AdaptVQE(estimator, optimizer, pool, gradient_threshold=0.0)
    with pytest.raises(ValueError, match="max_iterations"):
        AdaptVQE(estimator, optimizer, pool, max_iterations=0)
