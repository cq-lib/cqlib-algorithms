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

"""Contract and independent-reference tests for variational deflation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
import pytest
from cqlib import Circuit
from cqlib.qis import Hamiltonian, PauliString
from cqlib.qis.state import Statevector

from cqlib_algorithms.algorithms import VQD, VQDResult
from cqlib_algorithms.algorithms.variational.vqd import _DeflatedEstimator
from cqlib_algorithms.operators.ansatz import RealAmplitudesAnsatz
from cqlib_algorithms.operators.estimator import StatevectorEstimator
from cqlib_algorithms.operators.gradient import LinCombEstimatorGradient
from cqlib_algorithms.operators.optimizer import COBYLAOptimizer, GradientDescent
from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.operators.state_fidelity import (
    CircuitFidelityRequest,
    CircuitFidelityResult,
    ComputeUncomputeFidelity,
)
from cqlib_algorithms.protocols import (
    Ansatz,
    Estimator,
    EstimatorRequest,
    EstimatorResult,
    Optimizer,
    OptimizerRequest,
    OptimizerResult,
    OptimizerStatus,
    StateFidelity,
)


class _ExactFidelity(StateFidelity[CircuitFidelityRequest, CircuitFidelityResult]):
    def run(
        self,
        requests: Sequence[CircuitFidelityRequest],
        /,
    ) -> list[CircuitFidelityResult]:
        results: list[CircuitFidelityResult] = []
        for request in requests:
            request.validate()
            first = request.circuit_a.assign_parameters(
                dict(request.parameter_bindings_a)
            )
            second = request.circuit_b.assign_parameters(
                dict(request.parameter_bindings_b)
            )
            first_data = np.asarray(Statevector.from_circuit(first).data)
            second_data = np.asarray(Statevector.from_circuit(second).data)
            fidelity = float(abs(np.vdot(first_data, second_data)) ** 2)
            results.append(CircuitFidelityResult(fidelity, 0.0))
        return results


class _RecordingFidelity(StateFidelity[CircuitFidelityRequest, CircuitFidelityResult]):
    def __init__(self) -> None:
        self._delegate = ComputeUncomputeFidelity(StatevectorSampler())
        self.seeds: list[int | None] = []
        self.results: list[CircuitFidelityResult] = []

    def run(
        self,
        requests: Sequence[CircuitFidelityRequest],
        /,
    ) -> list[CircuitFidelityResult]:
        request_list = list(requests)
        self.seeds.extend(request.seed for request in request_list)
        results = self._delegate.run(request_list)
        self.results.extend(results)
        return results


class _EmptyEstimator(Estimator):
    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        return []


class _EmptyFidelity(StateFidelity[CircuitFidelityRequest, CircuitFidelityResult]):
    def run(
        self,
        requests: Sequence[CircuitFidelityRequest],
        /,
    ) -> list[CircuitFidelityResult]:
        return []


class _EmptyOptimizer(Optimizer):
    def run(
        self,
        estimator: Estimator,
        requests: Sequence[OptimizerRequest],
        /,
    ) -> list[OptimizerResult]:
        return []


class _NoEvaluationOptimizer(Optimizer):
    def run(
        self,
        estimator: Estimator,
        requests: Sequence[OptimizerRequest],
        /,
    ) -> list[OptimizerResult]:
        request_list = list(requests)
        return [
            OptimizerResult(
                request.parameter_names,
                tuple(
                    request.estimator_request.parameter_bindings[name]
                    for name in request.parameter_names
                ),
                0.0,
                0,
                OptimizerStatus.CONVERGED,
            )
            for request in request_list
        ]


class _PresetOptimizer(Optimizer):
    def __init__(self, points: Sequence[Mapping[str, float]]) -> None:
        self._points = tuple(dict(point) for point in points)
        self.calls = 0

    def run(
        self,
        estimator: Estimator,
        requests: Sequence[OptimizerRequest],
        /,
    ) -> list[OptimizerResult]:
        request_list = list(requests)
        if len(request_list) != 1:
            raise ValueError("preset optimizer expects one request")
        request = request_list[0]
        point = self._points[self.calls]
        self.calls += 1
        base = request.estimator_request
        result = estimator.run(
            [
                EstimatorRequest(
                    base.circuit,
                    base.observable,
                    point,
                    base.shots,
                    base.seed,
                )
            ]
        )[0]
        return [
            OptimizerResult(
                request.parameter_names,
                tuple(point[name] for name in request.parameter_names),
                result.expectation,
                0,
                OptimizerStatus.CONVERGED,
            )
        ]


class _StaticAnsatz(Ansatz):
    def validate(self) -> None:
        return None

    def build_circuit(self, prefix: str) -> Circuit:
        return Circuit(1)

    def num_parameters(self) -> int:
        return 0

    def num_qubits(self) -> int:
        return 1


def _preset_vqd(
    points: Sequence[Mapping[str, float]],
    *,
    betas: Sequence[float] | None = (4.0,),
    callback=None,
    fidelity: StateFidelity[CircuitFidelityRequest, CircuitFidelityResult]
    | None = None,
) -> VQD:
    return VQD(
        StatevectorEstimator(),
        _ExactFidelity() if fidelity is None else fidelity,
        RealAmplitudesAnsatz(1, reps=0),
        _PresetOptimizer(points),
        k=len(points),
        betas=betas,
        callback=callback,
    )


@pytest.mark.reference
def test_vqd_matches_independent_dense_spectrum_and_orthogonal_states() -> None:
    algorithm = _preset_vqd(
        [{"theta_0": math.pi}, {"theta_0": 0.0}],
        betas=(4.0,),
    )

    result = algorithm.compute_eigenvalues(
        PauliString.from_str("Z"),
        [{"theta_0": 2.2}, {"theta_0": 0.3}],
    )

    reference = np.linalg.eigvalsh(np.diag([1.0, -1.0]))
    assert result.eigenvalues == pytest.approx(tuple(reference), abs=1e-12)
    states = [
        np.asarray(Statevector.from_circuit(circuit).data)
        for circuit in result.optimal_circuits
    ]
    assert abs(np.vdot(states[0], states[1])) ** 2 == pytest.approx(0.0, abs=1e-12)
    assert result.optimal_values == pytest.approx((-1.0, 1.0), abs=1e-12)


@pytest.mark.reference
def test_vqd_real_optimizer_reaches_both_one_qubit_eigenstates() -> None:
    algorithm = VQD(
        StatevectorEstimator(),
        _ExactFidelity(),
        RealAmplitudesAnsatz(1, reps=0),
        COBYLAOptimizer(maxiter=100, tolerance=1e-9),
        k=2,
        betas=(4.0,),
    )

    result = algorithm.compute_eigenvalues(
        PauliString.from_str("Z"),
        [{"theta_0": 2.5}, {"theta_0": 0.3}],
        seed=7,
    )

    reference = np.linalg.eigvalsh(np.diag([1.0, -1.0]))
    assert result.eigenvalues == pytest.approx(tuple(reference), abs=1e-9)
    assert all(item.converged for item in result.optimizer_results)


@pytest.mark.reference
def test_vqd_preserves_distinct_states_in_a_degenerate_two_qubit_level() -> None:
    operator = Hamiltonian(2)
    operator.add_term(PauliString.from_str("ZI"), 1.0)
    operator.add_term(PauliString.from_str("IZ"), 1.0)
    points = (
        {"theta_0": math.pi, "theta_1": math.pi},
        {"theta_0": math.pi, "theta_1": 0.0},
        {"theta_0": 0.0, "theta_1": math.pi},
    )
    algorithm = VQD(
        StatevectorEstimator(),
        _ExactFidelity(),
        RealAmplitudesAnsatz(2, reps=0),
        _PresetOptimizer(points),
        k=3,
        betas=(6.0, 6.0),
    )

    result = algorithm.compute_eigenvalues(operator, points)

    reference = np.linalg.eigvalsh(operator.to_matrix())[:3]
    assert result.eigenvalues == pytest.approx(tuple(reference), abs=1e-12)
    states = [
        np.asarray(Statevector.from_circuit(circuit).data)
        for circuit in result.optimal_circuits
    ]
    for first in range(len(states)):
        for second in range(first):
            assert abs(np.vdot(states[first], states[second])) ** 2 == pytest.approx(
                0.0, abs=1e-12
            )


def test_vqd_reports_unpenalized_energy_and_operator_derived_beta() -> None:
    algorithm = _preset_vqd(
        [{"theta_0": math.pi}, {"theta_0": math.pi}],
        betas=None,
    )

    result = algorithm.compute_eigenvalues(
        PauliString.from_str("Z"),
        {"theta_0": 0.2},
    )

    assert result.betas == (10.0,)
    assert result.eigenvalues == pytest.approx((-1.0, -1.0), abs=1e-12)
    assert result.optimal_values == pytest.approx((-1.0, 9.0), abs=1e-12)


def test_vqd_default_beta_uses_multiterm_hamiltonian_coefficient_norm() -> None:
    operator = Hamiltonian(1)
    operator.add_term(PauliString.from_str("Z"), 2.0)
    operator.add_term(PauliString.from_str("X"), -0.5)
    algorithm = _preset_vqd(
        [{"theta_0": math.pi}, {"theta_0": 0.0}],
        betas=None,
    )

    result = algorithm.compute_eigenvalues(operator, {"theta_0": 0.2})

    assert result.betas == (25.0,)


def test_vqd_zero_hamiltonian_requires_explicit_beta_for_deflation() -> None:
    operator = Hamiltonian(1)
    default = _preset_vqd(
        [{"theta_0": math.pi}, {"theta_0": 0.0}],
        betas=None,
    )

    with pytest.raises(ValueError, match="zero operator requires explicit"):
        default.compute_eigenvalues(operator, {"theta_0": 0.2})

    explicit = _preset_vqd(
        [{"theta_0": math.pi}, {"theta_0": 0.0}],
        betas=(2.0,),
    ).compute_eigenvalues(operator, {"theta_0": 0.2})
    assert explicit.eigenvalues == (0.0, 0.0)
    assert explicit.betas == (2.0,)


def test_vqd_evaluates_auxiliaries_for_every_state_in_stable_order() -> None:
    algorithm = _preset_vqd(
        [{"theta_0": math.pi}, {"theta_0": 0.0}],
    )
    hamiltonian = Hamiltonian.from_pauli(PauliString.from_str("Z"))

    result = algorithm.compute_eigenvalues(
        hamiltonian,
        {"theta_0": 0.1},
        auxiliary_operators=(
            PauliString.from_str("Z"),
            PauliString.from_str("X"),
        ),
    )

    values = tuple(
        tuple(entry.expectation for entry in row) for row in result.auxiliary_results
    )
    assert np.asarray(values) == pytest.approx(
        np.asarray(((-1.0, 0.0), (1.0, 0.0))), abs=1e-12
    )


def test_vqd_callback_observes_immutable_points_and_level_order() -> None:
    observations = []

    def callback(count, parameters, value, level):
        observations.append((count, parameters, value, level))

    algorithm = _preset_vqd(
        [{"theta_0": math.pi}, {"theta_0": 0.0}],
        callback=callback,
    )
    algorithm.compute_eigenvalues(PauliString.from_str("Z"), {"theta_0": 0.2})

    assert [entry[0] for entry in observations] == [1, 2]
    assert [entry[3] for entry in observations] == [0, 1]
    with pytest.raises(TypeError):
        observations[0][1]["theta_0"] = 1.0


def test_vqd_integrates_compute_uncompute_with_reproducible_request_seeds() -> None:
    first_fidelity = _RecordingFidelity()
    first = _preset_vqd(
        [{"theta_0": 0.6}, {"theta_0": 1.8}],
        fidelity=first_fidelity,
    ).compute_eigenvalues(
        PauliString.from_str("Z"),
        {"theta_0": 0.1},
        fidelity_shots=257,
        seed=17,
    )
    second_fidelity = _RecordingFidelity()
    second = _preset_vqd(
        [{"theta_0": 0.6}, {"theta_0": 1.8}],
        fidelity=second_fidelity,
    ).compute_eigenvalues(
        PauliString.from_str("Z"),
        {"theta_0": 0.1},
        fidelity_shots=257,
        seed=17,
    )

    assert first_fidelity.seeds == second_fidelity.seeds
    assert first_fidelity.seeds and all(
        seed is not None for seed in first_fidelity.seeds
    )
    assert 0.0 < first_fidelity.results[0].fidelity < 1.0
    assert first_fidelity.results == second_fidelity.results
    assert first.eigenvalues == second.eigenvalues
    assert first.optimal_values == second.optimal_values


def test_deflated_estimator_rejects_width_changing_transformations() -> None:
    current = Circuit(2)
    request = EstimatorRequest(current, PauliString.from_str("ZZ"))
    deflated = _DeflatedEstimator(
        StatevectorEstimator(),
        _ExactFidelity(),
        (Circuit(1),),
        (2.0,),
        32,
        7,
        1,
        None,
    )

    with pytest.raises(ValueError, match="preserve the ansatz circuit width"):
        deflated.run([request])


def test_vqd_rejects_current_ancilla_based_lin_comb_gradient() -> None:
    algorithm = VQD(
        StatevectorEstimator(),
        _ExactFidelity(),
        RealAmplitudesAnsatz(1, reps=0),
        GradientDescent(LinCombEstimatorGradient(), maxiter=1),
        k=2,
        betas=(4.0,),
    )

    with pytest.raises(ValueError, match="preserve the ansatz circuit width"):
        algorithm.compute_eigenvalues(
            PauliString.from_str("Z"),
            [{"theta_0": 2.5}, {"theta_0": 0.3}],
        )


@pytest.mark.parametrize("k", [True, 0, -1, 1.5])
def test_vqd_rejects_invalid_state_counts(k) -> None:
    with pytest.raises((TypeError, ValueError)):
        VQD(
            StatevectorEstimator(),
            _ExactFidelity(),
            RealAmplitudesAnsatz(1, reps=0),
            _PresetOptimizer([{"theta_0": 0.0}]),
            k=k,
        )


@pytest.mark.parametrize(
    "betas",
    [(), (1.0, 2.0), (0.0,), (-1.0,), (float("nan"),), (True,)],
)
def test_vqd_rejects_invalid_penalties(betas) -> None:
    with pytest.raises((TypeError, ValueError)):
        VQD(
            StatevectorEstimator(),
            _ExactFidelity(),
            RealAmplitudesAnsatz(1, reps=0),
            _PresetOptimizer([{"theta_0": 0.0}, {"theta_0": 0.0}]),
            betas=betas,
        )


def test_vqd_rejects_more_states_than_the_hilbert_space() -> None:
    algorithm = VQD(
        StatevectorEstimator(),
        _ExactFidelity(),
        RealAmplitudesAnsatz(1, reps=0),
        _PresetOptimizer([{"theta_0": 0.0}] * 3),
        k=3,
        betas=(2.0, 2.0),
    )

    with pytest.raises(ValueError, match="Hilbert-space"):
        algorithm.compute_eigenvalues(
            PauliString.from_str("Z"),
            [{"theta_0": 0.0}] * 3,
        )


def test_vqd_rejects_static_ansatz_for_multiple_states() -> None:
    algorithm = VQD(
        StatevectorEstimator(),
        _ExactFidelity(),
        _StaticAnsatz(),
        _PresetOptimizer([{}, {}]),
        k=2,
        betas=(2.0,),
    )

    with pytest.raises(ValueError, match="active ansatz parameter"):
        algorithm.compute_eigenvalues(PauliString.from_str("Z"), {})


@pytest.mark.parametrize(
    "initial_points",
    [[], [{"theta_0": 0.0}], [{"other": 0.0}, {"theta_0": 0.0}]],
)
def test_vqd_rejects_invalid_initial_points(initial_points) -> None:
    algorithm = _preset_vqd(
        [{"theta_0": math.pi}, {"theta_0": 0.0}],
    )

    with pytest.raises((TypeError, ValueError)):
        algorithm.compute_eigenvalues(
            PauliString.from_str("Z"),
            initial_points,
        )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"shots": True}, TypeError),
        ({"shots": 0}, ValueError),
        ({"fidelity_shots": True}, TypeError),
        ({"fidelity_shots": 0}, ValueError),
        ({"seed": True}, TypeError),
        ({"seed": -1}, ValueError),
        ({"auxiliary_operators": object()}, TypeError),
        ({"auxiliary_operators": (object(),)}, TypeError),
        (
            {"auxiliary_operators": (PauliString.from_str("ZZ"),)},
            ValueError,
        ),
    ],
)
def test_vqd_rejects_invalid_execution_options(kwargs, error) -> None:
    algorithm = _preset_vqd([{"theta_0": math.pi}], betas=())

    with pytest.raises(error):
        algorithm.compute_eigenvalues(
            PauliString.from_str("Z"),
            {"theta_0": 0.2},
            **kwargs,
        )


def test_vqd_rejects_nonhermitian_auxiliary_operator() -> None:
    auxiliary = Hamiltonian(1)
    auxiliary.add_term(PauliString.from_str("X"), 1.0j)
    algorithm = _preset_vqd([{"theta_0": math.pi}], betas=())

    with pytest.raises(ValueError, match="Hermitian"):
        algorithm.compute_eigenvalues(
            PauliString.from_str("Z"),
            {"theta_0": 0.2},
            auxiliary_operators=(auxiliary,),
        )


def test_vqd_rejects_empty_optimizer_result_batch() -> None:
    algorithm = VQD(
        StatevectorEstimator(),
        _ExactFidelity(),
        RealAmplitudesAnsatz(1, reps=0),
        _EmptyOptimizer(),
        k=1,
        betas=(),
    )

    with pytest.raises(ValueError, match="exactly one result"):
        algorithm.compute_eigenvalues(PauliString.from_str("Z"), {"theta_0": 0.2})


def test_vqd_rejects_empty_estimator_result_batch() -> None:
    algorithm = VQD(
        _EmptyEstimator(),
        _ExactFidelity(),
        RealAmplitudesAnsatz(1, reps=0),
        _NoEvaluationOptimizer(),
        k=1,
        betas=(),
    )

    with pytest.raises(ValueError, match="one result per request"):
        algorithm.compute_eigenvalues(PauliString.from_str("Z"), {"theta_0": 0.2})


def test_vqd_rejects_empty_fidelity_result_batch() -> None:
    algorithm = _preset_vqd(
        [{"theta_0": math.pi}, {"theta_0": 0.0}],
        fidelity=_EmptyFidelity(),
    )

    with pytest.raises(ValueError, match="one result per request"):
        algorithm.compute_eigenvalues(PauliString.from_str("Z"), {"theta_0": 0.2})


def test_vqd_result_copies_parameter_mappings_and_has_concise_repr() -> None:
    point = {"theta_0": math.pi}
    result = _preset_vqd([point], betas=()).compute_eigenvalues(
        PauliString.from_str("Z"),
        point,
    )
    point["theta_0"] = 0.0

    assert result.optimal_parameters[0]["theta_0"] == pytest.approx(math.pi)
    assert isinstance(result, VQDResult)
    assert "num_states=1" in repr(result)
    assert "Circuit" not in repr(result)


def test_vqd_does_not_mutate_initial_points_or_operator() -> None:
    point = {"theta_0": 0.2}
    operator = Hamiltonian.from_pauli(PauliString.from_str("Z"))
    matrix = np.asarray(operator.to_matrix()).copy()

    _preset_vqd([{"theta_0": math.pi}], betas=()).compute_eigenvalues(
        operator,
        point,
    )

    assert point == {"theta_0": 0.2}
    assert np.asarray(operator.to_matrix()) == pytest.approx(matrix, abs=0.0)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda result: replace(result, betas=(1.0,)),
        lambda result: replace(result, optimal_values=(1.0,)),
        lambda result: replace(result, optimal_parameters=({"other": 0.0},)),
    ],
    ids=("beta-count", "optimizer-value", "parameter-names"),
)
def test_vqd_result_rejects_inconsistent_metadata(mutation) -> None:
    result = _preset_vqd([{"theta_0": math.pi}], betas=()).compute_eigenvalues(
        PauliString.from_str("Z"),
        {"theta_0": math.pi},
    )

    with pytest.raises((TypeError, ValueError)):
        mutation(result)


def test_vqd_result_rejects_unequal_auxiliary_row_widths() -> None:
    result = _preset_vqd(
        [{"theta_0": math.pi}, {"theta_0": 0.0}],
    ).compute_eigenvalues(
        PauliString.from_str("Z"),
        {"theta_0": 0.2},
    )

    with pytest.raises(ValueError, match="equal lengths"):
        replace(
            result,
            auxiliary_results=((), (EstimatorResult(0.0, 0.0),)),
        )
