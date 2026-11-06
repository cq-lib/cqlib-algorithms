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

"""Composition and exact-reference tests for electronic ground states."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from cqlib import Circuit

from cqlib_algorithms.algorithms import (
    ElectronicGroundState,
    ElectronicGroundStateResult,
)
from cqlib_algorithms.extension.electronic_structure import (
    ElectronicEnergy,
    ElectronicIntegrals,
    ElectronicOccupation,
    ElectronicSystem,
    SpinResolvedElectronicIntegrals,
)
from cqlib_algorithms.operators.estimator import StatevectorEstimator
from cqlib_algorithms.protocols import (
    Ansatz,
    Estimator,
    EstimatorRequest,
    EstimatorResult,
    Optimizer,
    OptimizerRequest,
    OptimizerResult,
    OptimizerStatus,
)


def _system() -> ElectronicSystem:
    energies = np.array([-0.5, 0.5])
    two_body = np.zeros((2, 2, 2, 2))
    for index in (
        (1, 0, 1, 0),
        (0, 1, 1, 0),
        (1, 0, 0, 1),
        (0, 1, 0, 1),
    ):
        two_body[index] = 0.2
    one_body = np.diag(energies)
    one_body -= 2.0 * two_body[:, :, 0, 0]
    one_body += two_body[:, 0, 0, :]
    return ElectronicSystem(
        ElectronicIntegrals(0.4, one_body, two_body),
        ElectronicOccupation.closed_shell(2, 2),
        energies,
        -0.6,
        nuclear_repulsion_energy=0.25,
        inactive_energy=0.15,
    )


class _FixedPointOptimizer(Optimizer):
    def run(
        self,
        estimator: Estimator,
        requests: Sequence[OptimizerRequest],
        /,
    ) -> list[OptimizerResult]:
        batch = list(requests)
        estimates = estimator.run([request.estimator_request for request in batch])
        return [
            OptimizerResult(
                request.parameter_names,
                tuple(
                    request.estimator_request.parameter_bindings[name]
                    for name in request.parameter_names
                ),
                estimate.expectation,
                0,
                OptimizerStatus.STEP_TOLERANCE,
            )
            for request, estimate in zip(batch, estimates, strict=True)
        ]


def _exact_uccsd_point(system: ElectronicSystem) -> tuple[dict[str, float], float]:
    matrix = ElectronicEnergy(system.integrals).to_qubit_hamiltonian().to_matrix()
    reference_index = 0b0011
    double_index = 0b1100
    block = matrix[
        np.ix_((reference_index, double_index), (reference_index, double_index))
    ]
    values, vectors = np.linalg.eigh(block)
    ground = vectors[:, 0]
    if ground[0].real < 0.0:
        ground = -ground

    state = reference_index
    transition_sign = 1.0
    for create, mode in ((False, 0), (False, 1), (True, 3), (True, 2)):
        occupied = bool(state & (1 << mode))
        assert occupied != create
        transition_sign *= -1.0 if (state & ((1 << mode) - 1)).bit_count() % 2 else 1.0
        state = state | (1 << mode) if create else state & ~(1 << mode)
    assert state == double_index
    theta = float(np.arctan2(ground[1].real / transition_sign, ground[0].real))
    return (
        {
            "chem_theta_0": 0.0,
            "chem_theta_1": 0.0,
            "chem_theta_2": theta,
        },
        float(values[0]),
    )


def test_ground_state_reaches_independent_two_determinant_reference():
    system = _system()
    point, reference_energy = _exact_uccsd_point(system)

    result = ElectronicGroundState(
        StatevectorEstimator(),
        _FixedPointOptimizer(),
        prefix="chem",
    ).compute(system, point)

    assert isinstance(result, ElectronicGroundStateResult)
    assert result.total_energy == pytest.approx(reference_energy, abs=1e-12)
    assert result.hamiltonian_constant == pytest.approx(0.4, abs=1e-12)
    assert result.active_space_energy == pytest.approx(
        reference_energy - 0.4,
        abs=1e-12,
    )
    assert result.nuclear_repulsion_energy == pytest.approx(0.25, abs=1e-12)
    assert result.inactive_energy == pytest.approx(0.15, abs=1e-12)
    assert result.electronic_energy == pytest.approx(
        reference_energy - 0.25,
        abs=1e-12,
    )
    assert result.correlation_energy == pytest.approx(
        reference_energy - system.hartree_fock_energy, abs=1e-12
    )
    assert result.particle_number.expectation == pytest.approx(2.0, abs=1e-12)
    assert result.magnetization.expectation == pytest.approx(0.0, abs=1e-12)
    assert result.angular_momentum.expectation == pytest.approx(0.0, abs=1e-12)
    assert all(item.variance == 0.0 for item in result.auxiliary_results.values())
    assert result.mapper is not None


def test_open_shell_ground_state_uses_alpha_beta_occupation_and_spin_observables():
    system = ElectronicSystem(
        ElectronicIntegrals(
            0.2,
            np.diag([-1.0, 0.7]),
            np.zeros((2,) * 4),
        ),
        ElectronicOccupation(2, 1, 0),
        np.array([-1.0, 0.7]),
        -0.8,
    )

    result = ElectronicGroundState(
        StatevectorEstimator(),
        _FixedPointOptimizer(),
        prefix="open",
    ).compute(system, {"open_theta_0": 0.0})

    assert result.total_energy == pytest.approx(-0.8, abs=1e-12)
    assert result.particle_number.expectation == pytest.approx(1.0, abs=1e-12)
    assert result.magnetization.expectation == pytest.approx(0.5, abs=1e-12)
    assert result.angular_momentum.expectation == pytest.approx(0.75, abs=1e-12)


@pytest.mark.reference
def test_uhf_ground_state_uses_spin_resolved_energy_and_overlap():
    angle = 0.34
    overlap = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    system = ElectronicSystem(
        SpinResolvedElectronicIntegrals(
            0.3,
            np.diag([-1.0, 0.7]),
            np.diag([-0.8, 0.9]),
            np.zeros((2,) * 4),
            np.zeros((2,) * 4),
            np.zeros((2,) * 4),
        ),
        ElectronicOccupation(2, 1, 1),
        np.array([-1.0, 0.7]),
        -1.5,
        np.array([-0.8, 0.9]),
        overlap,
    )

    result = ElectronicGroundState(
        StatevectorEstimator(),
        _FixedPointOptimizer(),
        prefix="uhf",
    ).compute(
        system,
        {
            "uhf_theta_0": 0.0,
            "uhf_theta_1": 0.0,
            "uhf_theta_2": 0.0,
        },
    )

    assert result.total_energy == pytest.approx(-1.5, abs=1e-12)
    assert result.particle_number.expectation == pytest.approx(2.0, abs=1e-12)
    assert result.magnetization.expectation == pytest.approx(0.0, abs=1e-12)
    assert result.angular_momentum.expectation == pytest.approx(
        np.sin(angle) ** 2,
        abs=1e-12,
    )


class _RecordingEstimator(Estimator):
    def __init__(self, results: list[EstimatorResult] | None = None) -> None:
        self.batches: list[list[EstimatorRequest]] = []
        self.results = results

    def run(self, requests: Sequence[EstimatorRequest], /) -> list[EstimatorResult]:
        batch = list(requests)
        self.batches.append(batch)
        if self.results is not None:
            return self.results
        return [EstimatorResult(float(index), 0.25) for index, _ in enumerate(batch)]


class _RecordingOptimizer(Optimizer):
    def __init__(self) -> None:
        self.estimator: Estimator | None = None
        self.requests: list[OptimizerRequest] = []

    def run(
        self,
        estimator: Estimator,
        requests: Sequence[OptimizerRequest],
        /,
    ) -> list[OptimizerResult]:
        self.estimator = estimator
        self.requests = list(requests)
        return [
            OptimizerResult(
                request.parameter_names,
                tuple(
                    request.estimator_request.parameter_bindings[name]
                    for name in request.parameter_names
                ),
                -1.25,
                3,
                OptimizerStatus.MAX_ITERATIONS,
            )
            for request in self.requests
        ]


def test_ground_state_reuses_vqe_and_batches_auxiliary_observables():
    estimator = _RecordingEstimator()
    optimizer = _RecordingOptimizer()
    algorithm = ElectronicGroundState(estimator, optimizer, prefix="theta")

    result = algorithm.compute(_system(), shots=123)

    assert optimizer.estimator is estimator
    assert len(optimizer.requests) == 1
    request = optimizer.requests[0]
    assert request.parameter_names == (
        "theta_theta_0",
        "theta_theta_1",
        "theta_theta_2",
    )
    assert set(request.estimator_request.parameter_bindings.values()) == {0.0}
    assert request.estimator_request.shots == 123
    assert len(estimator.batches) == 1
    assert len(estimator.batches[0]) == 3
    assert all(
        item.circuit is result.vqe_result.optimal_circuit
        for item in estimator.batches[0]
    )
    assert all(item.shots == 123 for item in estimator.batches[0])
    assert result.particle_number.expectation == 0.0
    assert result.magnetization.expectation == 1.0
    assert result.angular_momentum.expectation == 2.0


class _WrongWidthAnsatz(Ansatz):
    def validate(self) -> None:
        return None

    def build_circuit(self, prefix: str) -> Circuit:
        return Circuit(1)

    def num_parameters(self) -> int:
        return 0

    def num_qubits(self) -> int:
        return 1


def test_ground_state_rejects_invalid_inputs_and_widths():
    estimator = _RecordingEstimator()
    optimizer = _RecordingOptimizer()
    with pytest.raises(TypeError, match="estimator"):
        ElectronicGroundState(object(), optimizer)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="optimizer"):
        ElectronicGroundState(estimator, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="mapper"):
        ElectronicGroundState(estimator, optimizer, mapper=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="prefix"):
        ElectronicGroundState(estimator, optimizer, prefix=1)  # type: ignore[arg-type]

    algorithm = ElectronicGroundState(estimator, optimizer)
    with pytest.raises(TypeError, match="system"):
        algorithm.compute(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ansatz"):
        algorithm.compute(_system(), ansatz=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="same width"):
        algorithm.compute(_system(), ansatz=_WrongWidthAnsatz())


@pytest.mark.parametrize(
    ("results", "error", "message"),
    [
        ([EstimatorResult(0.0)], ValueError, "one result"),
        (
            [EstimatorResult(0.0), EstimatorResult(0.0), object()],
            TypeError,
            "EstimatorResult",
        ),
    ],
)
def test_ground_state_rejects_invalid_auxiliary_estimator_responses(
    results, error, message
):
    algorithm = ElectronicGroundState(
        _RecordingEstimator(results),
        _RecordingOptimizer(),
    )
    with pytest.raises(error, match=message):
        algorithm.compute(_system())


def test_ground_state_result_validates_and_freezes_auxiliary_results():
    system = _system()
    point, _ = _exact_uccsd_point(system)
    result = ElectronicGroundState(
        StatevectorEstimator(),
        _FixedPointOptimizer(),
        prefix="chem",
    ).compute(system, point)

    with pytest.raises(TypeError):
        result.auxiliary_results["particle_number"] = EstimatorResult(0.0)  # type: ignore[index]
    with pytest.raises(ValueError, match="must contain"):
        ElectronicGroundStateResult(system, result.vqe_result, {})
    with pytest.raises(TypeError, match="mapper"):
        ElectronicGroundStateResult(
            system,
            result.vqe_result,
            result.auxiliary_results,
            mapper=object(),  # type: ignore[arg-type]
        )
    no_reference = ElectronicSystem(system.integrals, system.occupation)
    no_reference_result = ElectronicGroundStateResult(
        no_reference,
        result.vqe_result,
        result.auxiliary_results,
    )
    assert no_reference_result.correlation_energy is None
    assert no_reference_result.nuclear_repulsion_energy is None
    assert no_reference_result.inactive_energy is None
    assert no_reference_result.electronic_energy is None
