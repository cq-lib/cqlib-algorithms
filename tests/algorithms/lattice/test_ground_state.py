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

"""End-to-end and exact-reference tests for lattice ground states."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from cqlib import Circuit

from cqlib_algorithms.algorithms.lattice import (
    FermionicLatticeMeasurements,
    LatticeGroundState,
    LatticeGroundStateResult,
    SpinLatticeMeasurements,
)
from cqlib_algorithms.extension.lattice import (
    FermiHubbardModel,
    FermionicLatticeConfiguration,
    FermionicSector,
    HeisenbergModel,
    IsingModel,
    Lattice,
    LatticeSystem,
    LineLattice,
    SpinLatticeConfiguration,
    SpinMagnetizationSector,
)
from cqlib_algorithms.operators.ansatz import EfficientSU2Ansatz
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


class _FixedPointOptimizer(Optimizer):
    """Evaluate and return the supplied initial point without updating it."""

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


def _fermionic_system() -> tuple[
    LatticeSystem,
    FermionicLatticeConfiguration,
    dict[str, float],
    np.ndarray,
    float,
]:
    lattice = Lattice.from_nodes_and_edges(
        2,
        ((0, 0, 0.3), (0, 1, -1.0), (1, 1, -0.3)),
    )
    model = FermiHubbardModel(lattice)
    system = LatticeSystem(model, FermionicSector(2, 1, 0))
    configuration = FermionicLatticeConfiguration(2, (0,), ())

    independent_block = np.array(((0.3, -1.0), (-1.0, -0.3)))
    eigenvalues, eigenvectors = np.linalg.eigh(independent_block)
    ground = eigenvectors[:, 0]
    if ground[0] < 0.0:
        ground = -ground
    hopping_angle = float(np.arctan2(ground[1], ground[0]))
    diagonal_angle = -np.pi / 1.2
    point = {
        "lat_0": hopping_angle,
        "lat_1": 0.0,
        "lat_2": diagonal_angle,
    }
    return system, configuration, point, ground, float(eigenvalues[0])


@pytest.mark.reference
def test_fermi_hubbard_closure_matches_independent_sector_diagonalization():
    system, configuration, point, ground, exact_energy = _fermionic_system()

    result = LatticeGroundState(
        StatevectorEstimator(),
        _FixedPointOptimizer(),
        prefix="lat",
    ).compute(
        system,
        point,
        configuration=configuration,
    )

    assert isinstance(result, LatticeGroundStateResult)
    assert isinstance(result.measurements, FermionicLatticeMeasurements)
    assert result.variational_energy == pytest.approx(exact_energy, abs=1e-12)
    assert result.measurements.total_particle_number.expectation == pytest.approx(1.0)
    assert result.measurements.spin_up_particle_number.expectation == pytest.approx(1.0)
    assert result.measurements.spin_down_particle_number.expectation == pytest.approx(
        0.0
    )
    assert result.measurements.magnetization.expectation == pytest.approx(0.5)
    expected_occupations = tuple(float(abs(value) ** 2) for value in ground)
    assert tuple(
        value.expectation for value in result.measurements.site_occupations
    ) == pytest.approx(expected_occupations, abs=1e-12)
    assert tuple(
        value.expectation for value in result.measurements.double_occupations
    ) == pytest.approx((0.0, 0.0), abs=1e-12)
    assert result.system.sector == FermionicSector(2, 1, 0)


def _spin_system() -> tuple[
    LatticeSystem,
    SpinLatticeConfiguration,
    dict[str, float],
    np.ndarray,
    float,
]:
    model = HeisenbergModel(
        LineLattice(2, edge_parameter=1.0),
        coupling=(1.0, 1.0, 0.5),
        external_field=((0.0, 0.0, 0.2), (0.0, 0.0, -0.1)),
    )
    system = LatticeSystem(model, SpinMagnetizationSector(2, 1))
    configuration = SpinLatticeConfiguration(2, (0,))

    independent_block = np.array(((-0.8, 2.0), (2.0, -0.2)))
    eigenvalues, eigenvectors = np.linalg.eigh(independent_block)
    ground = eigenvectors[:, 0]
    exchange_angle = -2.0 * np.arctan(abs(ground[0] / ground[1]))
    point = {
        "lat_theta_0": 0.0,
        "lat_theta_1": 0.0,
        "lat_theta_2": float(exchange_angle),
        "lat_theta_3": np.pi / 4.0,
        "lat_theta_4": -np.pi / 4.0,
    }
    return system, configuration, point, ground, float(eigenvalues[0])


@pytest.mark.reference
def test_spin_closure_matches_independent_fixed_magnetization_block():
    system, configuration, point, ground, exact_energy = _spin_system()

    result = LatticeGroundState(
        StatevectorEstimator(),
        _FixedPointOptimizer(),
        prefix="lat",
    ).compute(
        system,
        point,
        configuration=configuration,
    )

    assert isinstance(result.measurements, SpinLatticeMeasurements)
    assert result.variational_energy == pytest.approx(exact_energy, abs=1e-12)
    assert result.measurements.total_magnetization.expectation == pytest.approx(
        0.0, abs=1e-12
    )
    probabilities = np.abs(ground) ** 2
    expected_sites = (
        0.5 * (probabilities[1] - probabilities[0]),
        0.5 * (probabilities[0] - probabilities[1]),
    )
    assert tuple(
        value.expectation for value in result.measurements.site_magnetizations
    ) == pytest.approx(expected_sites, abs=1e-12)

    expected_xy = float(2.0 * ground[0] * ground[1])
    expected_correlations = (expected_xy, expected_xy, -1.0)
    assert tuple(
        item.result.expectation for item in result.measurements.correlations
    ) == pytest.approx(expected_correlations, abs=1e-12)
    assert tuple(
        (item.source, item.target, item.axis)
        for item in result.measurements.correlations
    ) == ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z"))
    expected_interaction = 2.0 * expected_xy + 0.5 * -1.0
    assert result.measurements.interaction_energy.expectation == pytest.approx(
        expected_interaction,
        abs=1e-12,
    )


@pytest.mark.reference
def test_nonconserving_spin_model_accepts_explicit_general_ansatz():
    model = IsingModel(
        Lattice.from_nodes_and_edges(1, ()),
        transverse_field=-1.0,
    )
    system = LatticeSystem(model)
    ansatz = EfficientSU2Ansatz(1, reps=0)

    result = LatticeGroundState(
        StatevectorEstimator(),
        _FixedPointOptimizer(),
        prefix="lat",
    ).compute(
        system,
        {"lat_0": np.pi / 2.0, "lat_1": 0.0},
        ansatz=ansatz,
    )

    assert result.variational_energy == pytest.approx(-1.0, abs=1e-12)
    assert isinstance(result.measurements, SpinLatticeMeasurements)
    assert result.measurements.total_magnetization.expectation == pytest.approx(
        0.0, abs=1e-12
    )
    assert result.measurements.interaction_energy.expectation == pytest.approx(
        0.0, abs=1e-12
    )
    assert result.measurements.correlations == ()


class _RecordingEstimator(Estimator):
    def __init__(self, response: object | None = None) -> None:
        self.batches: list[list[EstimatorRequest]] = []
        self.response = response

    def run(self, requests: Sequence[EstimatorRequest], /) -> list[EstimatorResult]:
        batch = list(requests)
        self.batches.append(batch)
        if self.response is not None:
            return self.response  # type: ignore[return-value]
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


def test_auxiliary_observables_are_one_ordered_batch_with_shots():
    system, configuration, _, _, _ = _fermionic_system()
    estimator = _RecordingEstimator()
    optimizer = _RecordingOptimizer()
    algorithm = LatticeGroundState(estimator, optimizer, prefix="theta")

    assert algorithm.estimator is estimator
    assert algorithm.optimizer is optimizer
    assert algorithm.prefix == "theta"

    result = algorithm.compute(system, configuration=configuration, shots=321)

    assert optimizer.estimator is estimator
    assert len(optimizer.requests) == 1
    objective = optimizer.requests[0].estimator_request
    assert objective.shots == 321
    assert set(objective.parameter_bindings.values()) == {0.0}
    assert len(estimator.batches) == 1
    assert len(estimator.batches[0]) == 8
    assert all(item.shots == 321 for item in estimator.batches[0])
    assert all(
        item.circuit is result.vqe_result.optimal_circuit
        for item in estimator.batches[0]
    )
    assert isinstance(result.measurements, FermionicLatticeMeasurements)
    assert result.measurements.total_particle_number.expectation == 0.0
    assert result.measurements.spin_up_particle_number.expectation == 1.0
    assert tuple(
        value.expectation for value in result.measurements.site_occupations
    ) == (4.0, 5.0)
    assert tuple(
        value.expectation for value in result.measurements.double_occupations
    ) == (6.0, 7.0)


@pytest.mark.parametrize("family", ["fermion", "spin"])
def test_default_ansatz_rejects_disconnected_nontrivial_sector(family: str):
    if family == "fermion":
        model = FermiHubbardModel(
            Lattice.from_nodes_and_edges(2, ((0, 0, 1.0), (1, 1, -1.0)))
        )
        system = LatticeSystem(model, FermionicSector(2, 1, 0))
        configuration = FermionicLatticeConfiguration(2, (0,), ())
    else:
        model = HeisenbergModel(
            Lattice.from_nodes_and_edges(2, ()),
            external_field=((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        )
        system = LatticeSystem(model, SpinMagnetizationSector(2, 1))
        configuration = SpinLatticeConfiguration(2, (0,))

    with pytest.raises(ValueError, match="connected nonzero lattice-edge graph"):
        LatticeGroundState(_RecordingEstimator(), _RecordingOptimizer()).compute(
            system,
            configuration=configuration,
        )


def test_unique_sector_is_not_over_rejected_on_a_disconnected_graph():
    model = FermiHubbardModel(
        Lattice.from_nodes_and_edges(2, ((0, 0, 0.2), (1, 1, -0.3)))
    )
    system = LatticeSystem(model, FermionicSector(2, 2, 0))
    configuration = FermionicLatticeConfiguration(2, (0, 1), ())

    result = LatticeGroundState(
        StatevectorEstimator(),
        _FixedPointOptimizer(),
    ).compute(system, configuration=configuration)

    assert result.variational_energy == pytest.approx(-0.1, abs=1e-12)


class _WrongWidthAnsatz(Ansatz):
    def validate(self) -> None:
        return None

    def build_circuit(self, prefix: str) -> Circuit:
        return Circuit(1)

    def num_parameters(self) -> int:
        return 0

    def num_qubits(self) -> int:
        return 1


def test_algorithm_rejects_ambiguous_or_incompatible_inputs():
    estimator = _RecordingEstimator()
    optimizer = _RecordingOptimizer()
    with pytest.raises(TypeError, match="estimator"):
        LatticeGroundState(object(), optimizer)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="optimizer"):
        LatticeGroundState(estimator, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="prefix"):
        LatticeGroundState(estimator, optimizer, prefix=1)  # type: ignore[arg-type]

    algorithm = LatticeGroundState(estimator, optimizer)
    system, configuration, _, _, _ = _fermionic_system()
    with pytest.raises(TypeError, match="system"):
        algorithm.compute(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="configuration is required"):
        algorithm.compute(system)
    with pytest.raises(TypeError, match="reps"):
        algorithm.compute(
            system,
            configuration=configuration,
            reps=True,
        )
    with pytest.raises(ValueError, match="non-negative"):
        algorithm.compute(
            system,
            configuration=configuration,
            reps=-1,
        )
    with pytest.raises(TypeError, match="FermionicLatticeConfiguration"):
        algorithm.compute(
            system,
            configuration=SpinLatticeConfiguration(2, (0,)),
        )
    with pytest.raises(TypeError, match="ansatz"):
        algorithm.compute(system, ansatz=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="configuration must be omitted"):
        algorithm.compute(
            system,
            configuration=configuration,
            ansatz=_WrongWidthAnsatz(),
        )
    with pytest.raises(ValueError, match="reps applies only"):
        algorithm.compute(system, ansatz=_WrongWidthAnsatz(), reps=2)
    with pytest.raises(ValueError, match="same width"):
        algorithm.compute(system, ansatz=_WrongWidthAnsatz())

    spin = LatticeSystem(IsingModel(LineLattice(2), transverse_field=0.4))
    with pytest.raises(ValueError, match="requires a custom ansatz"):
        algorithm.compute(
            spin,
            configuration=SpinLatticeConfiguration(2, (0,)),
        )

    spin_system, _, _, _, _ = _spin_system()
    with pytest.raises(TypeError, match="SpinLatticeConfiguration"):
        algorithm.compute(
            spin_system,
            configuration=FermionicLatticeConfiguration(2, (0,), ()),
        )


@pytest.mark.parametrize(
    ("response", "error", "message"),
    [
        ([EstimatorResult(0.0)], ValueError, "one result"),
        (
            [EstimatorResult(0.0)] * 7 + [object()],
            TypeError,
            "EstimatorResult",
        ),
        ((EstimatorResult(0.0),) * 8, TypeError, "list"),
    ],
)
def test_algorithm_rejects_invalid_auxiliary_estimator_responses(
    response: object,
    error: type[Exception],
    message: str,
):
    system, configuration, _, _, _ = _fermionic_system()
    algorithm = LatticeGroundState(
        _RecordingEstimator(response),
        _RecordingOptimizer(),
    )

    with pytest.raises(error, match=message):
        algorithm.compute(system, configuration=configuration)
