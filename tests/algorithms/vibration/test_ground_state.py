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

"""Composition and independent-reference tests for vibrational ground states."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from cqlib import Circuit

from cqlib_algorithms.algorithms.vibration import (
    VibrationalGroundState,
    VibrationalGroundStateResult,
)
from cqlib_algorithms.extension.vibration import (
    ModalIntegrals,
    ModalSpace,
    ModalTransition,
    VibrationalEnergy,
    VibrationalSystem,
)
from cqlib_algorithms.operators.ansatz import UVCCSDAnsatz
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


def _projector_system(
    angles: tuple[float, float, float] = (0.31, -0.27, 0.42),
) -> tuple[VibrationalSystem, np.ndarray, dict[str, float]]:
    """Return H = 0.37 I - |psi><psi| without production mapping logic."""
    theta_0, theta_1, theta_2 = angles
    c0, c1, c2 = np.cos((theta_0, theta_1, theta_2))
    s0, s1, s2 = np.sin((theta_0, theta_1, theta_2))
    reference_amplitude = c0 * c1
    double_amplitude = s0 * s1
    target = np.array(
        (
            c2 * reference_amplitude - s2 * double_amplitude,
            s0 * c1,
            c0 * s1,
            s2 * reference_amplitude + c2 * double_amplitude,
        ),
        dtype=np.complex128,
    )
    np.testing.assert_allclose(np.linalg.norm(target), 1.0, rtol=0.0, atol=1e-14)

    space = ModalSpace((2, 2))
    configurations = ((0, 0), (1, 0), (0, 1), (1, 1))
    matrix = -np.outer(target, target.conj())
    terms = []
    for bra, bra_configuration in enumerate(configurations):
        for ket, ket_configuration in enumerate(configurations):
            coefficient = matrix[bra, ket]
            if coefficient == 0.0:
                continue
            transitions = tuple(
                ModalTransition(mode, bra_configuration[mode], ket_configuration[mode])
                for mode in range(space.num_modes)
            )
            terms.append((transitions, coefficient))
    system = VibrationalSystem(
        VibrationalEnergy(
            ModalIntegrals.from_terms(space, terms, constant=0.37),
        )
    )
    point = {
        "exact_theta_0": theta_0,
        "exact_theta_1": theta_1,
        "exact_theta_2": theta_2,
    }
    return system, target, point


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


@pytest.mark.reference
@pytest.mark.parametrize("entry", ["run", "compute"])
def test_ground_state_matches_independent_two_mode_projector_reference(entry):
    system, target, point = _projector_system()
    modal_matrix = 0.37 * np.eye(4) - np.outer(target, target.conj())
    eigenvalues = np.linalg.eigvalsh(modal_matrix)

    solver = VibrationalGroundState(
        StatevectorEstimator(),
        _FixedPointOptimizer(),
        prefix="exact",
    )
    result = getattr(solver, entry)(system, point)

    assert isinstance(result, VibrationalGroundStateResult)
    assert result.variational_energy == pytest.approx(eigenvalues[0], abs=1e-12)
    assert result.hamiltonian_constant == pytest.approx(0.37, abs=1e-15)
    assert result.nonconstant_energy == pytest.approx(-1.0, abs=1e-12)

    encoded = result.vqe_result.optimal_circuit.to_matrix()[:, 0]
    assert system.is_physical_statevector(encoded)
    physical = encoded[list(system.modal_space.physical_basis_indices)]
    phase = physical[0] / target[0]
    np.testing.assert_allclose(physical / phase, target, rtol=0.0, atol=1e-12)
    forbidden = [
        index
        for index in range(encoded.size)
        if not system.modal_space.is_physical_basis_index(index)
    ]
    np.testing.assert_allclose(encoded[forbidden], 0.0, rtol=0.0, atol=1e-12)

    probabilities = np.abs(target) ** 2
    expected_rows = (
        (probabilities[0] + probabilities[2], probabilities[1] + probabilities[3]),
        (probabilities[0] + probabilities[1], probabilities[2] + probabilities[3]),
    )
    for actual_row, expected_row in zip(
        result.measurements.modal_occupations,
        expected_rows,
        strict=True,
    ):
        np.testing.assert_allclose(
            [item.expectation for item in actual_row],
            expected_row,
            rtol=0.0,
            atol=1e-12,
        )
    np.testing.assert_allclose(
        [item.expectation for item in result.measurements.mode_occupations],
        (1.0, 1.0),
        rtol=0.0,
        atol=1e-12,
    )
    assert all(
        item.variance == 0.0
        for row in result.measurements.modal_occupations
        for item in row
    )
    assert all(item.variance == 0.0 for item in result.measurements.mode_occupations)
    np.testing.assert_allclose(
        [item.expectation for item in result.measurements.mode_leakages],
        (0.0, 0.0),
        rtol=0.0,
        atol=1e-12,
    )
    assert all(item.variance == 0.0 for item in result.measurements.mode_leakages)


@pytest.mark.reference
def test_default_zero_point_is_exact_selected_vscf_reference():
    system, _, _ = _projector_system()
    result = VibrationalGroundState(
        StatevectorEstimator(),
        _FixedPointOptimizer(),
        prefix="zero",
    ).compute(system, reference_modals=(1, 0))

    expected = np.zeros(1 << system.num_qubits, dtype=np.complex128)
    expected[system.modal_space.encoded_basis_index((1, 0))] = 1.0
    state = result.vqe_result.optimal_circuit.to_matrix()[:, 0]
    np.testing.assert_allclose(state, expected, rtol=0.0, atol=1e-12)
    assert set(result.vqe_result.optimal_parameters.values()) == {0.0}
    assert result.measurements.modal_occupations[0][1].expectation == pytest.approx(1.0)
    assert result.measurements.modal_occupations[1][0].expectation == pytest.approx(1.0)
    assert all(
        item.expectation == pytest.approx(1.0)
        for item in result.measurements.mode_occupations
    )
    assert all(
        item.expectation == pytest.approx(0.0, abs=1e-12)
        for item in result.measurements.mode_leakages
    )


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
                -0.75,
                2,
                OptimizerStatus.MAX_ITERATIONS,
            )
            for request in self.requests
        ]


def test_workflow_uses_vqe_and_one_stable_auxiliary_batch():
    system, _, _ = _projector_system()
    estimator = _RecordingEstimator()
    optimizer = _RecordingOptimizer()
    algorithm = VibrationalGroundState(estimator, optimizer, prefix="vib")

    result = algorithm.compute(system, reps=2, shots=321)

    assert algorithm.estimator is estimator
    assert algorithm.optimizer is optimizer
    assert algorithm.prefix == "vib"
    assert optimizer.estimator is estimator
    assert len(optimizer.requests) == 1
    objective = optimizer.requests[0]
    assert objective.parameter_names == tuple(
        f"vib_theta_{index}" for index in range(6)
    )
    assert set(objective.estimator_request.parameter_bindings.values()) == {0.0}
    assert objective.estimator_request.shots == 321

    assert len(estimator.batches) == 1
    requests = estimator.batches[0]
    assert len(requests) == sum(system.modal_space.num_modals) + 2 * system.num_modes
    assert all(
        request.circuit is result.vqe_result.optimal_circuit for request in requests
    )
    assert all(request.shots == 321 for request in requests)
    assert tuple(
        result_value.expectation
        for row in result.measurements.modal_occupations
        for result_value in row
    ) == (0.0, 1.0, 2.0, 3.0)
    assert tuple(
        result_value.expectation
        for result_value in result.measurements.mode_occupations
    ) == (4.0, 5.0)
    assert tuple(
        result_value.expectation for result_value in result.measurements.mode_leakages
    ) == (6.0, 7.0)

    indices = np.arange(1 << system.num_qubits)
    number_diagonals = tuple(
        ((indices >> qubit) & 1).astype(np.float64)
        for qubit in range(system.num_qubits)
    )
    mode_diagonals = (
        number_diagonals[0] + number_diagonals[1],
        number_diagonals[2] + number_diagonals[3],
    )
    expected_diagonals = (
        *number_diagonals,
        *mode_diagonals,
        *((diagonal - 1.0) ** 2 for diagonal in mode_diagonals),
    )
    for request, expected_diagonal in zip(requests, expected_diagonals, strict=True):
        np.testing.assert_array_equal(
            request.observable.to_matrix(),
            np.diag(expected_diagonal),
        )


class _WrongWidthAnsatz(Ansatz):
    def validate(self) -> None:
        return None

    def build_circuit(self, prefix: str) -> Circuit:
        return Circuit(1)

    def num_parameters(self) -> int:
        return 0

    def num_qubits(self) -> int:
        return 1


class _VacuumAnsatz(Ansatz):
    def __init__(self, num_qubits: int) -> None:
        self._num_qubits = num_qubits

    def validate(self) -> None:
        return None

    def build_circuit(self, prefix: str) -> Circuit:
        return Circuit(self._num_qubits)

    def num_parameters(self) -> int:
        return 0

    def num_qubits(self) -> int:
        return self._num_qubits


def test_component_run_accepts_explicit_ansatz_and_reports_its_physical_sector():
    system, _, point = _projector_system()
    saved_point = dict(point)
    solver = VibrationalGroundState(
        StatevectorEstimator(), _FixedPointOptimizer(), prefix="exact"
    )
    physical = solver.run(system, point, ansatz=UVCCSDAnsatz(system.modal_space))
    vacuum = solver.run(system, ansatz=_VacuumAnsatz(system.num_qubits))
    assert physical.variational_energy == pytest.approx(-0.63, abs=1e-12)
    assert vacuum.variational_energy == pytest.approx(0.37, abs=1e-12)
    assert all(
        x.expectation == pytest.approx(0.0, abs=1e-12)
        for x in physical.measurements.mode_leakages
    )
    assert all(
        x.expectation == pytest.approx(1.0, abs=1e-12)
        for x in vacuum.measurements.mode_leakages
    )
    assert point == saved_point


class _ZeroAndDoubleHotSuperpositionAnsatz(Ansatz):
    def validate(self) -> None:
        return None

    def build_circuit(self, prefix: str) -> Circuit:
        circuit = Circuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        return circuit

    def num_parameters(self) -> int:
        return 0

    def num_qubits(self) -> int:
        return 2


def test_custom_ansatz_is_width_checked_but_reports_physical_leakage():
    system, _, _ = _projector_system()
    result = VibrationalGroundState(
        StatevectorEstimator(),
        _FixedPointOptimizer(),
    ).compute(system, ansatz=_VacuumAnsatz(system.num_qubits))

    assert result.variational_energy == pytest.approx(0.37, abs=1e-12)
    assert all(
        item.expectation == pytest.approx(0.0, abs=1e-12)
        for item in result.measurements.mode_occupations
    )
    assert all(
        item.expectation == pytest.approx(1.0, abs=1e-12)
        for item in result.measurements.mode_leakages
    )


@pytest.mark.reference
def test_strict_leakage_detects_equal_zero_and_double_hot_superposition():
    space = ModalSpace((2,))
    system = VibrationalSystem(VibrationalEnergy(ModalIntegrals.from_terms(space, ())))
    result = VibrationalGroundState(
        StatevectorEstimator(),
        _FixedPointOptimizer(),
    ).compute(system, ansatz=_ZeroAndDoubleHotSuperpositionAnsatz())

    assert result.measurements.mode_occupations[0].expectation == pytest.approx(
        1.0,
        abs=1e-12,
    )
    assert result.measurements.mode_leakages[0].expectation == pytest.approx(
        1.0,
        abs=1e-12,
    )


def test_zero_repetitions_evaluates_fixed_vscf_state_without_parameters():
    system, _, _ = _projector_system()
    result = VibrationalGroundState(
        StatevectorEstimator(),
        _FixedPointOptimizer(),
        prefix="fixed",
    ).compute(system, reps=0)

    assert result.vqe_result.optimal_parameters == {}
    assert result.vqe_result.optimizer_result.status is OptimizerStatus.STEP_TOLERANCE
    expected = np.zeros(1 << system.num_qubits, dtype=np.complex128)
    expected[system.modal_space.encoded_basis_index((0, 0))] = 1.0
    np.testing.assert_allclose(
        result.vqe_result.optimal_circuit.to_matrix()[:, 0],
        expected,
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.reference
def test_single_modal_space_uses_fixed_vscf_default_path():
    space = ModalSpace((1,))
    system = VibrationalSystem(
        VibrationalEnergy(
            ModalIntegrals.from_terms(
                space,
                (((ModalTransition(0, 0, 0),), -0.75),),
                constant=0.2,
            )
        )
    )

    result = VibrationalGroundState(
        StatevectorEstimator(),
        _FixedPointOptimizer(),
        prefix="single",
    ).compute(system, reps=3)

    assert result.vqe_result.optimal_parameters == {}
    assert result.variational_energy == pytest.approx(-0.55, abs=1e-12)
    np.testing.assert_allclose(
        result.vqe_result.optimal_circuit.to_matrix()[:, 0],
        (0.0, 1.0),
        rtol=0.0,
        atol=1e-12,
    )
    assert result.measurements.modal_occupations[0][0].expectation == pytest.approx(1.0)
    assert result.measurements.mode_occupations[0].expectation == pytest.approx(1.0)
    assert result.measurements.mode_leakages[0].expectation == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_invalid_dependencies_inputs_and_custom_ansatz_options_are_rejected():
    estimator = _RecordingEstimator()
    optimizer = _RecordingOptimizer()
    with pytest.raises(TypeError, match="estimator"):
        VibrationalGroundState(object(), optimizer)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="optimizer"):
        VibrationalGroundState(estimator, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="prefix"):
        VibrationalGroundState(estimator, optimizer, prefix=1)  # type: ignore[arg-type]

    algorithm = VibrationalGroundState(estimator, optimizer)
    system, _, _ = _projector_system()
    with pytest.raises(TypeError, match="system"):
        algorithm.compute(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="reps"):
        algorithm.compute(system, reps=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        algorithm.compute(system, reps=-1)
    with pytest.raises(TypeError, match="ansatz"):
        algorithm.compute(system, ansatz=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reference_modals"):
        algorithm.compute(
            system,
            ansatz=_WrongWidthAnsatz(),
            reference_modals=(0, 0),
        )
    with pytest.raises(ValueError, match="reps applies"):
        algorithm.compute(system, ansatz=_WrongWidthAnsatz(), reps=2)
    with pytest.raises(ValueError, match="same width"):
        algorithm.compute(system, ansatz=_WrongWidthAnsatz())
    with pytest.raises(ValueError, match="length"):
        algorithm.compute(system, reference_modals=(0,))
    with pytest.raises(ValueError, match="outside"):
        algorithm.compute(system, reference_modals=(0, 2))


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
def test_invalid_auxiliary_estimator_responses_are_rejected(
    response: object,
    error: type[Exception],
    message: str,
):
    system, _, _ = _projector_system()
    algorithm = VibrationalGroundState(
        _RecordingEstimator(response),
        _RecordingOptimizer(),
    )

    with pytest.raises(error, match=message):
        algorithm.compute(system)
