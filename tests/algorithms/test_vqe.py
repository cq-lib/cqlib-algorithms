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

"""Integration, reference and boundary tests for VQE."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pytest
from cqlib import Circuit, Parameter
from cqlib.qis import Hamiltonian, PauliString

from cqlib_algorithms.algorithms import VQE, VQEResult
from cqlib_algorithms.operators.ansatz import RealAmplitudesAnsatz
from cqlib_algorithms.operators.estimator import StatevectorEstimator
from cqlib_algorithms.operators.gradient import ParameterShiftGradient
from cqlib_algorithms.operators.optimizer import GradientDescent
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


class OrderedAnsatz(Ansatz):
    """Two active symbols and one deliberately inactive logical slot."""

    def validate(self) -> None:
        return None

    def build_circuit(self, prefix: str) -> Circuit:
        circuit = Circuit(2)
        circuit.ry(0, Parameter(f"{prefix}_beta"))
        circuit.rz(1, Parameter(f"{prefix}_alpha"))
        return circuit

    def num_parameters(self) -> int:
        return 3

    def num_qubits(self) -> int:
        return 2


class StaticAnsatz(Ansatz):
    def validate(self) -> None:
        return None

    def build_circuit(self, prefix: str) -> Circuit:
        circuit = Circuit(1)
        circuit.x(0)
        return circuit

    def num_parameters(self) -> int:
        return 0

    def num_qubits(self) -> int:
        return 1


class RecordingEstimator(Estimator):
    def __init__(self) -> None:
        self.batches: list[list[EstimatorRequest]] = []

    def run(self, requests: Sequence[EstimatorRequest], /) -> list[EstimatorResult]:
        batch = list(requests)
        self.batches.append(batch)
        return [EstimatorResult(0.0, 0.0) for _ in batch]


class RecordingOptimizer(Optimizer):
    def __init__(self) -> None:
        self.estimator: Estimator | None = None
        self.batches: list[list[OptimizerRequest]] = []

    def run(
        self,
        estimator: Estimator,
        requests: Sequence[OptimizerRequest],
        /,
    ) -> list[OptimizerResult]:
        self.estimator = estimator
        batch = list(requests)
        self.batches.append(batch)
        return [
            OptimizerResult(
                request.parameter_names,
                tuple(
                    request.estimator_request.parameter_bindings[name] + 0.5
                    for name in request.parameter_names
                ),
                -0.75,
                2,
                OptimizerStatus.MAX_ITERATIONS,
            )
            for request in batch
        ]


def _recording_vqe() -> tuple[VQE, RecordingEstimator, RecordingOptimizer]:
    estimator = RecordingEstimator()
    optimizer = RecordingOptimizer()
    return (
        VQE(estimator, OrderedAnsatz(), optimizer, prefix="phi"),
        estimator,
        optimizer,
    )


def test_vqe_bridges_ansatz_estimator_and_optimizer_contracts_exactly() -> None:
    vqe, estimator, optimizer = _recording_vqe()
    initial = {"phi_beta": 0.1, "phi_alpha": -0.2}

    result = vqe.compute_minimum_eigenvalue(
        PauliString.from_str("ZI"), initial, shots=321
    )

    assert optimizer.estimator is estimator
    assert len(optimizer.batches) == 1
    assert len(optimizer.batches[0]) == 1
    request = optimizer.batches[0][0]
    assert request.parameter_names == ("phi_beta", "phi_alpha")
    assert request.estimator_request.parameter_bindings == initial
    assert request.estimator_request.shots == 321
    assert result.eigenvalue == -0.75
    assert result.optimal_parameters == {
        "phi_beta": pytest.approx(0.6),
        "phi_alpha": pytest.approx(0.3),
    }
    assert result.optimizer_result.iterations == 2
    assert result.optimizer_result.status is OptimizerStatus.MAX_ITERATIONS


def test_result_parameters_are_immutable_and_circuit_is_independent_and_bound() -> None:
    vqe, _, optimizer = _recording_vqe()
    result = vqe.compute_minimum_eigenvalue(
        PauliString.from_str("ZI"),
        {"phi_beta": 0.1, "phi_alpha": -0.2},
    )
    submitted = optimizer.batches[0][0].estimator_request.circuit

    with pytest.raises(TypeError):
        result.optimal_parameters["phi_beta"] = 1.0  # type: ignore[index]
    assert result.optimal_circuit is not submitted
    assert result.optimal_circuit.used_symbols == []
    assert submitted.used_symbols == ["phi_beta", "phi_alpha"]
    result.optimal_circuit.x(0)
    assert len(result.optimal_circuit.operations) == len(submitted.operations) + 1
    assert len(submitted.operations) == 2


def test_one_qubit_real_components_reach_the_z_ground_energy() -> None:
    vqe = VQE(
        StatevectorEstimator(),
        RealAmplitudesAnsatz(1, reps=0),
        GradientDescent(
            ParameterShiftGradient(),
            learning_rate=0.4,
            maxiter=80,
            tolerance=1e-10,
        ),
    )
    result = vqe.compute_minimum_eigenvalue(PauliString.from_str("Z"), {"theta_0": 0.4})

    assert result.eigenvalue == pytest.approx(-1.0, abs=1e-10)
    state = result.optimal_circuit.to_matrix()[:, 0]
    assert abs(state[1]) ** 2 == pytest.approx(1.0, abs=1e-10)


@pytest.mark.reference
def test_two_qubit_energy_matches_independent_dense_eigenvalue_reference() -> None:
    observable = Hamiltonian(2)
    observable.add_term(PauliString.from_str("ZI"), 1.0)
    observable.add_term(PauliString.from_str("IZ"), 1.0)
    reference = float(np.linalg.eigvalsh(observable.to_matrix())[0])
    ansatz = RealAmplitudesAnsatz(2, reps=0)
    vqe = VQE(
        StatevectorEstimator(),
        ansatz,
        GradientDescent(
            ParameterShiftGradient(),
            learning_rate=0.4,
            maxiter=100,
            tolerance=1e-10,
        ),
    )

    result = vqe.compute_minimum_eigenvalue(
        observable,
        {"theta_0": 0.4, "theta_1": -0.7},
    )

    assert reference == pytest.approx(-2.0, abs=1e-14)
    assert result.eigenvalue == pytest.approx(reference, abs=1e-10)


def test_parameter_free_ansatz_uses_optimizer_no_parameters_path() -> None:
    result = VQE(
        StatevectorEstimator(),
        StaticAnsatz(),
        GradientDescent(ParameterShiftGradient()),
    ).compute_minimum_eigenvalue(PauliString.from_str("Z"), {})

    assert result.eigenvalue == pytest.approx(-1.0)
    assert result.optimal_parameters == {}
    assert result.optimizer_result.parameter_names == ()
    assert result.optimizer_result.status is OptimizerStatus.NO_PARAMETERS


@pytest.mark.parametrize(
    ("position", "value", "message"),
    [
        ("estimator", object(), "estimator must be an Estimator"),
        ("ansatz", object(), "ansatz must be an Ansatz"),
        ("optimizer", object(), "optimizer must be an Optimizer"),
    ],
)
def test_constructor_requires_public_component_types(
    position: str, value: object, message: str
) -> None:
    arguments: dict[str, object] = {
        "estimator": RecordingEstimator(),
        "ansatz": OrderedAnsatz(),
        "optimizer": RecordingOptimizer(),
    }
    arguments[position] = value
    with pytest.raises(TypeError, match=message):
        VQE(**arguments)  # type: ignore[arg-type]


def test_constructor_rejects_non_string_prefix() -> None:
    with pytest.raises(TypeError, match="prefix must be a str"):
        VQE(RecordingEstimator(), OrderedAnsatz(), RecordingOptimizer(), prefix=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("initial_point", "error", "message"),
    [
        ([], TypeError, "must be a mapping"),
        ({"phi_beta": 0.1}, ValueError, "missing"),
        (
            {"phi_beta": 0.1, "phi_alpha": -0.2, "extra": 0.0},
            ValueError,
            "unexpected",
        ),
        ({1: 0.1, "phi_alpha": -0.2}, TypeError, "names must be strings"),
        (
            {"phi_beta": True, "phi_alpha": -0.2},
            TypeError,
            "values must be real numbers",
        ),
        (
            {"phi_beta": math.inf, "phi_alpha": -0.2},
            ValueError,
            "values must be finite",
        ),
        (
            {"phi_beta": math.nan, "phi_alpha": -0.2},
            ValueError,
            "values must be finite",
        ),
    ],
)
def test_initial_point_validation(
    initial_point: Mapping[str, float] | object,
    error: type[Exception],
    message: str,
) -> None:
    vqe, _, _ = _recording_vqe()
    with pytest.raises(error, match=message):
        vqe.compute_minimum_eigenvalue(
            PauliString.from_str("ZI"),
            initial_point,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("shots", "error", "message"),
    [
        (True, TypeError, "shots must be an integer or None"),
        (1.5, TypeError, "shots must be an integer or None"),
        (0, ValueError, "shots must be greater than zero"),
        (-1, ValueError, "shots must be greater than zero"),
    ],
)
def test_shots_validation(
    shots: int | object, error: type[Exception], message: str
) -> None:
    vqe, _, _ = _recording_vqe()
    with pytest.raises(error, match=message):
        vqe.compute_minimum_eigenvalue(
            PauliString.from_str("ZI"),
            {"phi_beta": 0.1, "phi_alpha": -0.2},
            shots=shots,  # type: ignore[arg-type]
        )


def test_operator_type_width_and_hermiticity_are_validated_before_optimizer() -> None:
    vqe, _, optimizer = _recording_vqe()
    initial = {"phi_beta": 0.1, "phi_alpha": -0.2}

    with pytest.raises(TypeError, match="Hamiltonian or PauliString"):
        vqe.compute_minimum_eigenvalue(object(), initial)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="same qubit count"):
        vqe.compute_minimum_eigenvalue(PauliString.from_str("Z"), initial)
    non_hermitian = Hamiltonian(2)
    non_hermitian.add_term(PauliString.from_str("ZI"), 1j)
    with pytest.raises(ValueError, match="Hermitian"):
        vqe.compute_minimum_eigenvalue(non_hermitian, initial)
    assert optimizer.batches == []


class BrokenOptimizer(Optimizer):
    def __init__(self, response: object) -> None:
        self.response = response

    def run(
        self,
        estimator: Estimator,
        requests: Sequence[OptimizerRequest],
        /,
    ) -> list[OptimizerResult]:
        return self.response  # type: ignore[return-value]


def _vqe_with_broken_optimizer(response: object) -> VQE:
    return VQE(RecordingEstimator(), OrderedAnsatz(), BrokenOptimizer(response))


def test_optimizer_response_container_count_and_item_type_are_checked() -> None:
    point = {"theta_beta": 0.1, "theta_alpha": -0.2}
    operator = PauliString.from_str("ZI")
    with pytest.raises(TypeError, match="return a list"):
        _vqe_with_broken_optimizer(()).compute_minimum_eigenvalue(operator, point)
    with pytest.raises(ValueError, match="exactly one result"):
        _vqe_with_broken_optimizer([]).compute_minimum_eigenvalue(operator, point)
    with pytest.raises(TypeError, match="OptimizerResult values"):
        _vqe_with_broken_optimizer([object()]).compute_minimum_eigenvalue(
            operator, point
        )


def test_optimizer_result_parameter_names_must_match_request_order() -> None:
    broken = OptimizerResult(
        ("theta_alpha", "theta_beta"),
        (0.1, 0.2),
        -1.0,
        1,
        OptimizerStatus.MAX_ITERATIONS,
    )
    with pytest.raises(ValueError, match="names must match"):
        _vqe_with_broken_optimizer([broken]).compute_minimum_eigenvalue(
            PauliString.from_str("ZI"),
            {"theta_beta": 0.1, "theta_alpha": -0.2},
        )


def test_vqe_result_validates_optimizer_consistency() -> None:
    optimizer_result = OptimizerResult(
        ("x",), (0.25,), -0.5, 1, OptimizerStatus.MAX_ITERATIONS
    )
    with pytest.raises(ValueError, match="names must match"):
        VQEResult(-0.5, {"y": 0.25}, Circuit(1), optimizer_result)
    with pytest.raises(ValueError, match="values must match"):
        VQEResult(-0.5, {"x": 0.5}, Circuit(1), optimizer_result)
    with pytest.raises(ValueError, match="eigenvalue must match"):
        VQEResult(-0.25, {"x": 0.25}, Circuit(1), optimizer_result)
    parameterized = Circuit(1)
    parameterized.ry(0, Parameter("x"))
    with pytest.raises(ValueError, match="fully bound"):
        VQEResult(-0.5, {"x": 0.25}, parameterized, optimizer_result)
