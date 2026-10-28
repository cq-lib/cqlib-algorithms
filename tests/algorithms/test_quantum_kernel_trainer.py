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

# Modified to use the shared fidelity quantum-kernel operator.

"""Balanced alignment references, composition and optimization boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from cqlib import Circuit, Parameter, Qubit
from cqlib.circuit import CircuitError
from cqlib.device import ExecutionResult

from cqlib_algorithms.algorithms import _ml_common as common
from cqlib_algorithms.algorithms.pegasos_qsvc import PegasosQSVC
from cqlib_algorithms.algorithms.qsvr import QSVR
from cqlib_algorithms.algorithms.quantum_kernel_trainer import (
    QuantumKernelTrainer,
    _alignment,
)
from cqlib_algorithms.operators.quantum_kernel import FidelityQuantumKernel
from cqlib_algorithms.protocols.sampler import Sampler


def trainer(**kwargs):
    circuit = Circuit(1)
    circuit.ry(0, Parameter("w") * Parameter("x"))
    return QuantumKernelTrainer(circuit, ["x"], ["w"], **kwargs)


def fixed_solver(monkeypatch, point=None, **changes):
    import scipy.optimize

    def solve(fun, x0, **kwargs):
        assert kwargs["method"] == "COBYLA"
        x = np.array(x0 if point is None else point, dtype=float)
        result = {
            "x": x,
            "fun": fun(x),
            "success": False,
            "status": 3,
            "message": "Budget exhausted",
            "nfev": 1,
        }
        result.update(changes)
        return SimpleNamespace(**result)

    monkeypatch.setattr(scipy.optimize, "minimize", solve)


def analytic_alignment(X, y, weight):
    angle = weight * np.asarray(X)[:, 0]
    matrix = np.cos((angle[:, None] - angle[None, :]) / 2) ** 2
    y = np.asarray(y)
    labels, counts = np.unique(y, return_counts=True)
    scaled = np.where(y == labels[0], -1 / counts[0], 1 / counts[1])
    numerator = sum(
        matrix[i, j] * scaled[i] * scaled[j]
        for i in range(len(y))
        for j in range(len(y))
    )
    denominator = np.sqrt(np.sum(matrix**2)) * np.sum(scaled**2)
    return matrix, numerator / denominator


@pytest.mark.reference
def test_real_training_two_points_closed_form_and_downstream_composition():
    training = trainer(max_evaluations=120).fit(
        [[0.0], [1.0]], [-1.0, 1.0], initial_point={"w": 0.2}
    )
    assert training.initial_alignment_ < 0.01
    assert training.alignment_ > 0.70
    w = training.optimal_parameters_["w"]
    k = np.cos(w / 2) ** 2
    assert training.alignment_ == pytest.approx(
        (1 - k) / np.sqrt(2 * (1 + k * k)), abs=1e-13
    )
    assert training.loss_ == -training.alignment_
    assert training.objective_evaluations_ == training.optimizer_evaluations_ + 2
    assert 0 < training.optimizer_evaluations_ <= 120
    assert isinstance(training.optimizer_success_, bool)
    assert isinstance(training.optimizer_status_, int)
    assert isinstance(training.optimizer_message_, str)
    assert training.n_features_in_ == 1
    regressor = QSVR(training.quantum_kernel_, C=10.0, epsilon=0.0).fit(
        [[0.0], [1.0]], [0.0, 1.0]
    )
    np.testing.assert_allclose(regressor.predict([[0.0], [1.0]]), [0.0, 1.0], atol=1e-5)
    classifier = PegasosQSVC(training.quantum_kernel_, num_steps=100, seed=4).fit(
        [[0.0], [1.0]], [-1.0, 1.0]
    )
    assert classifier.score([[0.0], [1.0]], [-1.0, 1.0]) == 1.0
    rebuilt = FidelityQuantumKernel(
        training.quantum_kernel_.feature_map,
        ["x"],
        parameter_bindings=dict(training.optimal_parameters_),
    )
    np.testing.assert_allclose(
        rebuilt.evaluate([[0.0], [1.0]]), training.kernel_matrix_, atol=1e-14
    )


@pytest.mark.reference
@pytest.mark.parametrize("weight", [0.0, 0.4, 1.7, np.pi])
def test_unbalanced_uncentered_alignment_matches_analytic_reference(
    monkeypatch, weight
):
    fixed_solver(monkeypatch)
    X, y = [[0.0], [0.2], [0.8], [1.4]], [3.0, 3.0, 3.0, 8.0]
    training = trainer().fit(X, y, initial_point={"w": weight})
    expected, alignment = analytic_alignment(X, y, weight)
    np.testing.assert_allclose(training.kernel_matrix_, expected, atol=1e-13)
    assert training.alignment_ == pytest.approx(alignment, abs=1e-13)
    assert training.classes_ == (3.0, 8.0)


def test_multiple_parameters_expression_order_and_snapshot(monkeypatch):
    circuit = Circuit([9, 3])
    circuit.ry(Qubit(9), Parameter("u") * Parameter("z") + Parameter("fixed"))
    circuit.ry(Qubit(3), Parameter("v") * Parameter("a"))
    inputs, weights, fixed = ["z", "a"], ["v", "u"], {"fixed": 0.2}
    training = QuantumKernelTrainer(circuit, inputs, weights, parameter_bindings=fixed)
    inputs.reverse()
    weights.reverse()
    fixed["fixed"] = 3.0
    circuit.x(Qubit(9))
    fixed_solver(monkeypatch, [0.4, 1.3])
    X = np.array([[0.1, 0.7], [1.0, -0.2], [0.2, 0.9]])
    training.fit(X, [-1.0, 1.0, -1.0], initial_point={"u": 0.5, "v": 0.1})
    first = 1.3 * (X[:, None, 0] - X[None, :, 0])
    second = 0.4 * (X[:, None, 1] - X[None, :, 1])
    expected = np.cos(first / 2) ** 2 * np.cos(second / 2) ** 2
    np.testing.assert_allclose(training.kernel_matrix_, expected, atol=1e-13)
    assert tuple(training.optimal_parameters_) == ("v", "u")
    assert training.quantum_kernel_.parameter_bindings == {
        "fixed": 0.2,
        "v": 0.4,
        "u": 1.3,
    }
    with pytest.raises(TypeError):
        training.optimal_parameters_["u"] = 2
    saved = training.kernel_matrix_
    training.kernel_matrix_[:] = 0
    training.quantum_kernel_.feature_map.x(0)
    np.testing.assert_array_equal(training.kernel_matrix_, saved)
    X[:] = 0
    assert np.any(training.kernel_matrix_ != 1)


class PairSampler(Sampler):
    def __init__(self, fractions):
        self.fractions = fractions
        self.batches = []

    def run(self, requests, /):
        self.batches.append(tuple(requests))
        return [
            ExecutionResult.from_counts(
                "pair",
                list(request.qubits),
                request.shots,
                len(request.qubits),
                {
                    "0": round(request.shots * p),
                    "1": request.shots - round(request.shots * p),
                },
                "fixed",
            )
            for request, p in zip(requests, self.fractions)
        ]


@pytest.mark.reference
def test_sampled_raw_indefinite_gram_is_not_projected(monkeypatch):
    sampler = PairSampler([0.9, 0.9, 0.0])
    fixed_solver(monkeypatch)
    training = trainer(sampler=sampler, shots=10).fit(
        [[0.0], [1.0], [2.0]], [-1.0, 1.0, 1.0], initial_point={"w": 0.5}
    )
    raw = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, 0.0], [0.9, 0.0, 1.0]])
    np.testing.assert_array_equal(training.kernel_matrix_, raw)
    assert np.linalg.eigvalsh(training.kernel_matrix_)[0] < 0
    target = np.outer([-1.0, 0.5, 0.5], [-1.0, 0.5, 0.5])
    expected = np.sum(raw * target) / (np.linalg.norm(raw) * np.linalg.norm(target))
    assert training.alignment_ == pytest.approx(expected)
    assert expected < 0
    assert len(sampler.batches) == training.objective_evaluations_
    assert all(request.shots == 10 for batch in sampler.batches for request in batch)


def test_final_sampled_matrix_and_alignment_from_same_evaluation(monkeypatch):
    class Changing(PairSampler):
        def run(self, requests, /):
            self.fractions = [(0.2, 0.5, 0.8)[len(self.batches)]]
            return super().run(requests)

    sampler = Changing([])
    fixed_solver(monkeypatch)
    training = trainer(sampler=sampler, shots=10).fit(
        [[0.0], [1.0]], [-1.0, 1.0], initial_point={"w": 0.1}
    )
    np.testing.assert_array_equal(training.kernel_matrix_, [[1.0, 0.8], [0.8, 1.0]])
    assert training.initial_alignment_ == pytest.approx(0.8 / np.sqrt(2 * 1.04))
    assert training.alignment_ == pytest.approx(0.2 / np.sqrt(2 * 1.64))
    assert len(sampler.batches) == 3


def test_flat_legal_objective_and_actual_parameter_rebinding():
    circuit = Circuit(1)
    circuit.ry(0, Parameter("x"))
    circuit.rz(0, Parameter("w"))
    training = QuantumKernelTrainer(circuit, ["x"], ["w"], max_evaluations=40).fit(
        [[0.0], [1.0]], [-1.0, 1.0], initial_point={"w": 0.4}
    )
    assert training.alignment_ == pytest.approx(training.initial_alignment_, abs=1e-14)
    assert (
        training.quantum_kernel_.parameter_bindings["w"]
        == training.optimal_parameters_["w"]
    )


@pytest.mark.parametrize(
    "X",
    [
        np.empty((0, 1)),
        [[True], [False]],
        [[np.nan], [1.0]],
        [[1j], [0j]],
        [["0"], ["1"]],
        [0.0, 1.0],
        [[0.0, 1.0], [1.0, 2.0]],
    ],
)
def test_invalid_data(X):
    training = trainer()
    with pytest.raises((TypeError, ValueError)):
        training.fit(X, [-1.0, 1.0], initial_point={"w": 0.4})
    assert not training.is_fitted


@pytest.mark.parametrize(
    "y",
    [
        [0.0, 0.0],
        [0.0, 1.0, 2.0],
        [True, False],
        [0j, 1j],
        ["a", "b"],
        [np.nan, 1.0],
        [np.inf, 1.0],
        [[0.0], [1.0]],
        [0.0],
    ],
)
def test_invalid_labels(y):
    with pytest.raises((TypeError, ValueError)):
        trainer().fit([[0.0], [1.0]], y, initial_point={"w": 0.3})


@pytest.mark.parametrize(
    "initial",
    [{}, {"bad": 1.0}, {"w": True}, {"w": np.inf}, {"w": 1j}, {"w": "1"}, None],
)
def test_invalid_initial_point(initial):
    with pytest.raises((TypeError, ValueError)):
        trainer().fit([[0.0], [1.0]], [-1.0, 1.0], initial_point=initial)


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("max_evaluations", 2),
        ("max_evaluations", True),
        ("tolerance", 0),
        ("initial_step", np.inf),
        ("shots", 0),
        ("sampler", object()),
        ("max_statevector_qubits", 0),
        ("max_statevector_bytes", 0),
    ],
)
def test_invalid_configuration(keyword, value):
    with pytest.raises((TypeError, ValueError)):
        trainer(**{keyword: value})


@pytest.mark.parametrize(
    "name",
    [
        "quantum_kernel_",
        "optimal_parameters_",
        "classes_",
        "n_features_in_",
        "kernel_matrix_",
        "initial_alignment_",
        "alignment_",
        "loss_",
        "optimizer_success_",
        "optimizer_status_",
        "optimizer_message_",
        "optimizer_evaluations_",
        "objective_evaluations_",
    ],
)
def test_unfitted_diagnostics(name):
    with pytest.raises(ValueError, match="fitted"):
        getattr(trainer(), name)


def test_nonzero_domain_singular_point_and_failed_refit(monkeypatch):
    circuit = Circuit(1)
    circuit.ry(0, 1 / Parameter("x") / Parameter("w"))
    training = QuantumKernelTrainer(circuit, ["x"], ["w"])
    fixed_solver(monkeypatch)
    training.fit([[1.0], [2.0]], [-1.0, 1.0], initial_point={"w": 2.0})
    matrix = training.kernel_matrix_
    for X, initial in [([[0.0], [1.0]], {"w": 2.0}), ([[1.0], [1.0]], {"w": 0.0})]:
        with pytest.raises(CircuitError):
            training.fit(X, [-1.0, 1.0], initial_point=initial)
        np.testing.assert_array_equal(training.kernel_matrix_, matrix)


def test_resource_limit_rejected_without_fitted_state():
    training = trainer(max_statevector_bytes=1)
    with pytest.raises(ValueError, match="max_statevector_bytes"):
        training.fit([[0.0], [1.0]], [-1.0, 1.0], initial_point={"w": 0.5})
    assert not training.is_fitted


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, 2)),
        np.ones((3, 3)),
        [[1.0, 0.2], [0.3, 1.0]],
        [[1.0, np.nan], [np.nan, 1.0]],
        [[1.0, 1.1], [1.1, 1.0]],
    ],
)
def test_bad_kernel_results(monkeypatch, bad):
    monkeypatch.setattr(
        FidelityQuantumKernel, "evaluate", lambda *args: np.asarray(bad)
    )
    with pytest.raises((TypeError, ValueError)):
        trainer().fit([[0.0], [1.0]], [-1.0, 1.0], initial_point={"w": 0.1})


@pytest.mark.parametrize(
    "changes",
    [
        {"x": [np.inf]},
        {"x": []},
        {"success": 1},
        {"status": 1.2},
        {"message": None},
        {"nfev": True},
        {"fun": 1j},
    ],
)
def test_bad_optimizer_result_and_atomic_refit(monkeypatch, changes):
    fixed_solver(monkeypatch)
    training = trainer().fit([[0.0], [1.0]], [-1.0, 1.0], initial_point={"w": 0.2})
    previous = training.quantum_kernel_
    fixed_solver(monkeypatch, **changes)
    with pytest.raises((TypeError, ValueError)):
        training.fit([[0.0], [1.0]], [-1.0, 1.0], initial_point={"w": 0.7})
    assert training.quantum_kernel_ is previous


def test_missing_scipy_and_evaluation_budget(monkeypatch):
    training = trainer(max_evaluations=3).fit(
        [[0.0], [1.0]], [-1.0, 1.0], initial_point={"w": 0.3}
    )
    assert training.is_fitted and not training.optimizer_success_
    assert training.optimizer_evaluations_ <= 3
    original = common.importlib.import_module

    def absent(name, *args, **kwargs):
        if name == "scipy.optimize":
            raise ModuleNotFoundError("missing scipy", name="scipy")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(common.importlib, "import_module", absent)
    with pytest.raises(ImportError, match="machine-learning"):
        trainer().fit([[0.0], [1.0]], [-1.0, 1.0], initial_point={"w": 0.2})


def test_optimizer_cannot_overrun_budget(monkeypatch):
    import scipy.optimize

    def runaway(fun, x0, **kwargs):
        for _ in range(4):
            fun(x0)

    monkeypatch.setattr(scipy.optimize, "minimize", runaway)
    with pytest.raises(ValueError, match="exceeded"):
        trainer(max_evaluations=3).fit(
            [[0.0], [1.0]], [-1.0, 1.0], initial_point={"w": 0.2}
        )


def test_alignment_defends_zero_nonfinite_and_wrong_values():
    for matrix in [np.zeros((2, 2)), np.full((2, 2), np.inf)]:
        with pytest.raises(ValueError, match="norm"):
            _alignment(matrix, np.eye(2))


def test_final_evaluation_failure_preserves_previous_kernel(monkeypatch):
    fixed_solver(monkeypatch)
    training = trainer().fit([[0.0], [1.0]], [-1.0, 1.0], initial_point={"w": 0.2})
    previous = training.quantum_kernel_
    original = FidelityQuantumKernel.evaluate
    calls = 0

    def terminal_failure(kernel, X, Y=None):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("terminal evaluation failed")
        return original(kernel, X, Y)

    monkeypatch.setattr(FidelityQuantumKernel, "evaluate", terminal_failure)
    with pytest.raises(RuntimeError, match="terminal"):
        training.fit([[0.0], [1.0]], [-1.0, 1.0], initial_point={"w": 0.7})
    assert training.quantum_kernel_ is previous


@pytest.mark.parametrize(
    "inputs,trainables,fixed",
    [
        ([], ["w"], None),
        (["x"], [], None),
        (["x"], ["x"], None),
        (["x"], ["bad"], None),
        (["x", "x"], ["w"], None),
        (["x"], ["w"], {"w": 0.2}),
    ],
)
def test_symbol_partition_validated(inputs, trainables, fixed):
    circuit = Circuit(1)
    circuit.ry(0, Parameter("w") * Parameter("x"))
    with pytest.raises((TypeError, ValueError)):
        QuantumKernelTrainer(circuit, inputs, trainables, parameter_bindings=fixed)


def test_parameter_overflow_rejected_even_for_duplicate_rows(monkeypatch):
    fixed_solver(monkeypatch)
    circuit = Circuit(1)
    circuit.ry(0, Parameter("x") * Parameter("w") * Parameter("w"))
    training = QuantumKernelTrainer(circuit, ["x"], ["w"])
    with pytest.raises((ValueError, CircuitError)):
        training.fit([[1.0], [1.0]], [-1.0, 1.0], initial_point={"w": 1e308})
    assert not training.is_fitted


@pytest.mark.parametrize("operation", ["measure", "reset"])
def test_nonunitary_structure_rejected(operation):
    circuit = Circuit(1)
    circuit.ry(0, Parameter("w") * Parameter("x"))
    getattr(circuit, operation)(0)
    with pytest.raises((ValueError, CircuitError)):
        QuantumKernelTrainer(circuit, ["x"], ["w"])
