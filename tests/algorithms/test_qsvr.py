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

"""Independent classical SVR references and quantum model-state contracts."""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest
from cqlib import Circuit, Parameter
from cqlib.circuit import CircuitError
from cqlib.device import ExecutionResult

from cqlib_algorithms.algorithms.machine_learning.qsvr import QSVR
from cqlib_algorithms.operators.quantum_kernel import FidelityQuantumKernel
from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.protocols import Sampler


def kernel(**kwargs):
    circuit = Circuit(1)
    circuit.ry(0, Parameter("x"))
    return FidelityQuantumKernel(circuit, ["x"], **kwargs)


def analytic(x, y):
    return np.cos((np.asarray(x)[:, None, 0] - np.asarray(y)[None, :, 0]) / 2) ** 2


@pytest.mark.reference
@pytest.mark.parametrize("enforce_psd", [False, True])
def test_against_independent_analytic_kernel_sklearn_reference(enforce_psd):
    SVR = pytest.importorskip("sklearn.svm").SVR
    x = np.array([[-1.2], [0.1], [1.3], [2.7], [-0.4]])
    y = np.array([-0.7, 0.2, 0.9, 0.3, -0.2])
    new = np.array([[0.7], [-0.6], [2.1]])
    weights = np.array([1, 2, 0.7, 1.3, 0.9])
    model = QSVR(kernel(), C=2.1, epsilon=0.03, tol=1e-9, enforce_psd=enforce_psd)
    expected = SVR(kernel="precomputed", C=2.1, epsilon=0.03, tol=1e-9)
    expected.fit(analytic(x, x), y, sample_weight=weights)
    assert model.fit(x, y, sample_weight=weights) is model
    np.testing.assert_allclose(
        model.predict(new), expected.predict(analytic(new, x)), atol=1e-7, rtol=0
    )
    np.testing.assert_allclose(model.dual_coef_, expected.dual_coef_, atol=1e-7, rtol=0)
    np.testing.assert_array_equal(model.support_, expected.support_)
    np.testing.assert_allclose(model.intercept_, expected.intercept_, atol=1e-7, rtol=0)
    assert model.score(new, [0.3, -0.4, 0.7]) == pytest.approx(
        expected.score(analytic(new, x), [0.3, -0.4, 0.7]), abs=1e-7
    )
    assert model.fit_status_ == 0
    assert model.n_features_in_ == 1
    assert model.n_iter_ == expected.n_iter_


@pytest.mark.reference
def test_hand_computable_identity_gram_two_point_svr():
    model = QSVR(kernel(), C=1, epsilon=0.1, tol=1e-10)
    model.fit([[0], [math.pi]], [0, 1])
    np.testing.assert_allclose(model.raw_kernel_matrix_, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(model.predict([[0], [math.pi]]), [0.1, 0.9], atol=1e-12)
    np.testing.assert_allclose(model.dual_coef_, [[-0.4, 0.4]], atol=1e-12)
    np.testing.assert_allclose(model.intercept_, [0.5], atol=1e-12)


class NoisySampler(Sampler):
    def __init__(self):
        self.calls = []

    def run(self, requests):
        self.calls.append(tuple(requests))
        zeros = [9, 9, 0] if len(self.calls) == 1 else [2, 5, 8]
        assert len(requests) == len(zeros)
        return [
            ExecutionResult.from_counts(
                "noisy",
                list(request.qubits),
                request.shots,
                len(request.qubits),
                {"0": zero, "1": request.shots - zero},
            )
            for request, zero in zip(requests, zeros)
        ]


@pytest.mark.parametrize("repair", [True, False])
def test_finite_shot_indefinite_training_gram_and_raw_rectangular_prediction(repair):
    sampler = NoisySampler()
    model = QSVR(kernel(sampler=sampler, shots=10), enforce_psd=repair)
    raw = np.array([[1, 0.9, 0.9], [0.9, 1, 0], [0.9, 0, 1]])
    model.fit([[0], [1], [2]], [0, 0.5, 1])
    # Closed-form eigensystem for I plus a two-leaf star adjacency.
    vectors = np.array(
        [
            [1 / np.sqrt(2), 1 / np.sqrt(2), 0],
            [0.5, -0.5, 1 / np.sqrt(2)],
            [0.5, -0.5, -1 / np.sqrt(2)],
        ]
    )
    eigenvalues = np.array([1 + 0.9 * np.sqrt(2), 1 - 0.9 * np.sqrt(2), 1])
    expected = (
        vectors @ np.diag(np.maximum(eigenvalues, 0)) @ vectors.T if repair else raw
    )
    np.testing.assert_array_equal(model.raw_kernel_matrix_, raw)
    np.testing.assert_allclose(model.kernel_matrix_, expected, atol=1e-14, rtol=0)
    assert model.minimum_raw_eigenvalue_ == pytest.approx(
        1 - 0.9 * np.sqrt(2), abs=1e-14
    )
    assert model.psd_correction_norm_ == pytest.approx(
        np.linalg.norm(expected - raw), abs=1e-14
    )
    if repair:
        assert np.linalg.eigvalsh(model.kernel_matrix_)[0] >= -10 * np.finfo(
            float
        ).eps * np.linalg.norm(expected, 2)
        assert np.max(np.diag(model.kernel_matrix_)) > 1
    prediction = model.predict([[3]])
    direct = (
        np.array([[0.2, 0.5, 0.8]])[:, model.support_] @ model.dual_coef_[0]
        + model.intercept_[0]
    )
    np.testing.assert_allclose(prediction, direct, atol=1e-14, rtol=0)
    assert len(sampler.calls) == 2


def test_reproducibility_training_order_and_all_diagnostics_are_copies():
    x = np.array([[0], [0.3], [1.2], [2.3]])
    y = np.array([0, 0.8, -0.1, 0.5])
    model = QSVR(kernel(), tol=1e-9).fit(x, y)
    expected = model.predict([[0.2], [1.1]])
    for diagnostic in (
        model.training_data_,
        model.raw_kernel_matrix_,
        model.kernel_matrix_,
        model.support_,
        model.dual_coef_,
        model.intercept_,
    ):
        diagnostic[...] = 0
    x[...] = 0
    y[...] = 0
    np.testing.assert_array_equal(model.predict([[0.2], [1.1]]), expected)
    model.fit([[0], [0.3], [1.2], [2.3]], [0, 0.8, -0.1, 0.5])
    np.testing.assert_array_equal(model.predict([[0.2], [1.1]]), expected)
    assert not hasattr(model, "support_vectors_")
    assert not hasattr(model, "solver")


def test_empty_predict_no_quantum_or_solver_calls(monkeypatch):
    model = QSVR(kernel()).fit([[0], [1]], [0, 1])
    monkeypatch.setattr(
        FidelityQuantumKernel,
        "evaluate",
        lambda *a: pytest.fail("no quantum evaluation"),
    )
    assert model.predict(np.empty((0, 1))).shape == (0,)


@pytest.mark.parametrize(
    "changes",
    [
        {"C": 0},
        {"C": True},
        {"C": math.inf},
        {"C": 1j},
        {"epsilon": -0.1},
        {"epsilon": math.nan},
        {"epsilon": False},
        {"tol": 0},
        {"tol": "x"},
        {"max_iter": 0},
        {"max_iter": -2},
        {"max_iter": True},
        {"max_iter": 0.5},
        {"enforce_psd": 1},
    ],
)
def test_invalid_hyperparameters(changes):
    with pytest.raises((TypeError, ValueError)):
        QSVR(kernel(), **changes)


@pytest.mark.parametrize(
    "x,y,weights",
    [
        (np.empty((0, 1)), [], None),
        ([[0]], [[0]], None),
        ([[0], [1]], [0], None),
        ([[0]], [math.nan], None),
        ([[0]], [1j], None),
        ([[0]], [True], None),
        ([[0]], ["1"], None),
        ([[True]], [1], None),
        ([[math.inf]], [1], None),
        ([[0, 1]], [1], None),
        ([[0]], [1], [0]),
        ([[0]], [1], [-1]),
        ([[0]], [1], [math.nan]),
        ([[0]], [1], [True]),
        ([[0]], [1], [[1]]),
        ([[0]], [1], [1, 2]),
    ],
)
def test_fit_validation_before_quantum(x, y, weights, monkeypatch):
    model = QSVR(kernel())
    monkeypatch.setattr(
        FidelityQuantumKernel,
        "evaluate",
        lambda *a: pytest.fail("invalid fit evaluated quantum circuit"),
    )
    with pytest.raises((TypeError, ValueError)):
        model.fit(x, y, sample_weight=weights)
    assert not model.is_fitted


def test_predict_before_fit_and_invalid_prediction_data():
    model = QSVR(kernel())
    assert not model.is_fitted
    with pytest.raises(ValueError, match="fitted"):
        model.predict([[0]])
    with pytest.raises(ValueError, match="fitted"):
        _ = model.support_
    model.fit([[0]], [1])
    for bad in ([[0, 1]], [[math.nan]], [], [0]):
        with pytest.raises(ValueError):
            model.predict(bad)


def test_failed_refit_preserves_old_state_including_execution_failure(monkeypatch):
    model = QSVR(kernel()).fit([[0], [1]], [0, 1])
    prediction = model.predict([[0.3]])
    old_training = model.training_data_
    with pytest.raises(ValueError):
        model.fit([[0]], [0, 1])

    def fail(*args, **kwargs):
        raise RuntimeError("backend failed")

    with monkeypatch.context() as context:
        context.setattr(FidelityQuantumKernel, "evaluate", fail)
        with pytest.raises(RuntimeError, match="backend failed"):
            model.fit([[2], [3]], [2, 3])
    np.testing.assert_array_equal(model.training_data_, old_training)
    np.testing.assert_array_equal(model.predict([[0.3]]), prediction)


@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize(
    "kind,value", [("singularity", 1.0), ("overflow", 1e308), ("reciprocal", 0.0)]
)
def test_invalid_bound_training_data_rejected_and_failed_refit_is_atomic(
    sampled, kind, value
):
    parameter = Parameter("x")
    circuit = Circuit(1)
    angle = (
        1 / parameter
        if kind == "reciprocal"
        else parameter / (parameter - 1)
        if kind == "singularity"
        else parameter * parameter
    )
    circuit.ry(0, angle)
    kwargs = {"sampler": StatevectorSampler(), "shots": 8} if sampled else {}
    model = QSVR(FidelityQuantumKernel(circuit, ["x"], **kwargs))
    invalid_data = [[value], [value]]
    with pytest.raises((CircuitError, ValueError)):
        model.fit(invalid_data, [0.0, 1.0])
    assert not model.is_fitted

    # Equal valid rows require no sampling and give a deterministic intercept.
    valid_value = 2.0 if kind == "reciprocal" else 0.0
    model.fit([[valid_value], [valid_value]], [0.0, 1.0])
    prediction = model.predict([[valid_value]])
    training_data = model.training_data_
    raw_matrix = model.raw_kernel_matrix_
    coefficients = model.dual_coef_
    with pytest.raises((CircuitError, ValueError)):
        model.fit(invalid_data, [10.0, 20.0])
    assert model.is_fitted
    np.testing.assert_array_equal(model.training_data_, training_data)
    np.testing.assert_array_equal(model.raw_kernel_matrix_, raw_matrix)
    np.testing.assert_array_equal(model.dual_coef_, coefficients)
    np.testing.assert_array_equal(model.predict([[valid_value]]), prediction)


def test_iteration_limit_warning_and_status_are_preserved():
    warning = pytest.importorskip("sklearn.exceptions").ConvergenceWarning
    x = np.linspace(-3, 3, 30).reshape(-1, 1)
    y = np.sin(np.arange(30) * 1.7)
    model = QSVR(kernel(), max_iter=1, tol=1e-12, epsilon=0)
    with pytest.warns(warning):
        model.fit(x, y)
    assert model.is_fitted
    assert model.fit_status_ == 1
    assert model.n_iter_ == 1


def test_score_constant_targets_weights_and_minimum_size():
    model = QSVR(kernel()).fit([[0], [1]], [2, 2])
    assert model.score([[0], [1]], [2, 2]) == 1
    assert model.score([[0], [1]], [3, 3]) == 0
    expected = pytest.importorskip("sklearn.metrics").r2_score(
        [0, 1, 4],
        model.predict([[0], [1], [2]]),
        sample_weight=[1, 2, 3],
    )
    assert model.score(
        [[0], [1], [2]], [0, 1, 4], sample_weight=[1, 2, 3]
    ) == pytest.approx(expected)
    for y, weights in (([2], None), ([2, 2], [0, 1]), ([2, 2], [math.inf, 1])):
        with pytest.raises(ValueError):
            model.score([[0]] if len(y) == 1 else [[0], [1]], y, sample_weight=weights)


def test_package_and_classes_import_without_sklearn_and_fit_has_actionable_error():
    code = """
import importlib.abc
import sys
class BlockSklearn(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sklearn" or fullname.startswith("sklearn."):
            raise ModuleNotFoundError("blocked optional dependency", name="sklearn")
sys.meta_path.insert(0, BlockSklearn())
import cqlib_algorithms
from cqlib_algorithms.algorithms import QBayesian, QSVR
from cqlib_algorithms.operators import FidelityQuantumKernel
assert not any(name == "sklearn" or name.startswith("sklearn.") for name in sys.modules)
from cqlib import Circuit, Parameter
circuit = Circuit(1)
circuit.ry(0, Parameter("x"))
model = QSVR(FidelityQuantumKernel(circuit, ["x"]))
try:
    model.fit([[0], [1]], [0, 1])
except ImportError as error:
    assert "machine-learning" in str(error)
else:
    raise AssertionError("missing optional dependency was not reported")
assert not model.is_fitted
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_unrelated_optional_solver_import_error_is_not_mislabeled(monkeypatch):
    def broken(name):
        raise ModuleNotFoundError("broken transitive dependency", name="unrelated")

    monkeypatch.setattr(
        "cqlib_algorithms.algorithms.machine_learning.qsvr.importlib.import_module",
        broken,
    )
    with pytest.raises(ModuleNotFoundError, match="broken transitive"):
        QSVR(kernel()).fit([[0]], [1])
