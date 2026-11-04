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

"""Pegasos verified against independent Euclidean feature-space updates."""

from __future__ import annotations

import math

import numpy as np
import pytest
from cqlib import Circuit, Parameter
from cqlib.device import ExecutionResult

from cqlib_algorithms.algorithms.machine_learning.pegasos_qsvc import (
    PegasosQSVC,
    _norm_squared,
)
from cqlib_algorithms.operators.quantum_kernel import FidelityQuantumKernel
from cqlib_algorithms.protocols.sampler import Sampler


def kernel(**kwargs):
    circuit = Circuit(1)
    circuit.ry(0, Parameter("x"))
    return FidelityQuantumKernel(circuit, ["x"], **kwargs)


def features(X):
    angles = np.asarray(X)[:, 0] / 2
    return np.column_stack(
        [
            np.cos(angles) ** 2,
            np.sqrt(2) * np.cos(angles) * np.sin(angles),
            np.sin(angles) ** 2,
        ]
    )


def euclidean_reference(phi, labels, regularization, steps, seed):
    w = np.zeros(phi.shape[1])
    rng = np.random.default_rng(seed)
    for t in range(1, steps + 1):
        i = rng.integers(len(labels))
        old_margin = labels[i] * np.dot(w, phi[i])
        w = (1 - 1 / t) * w
        if old_margin < 1:
            w = w + labels[i] * phi[i] / (regularization * t)
        norm = np.linalg.norm(w)
        if norm:
            w *= min(1.0, 1 / (np.sqrt(regularization) * norm))
    return w


@pytest.mark.reference
@pytest.mark.parametrize(
    "regularization,steps", [(0.1, 1), (0.3, 2), (1.0, 7), (0.2, 101), (2.0, 51)]
)
def test_real_kernel_matches_euclidean_projected_reference(regularization, steps):
    X = np.array([[0.1], [0.7], [2.5], [3.1], [0.1]])
    labels = np.array([-1.0, -1.0, 1.0, 1.0, 1.0])
    classifier = PegasosQSVC(
        kernel(), regularization=regularization, num_steps=steps, seed=23
    ).fit(X, labels)
    phi = features(X)
    w = euclidean_reference(phi, labels, regularization, steps, 23)
    holdout = np.array([[0.2], [1.4], [2.9], [-0.5]])
    np.testing.assert_allclose(
        classifier.decision_function(X), phi @ w, rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(
        classifier.decision_function(holdout), features(holdout) @ w, rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(
        classifier.weight_norm_, np.linalg.norm(w), rtol=0, atol=1e-10
    )
    expected_objective = regularization * np.dot(w, w) / 2 + np.mean(
        np.maximum(0.0, 1 - labels * (phi @ w))
    )
    assert classifier.training_objective_ == pytest.approx(
        expected_objective, abs=1e-11
    )
    assert classifier.weight_norm_ <= 1 / np.sqrt(regularization) + 1e-12
    assert classifier.n_iter_ == steps


def test_first_step_projection_is_mandatory():
    classifier = PegasosQSVC(kernel(), regularization=0.04, num_steps=1, seed=0).fit(
        [[0.0], [math.pi]], [-1.0, 1.0]
    )
    assert classifier.weight_norm_ == pytest.approx(5.0)
    assert np.count_nonzero(classifier.coefficients_) == 1
    assert max(abs(classifier.coefficients_)) == pytest.approx(5.0)


class MatrixKernel(FidelityQuantumKernel):
    def __init__(self, matrix):
        circuit = Circuit(1)
        circuit.ry(0, Parameter("x"))
        super().__init__(circuit, ["x"])
        self.matrix = np.array(matrix)
        self.cross = None

    def evaluate(self, X, Y=None):
        if Y is None:
            return self.matrix
        return self.cross if self.cross is not None else self.matrix[: len(X)]


@pytest.mark.reference
def test_psd_positive_part_not_pseudoinverse_and_raw_prediction():
    raw = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, 0.0], [0.9, 0.0, 1.0]])
    qkernel = MatrixKernel(raw)
    classifier = PegasosQSVC(qkernel, regularization=0.2, num_steps=70, seed=4).fit(
        [[0.0], [1.0], [2.0]], [-1.0, 1.0, 1.0]
    )
    # Closed-form eigenvectors/eigenvalues independent of production eigh.
    vectors = np.column_stack(
        [
            [1 / np.sqrt(2), 0.5, 0.5],
            [0.0, 1 / np.sqrt(2), -1 / np.sqrt(2)],
            [1 / np.sqrt(2), -0.5, -0.5],
        ]
    )
    eigenvalues = np.array([1 + 0.9 * np.sqrt(2), 1.0, 0.0])
    positive = (vectors * eigenvalues) @ vectors.T
    np.testing.assert_allclose(classifier.kernel_matrix_, positive, atol=1e-14)
    assert classifier.minimum_raw_eigenvalue_ == pytest.approx(1 - 0.9 * np.sqrt(2))
    assert classifier.psd_correction_norm_ == pytest.approx(0.9 * np.sqrt(2) - 1)
    # Independent Euclidean realization of the repaired training feature map.
    phi = vectors * np.sqrt(eigenvalues)
    w = euclidean_reference(phi, [-1.0, 1.0, 1.0], 0.2, 70, 4)
    np.testing.assert_allclose(positive @ classifier.coefficients_, phi @ w, atol=1e-11)
    np.testing.assert_allclose(
        classifier.decision_function([[0.0], [1.0], [2.0]]),
        raw @ classifier.coefficients_,
        atol=1e-14,
    )
    assert not np.allclose(
        raw @ classifier.coefficients_, positive @ classifier.coefficients_
    )
    qkernel.cross = np.array([[0.2, 0.3, 0.4]])
    np.testing.assert_allclose(
        classifier.decision_function([[5.0]]), qkernel.cross @ classifier.coefficients_
    )


def test_exact_margin_one_uses_strict_inequality():
    # Orthogonal unit features with lambda=1 reach exactly margin one at t=1;
    # when the same class is sampled at t=2, only shrinkage may occur.
    qkernel = MatrixKernel([[1.0, 0.0], [0.0, 1.0]])
    seed = next(
        seed
        for seed in range(100)
        if len(set(np.random.default_rng(seed).integers(2, size=2))) == 1
    )
    classifier = PegasosQSVC(qkernel, regularization=1.0, num_steps=2, seed=seed).fit(
        [[0.0], [0.0]], [-1.0, 1.0]
    )
    assert classifier.weight_norm_ == pytest.approx(0.5)


def test_separable_train_holdout_original_labels_and_seed():
    X = [[-0.2], [0.0], [0.2], [math.pi - 0.2], [math.pi], [math.pi + 0.2]]
    y = [3.0, 3.0, 3.0, 8.0, 8.0, 8.0]
    np.random.seed(192)
    state = np.random.get_state()
    classifier = PegasosQSVC(kernel(), num_steps=400, seed=19).fit(X, y)
    assert classifier.score(X, y) == 1.0
    assert classifier.score([[-0.1], [0.1], [3.0], [3.3]], [3.0, 3.0, 8.0, 8.0]) == 1.0
    assert classifier.classes_ == (3.0, 8.0)
    other = PegasosQSVC(kernel(), num_steps=400, seed=19).fit(X, y)
    np.testing.assert_array_equal(classifier.coefficients_, other.coefficients_)
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


def test_zero_score_tie_and_defensive_diagnostics():
    qkernel = MatrixKernel([[1.0, 0.0], [0.0, 1.0]])
    X = np.array([[0.0], [1.0]])
    classifier = PegasosQSVC(qkernel, num_steps=1, seed=0).fit(X, [-2.0, 5.0])
    before = classifier.coefficients_.copy()
    X[:] = 99
    for name in [
        "training_data_",
        "coefficients_",
        "raw_kernel_matrix_",
        "kernel_matrix_",
    ]:
        getattr(classifier, name)[:] = 19
    np.testing.assert_array_equal(classifier.coefficients_, before)
    np.testing.assert_array_equal(classifier.training_data_, [[0.0], [1.0]])
    qkernel.cross = np.zeros((1, 2))
    np.testing.assert_array_equal(classifier.predict([[2.0]]), [5.0])
    assert classifier.predict(np.empty((0, 1))).shape == (0,)


class ConstantPairSampler(Sampler):
    calls = 0

    def run(self, requests, /):
        self.calls += 1
        return [
            ExecutionResult.from_counts(
                "pair",
                list(request.qubits),
                request.shots,
                len(request.qubits),
                {"0": request.shots // 2, "1": request.shots - request.shots // 2},
                "fixed",
            )
            for request in requests
        ]


def test_real_sampled_kernel_training_path():
    sampler = ConstantPairSampler()
    classifier = PegasosQSVC(
        kernel(sampler=sampler, shots=20), num_steps=20, seed=3
    ).fit([[0.0], [1.0]], [-1.0, 1.0])
    np.testing.assert_allclose(classifier.raw_kernel_matrix_, [[1.0, 0.5], [0.5, 1.0]])
    assert sampler.calls == 1
    classifier.decision_function([[0.4]])
    assert sampler.calls == 2


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("regularization", 0),
        ("regularization", -1),
        ("regularization", np.inf),
        ("regularization", True),
        ("num_steps", 0),
        ("num_steps", True),
        ("num_steps", 1.2),
        ("seed", True),
        ("seed", -1),
        ("seed", 1.2),
    ],
)
def test_invalid_configurations(keyword, value):
    with pytest.raises((TypeError, ValueError)):
        PegasosQSVC(kernel(), **{keyword: value})


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
        PegasosQSVC(kernel()).fit([[0.0], [1.0]], y)


@pytest.mark.parametrize(
    "X",
    [
        np.empty((0, 1)),
        [[True], [False]],
        [[np.inf], [1.0]],
        [0.0, 1.0],
        [[0.0, 1.0], [1.0, 0.0]],
        [[1.0], [1.0, 2.0]],
    ],
)
def test_invalid_inputs(X):
    with pytest.raises((TypeError, ValueError)):
        PegasosQSVC(kernel()).fit(X, [0.0, 1.0])


@pytest.mark.parametrize(
    "bad",
    [
        np.ones((3, 3)),
        [[1.0, 0.2], [0.3, 1.0]],
        [[1.0, np.nan], [np.nan, 1.0]],
        [[1.0, 1.1], [1.1, 1.0]],
        [[1.0, -0.1], [-0.1, 1.0]],
        [[True, False], [False, True]],
        [1.0, 1.0],
    ],
)
def test_bad_kernel_and_atomic_refit(bad):
    qkernel = MatrixKernel([[1.0, 0.0], [0.0, 1.0]])
    classifier = PegasosQSVC(qkernel, num_steps=3, seed=1).fit(
        [[0.0], [1.0]], [0.0, 1.0]
    )
    before = classifier.coefficients_.copy()
    qkernel.matrix = np.array(bad)
    with pytest.raises((TypeError, ValueError)):
        classifier.fit([[0.0], [1.0]], [0.0, 1.0])
    np.testing.assert_array_equal(classifier.coefficients_, before)


@pytest.mark.parametrize(
    "name",
    [
        "classes_",
        "n_features_in_",
        "training_data_",
        "coefficients_",
        "raw_kernel_matrix_",
        "kernel_matrix_",
        "weight_norm_",
        "training_objective_",
        "n_iter_",
        "minimum_raw_eigenvalue_",
        "psd_correction_norm_",
    ],
)
def test_unfitted_diagnostics(name):
    with pytest.raises(ValueError, match="fitted"):
        getattr(PegasosQSVC(kernel()), name)


def test_unfitted_prediction_bad_prediction_and_score():
    with pytest.raises(TypeError):
        PegasosQSVC(object())
    classifier = PegasosQSVC(kernel())
    for action in [
        lambda: classifier.predict([[0.0]]),
        lambda: classifier.score([[0.0]], [1.0]),
    ]:
        with pytest.raises(ValueError, match="fitted"):
            action()
    qkernel = MatrixKernel([[1.0, 0.0], [0.0, 1.0]])
    classifier = PegasosQSVC(qkernel).fit([[0.0], [1.0]], [0.0, 1.0])
    qkernel.cross = np.ones((1, 3))
    with pytest.raises(ValueError, match="rectangular"):
        classifier.predict([[0.4]])
    with pytest.raises(ValueError, match="classes"):
        classifier.score([[0.4]], [3.0])
    with pytest.raises(ValueError):
        classifier.score(np.empty((0, 1)), [])


def test_significant_negative_norm_and_numeric_overflow():
    with pytest.raises(ValueError, match="negative"):
        _norm_squared(np.array([1.0]), np.array([[-1.0]]))
    with pytest.raises(ValueError, match="overflow|finite"):
        PegasosQSVC(kernel(), regularization=1e-320, num_steps=1).fit(
            [[0.0], [1.0]], [-1.0, 1.0]
        )
