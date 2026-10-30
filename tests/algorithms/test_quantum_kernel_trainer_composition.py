# This code is part of Cqlib.
# (C) Copyright China Telecom Quantum Group 2026
# Licensed under the Apache License, Version 2.0; see LICENSE.txt.
"""Independent feature-kernel, custom loss and minimizer integration."""

import ast
import importlib
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from cqlib_algorithms.algorithms import QuantumKernelTrainer
from cqlib_algorithms.algorithms import _ml_common as common
from cqlib_algorithms.algorithms import _ml_validation as validation
from cqlib_algorithms.protocols import (
    QuantumKernel,
    ScalarMinimizerResult,
    TrainableQuantumKernel,
)


class FeatureKernel(QuantumKernel):
    def __init__(self, weight):
        self.weight = float(weight)

    @property
    def num_features(self):
        return 1

    def evaluate(self, X, Y=None):
        x = np.asarray(X, dtype=float)[:, 0]
        y = x if Y is None else np.asarray(Y, dtype=float)[:, 0]
        return x[:, None] * y[None, :] + self.weight**2


class FeatureFamily(TrainableQuantumKernel):
    @property
    def num_features(self):
        return 1

    @property
    def trainable_parameters(self):
        return ("w",)

    def bind_parameters(self, values):
        return FeatureKernel(values["w"])


def square_loss(matrix, targets):
    return float(np.mean((matrix - np.outer(targets, targets)) ** 2))


class GridSearch:
    def __init__(self, points):
        self.points = points

    def __call__(self, objective, initial_point, /):
        initial_point[:] = 999.0
        values = [objective(np.array([point], dtype=float)) for point in self.points]
        index = int(np.argmin(values))
        return ScalarMinimizerResult(
            (self.points[index],), values[index], True, 0, "Grid completed", len(values)
        )


@pytest.mark.reference
def test_custom_kernel_regression_loss_and_replacement_optimizer_without_scipy(
    monkeypatch,
):
    original = common.importlib.import_module

    def absent(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise AssertionError("custom minimizer uses its own search")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(common.importlib, "import_module", absent)
    X = np.array([[-2.0], [0.5], [1.0]])
    y = np.array([-1.5, 0.25, 0.75])
    points = [0.0, 0.5, 1.0]
    model = QuantumKernelTrainer(
        quantum_kernel=FeatureFamily(),
        loss=square_loss,
        optimizer=GridSearch(points),
        max_evaluations=3,
    )
    model.fit(X, y, initial_point={"w": 0.2})
    candidates = []
    for w in points:
        features = [(float(x[0]), w) for x in X]
        matrix = np.array(
            [
                [sum(a * b for a, b in zip(left, right)) for right in features]
                for left in features
            ]
        )
        loss = (
            sum((matrix[i, j] - y[i] * y[j]) ** 2 for i in range(3) for j in range(3))
            / 9
        )
        candidates.append((loss, w, matrix))
    expected = min(candidates, key=lambda row: row[0])
    assert model.optimal_parameters_ == {"w": expected[1]}
    assert model.loss_ == pytest.approx(expected[0])
    np.testing.assert_array_equal(model.kernel_matrix_, expected[2])
    assert model.kernel_matrix_.min() < 0
    assert model.kernel_matrix_.max() > 1
    assert model.objective_evaluations_ == 5
    assert model.result_.optimizer_result.value == model.loss_
    np.testing.assert_array_equal(model.quantum_kernel_.evaluate(X), expected[2])
    np.testing.assert_array_equal(
        FeatureFamily().evaluate(X, parameter_bindings={"w": expected[1]}), expected[2]
    )
    for name in ["classes_", "alignment_", "initial_alignment_"]:
        with pytest.raises(ValueError, match="BalancedKernelAlignmentLoss"):
            getattr(model, name)


def test_custom_loss_and_optimizer_array_ownership_and_independent_results():
    X = np.array([[-2.0], [1.0]])
    y = np.array([0.25, 1.75])

    def mutating_loss(matrix, targets):
        result = square_loss(matrix, targets)
        matrix[:] = 0
        targets[:] = 0
        return result

    model = QuantumKernelTrainer(
        quantum_kernel=FeatureFamily(),
        loss=mutating_loss,
        optimizer=GridSearch([0.0, 1.0]),
    )
    model.fit(X, y, initial_point={"w": 0.2})
    saved = model.result_
    expected = FeatureKernel(saved.optimal_parameters["w"]).evaluate(X)
    np.testing.assert_array_equal(saved.kernel_matrix, expected)
    with pytest.raises(ValueError):
        saved.kernel_matrix[:] = 9
    np.testing.assert_array_equal(saved.kernel_matrix, expected)
    with pytest.raises(TypeError):
        saved.optimal_parameters["w"] = 9
    with pytest.raises(FrozenInstanceError):
        saved.loss = 9
    model.fit(X, [1.0, 3.0], initial_point={"w": 0.8})
    X[:] = 0
    y[:] = 0
    np.testing.assert_array_equal(saved.kernel_matrix, expected)
    np.testing.assert_array_equal(
        saved.quantum_kernel.evaluate([[-2.0], [1.0]]), expected
    )
    assert model.result_ is not saved


@pytest.mark.parametrize("value", [True, 1j, np.nan, np.inf, np.array([0.2])])
def test_bad_custom_loss_preserves_result(value):
    class Loss:
        output = 0.5

        def __call__(self, *args):
            return self.output

    loss = Loss()
    model = QuantumKernelTrainer(
        quantum_kernel=FeatureFamily(), loss=loss, optimizer=GridSearch([0.0])
    )
    model.fit([[-2.0], [1.0]], [0.2, 0.4], initial_point={"w": 0.2})
    previous = model.result_
    loss.output = value
    with pytest.raises((TypeError, ValueError)):
        model.fit([[-2.0], [1.0]], [0.2, 0.4], initial_point={"w": 0.2})
    assert model.result_ is previous


@pytest.mark.parametrize("kind", ["count", "dimension", "type", "budget", "candidate"])
def test_bad_custom_minimizer(kind):
    def optimizer(objective, point, /):
        if kind == "candidate":
            objective(np.array([1.0, 2.0]))
        value = objective(point)
        if kind == "budget":
            objective(point)
        if kind == "type":
            return object()
        return ScalarMinimizerResult(
            (0.2, 0.4) if kind == "dimension" else (0.2,),
            value,
            True,
            0,
            "Done",
            2 if kind == "count" else 1,
        )

    model = QuantumKernelTrainer(
        quantum_kernel=FeatureFamily(),
        loss=square_loss,
        optimizer=optimizer,
        max_evaluations=1,
    )
    with pytest.raises((TypeError, ValueError)):
        model.fit([[0.0], [1.0]], [0.25, 1.75], initial_point={"w": 0.2})


@pytest.mark.parametrize(
    "matrix",
    [
        np.ones((3, 3)),
        [[1, 0], [1, 1]],
        [[1, np.nan], [np.nan, 1]],
        [[True, False], [False, True]],
    ],
)
def test_bad_generic_gram(monkeypatch, matrix):
    monkeypatch.setattr(FeatureKernel, "evaluate", lambda *args: matrix)
    with pytest.raises((TypeError, ValueError)):
        QuantumKernelTrainer(
            quantum_kernel=FeatureFamily(),
            loss=square_loss,
            optimizer=GridSearch([0.2]),
        ).fit([[0.0], [1.0]], [0.25, 1.75], initial_point={"w": 0.2})


@pytest.mark.parametrize(
    "values,width,exception,message",
    [
        ([[True]], 1, TypeError, "X must contain real numeric values, excluding bool"),
        ([[1j]], 1, TypeError, "X must contain real numeric values, excluding bool"),
        ([[np.nan]], 1, ValueError, "X must contain finite values"),
        ([[1.0, 2.0]], 1, ValueError, "X feature width must match input_parameters"),
        (
            [[1.0], [2.0, 3.0]],
            1,
            ValueError,
            "X must be a rectangular real numeric array",
        ),
    ],
)
def test_shared_validation_messages(values, width, exception, message):
    with pytest.raises(exception) as info:
        validation.data(values, width)
    assert str(info.value) == message


def test_shared_module_dependency_direction():
    base = Path(common.__file__).parent
    for name in ["_ml_common", "_ml_validation", "vqr", "pegasos_qsvc"]:
        tree = ast.parse((base / f"{name}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "qsvr"
                assert not (
                    node.module
                    and node.module.endswith(
                        (
                            "compute_uncompute",
                            "amplitude_estimation.problem",
                            "quantum_kernel._validation",
                        )
                    )
                )
    tree = ast.parse((base / "qsvr.py").read_text(encoding="utf-8"))
    assert not any(
        isinstance(n, ast.Attribute) and n.attr == "_validate_data"
        for n in ast.walk(tree)
    )


def test_protocol_public_exports_and_minimizer_result_validation():
    import cqlib_algorithms as root
    from cqlib_algorithms import protocols

    for name in [
        "QuantumKernel",
        "TrainableQuantumKernel",
        "KernelLoss",
        "ScalarMinimizer",
        "ScalarMinimizerResult",
    ]:
        assert getattr(root, name) is getattr(protocols, name)
    result = ScalarMinimizerResult((0.2,), 0.3, True, 0, "Done", 1)
    for changes in [
        {"point": (np.inf,)},
        {"point": (True,)},
        {"value": np.nan},
        {"value": True},
        {"success": 1},
        {"status": True},
        {"message": None},
        {"evaluations": 0},
        {"evaluations": True},
    ]:
        with pytest.raises((TypeError, ValueError)):
            replace(result, **changes)
    assert (
        importlib.util.find_spec("cqlib_algorithms.algorithms.quantum_kernel") is None
    )
