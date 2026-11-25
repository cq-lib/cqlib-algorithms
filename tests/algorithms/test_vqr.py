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

"""Independent circuit-learning references and transactional boundary tests."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from cqlib import Circuit, Parameter, Qubit
from cqlib.circuit import CircuitError
from cqlib.device import ExecutionResult
from cqlib.qis import Hamiltonian, PauliString, Phase

from cqlib_algorithms.algorithms.machine_learning import _common as common
from cqlib_algorithms.algorithms.machine_learning.vqr import VQR
from cqlib_algorithms.operators.estimator import SamplingEstimator, StatevectorEstimator
from cqlib_algorithms.protocols.estimator import Estimator, EstimatorResult
from cqlib_algorithms.protocols.sampler import Sampler
from cqlib_algorithms.protocols.scalar_minimizer import ScalarMinimizerResult


def model(**kwargs):
    circuit = Circuit(1)
    circuit.ry(0, Parameter("x"))
    circuit.ry(0, Parameter("theta"))
    return VQR(
        circuit,
        ["x"],
        ["theta"],
        PauliString.from_str("Z"),
        kwargs.pop("estimator", StatevectorEstimator()),
        **kwargs,
    )


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


class RecordingEstimator(Estimator):
    def __init__(self):
        self.batches = []
        self.response = None

    def run(self, requests, /):
        self.batches.append(tuple(requests))
        if self.response is not None:
            return self.response
        return StatevectorEstimator().run(requests)


@pytest.mark.reference
def test_real_cobyla_learns_cosine_and_generalizes():
    X = np.linspace(-1.4, 1.5, 9)[:, None]
    y = np.cos(X[:, 0] + 0.43)
    regressor = model(max_evaluations=150).fit(X, y, initial_point={"theta": -0.7})
    holdout = np.array([[-1.21], [0.13], [1.03]])
    assert regressor.training_loss_ < 1e-8
    np.testing.assert_allclose(
        regressor.predict(holdout), np.cos(holdout[:, 0] + 0.43), atol=5e-4, rtol=0
    )
    assert regressor.score(X, y) > 0.999999
    assert regressor.n_features_in_ == 1
    assert 0 < regressor.optimizer_evaluations_ <= 150
    assert regressor.objective_evaluations_ == regressor.optimizer_evaluations_ + 2
    assert isinstance(regressor.optimizer_status_, int)
    assert isinstance(regressor.optimizer_message_, str)


def test_vqr_composes_features_and_ansatz_for_a_cosine_regression():
    from cqlib.circuit.gates import StandardGate

    from cqlib_algorithms.operators import AngleEncoding, RealAmplitudesAnsatz

    X = np.linspace(-1.4, 1.5, 9)[:, None]
    targets = np.cos(X[:, 0] + 0.43)
    regressor = VQR.from_components(
        AngleEncoding(1, StandardGate.RY),
        RealAmplitudesAnsatz(1, reps=0),
        PauliString.from_str("Z"),
        StatevectorEstimator(),
        max_evaluations=150,
    )
    assert regressor.input_parameters == ("x_0",)
    assert regressor.weight_parameters == ("theta_0",)
    regressor.fit(X, targets, initial_point={"theta_0": -0.7})
    legacy = model(max_evaluations=150).fit(X, targets, initial_point={"theta": -0.7})
    holdout = np.array([[-1.21], [0.13], [1.03]])
    np.testing.assert_allclose(
        regressor.predict(holdout), np.cos(holdout[:, 0] + 0.43), atol=5e-4
    )
    np.testing.assert_allclose(
        regressor.predict(holdout), legacy.predict(holdout), atol=1e-12
    )


def test_vqr_component_order_and_data_column_order(monkeypatch):
    from cqlib.circuit.gates import StandardGate

    from cqlib_algorithms.operators import AngleEncoding, RealAmplitudesAnsatz

    fixed_solver(monkeypatch, [0.4, 0.2])
    columns = ["x_1", "x_0"]
    regressor = VQR.from_components(
        AngleEncoding(2, StandardGate.RX),
        RealAmplitudesAnsatz(2, reps=0),
        PauliString.from_str("IX"),
        StatevectorEstimator(),
        input_parameters=columns,
    )
    columns.reverse()
    data = [[0.8, -0.7], [-0.1, 1.0]]
    # On qubit zero, RX(x_0) followed by RY(theta_0) yields <X> = sin(theta_0) cos(x_0).
    reference = np.sin(0.4) * np.cos(np.asarray(data)[:, 1])
    regressor.fit(data, reference, initial_point={"theta_0": 0.4, "theta_1": 0.2})
    np.testing.assert_allclose(regressor.predict(data), reference, atol=1e-12)


@pytest.mark.parametrize(
    "case,error",
    [
        ("wrong_feature_type", TypeError),
        ("wrong_ansatz_type", TypeError),
        ("width", ValueError),
        ("overlap", ValueError),
        ("invalid_order", ValueError),
        ("duplicate", ValueError),
        ("empty_features", ValueError),
        ("empty_weights", ValueError),
    ],
)
def test_vqr_component_configuration_boundaries(case, error):
    from cqlib.circuit.gates import StandardGate

    from cqlib_algorithms.operators import (
        AngleEncoding,
        RealAmplitudesAnsatz,
        UniformSuperposition,
        UserCircuitPreparation,
    )

    encoding = AngleEncoding(1, StandardGate.RY)
    ansatz = RealAmplitudesAnsatz(1, reps=0)
    order = None
    if case == "wrong_feature_type":
        encoding = Circuit(1)
    elif case == "wrong_ansatz_type":
        ansatz = Circuit(1)
    elif case == "width":
        ansatz = RealAmplitudesAnsatz(2, reps=0)
    elif case == "overlap":
        circuit = Circuit(1)
        circuit.ry(0, Parameter("theta_0"))
        encoding = UserCircuitPreparation(circuit)
    elif case == "invalid_order":
        order = ["theta_0"]
    elif case == "duplicate":
        order = ["x_0", "x_0"]
    elif case == "empty_features":
        encoding = UniformSuperposition(1)
    elif case == "empty_weights":
        ansatz = RealAmplitudesAnsatz(1, reps=0, skip_final_rotation_layer=True)
    with pytest.raises(error):
        VQR.from_components(
            encoding,
            ansatz,
            PauliString.from_str("Z"),
            StatevectorEstimator(),
            input_parameters=order,
        )


@pytest.mark.reference
@pytest.mark.parametrize(
    "pauli,expected",
    [
        ("IZ", lambda x: np.cos(2 * x[:, 0] + 0.4)),
        ("ZI", lambda x: np.cos(x[:, 1] - 0.2)),
    ],
)
def test_qubit_insertion_order_explicit_columns_and_snapshot(
    monkeypatch, pauli, expected
):
    circuit = Circuit([7, 2])
    circuit.ry(Qubit(7), Parameter("z") * 2 + Parameter("w"))
    circuit.ry(Qubit(2), Parameter("a") + Parameter("fixed"))
    estimator = RecordingEstimator()
    inputs, weights, fixed = ["z", "a"], ["w"], {"fixed": -0.2}
    observable = PauliString.from_str(pauli)
    regressor = VQR(
        circuit, inputs, weights, observable, estimator, parameter_bindings=fixed
    )
    observable.phase = Phase.minus()
    inputs.reverse()
    weights[0] = "bad"
    fixed["fixed"] = 2
    circuit.x(Qubit(7))
    fixed_solver(monkeypatch, [0.4])
    X = np.array([[0.1, 0.8], [-0.2, 0.1], [0.5, -0.3]])
    regressor.fit(X, expected(X), initial_point={"w": 0.4})
    np.testing.assert_allclose(regressor.predict(X), expected(X), atol=1e-12, rtol=0)
    assert len(estimator.batches) == 4
    for batch in estimator.batches:
        assert len(batch) == 3
        assert all(
            not request.circuit.used_symbols and not request.parameter_bindings
            for request in batch
        )
    with pytest.raises(TypeError):
        regressor.optimal_parameters_["w"] = 2


@pytest.mark.reference
def test_y_observable_uses_existing_correct_estimator(monkeypatch):
    circuit = Circuit(1)
    circuit.rx(0, Parameter("x") + Parameter("w"))
    regressor = VQR(
        circuit, ["x"], ["w"], PauliString.from_str("Y"), StatevectorEstimator()
    )
    fixed_solver(monkeypatch)
    X = np.array([[0.2], [0.8]])
    regressor.fit(X, -np.sin(X[:, 0] + 0.3), initial_point={"w": 0.3})
    np.testing.assert_allclose(regressor.predict(X), -np.sin(X[:, 0] + 0.3), atol=1e-12)


class ConstantCountSampler(Sampler):
    def __init__(self):
        self.requests = []

    def run(self, requests, /):
        self.requests.extend(requests)
        return [
            ExecutionResult.from_counts(
                "fixed",
                list(request.qubits),
                request.shots,
                len(request.qubits),
                {"0": 7, "1": 3},
                "fixed",
            )
            for request in requests
        ]


def test_sampling_estimator_real_shot_path(monkeypatch):
    sampler = ConstantCountSampler()
    fixed_solver(monkeypatch)
    regressor = model(estimator=SamplingEstimator(sampler), shots=10).fit(
        [[0.0], [1.0]], [0.4, 0.4], initial_point={"theta": 0.3}
    )
    np.testing.assert_allclose(regressor.predict([[2.0], [3.0]]), [0.4, 0.4])
    assert regressor.training_loss_ < 1e-30
    assert sampler.requests and all(request.shots == 10 for request in sampler.requests)


@pytest.mark.parametrize(
    "X",
    [
        [],
        [[True]],
        [[1j]],
        [["1"]],
        [[object()]],
        [[np.inf]],
        [[np.nan]],
        [1.0],
        [[1.0, 2.0]],
        [[1.0], [1.0, 2.0]],
    ],
)
def test_invalid_data_never_executes(X):
    estimator = RecordingEstimator()
    regressor = model(estimator=estimator)
    with pytest.raises((TypeError, ValueError)):
        regressor.fit(X, [0.0], initial_point={"theta": 0.0})
    assert not regressor.is_fitted and not estimator.batches


@pytest.mark.parametrize(
    "y", [[], [[1.0]], [True], [1j], ["1"], [np.inf], [np.nan], [1.0, 2.0]]
)
def test_invalid_targets(y):
    with pytest.raises((TypeError, ValueError)):
        model().fit([[0.0]], y, initial_point={"theta": 0.0})


@pytest.mark.parametrize(
    "initial",
    [
        {},
        {"bad": 0.0},
        {"theta": True},
        {"theta": np.inf},
        {"theta": 1j},
        {"theta": "1"},
        [("theta", 1.0)],
        None,
    ],
)
def test_invalid_initial_points(initial):
    with pytest.raises((TypeError, ValueError)):
        model().fit([[0.0]], [0.0], initial_point=initial)


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("max_evaluations", True),
        ("max_evaluations", 2),
        ("max_evaluations", 1.2),
        ("tolerance", 0),
        ("tolerance", np.inf),
        ("tolerance", True),
        ("initial_step", -1),
        ("initial_step", np.nan),
        ("shots", 0),
        ("shots", True),
    ],
)
def test_invalid_configurations(keyword, value):
    with pytest.raises((TypeError, ValueError)):
        model(**{keyword: value})


@pytest.mark.parametrize(
    "inputs,weights,fixed",
    [
        ("x", ["theta"], None),
        ([], ["theta"], None),
        (["x", "x"], ["theta"], None),
        ([1], ["theta"], None),
        (["x"], [], None),
        (["x"], ["x"], None),
        (["x"], ["bad"], None),
        (["x"], ["theta"], {"x": 1.0}),
    ],
)
def test_bad_symbol_partition(inputs, weights, fixed):
    circuit = Circuit(1)
    circuit.ry(0, Parameter("x") + Parameter("theta"))
    with pytest.raises((TypeError, ValueError)):
        VQR(
            circuit,
            inputs,
            weights,
            PauliString.from_str("Z"),
            StatevectorEstimator(),
            parameter_bindings=fixed,
        )


@pytest.mark.parametrize(
    "name",
    [
        "predict",
        "score",
        "n_features_in_",
        "optimal_parameters_",
        "training_loss_",
        "initial_loss_",
        "optimizer_success_",
        "optimizer_status_",
        "optimizer_message_",
        "optimizer_evaluations_",
        "objective_evaluations_",
    ],
)
def test_unfitted_errors(name):
    regressor = model()
    with pytest.raises(ValueError, match="fitted"):
        if name == "predict":
            regressor.predict([[0.0]])
        elif name == "score":
            regressor.score([[0.0], [1.0]], [0.0, 1.0])
        else:
            getattr(regressor, name)


@pytest.mark.parametrize(
    "response",
    [(), [EstimatorResult(0.0)], [object(), object()], [EstimatorResult(0.0)] * 3],
)
def test_malformed_estimator_results(response):
    estimator = RecordingEstimator()
    estimator.response = response
    regressor = model(estimator=estimator)
    with pytest.raises((TypeError, ValueError)):
        regressor.fit([[0.0], [1.0]], [0.0, 1.0], initial_point={"theta": 0.2})
    assert not regressor.is_fitted


@pytest.mark.parametrize(
    "expression,bad", [(lambda x: 1 / x, 0.0), (lambda x: x * x, 1e308)]
)
def test_all_rows_bound_before_any_estimator_execution(expression, bad):
    x = Parameter("x")
    circuit = Circuit(1)
    circuit.ry(0, expression(x) + Parameter("w"))
    estimator = RecordingEstimator()
    regressor = VQR(circuit, ["x"], ["w"], PauliString.from_str("Z"), estimator)
    values = [[1.0], [bad]]
    with pytest.raises((ValueError, CircuitError, OverflowError)):
        regressor.fit(values, [0.0, 0.0], initial_point={"w": 0.1})
    assert not estimator.batches


def test_domain_is_not_probed_at_zero_and_failed_refit_is_atomic(monkeypatch):
    circuit = Circuit(1)
    circuit.ry(0, 1 / Parameter("x") + 1 / Parameter("w"))
    regressor = VQR(
        circuit, ["x"], ["w"], PauliString.from_str("Z"), StatevectorEstimator()
    )
    fixed_solver(monkeypatch)
    X = [[1.0], [2.0]]
    regressor.fit(X, [0.0, 1.0], initial_point={"w": 2.0})
    before = regressor.predict(X)
    with pytest.raises(CircuitError):
        regressor.fit(X, [0.0, 1.0], initial_point={"w": 0.0})
    np.testing.assert_array_equal(regressor.predict(X), before)


def test_final_independent_evaluation_and_empty_prediction(monkeypatch):
    class Counter(Estimator):
        calls = 0

        def run(self, requests, /):
            self.calls += 1
            return [EstimatorResult(float(self.calls)) for request in requests]

    estimator = Counter()
    fixed_solver(monkeypatch)
    regressor = model(estimator=estimator).fit(
        [[0.0], [1.0]], [0.0, 0.0], initial_point={"theta": 0.0}
    )
    assert regressor.initial_loss_ == 1.0 and regressor.training_loss_ == 9.0
    assert regressor.objective_evaluations_ == 3
    assert not regressor.optimizer_success_
    assert regressor.predict(np.empty((0, 1))).shape == (0,)
    assert estimator.calls == 3


@pytest.mark.parametrize(
    "changes",
    [
        {"x": [np.nan]},
        {"x": [0.0, 1.0]},
        {"success": "yes"},
        {"status": True},
        {"message": 2},
        {"nfev": 2},
        {"fun": np.inf},
    ],
)
def test_invalid_optimizer_result_keeps_previous_model(monkeypatch, changes):
    fixed_solver(monkeypatch)
    regressor = model().fit([[0.0], [1.0]], [0.0, 1.0], initial_point={"theta": 0.3})
    before = regressor.predict([[0.2]])
    fixed_solver(monkeypatch, **changes)
    with pytest.raises((TypeError, ValueError)):
        regressor.fit([[0.0], [1.0]], [0.0, 1.0], initial_point={"theta": 0.7})
    np.testing.assert_array_equal(regressor.predict([[0.2]]), before)


def test_budget_truthful_nonconvergence_and_constant_r2():
    regressor = model(max_evaluations=3).fit(
        [[0.0], [1.0]], [0.0, 1.0], initial_point={"theta": 0.2}
    )
    assert regressor.is_fitted and not regressor.optimizer_success_
    assert regressor.optimizer_evaluations_ <= 3
    prediction = regressor.predict([[0.4], [0.4]])
    assert regressor.score([[0.4], [0.4]], prediction) == 1.0
    assert regressor.score([[0.4], [0.4]], [100.0, 100.0]) == 0.0
    with pytest.raises(ValueError):
        regressor.score([[0.0]], [1.0])


def test_loss_overflow_and_missing_scipy(monkeypatch):
    with pytest.raises(ValueError, match="finite|overflow"):
        model().fit([[0.0]], [1e308], initial_point={"theta": 0.0})
    original = common.importlib.import_module

    def absent(name, *args, **kwargs):
        if name == "scipy.optimize":
            raise ModuleNotFoundError("missing scipy", name="scipy")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(common.importlib, "import_module", absent)
    with pytest.raises(ImportError, match="machine-learning"):
        model().fit([[0.0]], [0.0], initial_point={"theta": 0.0})


@pytest.mark.parametrize("kind", ["measure", "reset"])
def test_rejects_nonunitary_structure(kind):
    circuit = Circuit(1)
    circuit.ry(0, Parameter("x") + Parameter("theta"))
    if kind == "measure":
        circuit.measure(0)
    else:
        circuit.reset(0)
    with pytest.raises((ValueError, CircuitError)):
        VQR(
            circuit, ["x"], ["theta"], PauliString.from_str("Z"), StatevectorEstimator()
        )


def test_rejects_bad_observable_and_estimator():
    circuit = Circuit(1)
    circuit.ry(0, Parameter("x") + Parameter("theta"))
    nonhermitian = PauliString.from_str("Z")
    nonhermitian.phase = Phase.i()
    for observable in [object(), PauliString.from_str("ZZ"), nonhermitian]:
        with pytest.raises((TypeError, ValueError)):
            VQR(circuit, ["x"], ["theta"], observable, StatevectorEstimator())
    with pytest.raises(TypeError):
        VQR(circuit, ["x"], ["theta"], PauliString.from_str("Z"), object())


def test_hamiltonian_prediction_and_defensive_copy(monkeypatch):
    circuit = Circuit(1)
    circuit.ry(0, Parameter("x") + Parameter("w"))
    observable = Hamiltonian(1)
    observable.add_term(PauliString.from_str("Z"), 2.0)
    observable.add_term(PauliString.from_str("X"), 0.3)
    regressor = VQR(circuit, ["x"], ["w"], observable, StatevectorEstimator())
    observable.add_term(PauliString.from_str("I"), 100.0)
    fixed_solver(monkeypatch)
    X = np.array([[0.1], [0.7]])
    expected = 2 * np.cos(X[:, 0] + 0.2) + 0.3 * np.sin(X[:, 0] + 0.2)
    regressor.fit(X, expected, initial_point={"w": 0.2})
    np.testing.assert_allclose(regressor.predict(X), expected, atol=1e-12)


@pytest.mark.parametrize(
    "field,value",
    [
        ("expectation", True),
        ("expectation", np.nan),
        ("variance", -1.0),
        ("variance", True),
    ],
)
def test_mutated_estimator_result_revalidated(field, value):
    result = EstimatorResult(0.0, 0.0)
    object.__setattr__(result, field, value)
    estimator = RecordingEstimator()
    estimator.response = [result]
    with pytest.raises((TypeError, ValueError)):
        model(estimator=estimator).fit([[0.0]], [0.0], initial_point={"theta": 0.1})


def test_final_evaluation_failure_preserves_existing_model(monkeypatch):
    estimator = RecordingEstimator()
    fixed_solver(monkeypatch)
    regressor = model(estimator=estimator).fit(
        [[0.0]], [0.0], initial_point={"theta": 0.3}
    )
    old = dict(regressor.optimal_parameters_)
    original = estimator.run
    calls = 0

    def terminal_failure(requests):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("terminal execution failed")
        return original(requests)

    estimator.run = terminal_failure
    with pytest.raises(RuntimeError, match="terminal"):
        regressor.fit([[0.0]], [0.0], initial_point={"theta": 0.7})
    assert dict(regressor.optimal_parameters_) == old


@pytest.mark.parametrize("field", ["rotation", "phase"])
def test_invalid_fixed_expression_rejected_without_domain_probe(field):
    circuit = Circuit(1)
    circuit.ry(0, Parameter("x") + Parameter("theta"))
    if field == "rotation":
        circuit.rz(0, Parameter("fixed") * Parameter("fixed"))
    else:
        circuit.set_global_phase(Parameter("fixed") * Parameter("fixed"))
    with pytest.raises((ValueError, CircuitError)):
        VQR(
            circuit,
            ["x"],
            ["theta"],
            PauliString.from_str("Z"),
            StatevectorEstimator(),
            parameter_bindings={"fixed": 1e308},
        )


@pytest.mark.reference
def test_custom_scalar_optimizer_drives_predictions_and_mse(monkeypatch):
    observed = []
    candidates = [-0.4, 0.43, 1.2]

    def search(objective, initial_point):
        initial_point[:] = 9.0
        values = [objective(np.array([theta])) for theta in candidates]
        observed.extend(values)
        selected = int(np.argmin(values))
        return ScalarMinimizerResult(
            (candidates[selected],), values[selected], True, 7, "Search completed", 3
        )

    original_import = common.importlib.import_module

    def import_module(name, *args, **kwargs):
        if name == "scipy.optimize":
            raise AssertionError("The custom search evaluates the scalar objective")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(common.importlib, "import_module", import_module)
    X = np.array([[-1.1], [-0.2], [0.6], [1.3]])
    y = [math.cos(float(row[0]) + 0.43) for row in X]
    initial = {"theta": -0.7}
    regressor = model(optimizer=search, max_evaluations=3)
    assert regressor.fit(X, y, initial_point=initial) is regressor
    expected = [
        sum(
            (math.cos(float(row[0]) + theta) - target) ** 2 for row, target in zip(X, y)
        )
        / len(y)
        for theta in candidates
    ]
    np.testing.assert_allclose(observed, expected, atol=1e-14)
    assert initial == {"theta": -0.7}
    assert regressor.optimal_parameters_ == {"theta": 0.43}
    np.testing.assert_allclose(regressor.predict(X), y, atol=1e-14)
    assert regressor.training_loss_ == pytest.approx(expected[1], abs=1e-28)
    assert regressor.optimizer_evaluations_ == 3
    assert regressor.objective_evaluations_ == 5
    assert regressor.optimizer_status_ == 7


def one_point_search(objective, initial_point):
    value = objective(initial_point)
    return ScalarMinimizerResult(
        tuple(float(value) for value in initial_point),
        value,
        True,
        0,
        "Point evaluated",
        1,
    )


def test_custom_optimizer_accepts_single_evaluation_budget():
    regressor = model(optimizer=one_point_search, max_evaluations=1).fit(
        [[0.0]], [0.5], initial_point={"theta": 0.3}
    )
    assert regressor.optimizer_evaluations_ == 1
    assert regressor.training_loss_ == pytest.approx((math.cos(0.3) - 0.5) ** 2)


@pytest.mark.reference
def test_explicit_none_retains_cobyla_regression():
    X = np.linspace(-1.2, 1.2, 7)[:, None]
    y = np.cos(X[:, 0] + 0.43)
    regressor = model(optimizer=None, max_evaluations=120).fit(
        X, y, initial_point={"theta": -0.7}
    )
    assert regressor.training_loss_ < 1e-10
    np.testing.assert_allclose(
        regressor.predict([[-0.8], [0.3]]),
        np.cos(np.array([-0.8, 0.3]) + 0.43),
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"point": ()},
        {"point": (0.1, 0.2)},
        {"point": (np.inf,)},
        {"point": (np.nan,)},
        {"value": np.inf},
        {"value": 1j},
        {"evaluations": 2},
        {"evaluations": 0},
        {"return_object": True},
        {"skip_call": True},
    ],
)
def test_custom_optimizer_invalid_result_preserves_model(changes):
    class Search:
        def __init__(self):
            self.changes = {}

        def __call__(self, objective, initial_point):
            update = self.changes
            value = 0.0 if update.get("skip_call") else objective(initial_point)
            if update.get("return_object"):
                return object()
            fields = {
                "point": tuple(float(x) for x in initial_point),
                "value": value,
                "success": True,
                "status": 0,
                "message": "Point evaluated",
                "evaluations": 1,
            }
            fields.update(
                {key: value for key, value in update.items() if key in fields}
            )
            return ScalarMinimizerResult(**fields)

    search = Search()
    regressor = model(optimizer=search, max_evaluations=1).fit(
        [[0.0], [0.4]], [0.4, 0.1], initial_point={"theta": 0.2}
    )
    previous = regressor.predict([[0.3]])
    previous_loss = regressor.training_loss_
    search.changes = changes
    with pytest.raises((TypeError, ValueError)):
        regressor.fit([[0.0], [0.4]], [0.4, 0.1], initial_point={"theta": 0.8})
    assert regressor.optimal_parameters_ == {"theta": 0.2}
    np.testing.assert_array_equal(regressor.predict([[0.3]]), previous)
    assert regressor.training_loss_ == previous_loss


def test_custom_optimizer_budget_overrun():
    def search(objective, initial_point):
        objective(initial_point)
        objective(initial_point)

    with pytest.raises(ValueError, match="exceeded max_evaluations"):
        model(optimizer=search, max_evaluations=1).fit(
            [[0.0]], [0.0], initial_point={"theta": 0.1}
        )


@pytest.mark.parametrize(
    "keyword,value",
    [("optimizer", object()), ("max_evaluations", 0), ("max_evaluations", True)],
)
def test_custom_optimizer_configuration(keyword, value):
    arguments = {"optimizer": one_point_search, "max_evaluations": 1}
    arguments[keyword] = value
    with pytest.raises((TypeError, ValueError)):
        model(**arguments)


def test_custom_optimizer_final_estimator_failure_preserves_model():
    estimator = RecordingEstimator()
    regressor = model(
        optimizer=one_point_search, max_evaluations=1, estimator=estimator
    )
    regressor.fit([[0.0]], [0.0], initial_point={"theta": 0.2})
    old = regressor.predict([[0.1]])
    original = estimator.run
    calls = 0

    def run(requests):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("Final evaluation failed")
        return original(requests)

    estimator.run = run
    with pytest.raises(RuntimeError, match="Final evaluation"):
        regressor.fit([[0.0]], [0.0], initial_point={"theta": 0.7})
    assert regressor.optimal_parameters_ == {"theta": 0.2}
    np.testing.assert_array_equal(regressor.predict([[0.1]]), old)
