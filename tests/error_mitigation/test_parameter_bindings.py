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

"""Component, parameter-binding, and algorithm-integration tests for mitigation."""

from __future__ import annotations

import numpy as np
import pytest
from cqlib import Circuit, Parameter
from cqlib.circuit import StandardGate
from cqlib.qis import Hamiltonian, PauliString

import cqlib_algorithms.error_mitigation as mitigation
from cqlib_algorithms import QAOA, VQE
from cqlib_algorithms.error_mitigation import (
    XX,
    QemExecutor,
    cdr_executor,
    ddd_executor,
    deb_executor,
    lre_executor,
    pea_executor,
    pec_executor,
    pt_executor,
    qse_executor,
    rem_executor,
    shadows_executor,
    trex_executor,
    vd_executor,
    zne_executor,
)
from cqlib_algorithms.error_mitigation._circuits import bind_request_circuit
from cqlib_algorithms.error_mitigation.quasiprobability import (
    depolarizing_quasiprobability_of_operation,
)
from cqlib_algorithms.operators import (
    AngleEncoding,
    ParameterShiftGradient,
    RealAmplitudesAnsatz,
    SamplingEstimator,
    StatevectorEstimator,
    StatevectorSampler,
)
from cqlib_algorithms.operators.optimizer import GradientDescent
from cqlib_algorithms.protocols import (
    Estimator,
    EstimatorRequest,
    EstimatorResult,
    Optimizer,
    OptimizerResult,
    OptimizerStatus,
    Sampler,
    StatePreparation,
)

_METHODS = (
    "zne",
    "zne_local",
    "lre",
    "cdr",
    "vd",
    "pec",
    "pea",
    "rem",
    "trex",
    "shadows",
    "qse",
    "ddd",
    "pt",
    "deb",
)


class _RecordingEstimator(Estimator):
    def __init__(self):
        self.calls = []

    def run(self, requests):
        for request in requests:
            request.validate()
            assert not request.circuit.used_symbols
            assert not request.parameter_bindings
        self.calls.extend(requests)
        return [
            (
                StatevectorEstimator()
                if request.shots is None
                else SamplingEstimator(StatevectorSampler())
            ).run([request])[0]
            for request in requests
        ]


class _RecordingSampler(Sampler):
    def __init__(self):
        self.calls = []

    def run(self, requests):
        for request in requests:
            request.validate()
            assert not request.circuit.used_symbols
        self.calls.extend(requests)
        return StatevectorSampler().run(requests)


def _source():
    circuit = RealAmplitudesAnsatz(2, reps=1).build_circuit("theta")
    point = dict(zip(circuit.used_symbols, (0.4, 0.9, 1.2, -0.7), strict=True))
    return circuit, point


def _configuration(method, bound):
    options = {"seed": 7}
    if method in {"zne", "zne_local"}:
        scales = (1.0, 3.0, 5.0) if method == "zne" else (1.0, 2.0, 3.0)
        return (zne_executor, {"scale_factors": scales, **options})
    if method == "lre":
        return (
            lre_executor,
            {
                "degree": 1,
                "fold_multiplier": 1,
                "num_chunks": min(2, len(bound.operations)),
                **options,
            },
        )
    if method == "cdr":
        return (
            cdr_executor,
            {
                "ideal_estimator": StatevectorEstimator(),
                "num_training_circuits": 12,
                "fraction_non_clifford": 0.5,
                **options,
            },
        )
    if method == "vd":
        return (
            vd_executor,
            {"copies": 2, "shots_numerator": 128, "shots_denominator": 128, **options},
        )
    if method == "pec":
        representations = [
            depolarizing_quasiprobability_of_operation(operation.instruction, 0.0)
            for operation in bound.operations
        ]
        return (
            pec_executor,
            {"quasiprobabilities": representations, "precision": 0.5, **options},
        )
    if method == "pea":
        return (
            pea_executor,
            {
                "noise_level": 0.0,
                "scale_factors": (1.0, 2.0, 3.0),
                "extrapolation_method": "richardson",
                "precision": 0.5,
                **options,
            },
        )
    if method == "rem":
        return (rem_executor, {"shots": 128, "p0": 0.0, "p1": 0.0, **options})
    if method == "trex":
        return (trex_executor, {"shots": 128, "num_randomizations": 2, **options})
    if method == "shadows":
        return (shadows_executor, {"num_total_measurements_shadow": 32, **options})
    if method == "qse":
        hamiltonian = Hamiltonian(2)
        hamiltonian.add_term(PauliString.from_str("ZI"), 1.0)
        hamiltonian.add_term(PauliString.from_str("XI"), 0.5)
        return (
            qse_executor,
            {
                "check_operators": (PauliString.from_str("XI"),),
                "code_hamiltonian": hamiltonian,
                **options,
            },
        )
    if method == "ddd":
        return (ddd_executor, {"rule": XX, **options})
    if method == "pt":
        return (pt_executor, {"num_randomizations": 2, **options})
    if method == "deb":
        return (
            deb_executor,
            {"mode": "averaging", "shots": 128, "num_variants": 3, **options},
        )
    raise AssertionError(method)


def _pipeline(method, base, bound):
    factory, options = _configuration(method, bound)
    return factory(base, **options)


@pytest.mark.parametrize("method", _METHODS)
def test_symbolic_request_matches_prebound_circuit_for_every_technique(method):
    circuit, point = _source()
    original_point = dict(point)
    bound = circuit.assign_parameters(point)
    original_matrix = bound.to_matrix().copy()
    estimator, sampler = _RecordingEstimator(), _RecordingSampler()
    pipeline = _pipeline(
        method, QemExecutor(estimator=estimator, sampler=sampler), bound
    )
    observable = PauliString.from_str("ZI")
    expected = pipeline.estimate([EstimatorRequest(bound, observable, seed=19)])[0]
    estimator.calls.clear()
    sampler.calls.clear()
    actual = pipeline.estimate([EstimatorRequest(circuit, observable, point, seed=19)])[
        0
    ]

    assert actual.expectation == pytest.approx(expected.expectation, abs=1e-12)
    assert actual.variance == expected.variance
    assert estimator.calls or sampler.calls
    assert circuit.used_symbols
    assert point == original_point
    np.testing.assert_array_equal(
        circuit.assign_parameters(point).to_matrix(), original_matrix
    )


@pytest.mark.parametrize("method", _METHODS)
def test_mutated_symbol_set_is_rejected_before_any_backend_execution(method):
    circuit, point = _source()
    estimator, sampler = _RecordingEstimator(), _RecordingSampler()
    pipeline = _pipeline(
        method,
        QemExecutor(estimator=estimator, sampler=sampler),
        circuit.assign_parameters(point),
    )
    request = EstimatorRequest(circuit, PauliString.from_str("ZI"), point)
    circuit.rz(0, Parameter("late"))
    with pytest.raises(ValueError, match="missing=.*late"):
        pipeline.estimate([request])
    assert estimator.calls == []
    assert sampler.calls == []


@pytest.mark.parametrize("method", ["zne", "zne_local", "lre"])
def test_parameterized_vqe_matches_the_ideal_reference(method):
    estimator = _RecordingEstimator()
    base = QemExecutor(estimator=estimator)
    circuit = RealAmplitudesAnsatz(1, reps=0).build_circuit("theta")
    pipeline = _pipeline(method, base, circuit.assign_parameters({"theta_0": 0.4}))

    def solve(backend):
        return VQE(
            backend,
            RealAmplitudesAnsatz(1, reps=0),
            GradientDescent(ParameterShiftGradient(), maxiter=2),
        ).run(PauliString.from_str("Z"), {"theta_0": 0.4})

    reference = solve(StatevectorEstimator())
    actual = solve(pipeline.estimator)
    assert actual.optimal_value == pytest.approx(reference.optimal_value, abs=1e-12)
    assert actual.optimal_parameters == pytest.approx(
        reference.optimal_parameters, abs=1e-12
    )
    assert len(estimator.calls) > 3


@pytest.mark.parametrize("outer", ["zne", "vd"])
def test_nested_mitigation_binds_parameters_once(outer):
    circuit, point = _source()
    bound = circuit.assign_parameters(point)
    base = QemExecutor(estimator=_RecordingEstimator())
    inner = "vd" if outer == "zne" else "zne"
    pipeline = _pipeline(outer, _pipeline(inner, base, bound), bound)
    observable = PauliString.from_str("ZI")
    expected = pipeline.estimate([EstimatorRequest(bound, observable, seed=7)])[0]
    actual = pipeline.estimate([EstimatorRequest(circuit, observable, point, seed=7)])[
        0
    ]
    assert actual.expectation == pytest.approx(expected.expectation, abs=1e-12)


@pytest.mark.parametrize("method", _METHODS)
def test_parameter_batches_keep_order_and_reproduce_the_same_draws(method):
    circuit, point = _source()
    points = (point, {name: value + 0.2 for name, value in point.items()})
    bound = circuit.assign_parameters(point)
    base = QemExecutor(estimator=_RecordingEstimator(), sampler=_RecordingSampler())
    pipeline = _pipeline(method, base, bound)
    observable = PauliString.from_str("ZI")
    requests = [
        EstimatorRequest(circuit, observable, values, seed=20 + index)
        for index, values in enumerate(points)
    ]
    expected = [
        pipeline.estimate(
            [
                EstimatorRequest(
                    circuit.assign_parameters(values),
                    observable,
                    seed=20 + index,
                )
            ]
        )[0].expectation
        for index, values in enumerate(points)
    ]
    assert pipeline.estimate([]) == []
    for _ in range(2):
        actual = pipeline.estimate(requests)
        assert [value.expectation for value in actual] == pytest.approx(
            expected, abs=1e-12
        )


@pytest.mark.parametrize("already_bound", [False, True])
def test_binding_preserves_global_phase_and_source_qubit_identifiers(already_bound):
    source = Circuit([7, 2])
    source.ry(source.qubits[0], Parameter("angle"))
    source.set_global_phase(Parameter("phase"))
    point = {"angle": 0.4, "phase": 0.7}
    if already_bound:
        source = source.assign_parameters(point)
    request = EstimatorRequest(
        source, PauliString.from_str("ZI"), {} if already_bound else point
    )
    bound = bind_request_circuit(request)
    rotation = np.array([[np.cos(0.2), -np.sin(0.2)], [np.sin(0.2), np.cos(0.2)]])
    # Core matrix axes follow physical IDs (2, 7); the rotation acts on ID 7.
    expected = np.exp(0.7j) * np.kron(rotation, np.eye(2))
    assert bound is not source
    assert bound.qubits == source.qubits
    assert not bound.used_symbols
    np.testing.assert_allclose(bound.to_matrix(), expected, rtol=0.0, atol=1e-12)
    bound.x(bound.qubits[0])
    assert set(source.used_symbols) == (set() if already_bound else {"angle", "phase"})
    np.testing.assert_allclose(
        source.assign_parameters(dict(request.parameter_bindings)).to_matrix(),
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_cache_distinguishes_bindings_and_reuses_the_same_request():
    circuit = RealAmplitudesAnsatz(1, reps=0).build_circuit("theta")
    backend = _RecordingEstimator()
    pipeline = zne_executor(
        QemExecutor(estimator=backend, use_cache=True),
        scale_factors=(1.0, 3.0, 5.0),
    )
    requests = [
        EstimatorRequest(circuit, PauliString.from_str("Z"), {"theta_0": angle}, seed=7)
        for angle in (0.4, 1.1)
    ]
    first = pipeline.estimate([requests[0]])[0]
    executed = len(backend.calls)
    assert pipeline.estimate([requests[0]])[0] == first
    assert len(backend.calls) == executed
    second = pipeline.estimate([requests[1]])[0]
    assert len(backend.calls) > executed
    assert [first.expectation, second.expectation] == pytest.approx(
        np.cos((0.4, 1.1)),
        abs=1e-12,
    )


def test_request_shots_and_seed_reach_the_transformed_backend_tasks():
    circuit = RealAmplitudesAnsatz(1, reps=0).build_circuit("theta")
    backend = _RecordingEstimator()
    pipeline = zne_executor(
        QemExecutor(estimator=backend), scale_factors=(1.0, 3.0, 5.0)
    )
    request = EstimatorRequest(
        circuit, PauliString.from_str("Z"), {"theta_0": 0.4}, shots=128, seed=17
    )
    first = pipeline.estimate([request])[0]
    metadata = [(task.shots, task.seed) for task in backend.calls]
    assert metadata and all(
        shots == 128 and seed is not None for shots, seed in metadata
    )
    backend.calls.clear()
    assert pipeline.estimate([request])[0] == first
    assert [(task.shots, task.seed) for task in backend.calls] == metadata


def test_binding_rejects_an_invalid_request_type():
    with pytest.raises(TypeError, match="EstimatorRequest"):
        bind_request_circuit(object())


@pytest.mark.parametrize("method", _METHODS)
def test_component_run_and_immediate_execution_match_the_legacy_circuit(method):
    preparation = AngleEncoding(2, StandardGate.RY)
    point = {"features_0": 0.4, "features_1": 0.9}
    bound = preparation.build_circuit("features").assign_parameters(point)
    base = QemExecutor(estimator=_RecordingEstimator(), sampler=_RecordingSampler())
    factory, options = _configuration(method, bound)
    pipeline = factory(base, **options)
    immediate = getattr(
        mitigation, "execute_with_" + factory.__name__.removesuffix("_executor")
    )
    observable = PauliString.from_str("ZI")
    legacy = immediate(bound, base, observable, **options)
    actual = pipeline.run(
        preparation, observable, parameter_bindings=point, prefix="features"
    )
    direct = immediate(
        preparation,
        base,
        observable,
        parameter_bindings=point,
        prefix="features",
        **options,
    )
    assert isinstance(actual, EstimatorResult)
    assert actual.expectation == pytest.approx(legacy, abs=1e-12)
    assert direct == pytest.approx(legacy, abs=1e-12)
    assert pipeline.run(bound, observable).expectation == pytest.approx(
        legacy, abs=1e-12
    )


class _FixedPointOptimizer(Optimizer):
    def run(self, estimator, requests):
        estimates = estimator.run([request.estimator_request for request in requests])
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
            for request, estimate in zip(requests, estimates, strict=True)
        ]


@pytest.mark.parametrize("method", _METHODS)
def test_every_mitigation_can_supply_the_vqe_estimator_component(method):
    circuit, point = _source()
    bound = circuit.assign_parameters(point)
    base = QemExecutor(estimator=_RecordingEstimator(), sampler=_RecordingSampler())
    pipeline = _pipeline(method, base, bound)
    observable = PauliString.from_str("ZI")
    reference = pipeline.estimate([EstimatorRequest(bound, observable)])[0].expectation
    result = VQE(
        pipeline.as_estimator(), RealAmplitudesAnsatz(2, reps=1), _FixedPointOptimizer()
    ).run(observable, point)
    assert result.optimal_value == pytest.approx(reference, abs=1e-12)
    assert result.optimal_parameters == point


def test_estimator_adapter_preserves_cache_and_validates_before_cache_lookup():
    backend = _RecordingEstimator()
    pipeline = zne_executor(
        QemExecutor(estimator=backend, use_cache=True), scale_factors=(1.0, 3.0, 5.0)
    )
    estimator = pipeline.as_estimator()
    assert isinstance(estimator, Estimator)
    circuit, point = _source()
    request = EstimatorRequest(circuit, PauliString.from_str("ZI"), point)
    first = estimator.run([request])
    count = len(backend.calls)
    assert estimator.run([request, request]) == first * 2
    assert len(backend.calls) == count
    assert estimator.run([]) == []
    assert len(backend.calls) == count
    circuit.rz(0, Parameter("late"))
    with pytest.raises(ValueError, match="missing=.*late"):
        estimator.run([request])
    assert len(backend.calls) == count


@pytest.mark.parametrize("use_cache", [False, True])
def test_component_execution_reports_missing_backends_and_bad_result_batches(use_cache):
    preparation = AngleEncoding(1, StandardGate.RY)
    with pytest.raises(ValueError, match="no estimator"):
        QemExecutor(sampler=StatevectorSampler()).as_estimator()
    with pytest.raises(ValueError, match="no estimator"):
        QemExecutor(sampler=StatevectorSampler()).run(
            preparation, PauliString.from_str("Z"), parameter_bindings={"qem_0": 0.4}
        )

    class MissingResults(Estimator):
        def run(self, requests):
            return []

    executor = QemExecutor(estimator=MissingResults(), use_cache=use_cache)
    with pytest.raises(ValueError, match="one result per request"):
        executor.run(
            preparation, PauliString.from_str("Z"), parameter_bindings={"qem_0": 0.4}
        )


def test_component_run_uses_a_defensive_copy_of_a_cached_preparation():
    cached = Circuit(1)
    cached.ry(0, 0.4)
    original = cached.to_matrix().copy()

    class CachedPreparation(StatePreparation):
        def validate(self):
            pass

        def num_qubits(self):
            return 1

        def num_parameters(self):
            return 0

        def build_circuit(self, prefix):
            return cached

    class MutatingEstimator(Estimator):
        def run(self, requests):
            results = StatevectorEstimator().run(requests)
            for request in requests:
                request.circuit.x(0)
            return results

    executor = QemExecutor(estimator=MutatingEstimator())
    for source in (cached, CachedPreparation()):
        first = executor.run(source, PauliString.from_str("Z"))
        second = executor.run(source, PauliString.from_str("Z"))
        assert first == second
        np.testing.assert_array_equal(cached.to_matrix(), original)


@pytest.mark.parametrize("method", ["zne", "zne_local", "lre", "cdr"])
def test_estimator_component_preserves_vqe_parameter_updates(method):
    circuit, point = _source()
    pipeline = _pipeline(
        method,
        QemExecutor(estimator=_RecordingEstimator()),
        circuit.assign_parameters(point),
    )

    def solve(estimator):
        return VQE(
            estimator,
            RealAmplitudesAnsatz(2, reps=1),
            GradientDescent(ParameterShiftGradient(), maxiter=2),
        ).run(PauliString.from_str("ZI"), point)

    reference = solve(StatevectorEstimator())
    actual = solve(pipeline.as_estimator())
    assert actual.optimal_value == pytest.approx(reference.optimal_value, abs=1e-11)
    assert actual.optimal_parameters == pytest.approx(
        reference.optimal_parameters, abs=1e-11
    )


def test_qaoa_uses_the_mitigated_estimator_component_without_circuit_assembly():
    pipeline = zne_executor(
        QemExecutor(estimator=StatevectorEstimator()), scale_factors=(1.0, 3.0, 5.0)
    )
    cost = Hamiltonian(1)
    cost.add_term(PauliString.from_str("Z"), 1.0)

    def solve(estimator):
        return QAOA(
            estimator, GradientDescent(ParameterShiftGradient(), maxiter=2), reps=1
        ).run(cost, {"qaoa_0_gamma": 0.4, "qaoa_0_beta": 0.7})

    reference = solve(StatevectorEstimator())
    actual = solve(pipeline.as_estimator())
    assert actual.optimal_value == pytest.approx(reference.optimal_value, abs=1e-11)
    assert actual.optimal_parameters == pytest.approx(
        reference.optimal_parameters, abs=1e-11
    )


@pytest.mark.parametrize("method", ["rem", "trex", "shadows"])
def test_sampling_based_methods_expose_corrected_expectations_and_keep_raw_sampler(
    method,
):
    circuit, point = _source()
    sampler = _RecordingSampler()
    pipeline = _pipeline(
        method, QemExecutor(sampler=sampler), circuit.assign_parameters(point)
    )
    assert pipeline.sampler is sampler
    estimator = pipeline.as_estimator()
    result = estimator.run(
        [EstimatorRequest(circuit, PauliString.from_str("ZI"), point)]
    )[0]
    assert isinstance(result, EstimatorResult)
    assert sampler.calls


@pytest.mark.parametrize(
    "preparation, bindings, prefix, observable, error, message",
    [
        (object(), {}, "qem", "Z", TypeError, "StatePreparation"),
        (AngleEncoding(1, StandardGate.RY), {}, "qem", "Z", ValueError, "missing"),
        (
            AngleEncoding(1, StandardGate.RY),
            {"qem_0": 0.4},
            1,
            "Z",
            TypeError,
            "prefix",
        ),
        (
            AngleEncoding(1, StandardGate.RY),
            {"qem_0": float("nan")},
            "qem",
            "Z",
            ValueError,
            "finite",
        ),
        (
            AngleEncoding(1, StandardGate.RY),
            {"qem_0": 0.4},
            "qem",
            "ZZ",
            ValueError,
            "same qubit count",
        ),
    ],
)
def test_component_input_contract_is_checked_before_execution(
    preparation, bindings, prefix, observable, error, message
):
    backend = _RecordingEstimator()
    pipeline = zne_executor(QemExecutor(estimator=backend))
    with pytest.raises(error, match=message):
        pipeline.run(
            preparation,
            PauliString.from_str(observable),
            parameter_bindings=bindings,
            prefix=prefix,
        )
    assert backend.calls == []
