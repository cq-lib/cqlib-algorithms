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

"""Reference, integration, and boundary tests for CHSH evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from cqlib import Circuit, Parameter, Qubit

from cqlib_algorithms.algorithms import CHSH, CHSHRequest, CHSHResult
from cqlib_algorithms.operators.estimator import (
    SamplingEstimator,
    StatevectorEstimator,
)
from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.operators.state_preparation import UserCircuitPreparation
from cqlib_algorithms.protocols import Estimator, EstimatorRequest, EstimatorResult

_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def _measurement(angle: float) -> np.ndarray:
    return math.cos(angle) * _Z + math.sin(angle) * _X


def _product_matrix(alice: float, bob: float) -> np.ndarray:
    # Core qubit 0 is the least-significant tensor factor.
    return np.kron(_measurement(bob), _measurement(alice))


class RecordingEstimator(Estimator):
    def __init__(self, results: Sequence[EstimatorResult] | None = None) -> None:
        self.batches: list[list[EstimatorRequest]] = []
        self.results = None if results is None else list(results)

    def run(self, requests: Sequence[EstimatorRequest], /) -> list[EstimatorResult]:
        batch = list(requests)
        self.batches.append(batch)
        if self.results is not None:
            return list(self.results)
        return [EstimatorResult(0.0, 0.0) for _ in batch]


class WrongContainerEstimator(Estimator):
    def run(self, requests: Sequence[EstimatorRequest], /) -> list[EstimatorResult]:
        return tuple(EstimatorResult(0.0) for _ in requests)  # type: ignore[return-value]


class WrongCountEstimator(Estimator):
    def run(self, requests: Sequence[EstimatorRequest], /) -> list[EstimatorResult]:
        return [EstimatorResult(0.0)]


class WrongEntryEstimator(Estimator):
    def run(self, requests: Sequence[EstimatorRequest], /) -> list[EstimatorResult]:
        return [None for _ in requests]  # type: ignore[list-item]


def _run_exact(request: CHSHRequest) -> CHSHResult:
    return CHSH(StatevectorEstimator()).run([request])[0]


def test_default_bell_state_reaches_positive_tsirelson_bound() -> None:
    result = _run_exact(CHSHRequest())
    root_half = 1.0 / math.sqrt(2.0)

    assert result.correlators == pytest.approx(
        (root_half, root_half, root_half, -root_half), abs=1e-12
    )
    assert result.chsh_value == pytest.approx(2.0 * math.sqrt(2.0), abs=1e-12)
    assert result.absolute_value == pytest.approx(result.tsirelson_bound, abs=1e-12)
    assert result.violation_margin == pytest.approx(2.0 * math.sqrt(2.0) - 2.0)
    assert result.violates_classical_bound
    assert result.classical_bound == 2.0
    assert result.variances == (0.0, 0.0, 0.0, 0.0)
    assert result.standard_deviation_upper_bound == 0.0


@pytest.mark.parametrize(
    ("alice_angles", "bob_angles"),
    [
        ((0.13, -0.72), (1.11, -2.01)),
        ((math.pi, math.pi / 3.0), (-math.pi / 6.0, 2.4)),
        ((-1.7, 0.0), (0.81, math.pi / 2.0)),
    ],
)
def test_bell_correlators_match_independent_numpy_reference(
    alice_angles: tuple[float, float],
    bob_angles: tuple[float, float],
) -> None:
    result = _run_exact(CHSHRequest(alice_angles=alice_angles, bob_angles=bob_angles))
    state = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / math.sqrt(2.0)
    expected = tuple(
        float(np.vdot(state, _product_matrix(alice, bob) @ state).real)
        for alice in alice_angles
        for bob in bob_angles
    )

    assert result.correlators == pytest.approx(expected, abs=1e-12)


def test_product_zero_state_has_no_default_violation() -> None:
    result = _run_exact(
        CHSHRequest(state_preparation=UserCircuitPreparation(Circuit(2)))
    )

    assert result.chsh_value == pytest.approx(math.sqrt(2.0), abs=1e-12)
    assert not result.violates_classical_bound
    assert result.violation_margin == pytest.approx(math.sqrt(2.0) - 2.0)


def test_negative_chsh_value_uses_absolute_value_for_violation() -> None:
    root_half = 1.0 / math.sqrt(2.0)
    estimator = RecordingEstimator(
        [
            EstimatorResult(-root_half),
            EstimatorResult(-root_half),
            EstimatorResult(-root_half),
            EstimatorResult(root_half),
        ]
    )

    result = CHSH(estimator).run([CHSHRequest()])[0]

    assert result.chsh_value == pytest.approx(-2.0 * math.sqrt(2.0))
    assert result.absolute_value == pytest.approx(2.0 * math.sqrt(2.0))
    assert result.violates_classical_bound


def test_statevector_estimator_observes_negative_tsirelson_value() -> None:
    request = CHSHRequest(
        alice_angles=(math.pi, 3.0 * math.pi / 2.0),
    )

    result = _run_exact(request)

    assert result.chsh_value == pytest.approx(-2.0 * math.sqrt(2.0), abs=1e-12)
    assert result.absolute_value == pytest.approx(result.tsirelson_bound, abs=1e-12)
    assert result.violates_classical_bound


def test_tolerance_controls_only_numeric_threshold() -> None:
    below = CHSHResult((2.0 + 5e-7, 0.0, 0.0, 0.0), (0.0,) * 4, 1e-6)
    above = replace(below, tolerance=1e-8)

    assert not below.violates_classical_bound
    assert above.violates_classical_bound


def test_build_circuit_returns_fresh_default_bell_preparations() -> None:
    algorithm = CHSH(RecordingEstimator())
    first = algorithm.build_circuit()
    second = algorithm.build_circuit()

    first.x(0)
    assert first is not second
    assert second.to_matrix()[:, 0] == pytest.approx(
        np.array([1.0, 0.0, 0.0, 1.0]) / math.sqrt(2.0), abs=1e-12
    )


def test_custom_non_contiguous_qubits_are_mapped_by_insertion_order() -> None:
    source = Circuit([Qubit(9), Qubit(2)])
    source.h(Qubit(9))
    source.cx(Qubit(9), Qubit(2))

    built = CHSH(RecordingEstimator()).build_circuit(UserCircuitPreparation(source))

    assert built.qubits == [Qubit(0), Qubit(1)]
    assert built.to_matrix([0, 1]) == pytest.approx(source.to_matrix([9, 2]), abs=1e-12)
    result = _run_exact(CHSHRequest(state_preparation=UserCircuitPreparation(source)))
    assert result.chsh_value == pytest.approx(2.0 * math.sqrt(2.0), abs=1e-12)


def test_custom_state_preparation_is_not_modified() -> None:
    source = Circuit([Qubit(7), Qubit(3)])
    source.ry(Qubit(7), 0.37)
    source.cx(Qubit(7), Qubit(3))
    before = source.to_matrix().copy()
    before_qubits = list(source.qubits)

    CHSH(StatevectorEstimator()).run(
        [CHSHRequest(state_preparation=UserCircuitPreparation(source))]
    )

    assert source.qubits == before_qubits
    assert source.to_matrix() == pytest.approx(before, abs=1e-12)


def test_recording_estimator_receives_four_observables_in_defined_order() -> None:
    alice = (0.27, 1.49)
    bob = (-0.61, 2.02)
    estimator = RecordingEstimator()

    CHSH(estimator).run([CHSHRequest(alice_angles=alice, bob_angles=bob, shots=123)])

    assert len(estimator.batches) == 1
    batch = estimator.batches[0]
    assert len(batch) == 4
    expected_pairs = (
        (alice[0], bob[0]),
        (alice[0], bob[1]),
        (alice[1], bob[0]),
        (alice[1], bob[1]),
    )
    for request, (alice_angle, bob_angle) in zip(batch, expected_pairs, strict=True):
        assert request.shots == 123
        assert request.parameter_bindings == {}
        assert request.observable.to_matrix() == pytest.approx(
            _product_matrix(alice_angle, bob_angle), abs=1e-12
        )


def test_batch_is_flattened_into_one_estimator_call() -> None:
    estimator = RecordingEstimator()
    requests = [
        CHSHRequest(shots=7),
        CHSHRequest(alice_angles=(0.1, 0.2), bob_angles=(0.3, 0.4), shots=9),
    ]

    results = CHSH(estimator).run(requests)

    assert len(results) == 2
    assert len(estimator.batches) == 1
    assert [request.shots for request in estimator.batches[0]] == [7] * 4 + [9] * 4


def test_empty_batch_does_not_call_estimator() -> None:
    estimator = RecordingEstimator()

    assert CHSH(estimator).run([]) == []
    assert estimator.batches == []


def test_variance_upper_bound_and_unknown_variance() -> None:
    known = CHSHResult((0.0,) * 4, (0.01, 0.04, 0.09, 0.16))
    unknown = CHSHResult((0.0,) * 4, (0.01, None, 0.09, 0.16))

    assert known.standard_deviation_upper_bound == pytest.approx(1.0)
    assert unknown.standard_deviation_upper_bound is None


def test_result_keeps_finite_values_beyond_ideal_physical_bounds() -> None:
    result = CHSHResult((1.01, 1.02, 1.03, -1.04), (0.0,) * 4)

    assert result.chsh_value == pytest.approx(4.1)
    assert result.absolute_value > result.tsirelson_bound


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alice_angles": [0.0, 1.0]},
        {"alice_angles": (0.0,)},
        {"alice_angles": (True, 1.0)},
        {"bob_angles": (0.0, math.inf)},
        {"shots": True},
        {"shots": 0},
        {"tolerance": True},
        {"tolerance": -1.0},
        {"tolerance": math.nan},
        {"state_preparation": "circuit"},
        {"state_preparation": Circuit(2)},
    ],
)
def test_request_rejects_invalid_configuration(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        CHSHRequest(**kwargs)  # type: ignore[arg-type]


def test_request_rejects_unbound_custom_circuit() -> None:
    circuit = Circuit(2)
    circuit.ry(0, Parameter("theta"))

    with pytest.raises(ValueError, match="active symbols"):
        CHSHRequest(state_preparation=UserCircuitPreparation(circuit))


@pytest.mark.parametrize("directive", ["measure", "reset", "delay"])
def test_request_rejects_nonunitary_custom_circuit(directive: str) -> None:
    request = CHSHRequest(UserCircuitPreparation(Circuit(2)))
    circuit = request.preparation_circuit
    assert circuit is not None
    if directive == "measure":
        circuit.measure(0)
    elif directive == "reset":
        circuit.reset(0)
    else:
        circuit.delay(0, 1.0)

    with pytest.raises(ValueError, match="unitary circuit"):
        request.validate()


def test_request_rejects_nested_nonunitary_custom_circuit() -> None:
    inner = Circuit(1)
    inner.reset(0)
    request = CHSHRequest(UserCircuitPreparation(Circuit(2)))
    outer = request.preparation_circuit
    assert outer is not None
    outer.append_circuit_gate(inner.to_gate("hidden_reset"), [0])

    with pytest.raises(ValueError, match="unitary circuit"):
        request.validate()


def test_mutated_request_circuit_is_revalidated_before_execution() -> None:
    circuit = Circuit(2)
    request = CHSHRequest(state_preparation=UserCircuitPreparation(circuit))
    assert request.preparation_circuit is not None
    request.preparation_circuit.ry(0, Parameter("theta"))

    with pytest.raises(ValueError, match="active symbols"):
        CHSH(RecordingEstimator()).run([request])


def test_nonunitary_mutation_is_revalidated_before_execution() -> None:
    circuit = Circuit(2)
    request = CHSHRequest(state_preparation=UserCircuitPreparation(circuit))
    assert request.preparation_circuit is not None
    request.preparation_circuit.reset(0)

    with pytest.raises(ValueError, match="unitary circuit"):
        CHSH(RecordingEstimator()).run([request])


def test_request_and_result_are_frozen_and_replaceable() -> None:
    request = CHSHRequest()
    result = CHSHResult((0.0,) * 4, (0.0,) * 4)

    with pytest.raises(FrozenInstanceError):
        request.shots = 10  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.tolerance = 0.1  # type: ignore[misc]
    assert replace(request, shots=5).shots == 5
    assert replace(result, tolerance=0.2).tolerance == 0.2


@pytest.mark.parametrize(
    ("correlators", "variances", "tolerance"),
    [
        ([0.0] * 4, (0.0,) * 4, 0.0),
        ((0.0,) * 3, (0.0,) * 4, 0.0),
        ((0.0, 0.0, 0.0, math.inf), (0.0,) * 4, 0.0),
        ((0.0,) * 4, [0.0] * 4, 0.0),
        ((0.0,) * 4, (0.0,) * 3, 0.0),
        ((0.0,) * 4, (0.0, 0.0, 0.0, -1.0), 0.0),
        ((0.0,) * 4, (0.0, 0.0, 0.0, math.nan), 0.0),
        ((0.0,) * 4, (0.0,) * 4, -1.0),
    ],
)
def test_result_rejects_invalid_values(
    correlators: object,
    variances: object,
    tolerance: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        CHSHResult(correlators, variances, tolerance)  # type: ignore[arg-type]


def test_algorithm_rejects_wrong_estimator_and_batch_entries() -> None:
    with pytest.raises(TypeError, match="estimator"):
        CHSH(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="CHSHRequest"):
        CHSH(RecordingEstimator()).run([object()])  # type: ignore[list-item]


@pytest.mark.parametrize(
    ("estimator", "error", "message"),
    [
        (WrongContainerEstimator(), TypeError, "return a list"),
        (WrongCountEstimator(), ValueError, "unexpected result count"),
        (WrongEntryEstimator(), TypeError, "EstimatorResult"),
    ],
)
def test_algorithm_rejects_invalid_estimator_responses(
    estimator: Estimator,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        CHSH(estimator).run([CHSHRequest()])


@pytest.mark.integration
def test_sampling_estimator_preserves_finite_shot_semantics() -> None:
    algorithm = CHSH(SamplingEstimator(StatevectorSampler()))

    result = algorithm.run([CHSHRequest(shots=4000)])[0]

    assert all(math.isfinite(value) for value in result.correlators)
    assert all(
        variance is not None and variance >= 0.0 for variance in result.variances
    )
    assert result.standard_deviation_upper_bound is not None
    assert abs(result.chsh_value - 2.0 * math.sqrt(2.0)) < 0.35


def test_component_single_run_matches_component_batch_and_preparation_replacement():
    from cqlib_algorithms.operators import BasisStateInitialization, GHZStatePreparation

    algorithm = CHSH(StatevectorEstimator())
    for preparation, expected in (
        (GHZStatePreparation(2), 2 * math.sqrt(2)),
        (BasisStateInitialization(2, 0), math.sqrt(2)),
    ):
        actual = algorithm.run(preparation)
        component_batch = algorithm.run([CHSHRequest(preparation)])[0]
        assert actual == component_batch
        assert actual.chsh_value == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "case", ["width", "symbol", "missing_type", "batch_shots", "angles"]
)
def test_single_preparation_validation_precedes_estimation(case):
    from cqlib.circuit.gates import StandardGate

    from cqlib_algorithms.operators import AngleEncoding, GHZStatePreparation

    estimator = RecordingEstimator()
    algorithm = CHSH(estimator)
    with pytest.raises((TypeError, ValueError)):
        if case == "width":
            algorithm.run(GHZStatePreparation(3))
        elif case == "symbol":
            algorithm.run(AngleEncoding(2, StandardGate.RY))
        elif case == "missing_type":
            algorithm.run("preparation")
        elif case == "batch_shots":
            algorithm.run([], shots=10)
        else:
            algorithm.run(GHZStatePreparation(2), alice_angles=(0.0, math.nan))
    assert estimator.batches == []


def test_single_request_forwards_angles_shots_and_tolerance():
    from cqlib_algorithms.operators import GHZStatePreparation

    values = [EstimatorResult(value, 0.01) for value in (0.3, 0.2, -0.1, 0.5)]
    estimator = RecordingEstimator(values)
    algorithm = CHSH(estimator)
    alice, bob = (0.2, 1.1), (-0.3, 0.8)
    result = algorithm.run(
        GHZStatePreparation(2),
        alice_angles=alice,
        bob_angles=bob,
        shots=91,
        tolerance=0.02,
    )
    assert result == CHSHResult((0.3, 0.2, -0.1, 0.5), (0.01,) * 4, 0.02)
    assert len(estimator.batches) == 1
    for index, request in enumerate(estimator.batches[0]):
        assert request.shots == 91
        np.testing.assert_allclose(
            request.observable.to_matrix(),
            _product_matrix(alice[index // 2], bob[index % 2]),
            atol=1e-12,
        )


def test_component_requests_preserve_mixed_batch_order_and_complex_correlators():
    from cqlib_algorithms.operators import (
        ArbitraryStatePreparation,
        GHZStatePreparation,
    )

    state = np.array([math.sqrt(0.3), 0.0, 0.0, 1j * math.sqrt(0.7)])
    alice, bob = (0.21, 1.4), (-0.32, 0.81)
    circuit = Circuit(2)
    requests = [
        CHSHRequest(ArbitraryStatePreparation(state.tolist()), alice, bob),
        CHSHRequest(GHZStatePreparation(2)),
        CHSHRequest(UserCircuitPreparation(circuit)),
        CHSHRequest(),
    ]

    results = CHSH(StatevectorEstimator()).run(requests)
    expected = tuple(
        float(np.vdot(state, _product_matrix(a, b) @ state).real)
        for a in alice
        for b in bob
    )
    assert len(results) == 4
    assert results[0].correlators == pytest.approx(expected, abs=1e-12)
    assert [item.chsh_value for item in results[1:]] == pytest.approx(
        [2 * math.sqrt(2), math.sqrt(2), 2 * math.sqrt(2)], abs=1e-12
    )
    assert requests[2].preparation_circuit is not circuit
    np.testing.assert_allclose(
        requests[2].preparation_circuit.to_matrix(), circuit.to_matrix(), atol=1e-12
    )


def test_component_request_snapshots_once_and_preserves_phase_and_qubit_order(
    monkeypatch: pytest.MonkeyPatch,
):
    from cqlib_algorithms.operators import GHZStatePreparation

    source = Circuit([Qubit(9), Qubit(2)])
    source.h(Qubit(9))
    source.cx(Qubit(9), Qubit(2))
    source.set_global_phase(0.37)
    expected = source.to_matrix([9, 2]).copy()
    prefixes: list[str] = []

    def build(_self: GHZStatePreparation, prefix: str) -> Circuit:
        prefixes.append(prefix)
        return source

    monkeypatch.setattr(GHZStatePreparation, "build_circuit", build)
    request = CHSHRequest(GHZStatePreparation(2))
    assert isinstance(request.preparation_circuit, Circuit)
    assert request.preparation_circuit is not source
    assert request.preparation_circuit.to_matrix() == pytest.approx(expected, abs=1e-12)
    source.reset(Qubit(9))

    algorithm = CHSH(StatevectorEstimator())
    for _ in range(2):
        result = algorithm.run([request])[0]
        assert result.chsh_value == pytest.approx(2 * math.sqrt(2), abs=1e-12)
    assert prefixes == ["chsh"]
    copied = replace(request, shots=17)
    assert copied.preparation_circuit is not request.preparation_circuit
    np.testing.assert_allclose(
        copied.preparation_circuit.to_matrix(), expected, atol=1e-12
    )
    assert prefixes == ["chsh"]


@pytest.mark.parametrize("case", ["width", "symbol"])
def test_component_request_rejects_incompatible_preparations(case: str):
    from cqlib.circuit.gates import StandardGate

    from cqlib_algorithms.operators import AngleEncoding, GHZStatePreparation

    preparation = (
        GHZStatePreparation(1) if case == "width" else AngleEncoding(2, StandardGate.RY)
    )
    with pytest.raises(ValueError, match="two qubits|active symbols"):
        CHSHRequest(preparation)


@pytest.mark.parametrize(
    ("case", "error", "message"),
    [
        ("type", TypeError, "return a Circuit"),
        ("width", ValueError, "qubit count"),
        ("reset", ValueError, "unitary circuit"),
    ],
)
def test_component_request_checks_built_circuit(
    case: str, error: type[Exception], message: str, monkeypatch: pytest.MonkeyPatch
):
    from cqlib_algorithms.operators import GHZStatePreparation

    def build(_self: GHZStatePreparation, _prefix: str) -> Circuit:
        if case == "type":
            return object()  # type: ignore[return-value]
        circuit = Circuit(1 if case == "width" else 2)
        if case == "reset":
            circuit.reset(0)
        return circuit

    monkeypatch.setattr(GHZStatePreparation, "build_circuit", build)
    with pytest.raises(error, match=message):
        CHSHRequest(GHZStatePreparation(2))
