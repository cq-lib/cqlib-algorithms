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

"""Tests for the public estimator contract."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from cqlib import Circuit, Parameter, Qubit
from cqlib.error_mitigation import Estimator as ErrorMitigationEstimator
from cqlib.qis import Hamiltonian, PauliString, Phase

from cqlib_algorithms.protocols.estimator import (
    Estimator,
    EstimatorRequest,
    EstimatorResult,
)


def _symbolic_request(*, shots: int | None = None) -> EstimatorRequest:
    circuit = Circuit(1)
    circuit.ry(0, Parameter("theta"))
    return EstimatorRequest(
        circuit,
        PauliString.from_str("Z"),
        {"theta": 0.25},
        shots,
    )


class _FakeEstimator(Estimator):
    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        results: list[EstimatorResult] = []
        for request in requests:
            request.validate()
            results.append(
                EstimatorResult(sum(request.parameter_bindings.values()), 0.0)
            )
        return results


class _ExactOnlyEstimator(Estimator):
    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        results: list[EstimatorResult] = []
        for request in requests:
            request.validate()
            if request.shots is not None:
                raise ValueError("shot-based estimation is not supported")
            results.append(EstimatorResult(0.0, 0.0))
        return results


def _run_estimator(
    estimator: Estimator,
    requests: Sequence[EstimatorRequest],
) -> list[EstimatorResult]:
    return estimator.run(requests)


def test_estimator_request_accepts_pauli_and_hamiltonian_observables():
    pauli_request = _symbolic_request()
    hamiltonian = Hamiltonian.from_pauli(PauliString.from_str("Z"))
    hamiltonian_request = EstimatorRequest(
        pauli_request.circuit,
        hamiltonian,
        {"theta": 0.25},
    )

    assert isinstance(pauli_request.observable, PauliString)
    assert isinstance(hamiltonian_request.observable, Hamiltonian)
    assert pauli_request.shots is None
    with pytest.raises(TypeError):
        hash(pauli_request)


def test_estimator_request_copies_bindings_into_an_immutable_mapping():
    bindings = {"theta": 0.25}
    original = _symbolic_request()
    request = EstimatorRequest(original.circuit, original.observable, bindings)

    bindings["theta"] = 0.75
    assert request.parameter_bindings["theta"] == pytest.approx(0.25)
    with pytest.raises(TypeError):
        request.parameter_bindings["theta"] = 1.0  # type: ignore[index]


@pytest.mark.parametrize(
    ("bindings", "message"),
    [
        ({}, "missing"),
        ({"theta": 0.25, "other": 1.0}, "unexpected"),
    ],
)
def test_estimator_request_requires_exact_binding_names(bindings, message):
    request = _symbolic_request()

    with pytest.raises(ValueError, match=message):
        EstimatorRequest(request.circuit, request.observable, bindings)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_estimator_request_rejects_non_finite_bindings(value):
    request = _symbolic_request()

    with pytest.raises(ValueError, match="finite"):
        EstimatorRequest(request.circuit, request.observable, {"theta": value})


@pytest.mark.parametrize("value", [True, "0.25", object()])
def test_estimator_request_rejects_non_real_bindings(value):
    request = _symbolic_request()

    with pytest.raises(TypeError, match="real numbers"):
        EstimatorRequest(request.circuit, request.observable, {"theta": value})


def test_estimator_request_rejects_non_string_binding_names():
    request = _symbolic_request()

    with pytest.raises(TypeError, match="names must be strings"):
        EstimatorRequest(
            request.circuit,
            request.observable,
            {1: 0.25},  # type: ignore[dict-item]
        )


@pytest.mark.parametrize("shots", [0, -1])
def test_estimator_request_rejects_non_positive_shots(shots):
    request = _symbolic_request()

    with pytest.raises(ValueError, match="greater than zero"):
        EstimatorRequest(
            request.circuit,
            request.observable,
            request.parameter_bindings,
            shots,
        )


@pytest.mark.parametrize("shots", [True, 1.5, "10"])
def test_estimator_request_rejects_invalid_shot_types(shots):
    request = _symbolic_request()

    with pytest.raises(TypeError, match="integer or None"):
        EstimatorRequest(
            request.circuit,
            request.observable,
            request.parameter_bindings,
            shots,
        )


@pytest.mark.parametrize("seed", [True, 1.5, "1"])
def test_estimator_request_rejects_invalid_seed_types(seed):
    request = _symbolic_request()

    with pytest.raises(TypeError, match="seed must be an integer or None"):
        EstimatorRequest(
            request.circuit,
            request.observable,
            request.parameter_bindings,
            seed=seed,
        )


def test_estimator_request_rejects_negative_seed():
    request = _symbolic_request()

    with pytest.raises(ValueError, match="seed must be non-negative"):
        EstimatorRequest(
            request.circuit,
            request.observable,
            request.parameter_bindings,
            seed=-1,
        )


def test_estimator_request_requires_matching_qubit_counts():
    circuit = Circuit(1)

    with pytest.raises(ValueError, match="same qubit count"):
        EstimatorRequest(circuit, PauliString.from_str("ZZ"))


def test_estimator_mapping_uses_circuit_qubit_insertion_order():
    circuit = Circuit([7, 2])
    observable = PauliString.from_str("XZ")
    request = EstimatorRequest(circuit, observable)

    mapping = tuple(zip(range(observable.num_qubits), request.circuit.qubits))
    assert mapping == ((0, Qubit(7)), (1, Qubit(2)))


@pytest.mark.parametrize("phase", [Phase.i(), Phase.minus_i()])
def test_estimator_request_rejects_non_real_standalone_pauli_phase(phase):
    circuit = Circuit(1)
    pauli = PauliString.from_str("Z")
    pauli.phase = phase

    with pytest.raises(ValueError, match=r"phase must be \+1 or -1"):
        EstimatorRequest(circuit, pauli)


def test_estimator_request_accepts_negative_standalone_pauli_phase():
    circuit = Circuit(1)
    pauli = PauliString.from_str("Z")
    pauli.phase = Phase.minus()

    assert EstimatorRequest(circuit, pauli).observable is pauli


@pytest.mark.parametrize("coefficient", [1j, 1e-11j])
def test_estimator_request_strictly_rejects_imaginary_hamiltonian(coefficient):
    circuit = Circuit(1)
    hamiltonian = Hamiltonian(1)
    hamiltonian.add_term(PauliString.from_str("Z"), coefficient)

    with pytest.raises(ValueError, match="Hermitian"):
        EstimatorRequest(circuit, hamiltonian)


def test_estimator_request_rejects_non_finite_original_effective_coefficient():
    circuit = Circuit(1)
    hamiltonian = Hamiltonian(1)
    pauli = PauliString.from_str("Z")
    hamiltonian.add_term(pauli, complex(float("nan"), 0.0))
    hamiltonian.add_term(pauli, 1.0)

    with pytest.raises(ValueError, match="finite"):
        EstimatorRequest(circuit, hamiltonian)


def test_estimator_request_aggregates_exact_duplicate_cancellation():
    circuit = Circuit(1)
    hamiltonian = Hamiltonian(1)
    pauli = PauliString.from_str("Z")
    hamiltonian.add_term(pauli, 1j)
    hamiltonian.add_term(pauli, -1j)

    assert EstimatorRequest(circuit, hamiltonian).observable is hamiltonian


def test_estimator_request_aggregates_by_phase_normalized_pauli_key():
    circuit = Circuit(1)
    hamiltonian = Hamiltonian(1)
    positive = PauliString.from_str("Z")
    negative = PauliString.from_str("Z")
    negative.phase = Phase.minus()
    hamiltonian.add_term(positive, 1j)
    hamiltonian.add_term(negative, 1j)

    assert EstimatorRequest(circuit, hamiltonian).observable is hamiltonian


def test_estimator_request_absorbs_hamiltonian_term_phase_into_coefficient():
    circuit = Circuit(1)
    hamiltonian = Hamiltonian(1)
    pauli = PauliString.from_str("Z")
    pauli.phase = Phase.i()
    hamiltonian.add_term(pauli, -1j)

    assert EstimatorRequest(circuit, hamiltonian).observable is hamiltonian


def test_estimator_revalidates_mutable_observable_before_execution():
    request = EstimatorRequest(
        Circuit(1),
        Hamiltonian.from_pauli(PauliString.from_str("Z")),
    )
    request.observable.add_term(PauliString.from_str("X"), 1e-11j)

    with pytest.raises(ValueError, match="Hermitian"):
        _FakeEstimator().run([request])


def test_exact_only_estimator_may_reject_shot_based_requests():
    exact = _symbolic_request()
    sampled = EstimatorRequest(
        exact.circuit,
        exact.observable,
        exact.parameter_bindings,
        100,
    )

    assert _ExactOnlyEstimator().run([exact]) == [EstimatorResult(0.0, 0.0)]
    with pytest.raises(ValueError, match="not supported"):
        _ExactOnlyEstimator().run([sampled])


@pytest.mark.parametrize(
    ("expectation", "variance", "exception"),
    [
        (float("nan"), None, ValueError),
        (float("inf"), None, ValueError),
        (True, None, TypeError),
        (0.0, -1.0, ValueError),
        (0.0, float("nan"), ValueError),
        (0.0, True, TypeError),
    ],
)
def test_estimator_result_rejects_invalid_values(expectation, variance, exception):
    with pytest.raises(exception):
        EstimatorResult(expectation, variance)


def test_estimator_result_accepts_unknown_or_zero_estimate_variance():
    assert EstimatorResult(0, None) == EstimatorResult(0.0, None)
    assert EstimatorResult(1, 0).variance == 0.0


def test_estimator_preserves_batch_order_and_supports_empty_batches():
    estimator = _FakeEstimator()
    first = _symbolic_request()
    second = EstimatorRequest(
        first.circuit,
        first.observable,
        {"theta": 0.75},
        100,
    )

    assert _run_estimator(estimator, []) == []
    results = _run_estimator(estimator, [first, second])

    assert [result.expectation for result in results] == pytest.approx([0.25, 0.75])
    assert [result.variance for result in results] == [0.0, 0.0]


def test_estimator_is_distinct_and_uses_nominal_runtime_membership():
    assert Estimator is not ErrorMitigationEstimator
    assert isinstance(_FakeEstimator(), Estimator)
