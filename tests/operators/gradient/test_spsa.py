# This code is part of Cqlib.
#
# (C) Copyright China Telecom Quantum Group 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# use this file except in compliance with the License, obtained from the
# LICENSE.txt file in the root directory of this source tree or
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""SPSA gradient tests for reproducibility, batching and unbiasedness."""

from __future__ import annotations

import numpy as np
import pytest
from cqlib import Circuit, Parameter
from cqlib.qis import PauliString

from cqlib_algorithms.operators.gradient import (
    ParameterShiftGradient,
    SPSAGradient,
)
from cqlib_algorithms.protocols import EstimatorRequest, GradientRequest


def _request(circuit, observable, bindings, names):
    return GradientRequest(EstimatorRequest(circuit, observable, bindings), names)


def _two_parameter_objective():
    first = Parameter("a")
    second = Parameter("b")
    circuit = Circuit(2)
    circuit.ry(0, first)
    circuit.rz(1, second)
    circuit.h(0)
    circuit.s(1)
    return _request(
        circuit,
        PauliString.from_str("ZX"),
        {"a": 0.3, "b": -0.4},
        ("a", "b"),
    )


class TestConstruction:
    def test_perturbation_must_be_positive(self) -> None:
        for perturbation in (0.0, -0.1, float("inf"), float("nan")):
            with pytest.raises(ValueError, match="finite positive"):
                SPSAGradient(perturbation=perturbation)

    def test_perturbation_type_is_checked(self) -> None:
        with pytest.raises(TypeError, match="real number"):
            SPSAGradient(perturbation="small")  # type: ignore[arg-type]

    def test_seed_type_is_checked(self) -> None:
        with pytest.raises(TypeError, match="int, a Generator"):
            SPSAGradient(perturbation=0.1, seed="fixed")  # type: ignore[arg-type]

    def test_accepts_generator_seed(self) -> None:
        generator = np.random.default_rng(7)
        gradient = SPSAGradient(0.1, seed=generator)
        assert gradient.perturbation == 0.1


class TestReproducibility:
    def test_same_seed_replays_same_sequence(self, matrix_estimator) -> None:
        request = _two_parameter_objective()
        first = SPSAGradient(0.05, seed=123).run(matrix_estimator, [request])[0]
        second = SPSAGradient(0.05, seed=123).run(matrix_estimator, [request])[0]
        assert first.values == second.values

    def test_generator_state_advances_across_runs(self, matrix_estimator) -> None:
        # delta and -delta always yield the same estimate (the difference
        # flips sign together with the perturbation), so detecting state
        # advance needs a seed whose consecutive draws are not antipodes;
        # seed 0 draws [1,1] then [1,-1].
        request = _two_parameter_objective()
        gradient = SPSAGradient(0.05, seed=0)
        first = gradient.run(matrix_estimator, [request])[0]
        second = gradient.run(matrix_estimator, [request])[0]
        assert first.values != second.values

    def test_shared_generator_stream_matches_rebuild(self, matrix_estimator) -> None:
        request = _two_parameter_objective()
        shared = np.random.default_rng(0)
        holder = SPSAGradient(0.05, seed=shared)
        holder.run(matrix_estimator, [request])
        fresh = SPSAGradient(0.05, seed=0).run(matrix_estimator, [request])[0]
        second = holder.run(matrix_estimator, [request])[0]
        assert second.values != fresh.values


class TestBatching:
    def test_two_evaluations_per_request_in_single_batch(
        self, recording_estimator_factory
    ) -> None:
        request = _two_parameter_objective()
        estimator = recording_estimator_factory()
        SPSAGradient(0.05, seed=1).run(estimator, [request])
        assert estimator.batch_sizes == [2]

    def test_multiple_requests_flatten_into_one_batch(
        self, recording_estimator_factory
    ) -> None:
        request = _two_parameter_objective()
        estimator = recording_estimator_factory()
        SPSAGradient(0.05, seed=1).run(estimator, [request, request])
        assert estimator.batch_sizes == [4]

    def test_empty_parameter_names_skip_estimator(
        self, constant_estimator_factory
    ) -> None:
        theta = Parameter("theta")
        circuit = Circuit(1)
        circuit.ry(0, theta)
        request = _request(circuit, PauliString.from_str("Z"), {"theta": 0.2}, ())
        estimator = constant_estimator_factory(1.0)
        result = SPSAGradient(0.05, seed=1).run(estimator, [request])[0]
        assert result.parameter_names == ()
        assert result.values == ()
        assert estimator.calls == 0

    def test_perturbation_moves_only_requested_parameters(
        self, matrix_estimator
    ) -> None:
        first = Parameter("a")
        fixed = Parameter("b")
        circuit = Circuit(2)
        circuit.ry(0, first)
        circuit.ry(1, fixed)
        request = _request(
            circuit,
            PauliString.from_str("ZZ"),
            {"a": 0.3, "b": 1.1},
            ("a",),
        )
        gradient = SPSAGradient(0.05, seed=3)
        result = gradient.run(matrix_estimator, [request])[0]
        assert result.parameter_names == ("a",)
        assert len(result.values) == 1

    def test_estimator_result_count_mismatch(
        self, constant_estimator_factory, exact_result_factory
    ) -> None:
        request = _two_parameter_objective()

        class Broken(constant_estimator_factory):
            def run(self, requests, /):
                self.calls += 1
                return [exact_result_factory(1.0)]

        with pytest.raises(ValueError, match="per evaluation"):
            SPSAGradient(0.05, seed=1).run(Broken(1.0), [request])

    def test_estimator_type_is_checked(self, matrix_estimator) -> None:
        request = _two_parameter_objective()
        with pytest.raises(TypeError, match="Estimator"):
            SPSAGradient(0.05, seed=1).run(object(), [request])  # type: ignore[arg-type]

    def test_request_type_is_checked(self, matrix_estimator) -> None:
        with pytest.raises(TypeError, match="GradientRequest"):
            SPSAGradient(0.05, seed=1).run(matrix_estimator, [object()])  # type: ignore[list-item]


@pytest.mark.slow
@pytest.mark.reference
class TestUnbiasedness:
    def test_mean_estimate_converges_to_true_gradient(self, matrix_estimator) -> None:
        request = _two_parameter_objective()
        exact = ParameterShiftGradient().run(matrix_estimator, [request])[0].values
        perturbation = 1e-4
        estimates = np.zeros(2)
        trials = 400
        for seed in range(trials):
            estimate = SPSAGradient(perturbation, seed=seed).run(
                matrix_estimator, [request]
            )[0]
            estimates += np.array(estimate.values)
        mean = estimates / trials
        # The bias is O(c); the statistical error shrinks like 1/sqrt(trials).
        np.testing.assert_allclose(mean, exact, atol=5e-3)
