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

"""QAOA integration tests for circuit-defined one-parameter mixers."""

import math

import numpy as np
import pytest
from cqlib.circuit import Circuit, Parameter
from cqlib.circuit.ansatz import EvolutionStrategy
from cqlib.qis import Hamiltonian, PauliString

from cqlib_algorithms.algorithms import QAOA
from cqlib_algorithms.operators import (
    BasisStateInitialization,
    COBYLAOptimizer,
    FullyConnectedMixer,
    GradientDescent,
    ParameterShiftGradient,
    ParityRingMixer,
    RingMixer,
    StatevectorEstimator,
)
from cqlib_algorithms.protocols import Ansatz


def _algorithm(**kwargs: object) -> QAOA:
    return QAOA(
        StatevectorEstimator(),
        GradientDescent(ParameterShiftGradient(), maxiter=1),
        **kwargs,  # type: ignore[arg-type]
    )


def test_qaoa_builds_each_custom_mixer_layer_with_the_layer_beta() -> None:
    algorithm = _algorithm(reps=2, mixer=ParityRingMixer(2), prefix="alt")
    cost = Hamiltonian.from_pauli(PauliString.from_str("ZZ"))
    circuit = algorithm.build_circuit(cost)
    assert circuit.used_symbols == [
        "alt_0_gamma",
        "alt_0_beta",
        "alt_1_gamma",
        "alt_1_beta",
    ]


@pytest.mark.parametrize("mixer_type", [RingMixer, FullyConnectedMixer])
def test_run_with_interchangeable_one_hot_mixers(mixer_type) -> None:
    algorithm = QAOA(
        StatevectorEstimator(),
        COBYLAOptimizer(maxiter=15),
        initial_state=BasisStateInitialization(4, 1),
        mixer=mixer_type(4),
    )
    cost = Hamiltonian.from_pauli(PauliString.from_str("IZII"))
    result = algorithm.run(cost, {"qaoa_0_gamma": 0.37, "qaoa_0_beta": -0.21})

    probabilities = np.abs(result.optimal_circuit.to_matrix()[:, 0]) ** 2
    assert sum(probabilities[[1, 2, 4, 8]]) == pytest.approx(1.0, abs=1e-12)
    assert result.optimal_value == pytest.approx(1 - 2 * probabilities[4], abs=1e-12)


@pytest.mark.reference
def test_parameter_shift_executes_one_round_with_a_circuit_mixer() -> None:
    learning_rate = 0.1
    algorithm = QAOA(
        StatevectorEstimator(),
        GradientDescent(
            ParameterShiftGradient(),
            learning_rate=learning_rate,
            maxiter=1,
        ),
        mixer=ParityRingMixer(2),
    )
    cost = Hamiltonian.from_pauli(PauliString.from_str("ZI"))
    initial_point = {"qaoa_0_gamma": 0.37, "qaoa_0_beta": -0.21}
    circuit = algorithm.build_circuit(cost)
    cost_matrix = cost.to_matrix()

    def minimized_objective(beta: float) -> float:
        bindings = dict(initial_point)
        bindings["qaoa_0_beta"] = beta
        state = circuit.assign_parameters(bindings).to_matrix()[:, 0]
        expectation = np.vdot(state, cost_matrix @ state).real
        return -float(expectation)

    epsilon = 1e-6
    finite_difference_beta = (
        minimized_objective(initial_point["qaoa_0_beta"] + epsilon)
        - minimized_objective(initial_point["qaoa_0_beta"] - epsilon)
    ) / (2.0 * epsilon)
    assert abs(finite_difference_beta) > 1e-3

    result = algorithm.compute_maximum(cost, initial_point)
    expected_beta = (
        initial_point["qaoa_0_beta"] - learning_rate * finite_difference_beta
    )

    assert result.optimizer_result.iterations == 1
    assert tuple(result.optimal_parameters) == (
        "qaoa_0_gamma",
        "qaoa_0_beta",
    )
    assert all(math.isfinite(value) for value in result.optimal_parameters.values())
    assert result.optimal_parameters["qaoa_0_beta"] != pytest.approx(
        initial_point["qaoa_0_beta"],
        abs=1e-8,
    )
    assert result.optimal_parameters["qaoa_0_beta"] == pytest.approx(
        expected_beta,
        abs=1e-8,
    )
    assert math.isfinite(result.optimal_value)
    assert result.optimal_circuit.used_symbols == []
    assert np.isfinite(result.optimal_circuit.to_matrix()).all()


@pytest.mark.reference
def test_two_layer_circuit_mixer_optimization_handles_all_four_parameters() -> None:
    learning_rate = 0.05
    algorithm = QAOA(
        StatevectorEstimator(),
        GradientDescent(
            ParameterShiftGradient(),
            learning_rate=learning_rate,
            maxiter=1,
        ),
        reps=2,
        mixer=ParityRingMixer(2),
        prefix="alt",
    )
    cost = Hamiltonian.from_pauli(PauliString.from_str("ZI"))
    initial_point = {
        "alt_0_gamma": 0.31,
        "alt_0_beta": -0.19,
        "alt_1_gamma": 0.23,
        "alt_1_beta": 0.17,
    }
    circuit = algorithm.build_circuit(cost)
    cost_matrix = cost.to_matrix()

    def minimized_objective(bindings: dict[str, float]) -> float:
        state = circuit.assign_parameters(bindings).to_matrix()[:, 0]
        expectation = np.vdot(state, cost_matrix @ state).real
        return -float(expectation)

    epsilon = 1e-6
    finite_difference: dict[str, float] = {}
    for name in initial_point:
        forward = dict(initial_point)
        backward = dict(initial_point)
        forward[name] += epsilon
        backward[name] -= epsilon
        finite_difference[name] = (
            minimized_objective(forward) - minimized_objective(backward)
        ) / (2.0 * epsilon)
        assert abs(finite_difference[name]) > 1e-3

    result = algorithm.compute_maximum(cost, initial_point)

    assert result.optimizer_result.iterations == 1
    assert tuple(result.optimal_parameters) == (
        "alt_0_gamma",
        "alt_0_beta",
        "alt_1_gamma",
        "alt_1_beta",
    )
    for name, initial_value in initial_point.items():
        expected = initial_value - learning_rate * finite_difference[name]
        assert result.optimal_parameters[name] == pytest.approx(expected, abs=1e-8)
    assert all(math.isfinite(value) for value in result.optimal_parameters.values())
    assert math.isfinite(result.optimal_value)
    assert result.optimal_circuit.used_symbols == []


def test_qaoa_keeps_the_hamiltonian_mixer_interface_backward_compatible() -> None:
    mixer_operator = Hamiltonian.from_pauli(PauliString.from_str("XI"))
    circuit = _algorithm(mixer_operator=mixer_operator).build_circuit(
        Hamiltonian.from_pauli(PauliString.from_str("ZZ"))
    )
    assert circuit.used_symbols == ["qaoa_0_gamma", "qaoa_0_beta"]


def test_qaoa_rejects_ambiguous_mixer_configuration() -> None:
    mixer_operator = Hamiltonian.from_pauli(PauliString.from_str("XI"))
    with pytest.raises(ValueError, match="mutually exclusive"):
        _algorithm(mixer=ParityRingMixer(2), mixer_operator=mixer_operator)
    with pytest.raises(ValueError, match="cannot be used"):
        _algorithm(
            mixer=ParityRingMixer(2),
            mixer_strategy=EvolutionStrategy.auto(2),
        )


def test_qaoa_rejects_invalid_custom_mixer_type_width_and_parameter_count() -> None:
    with pytest.raises(TypeError, match="mixer must be an Ansatz"):
        _algorithm(mixer=object())
    with pytest.raises(ValueError, match="same width"):
        _algorithm(mixer=ParityRingMixer(3)).build_circuit(
            Hamiltonian.from_pauli(PauliString.from_str("ZZ"))
        )

    class TwoParameterMixer(_BadMixer):
        def num_parameters(self) -> int:
            return 2

    with pytest.raises(ValueError, match="exactly one parameter"):
        _algorithm(mixer=TwoParameterMixer())


class _BadMixer(Ansatz):
    def validate(self) -> None:
        return None

    def build_circuit(self, prefix: str) -> Circuit:
        circuit = Circuit(2)
        circuit.rx(0, Parameter(f"{prefix}_wrong"))
        return circuit

    def num_parameters(self) -> int:
        return 1

    def num_qubits(self) -> int:
        return 2


class _NonCircuitMixer(_BadMixer):
    def build_circuit(self, prefix: str) -> Circuit:
        return object()  # type: ignore[return-value]


class _WidthMismatchMixer(_BadMixer):
    def build_circuit(self, prefix: str) -> Circuit:
        circuit = Circuit(1)
        circuit.rx(0, Parameter(f"{prefix}_beta"))
        return circuit


class _ExtraSymbolMixer(_BadMixer):
    def build_circuit(self, prefix: str) -> Circuit:
        circuit = Circuit(2)
        circuit.rx(0, Parameter(f"{prefix}_beta"))
        circuit.ry(1, Parameter(f"{prefix}_extra"))
        return circuit


def test_qaoa_rejects_a_custom_mixer_that_breaks_the_beta_contract() -> None:
    with pytest.raises(ValueError, match="layer beta"):
        _algorithm(mixer=_BadMixer()).build_circuit(
            Hamiltonian.from_pauli(PauliString.from_str("ZZ"))
        )


@pytest.mark.parametrize(
    ("mixer", "error", "message"),
    [
        (_NonCircuitMixer(), TypeError, "must build a cqlib.Circuit"),
        (_WidthMismatchMixer(), ValueError, "inconsistent width"),
        (_ExtraSymbolMixer(), ValueError, "exactly the layer beta"),
    ],
)
def test_qaoa_rejects_low_level_circuit_mixer_contract_violations(
    mixer: Ansatz,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _algorithm(mixer=mixer).build_circuit(
            Hamiltonian.from_pauli(PauliString.from_str("ZZ"))
        )
