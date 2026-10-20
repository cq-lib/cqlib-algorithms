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

"""Independent matrix and boundary tests for one-hot value mixers."""

import math

import numpy as np
import pytest
from cqlib.circuit import ParameterError
from cqlib.circuit.ansatz import EvolutionStrategy
from cqlib.qis import TrotterMode

from cqlib_algorithms.operators.mixer import (
    FullyConnectedMixer,
    NearbyValuesMixer,
    NullSwapMixer,
    ParityRingMixer,
    RingMixer,
)


def _exchange_matrix(num_qubits: int, left: int, right: int) -> np.ndarray:
    matrix = np.zeros((2**num_qubits, 2**num_qubits), dtype=np.complex128)
    for state in range(2**num_qubits):
        left_bit = (state >> left) & 1
        right_bit = (state >> right) & 1
        if left_bit != right_bit:
            matrix[state ^ (1 << left) ^ (1 << right), state] = 1.0
    return matrix


def _unitary(generator: np.ndarray, beta: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(generator)
    return (vectors * np.exp(-1j * beta * values)) @ vectors.conj().T


def _product_formula_reference(
    num_values: int,
    pairs: tuple[tuple[int, int], ...],
    beta: float,
    *,
    second_order: bool,
    steps: int,
) -> np.ndarray:
    """Construct the ordered exchange product without production helpers."""
    expected = np.eye(2**num_values, dtype=np.complex128)
    if second_order:
        ordered_pairs = (*pairs, *reversed(pairs))
        angle = beta / (2.0 * steps)
    else:
        ordered_pairs = pairs
        angle = beta / steps
    for _ in range(steps):
        for left, right in ordered_pairs:
            local = _unitary(_exchange_matrix(num_values, left, right), angle)
            expected = local @ expected
    return expected


@pytest.mark.reference
def test_two_value_simultaneous_mixers_have_the_intended_multiplicity() -> None:
    beta = 0.37
    exchange = _exchange_matrix(2, 0, 1)
    expected_ring = _unitary(2.0 * exchange, beta)
    expected_null = _unitary(exchange, beta)

    for mixer in (NearbyValuesMixer(2, 1), RingMixer(2), FullyConnectedMixer(2)):
        actual = mixer.build_circuit("m").assign_parameters({"m_beta": beta})
        np.testing.assert_allclose(
            actual.to_matrix(), expected_ring, rtol=0.0, atol=1e-12
        )

    actual_null = (
        NullSwapMixer(2).build_circuit("m").assign_parameters({"m_beta": beta})
    )
    np.testing.assert_allclose(
        actual_null.to_matrix(), expected_null, rtol=0.0, atol=1e-12
    )


@pytest.mark.reference
@pytest.mark.parametrize(
    ("mixer", "pairs", "second_order", "steps", "beta"),
    [
        (
            NearbyValuesMixer(
                4,
                2,
                strategy=EvolutionStrategy.trotter(
                    TrotterMode.first_order(),
                    3,
                ),
            ),
            (
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (0, 2),
                (1, 3),
                (2, 0),
                (3, 1),
            ),
            False,
            3,
            0.37,
        ),
        (
            FullyConnectedMixer(
                3,
                strategy=EvolutionStrategy.trotter(
                    TrotterMode.second_order(),
                    2,
                ),
            ),
            ((0, 1), (1, 2), (2, 0), (0, 2), (1, 0), (2, 1)),
            True,
            2,
            -0.29,
        ),
    ],
)
def test_simultaneous_mixers_match_independent_multistep_product_formulas(
    mixer: NearbyValuesMixer | FullyConnectedMixer,
    pairs: tuple[tuple[int, int], ...],
    second_order: bool,
    steps: int,
    beta: float,
) -> None:
    actual = mixer.build_circuit("m").assign_parameters({"m_beta": beta}).to_matrix()
    expected = _product_formula_reference(
        mixer.num_qubits(),
        pairs,
        beta,
        second_order=second_order,
        steps=steps,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.reference
def test_null_swap_generator_matches_all_null_to_value_transitions() -> None:
    epsilon = 1e-6
    mixer = NullSwapMixer(3, null_value=1)
    forward = mixer.build_circuit("m").assign_parameters({"m_beta": epsilon})
    backward = mixer.build_circuit("m").assign_parameters({"m_beta": -epsilon})
    numerical_generator = (
        1j * (forward.to_matrix() - backward.to_matrix()) / (2.0 * epsilon)
    )
    expected = _exchange_matrix(3, 1, 0) + _exchange_matrix(3, 1, 2)
    np.testing.assert_allclose(numerical_generator, expected, rtol=0.0, atol=2e-10)


@pytest.mark.reference
def test_three_value_ring_matches_independent_finite_beta_product() -> None:
    beta = 0.31
    mixer = RingMixer(3)
    actual = mixer.build_circuit("r").assign_parameters({"r_beta": beta}).to_matrix()
    expected = _product_formula_reference(
        3,
        ((0, 1), (1, 2), (2, 0)),
        beta,
        second_order=False,
        steps=1,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.reference
@pytest.mark.parametrize(
    ("mode", "steps"),
    [(TrotterMode.first_order(), 3), (TrotterMode.second_order(), 2)],
)
def test_null_swap_matches_independent_multistep_product_formula(
    mode: TrotterMode,
    steps: int,
) -> None:
    beta = -0.27
    mixer = NullSwapMixer(
        3,
        strategy=EvolutionStrategy.trotter(mode, steps),
    )
    actual = mixer.build_circuit("n").assign_parameters({"n_beta": beta}).to_matrix()
    expected = _product_formula_reference(
        3,
        ((0, 1), (0, 2)),
        beta,
        second_order=mode == TrotterMode.second_order(),
        steps=steps,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.reference
@pytest.mark.parametrize("num_values", [3, 4])
def test_parity_ring_matches_independent_ordered_exchange_product(
    num_values: int,
) -> None:
    beta = -0.29
    actual = (
        ParityRingMixer(num_values)
        .build_circuit("p")
        .assign_parameters({"p_beta": beta})
    )
    layers = [
        [(start, start + 1) for start in range(0, num_values - 1, 2)],
        [(start, (start + 1) % num_values) for start in range(1, num_values, 2)],
    ]
    if num_values % 2:
        layers.append([(num_values - 1, 0)])
    expected = np.eye(2**num_values, dtype=np.complex128)
    for layer in layers:
        for left, right in layer:
            expected = (
                _unitary(_exchange_matrix(num_values, left, right), beta) @ expected
            )
    np.testing.assert_allclose(actual.to_matrix(), expected, rtol=0.0, atol=1e-12)


@pytest.mark.reference
def test_repeated_five_value_parity_ring_is_pairwise_connected() -> None:
    num_values = 5
    repetitions = 3
    beta = 0.17
    mixer = ParityRingMixer(num_values, repetitions=repetitions)
    actual = mixer.build_circuit("p").assign_parameters({"p_beta": beta}).to_matrix()

    layers = (
        ((0, 1), (2, 3)),
        ((1, 2), (3, 4)),
        ((4, 0),),
    )
    one_repetition = np.eye(2**num_values, dtype=np.complex128)
    for layer in layers:
        for left, right in layer:
            local = _unitary(_exchange_matrix(num_values, left, right), beta)
            one_repetition = local @ one_repetition
    expected = np.linalg.matrix_power(one_repetition, repetitions)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)

    feasible = [1 << value for value in range(num_values)]
    assert np.all(np.abs(actual[np.ix_(feasible, feasible)]) > 1e-12)


def test_all_one_hot_mixers_expose_one_stable_beta_parameter() -> None:
    mixers = (
        NearbyValuesMixer(4, 2),
        RingMixer(4),
        FullyConnectedMixer(4),
        NullSwapMixer(4, null_value=2),
        ParityRingMixer(4, repetitions=2),
    )
    for mixer in mixers:
        first = mixer.build_circuit("theta")
        second = mixer.build_circuit("theta")
        assert mixer.num_parameters() == 1
        assert first.used_symbols == ["theta_beta"]
        assert first == second
        assert first is not second


@pytest.mark.parametrize(
    ("factory", "error"),
    [
        (lambda: NearbyValuesMixer(1, 1), ValueError),
        (lambda: NearbyValuesMixer(3, 0), ValueError),
        (lambda: NearbyValuesMixer(3, 3), ValueError),
        (lambda: RingMixer(True), TypeError),
        (lambda: FullyConnectedMixer(1), ValueError),
        (lambda: NullSwapMixer(3, null_value=3), ValueError),
        (lambda: NullSwapMixer(3, null_value=True), TypeError),
        (lambda: ParityRingMixer(3, repetitions=0), ValueError),
        (lambda: NearbyValuesMixer(3, 1, strategy=object()), TypeError),
    ],
)
def test_one_hot_mixer_boundaries(factory: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        factory()  # type: ignore[operator]


def test_exact_strategy_rejects_a_noncommuting_simultaneous_sum() -> None:
    with pytest.raises(ValueError, match="commut"):
        NullSwapMixer(3, strategy=EvolutionStrategy.exact())


@pytest.mark.parametrize(
    "strategy",
    [
        None,
        EvolutionStrategy.trotter(TrotterMode.first_order(), 2),
        EvolutionStrategy.trotter(TrotterMode.second_order(), 2),
    ],
)
def test_supported_product_formulas_preserve_the_one_hot_subspace(
    strategy: EvolutionStrategy | None,
) -> None:
    mixer = RingMixer(3, strategy=strategy)
    matrix = mixer.build_circuit("m").assign_parameters({"m_beta": 0.41}).to_matrix()
    feasible = [1, 2, 4]
    infeasible = sorted(set(range(8)) - set(feasible))
    np.testing.assert_allclose(
        matrix[np.ix_(infeasible, feasible)], 0.0, rtol=0.0, atol=1e-12
    )


def test_randomized_pauli_term_order_is_rejected_before_it_can_leak() -> None:
    strategy = EvolutionStrategy.trotter(TrotterMode.randomized(7), 2)
    with pytest.raises(ValueError, match="randomized evolution"):
        RingMixer(3, strategy=strategy)


def test_nonfinite_beta_is_rejected_by_core_binding() -> None:
    circuit = RingMixer(3).build_circuit("m")
    with pytest.raises(ParameterError):
        circuit.assign_parameters({"m_beta": math.inf})
