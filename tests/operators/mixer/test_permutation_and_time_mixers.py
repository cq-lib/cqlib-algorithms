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

"""Permutation and scheduling mixer correctness tests."""

from itertools import permutations, product

import numpy as np
import pytest
from cqlib.qis import Statevector

from cqlib_algorithms.operators.mixer import (
    ColorParityPermutationSwapMixer,
    TimeColorMixer,
    TimeSwapMixer,
    ValueIndependentPermutationSwapMixer,
    ValueSelectivePermutationSwapMixer,
)


def _unitary(generator: np.ndarray, beta: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(generator)
    return (vectors * np.exp(-1j * beta * values)) @ vectors.conj().T


def _left_apply_transition(
    matrix: np.ndarray,
    targets: tuple[int, int, int, int],
    beta: float,
) -> None:
    """Left-apply an independently constructed four-bit transition."""
    first_pair = (1 << targets[0]) | (1 << targets[1])
    second_pair = (1 << targets[2]) | (1 << targets[3])
    target_mask = first_pair | second_pair
    cosine = np.cos(beta)
    sine = -1j * np.sin(beta)
    for spectator in range(matrix.shape[0]):
        if spectator & target_mask:
            continue
        first = spectator | first_pair
        second = spectator | second_pair
        first_row = matrix[first].copy()
        second_row = matrix[second].copy()
        matrix[first] = cosine * first_row + sine * second_row
        matrix[second] = sine * first_row + cosine * second_row


def _apply_transition_to_state(
    state: np.ndarray,
    targets: tuple[int, int, int, int],
    beta: float,
) -> None:
    """Apply a four-bit two-state rotation to a test-side statevector."""
    first_pair = (1 << targets[0]) | (1 << targets[1])
    second_pair = (1 << targets[2]) | (1 << targets[3])
    target_mask = first_pair | second_pair
    indices = np.arange(state.size, dtype=np.int64)
    spectators = indices[(indices & target_mask) == 0]
    first = spectators | first_pair
    second = spectators | second_pair
    first_values = state[first].copy()
    second_values = state[second].copy()
    cosine = np.cos(beta)
    sine = -1j * np.sin(beta)
    state[first] = cosine * first_values + sine * second_values
    state[second] = sine * first_values + cosine * second_values


def _simulate(circuit: object, initial: np.ndarray) -> np.ndarray:
    """Run a bound Core circuit on an independently supplied input state."""
    state = Statevector.from_state(initial.size.bit_length() - 1, initial)
    state.apply_circuit(circuit)  # type: ignore[arg-type]
    return np.asarray(state.data, dtype=np.complex128)


def _weighted_state(width: int, basis_states: list[int]) -> np.ndarray:
    """Return a deterministic generic superposition over selected states."""
    state = np.zeros(2**width, dtype=np.complex128)
    for index, basis_state in enumerate(basis_states):
        state[basis_state] = (index + 1) + 1j * ((3 * index) % 7 + 1)
    return state / np.linalg.norm(state)


def _permutation_basis(num_items: int) -> list[int]:
    return [
        sum(1 << (position * num_items + value) for position, value in enumerate(order))
        for order in permutations(range(num_items))
    ]


def _absolute_offsets(
    processing_times: tuple[int, ...], horizon: int
) -> tuple[int, ...]:
    offsets: list[int] = []
    cursor = 0
    for duration in processing_times:
        offsets.append(cursor)
        cursor += horizon - duration + 1
    return tuple(offsets)


def _absolute_one_hot_basis(
    processing_times: tuple[int, ...], horizon: int
) -> list[int]:
    offsets = _absolute_offsets(processing_times, horizon)
    starts = [range(horizon - duration + 1) for duration in processing_times]
    return [
        sum(1 << (offsets[job] + start) for job, start in enumerate(schedule))
        for schedule in product(*starts)
    ]


@pytest.mark.reference
def test_value_selective_swap_is_exact_on_the_full_four_qubit_space() -> None:
    generator = np.zeros((16, 16), dtype=np.complex128)
    generator[3, 12] = generator[12, 3] = 1.0
    beta = 0.39
    actual = ValueSelectivePermutationSwapMixer().build_circuit("v")
    actual = actual.assign_parameters({"v_beta": beta})
    np.testing.assert_allclose(
        actual.to_matrix(), _unitary(generator, beta), rtol=0.0, atol=1e-12
    )


@pytest.mark.reference
def test_value_independent_swap_matches_exact_swap_blocks_on_feasible_space() -> None:
    num_values = 3
    beta = -0.31
    actual = ValueIndependentPermutationSwapMixer(num_values).build_circuit("v")
    matrix = actual.assign_parameters({"v_beta": beta}).to_matrix()
    feasible = [
        (1 << first) | (1 << (num_values + second))
        for first in range(num_values)
        for second in range(num_values)
    ]
    submatrix = matrix[np.ix_(feasible, feasible)]
    generator = np.zeros((num_values**2, num_values**2), dtype=np.complex128)
    for first in range(num_values):
        for second in range(num_values):
            if first != second:
                generator[second * num_values + first, first * num_values + second] = (
                    1.0
                )
    np.testing.assert_allclose(
        submatrix, _unitary(generator, beta), rtol=0.0, atol=1e-12
    )
    infeasible = sorted(set(range(2 ** (2 * num_values))) - set(feasible))
    np.testing.assert_allclose(
        matrix[np.ix_(infeasible, feasible)], 0.0, rtol=0.0, atol=1e-12
    )


@pytest.mark.reference
def test_two_value_independent_swap_matches_its_single_transition() -> None:
    beta = 0.24
    actual = ValueIndependentPermutationSwapMixer(2).build_circuit("v")
    actual = actual.assign_parameters({"v_beta": beta}).to_matrix()
    expected = np.eye(16, dtype=np.complex128)
    _left_apply_transition(expected, (0, 3, 2, 1), beta)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.reference
def test_color_parity_mixer_preserves_direct_one_hot_permutations() -> None:
    num_items = 3
    beta = 0.17
    matrix = ColorParityPermutationSwapMixer(num_items).build_circuit("c")
    matrix = matrix.assign_parameters({"c_beta": beta}).to_matrix()
    feasible: list[int] = []
    for first in range(num_items):
        for second in range(num_items):
            for third in range(num_items):
                if len({first, second, third}) == num_items:
                    feasible.append(
                        (1 << first)
                        | (1 << (num_items + second))
                        | (1 << (2 * num_items + third))
                    )
    infeasible = sorted(set(range(2 ** (num_items**2))) - set(feasible))
    np.testing.assert_allclose(
        matrix[np.ix_(infeasible, feasible)], 0.0, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        matrix[:, feasible].conj().T @ matrix[:, feasible],
        np.eye(len(feasible)),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.reference
def test_color_parity_mixer_matches_independent_color_then_parity_order() -> None:
    num_items = 3
    beta = -0.23
    actual = ColorParityPermutationSwapMixer(num_items).build_circuit("c")
    actual = actual.assign_parameters({"c_beta": beta}).to_matrix()

    # For K3 the deterministic color classes are its three individual edges.
    # The three-position ring is partitioned as odd, even, then closing edge.
    value_colors = (((0, 1),), ((0, 2),), ((1, 2),))
    position_layers = (((0, 1),), ((1, 2),), ((2, 0),))
    expected = np.eye(2 ** (num_items**2), dtype=np.complex128)
    for color in value_colors:
        for layer in position_layers:
            for position, next_position in layer:
                for first_value, second_value in color:
                    targets = (
                        position * num_items + first_value,
                        next_position * num_items + second_value,
                        next_position * num_items + first_value,
                        position * num_items + second_value,
                    )
                    _left_apply_transition(expected, targets, beta)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.reference
def test_repeated_color_parity_is_pairwise_connected_on_permutations() -> None:
    num_items = 3
    repetitions = 3
    beta = 0.17
    actual = ColorParityPermutationSwapMixer(
        num_items,
        repetitions=repetitions,
    ).build_circuit("c")
    actual = actual.assign_parameters({"c_beta": beta}).to_matrix()

    value_colors = (((0, 1),), ((0, 2),), ((1, 2),))
    position_layers = (((0, 1),), ((1, 2),), ((2, 0),))
    expected = np.eye(2 ** (num_items**2), dtype=np.complex128)
    for _ in range(repetitions):
        for color in value_colors:
            for layer in position_layers:
                for position, next_position in layer:
                    for first_value, second_value in color:
                        targets = (
                            position * num_items + first_value,
                            next_position * num_items + second_value,
                            next_position * num_items + first_value,
                            position * num_items + second_value,
                        )
                        _left_apply_transition(expected, targets, beta)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)

    feasible = _permutation_basis(num_items)
    assert np.all(np.abs(actual[np.ix_(feasible, feasible)]) > 1e-12)


@pytest.mark.reference
def test_two_item_color_parity_matches_both_position_edges() -> None:
    beta = -0.21
    actual = ColorParityPermutationSwapMixer(2).build_circuit("c")
    actual = actual.assign_parameters({"c_beta": beta}).to_matrix()
    expected = np.eye(16, dtype=np.complex128)
    for position, next_position in ((0, 1), (1, 0)):
        _left_apply_transition(
            expected,
            (
                position * 2,
                next_position * 2 + 1,
                next_position * 2,
                position * 2 + 1,
            ),
            beta,
        )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.reference
def test_k4_color_parity_matches_all_edges_in_each_color_on_a_statevector() -> None:
    num_items = 4
    beta = 0.13
    colors = (
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    )
    position_layers = (
        ((0, 1), (2, 3)),
        ((1, 2), (3, 0)),
    )
    initial = _weighted_state(num_items**2, _permutation_basis(num_items))
    expected = initial.copy()
    for color in colors:
        for layer in position_layers:
            for position, next_position in layer:
                for first_value, second_value in color:
                    targets = (
                        position * num_items + first_value,
                        next_position * num_items + second_value,
                        next_position * num_items + first_value,
                        position * num_items + second_value,
                    )
                    _apply_transition_to_state(expected, targets, beta)

    circuit = ColorParityPermutationSwapMixer(num_items).build_circuit("c")
    actual = _simulate(circuit.assign_parameters({"c_beta": beta}), initial)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.reference
def test_time_swap_and_single_part_time_color_match_exact_schedule_transition() -> None:
    processing_times = [1, 2]
    beta = 0.28
    partial = TimeSwapMixer(processing_times, 0, 1, 0)
    full = TimeColorMixer(processing_times)
    assert partial.num_qubits() == full.num_qubits() == 5
    partial_matrix = (
        partial.build_circuit("t").assign_parameters({"t_beta": beta}).to_matrix()
    )
    full_matrix = (
        full.build_circuit("t").assign_parameters({"t_beta": beta}).to_matrix()
    )
    np.testing.assert_allclose(full_matrix, partial_matrix, rtol=0.0, atol=1e-12)

    # Offsets are 0 and 3. The transition is between starts (0,1) and (2,0).
    source = (1 << 0) | (1 << 4)
    target = (1 << 2) | (1 << 3)
    generator = np.zeros((32, 32), dtype=np.complex128)
    spectator = 1
    for spectator_bit in (0, 1):
        spectator_mask = spectator_bit << spectator
        generator[source | spectator_mask, target | spectator_mask] = 1.0
        generator[target | spectator_mask, source | spectator_mask] = 1.0
    np.testing.assert_allclose(
        partial_matrix, _unitary(generator, beta), rtol=0.0, atol=1e-12
    )


@pytest.mark.reference
def test_time_color_matches_independent_time_then_color_order() -> None:
    processing_times = (1, 1, 1)
    horizon = 3
    beta = 0.19
    mixer = TimeColorMixer(processing_times, horizon=horizon)
    actual = mixer.build_circuit("t").assign_parameters({"t_beta": beta}).to_matrix()

    # Each duration-one job has starts 0, 1, 2. Independently enumerate time
    # before the three K3 color classes; only t=0,1 can contain both jobs.
    offsets = (0, 3, 6)
    value_colors = (((0, 1),), ((0, 2),), ((1, 2),))
    expected = np.eye(2**9, dtype=np.complex128)
    for start_time in (0, 1):
        for color in value_colors:
            for first_job, second_job in color:
                targets = (
                    offsets[first_job] + start_time + 1,
                    offsets[second_job] + start_time,
                    offsets[first_job] + start_time,
                    offsets[second_job] + start_time + 1,
                )
                _left_apply_transition(expected, targets, beta)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.reference
def test_repeated_time_color_matches_two_independent_passes() -> None:
    beta = 0.18
    actual = TimeColorMixer((1, 1), repetitions=2).build_circuit("t")
    actual = actual.assign_parameters({"t_beta": beta}).to_matrix()
    expected = np.eye(16, dtype=np.complex128)
    for _ in range(2):
        _left_apply_transition(expected, (1, 2, 0, 3), beta)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.reference
def test_k4_time_color_matches_all_job_edges_in_each_color() -> None:
    processing_times = (1, 1, 1, 1)
    horizon = 4
    beta = -0.11
    colors = (
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    )
    offsets = (0, 4, 8, 12)
    initial = _weighted_state(
        16,
        _absolute_one_hot_basis(processing_times, horizon),
    )
    expected = initial.copy()
    for start_time in range(3):
        for color in colors:
            for first_job, second_job in color:
                targets = (
                    offsets[first_job] + start_time + 1,
                    offsets[second_job] + start_time,
                    offsets[first_job] + start_time,
                    offsets[second_job] + start_time + 1,
                )
                _apply_transition_to_state(expected, targets, beta)

    circuit = TimeColorMixer(processing_times, horizon=horizon).build_circuit("t")
    actual = _simulate(circuit.assign_parameters({"t_beta": beta}), initial)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.reference
@pytest.mark.parametrize(
    ("first_job", "second_job", "targets"),
    [(0, 1, (2, 4, 1, 6)), (1, 0, (6, 1, 4, 2))],
)
def test_time_swap_with_extra_horizon_matches_independent_transition(
    first_job: int,
    second_job: int,
    targets: tuple[int, int, int, int],
) -> None:
    beta = 0.16
    mixer = TimeSwapMixer(
        (2, 1),
        first_job,
        second_job,
        start_time=1,
        horizon=4,
    )
    actual = mixer.build_circuit("t").assign_parameters({"t_beta": beta}).to_matrix()
    expected = np.eye(128, dtype=np.complex128)
    _left_apply_transition(expected, targets, beta)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.reference
def test_unequal_duration_time_color_matches_independent_ordered_statevector() -> None:
    processing_times = (1, 2, 1)
    horizon = 5
    beta = 0.12
    colors = (((0, 1),), ((0, 2),), ((1, 2),))
    offsets = _absolute_offsets(processing_times, horizon)
    width = sum(horizon - duration + 1 for duration in processing_times)
    initial = _weighted_state(
        width,
        _absolute_one_hot_basis(processing_times, horizon),
    )
    expected = initial.copy()
    for start_time in range(horizon + 1):
        for color in colors:
            for first_job, second_job in color:
                first_duration = processing_times[first_job]
                second_duration = processing_times[second_job]
                if start_time > horizon - first_duration - second_duration:
                    continue
                targets = (
                    offsets[first_job] + start_time + second_duration,
                    offsets[second_job] + start_time,
                    offsets[first_job] + start_time,
                    offsets[second_job] + start_time + first_duration,
                )
                _apply_transition_to_state(expected, targets, beta)

    circuit = TimeColorMixer(processing_times, horizon=horizon).build_circuit("t")
    actual = _simulate(circuit.assign_parameters({"t_beta": beta}), initial)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_processing_times_are_snapshotted() -> None:
    durations = [1, 2, 1]
    mixer = TimeColorMixer(durations)
    before = mixer.build_circuit("t")
    durations[0] = 9
    after = mixer.build_circuit("t")
    assert before == after


def test_time_swap_processing_times_are_snapshotted() -> None:
    durations = [2, 1]
    mixer = TimeSwapMixer(durations, 0, 1, 1, horizon=4)
    before = mixer.build_circuit("t")
    durations[:] = [9, 9]
    after = mixer.build_circuit("t")
    assert before == after


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ValueIndependentPermutationSwapMixer(1),
        lambda: ColorParityPermutationSwapMixer(1),
        lambda: TimeSwapMixer([1], 0, 0, 0),
        lambda: TimeSwapMixer([1, 2], 0, 0, 0),
        lambda: TimeSwapMixer([1, 2], 0, 1, 1),
        lambda: TimeColorMixer([1, 2], horizon=2),
        lambda: TimeColorMixer([1, True]),
        lambda: TimeColorMixer([1, 2], repetitions=0),
    ],
)
def test_permutation_and_time_boundaries(factory: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()  # type: ignore[operator]
