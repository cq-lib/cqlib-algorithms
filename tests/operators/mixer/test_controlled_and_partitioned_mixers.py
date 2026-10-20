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

"""Controlled partial mixer and ordered composition tests."""

import numpy as np
import pytest
from cqlib.circuit import Circuit, Parameter

from cqlib_algorithms.operators.mixer import (
    ControlledBitFlipMixer,
    ControlledNullSwapMixer,
    ControlledSwapMixer,
    MixerApplication,
    PartitionedMixer,
)
from cqlib_algorithms.protocols import Ansatz


def _unitary(generator: np.ndarray, beta: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(generator)
    return (vectors * np.exp(-1j * beta * values)) @ vectors.conj().T


def _controlled_generator(
    num_controls: int,
    control_state: int,
    *,
    two_targets: bool,
) -> np.ndarray:
    width = num_controls + (2 if two_targets else 1)
    matrix = np.zeros((2**width, 2**width), dtype=np.complex128)
    for state in range(2**width):
        if state & ((1 << num_controls) - 1) != control_state:
            continue
        if two_targets:
            left = num_controls
            right = num_controls + 1
            if ((state >> left) & 1) != ((state >> right) & 1):
                target = state ^ (1 << left) ^ (1 << right)
                matrix[target, state] = 1.0
        else:
            matrix[state ^ (1 << num_controls), state] = 1.0
    return matrix


def _mapped_controlled_x_generator(
    width: int,
    control: int,
    target: int,
    control_state: int,
) -> np.ndarray:
    """Construct a global controlled-X generator from bit predicates."""
    matrix = np.zeros((2**width, 2**width), dtype=np.complex128)
    for state in range(2**width):
        if (state >> control) & 1 == control_state:
            matrix[state ^ (1 << target), state] = 1.0
    return matrix


@pytest.mark.reference
def test_controlled_bit_flip_matches_projected_x_without_global_phase() -> None:
    beta = 0.41
    actual = ControlledBitFlipMixer(2, control_state=1).build_circuit("c")
    actual = actual.assign_parameters({"c_beta": beta}).to_matrix()
    expected = _unitary(_controlled_generator(2, 1, two_targets=False), beta)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.reference
def test_default_controlled_bit_flip_uses_all_zero_nor_predicate() -> None:
    beta = 0.29
    actual = ControlledBitFlipMixer(2).build_circuit("c")
    actual = actual.assign_parameters({"c_beta": beta}).to_matrix()
    expected = _unitary(_controlled_generator(2, 0, two_targets=False), beta)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.reference
@pytest.mark.parametrize("mixer_type", [ControlledNullSwapMixer, ControlledSwapMixer])
def test_controlled_exchange_matches_projected_xy(mixer_type: type[Ansatz]) -> None:
    beta = -0.33
    mixer = mixer_type(2, control_state=2)  # type: ignore[call-arg]
    actual = mixer.build_circuit("c").assign_parameters({"c_beta": beta})
    expected = _unitary(_controlled_generator(2, 2, two_targets=True), beta)
    np.testing.assert_allclose(actual.to_matrix(), expected, rtol=0.0, atol=1e-12)


@pytest.mark.reference
@pytest.mark.parametrize("mixer_type", [ControlledNullSwapMixer, ControlledSwapMixer])
def test_default_controlled_exchange_uses_all_zero_none_predicate(
    mixer_type: type[Ansatz],
) -> None:
    beta = -0.37
    mixer = mixer_type(2)  # type: ignore[call-arg]
    actual = mixer.build_circuit("c").assign_parameters({"c_beta": beta})
    expected = _unitary(_controlled_generator(2, 0, two_targets=True), beta)
    np.testing.assert_allclose(actual.to_matrix(), expected, rtol=0.0, atol=1e-12)


@pytest.mark.reference
def test_default_controlled_null_swap_preserves_one_hot_and_only_acts_at_zero() -> None:
    num_controls = 2
    width = num_controls + 2
    beta = 0.43
    matrix = (
        ControlledNullSwapMixer(num_controls)
        .build_circuit("c")
        .assign_parameters({"c_beta": beta})
        .to_matrix()
    )

    for controls in range(1 << num_controls):
        for target in (num_controls, num_controls + 1):
            basis_index = controls | (1 << target)
            column = matrix[:, basis_index]
            supported = np.flatnonzero(np.abs(column) > 1e-12)
            for output in supported:
                assert output & ((1 << num_controls) - 1) == controls
                target_bits = output >> num_controls
                assert target_bits.bit_count() == 1
            if controls:
                expected = np.zeros(2**width, dtype=np.complex128)
                expected[basis_index] = 1.0
                np.testing.assert_allclose(column, expected, rtol=0.0, atol=1e-12)

    zero_control_input = 1 << num_controls
    assert abs(matrix[1 << (num_controls + 1), zero_control_input]) > 1e-12


@pytest.mark.reference
def test_zero_control_bit_flip_is_an_unconditional_rotation() -> None:
    beta = 0.26
    actual = ControlledBitFlipMixer(0).build_circuit("c")
    actual = actual.assign_parameters({"c_beta": beta}).to_matrix()
    expected = _unitary(_controlled_generator(0, 0, two_targets=False), beta)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.reference
@pytest.mark.parametrize("mixer_type", [ControlledNullSwapMixer, ControlledSwapMixer])
def test_zero_control_exchange_is_an_unconditional_exchange(
    mixer_type: type[Ansatz],
) -> None:
    beta = -0.34
    mixer = mixer_type(0)  # type: ignore[call-arg]
    actual = mixer.build_circuit("c").assign_parameters({"c_beta": beta})
    expected = _unitary(_controlled_generator(0, 0, two_targets=True), beta)
    np.testing.assert_allclose(actual.to_matrix(), expected, rtol=0.0, atol=1e-12)


class _AxisMixer(Ansatz):
    def __init__(self, axis: str) -> None:
        self.axis = axis

    def validate(self) -> None:
        if self.axis not in {"x", "z"}:
            raise ValueError("unsupported axis")

    def build_circuit(self, prefix: str) -> Circuit:
        beta = Parameter(f"{prefix}_beta")
        circuit = Circuit(1)
        if self.axis == "x":
            circuit.rx(0, 2.0 * beta)
        else:
            circuit.rz(0, 2.0 * beta)
        return circuit

    def num_parameters(self) -> int:
        return 1

    def num_qubits(self) -> int:
        return 1


@pytest.mark.reference
def test_partitioned_mixer_preserves_application_order_and_shared_beta() -> None:
    parts = [
        MixerApplication(_AxisMixer("x"), (0,)),
        MixerApplication(_AxisMixer("z"), (0,)),
    ]
    mixer = PartitionedMixer(1, parts)
    parts.reverse()
    beta = 0.27
    actual = mixer.build_circuit("p").assign_parameters({"p_beta": beta})
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    z = np.diag([1.0, -1.0]).astype(np.complex128)
    expected = _unitary(z, beta) @ _unitary(x, beta)
    np.testing.assert_allclose(actual.to_matrix(), expected, rtol=0.0, atol=1e-12)


@pytest.mark.reference
def test_partitioned_mixer_maps_nontrivial_local_qubits_and_snapshots_lists() -> None:
    first_qubits = [2, 0]
    first = MixerApplication(
        ControlledBitFlipMixer(1, control_state=0),
        first_qubits,  # type: ignore[arg-type]
    )
    first_qubits[:] = [0, 2]
    second = MixerApplication(ControlledBitFlipMixer(1, control_state=1), (0, 1))
    parts = [first, second]
    mixer = PartitionedMixer(3, parts)
    parts.reverse()
    assert first.qubits == (2, 0)

    beta = 0.22
    actual = mixer.build_circuit("p").assign_parameters({"p_beta": beta}).to_matrix()
    first_generator = _mapped_controlled_x_generator(3, 2, 0, 0)
    second_generator = _mapped_controlled_x_generator(3, 0, 1, 1)
    expected = _unitary(second_generator, beta) @ _unitary(first_generator, beta)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ControlledBitFlipMixer(-1),
        lambda: ControlledBitFlipMixer(2, control_state=4),
        lambda: ControlledNullSwapMixer(1, control_state=True),
        lambda: MixerApplication(_AxisMixer("x"), (0, 0)),
        lambda: PartitionedMixer(1, []),
        lambda: PartitionedMixer(1, [MixerApplication(_AxisMixer("x"), (1,))]),
    ],
)
def test_controlled_and_partitioned_boundaries(factory: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()  # type: ignore[operator]
