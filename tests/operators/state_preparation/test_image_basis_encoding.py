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

"""Independent state tests for 8-bit grayscale BRQI encoding."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cqlib_algorithms import operators
from cqlib_algorithms.operators.state_preparation import BRQI, ImageBasisEncoding
from cqlib_algorithms.protocols import StatePreparation


def _expected_brqi_state(image_shape: tuple[int, int], pixels: list[int]) -> np.ndarray:
    height, width = image_shape
    row_width = height.bit_length() - 1
    column_width = width.bit_length() - 1
    num_qubits = row_width + column_width + 4
    expected = np.zeros(1 << num_qubits, dtype=np.complex128)
    amplitude = 1.0 / math.sqrt(8 * height * width)
    for row in range(height):
        for column in range(width):
            value = pixels[row * width + column]
            for plane in range(8):
                bit = (value >> plane) & 1
                index = bit + 2 * column + 2 * width * row + 2 * width * height * plane
                expected[index] = amplitude
    return expected


@pytest.mark.parametrize(
    ("image_shape", "pixels"),
    [
        ((1, 1), [0b10000101]),
        ((1, 2), [0b00000001, 0b10010000]),
        ((2, 2), [0b00000001, 0b00000110, 0b01010000, 0b10001000]),
    ],
)
def test_complete_brqi_state_matches_independent_reference(
    image_shape: tuple[int, int], pixels: list[int]
) -> None:
    component = ImageBasisEncoding(image_shape, [pixels])
    actual = component.build_circuit("ignored").to_matrix()[:, 0]
    expected = _expected_brqi_state(image_shape, pixels)

    assert isinstance(component, StatePreparation)
    component.validate()
    assert component.num_parameters() == 0
    assert (
        component.num_qubits()
        == int(math.log2(image_shape[0])) + int(math.log2(image_shape[1])) + 4
    )
    assert np.linalg.norm(actual) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_little_endian_value_column_row_plane_register_order() -> None:
    pixels = [1, 2, 4, 8]
    actual = ImageBasisEncoding((2, 2), [pixels]).build_circuit("x").to_matrix()[:, 0]
    support = set(np.flatnonzero(np.abs(actual) > 1e-12))
    expected_support = {
        ((pixels[row * 2 + column] >> plane) & 1) + 2 * column + 4 * row + 8 * plane
        for row in range(2)
        for column in range(2)
        for plane in range(8)
    }
    assert support == expected_support


def test_high_level_schedule_has_uniform_address_and_addressed_writes() -> None:
    pixels = [1, 3, 0, 255]
    component = ImageBasisEncoding((2, 2), [pixels])
    circuit = component.build_circuit("image")
    operations = list(circuit.operations)
    hadamards = [operation for operation in operations if operation.name == "H"]
    controlled_writes = [
        operation
        for operation in operations
        if len(operation.qubits) == component.num_qubits()
    ]

    assert len(hadamards) == 5
    assert [operation.qubits[0].index for operation in hadamards] == [1, 2, 3, 4, 5]
    assert len(controlled_writes) == sum(value.bit_count() for value in pixels)
    assert all(operation.qubits[-1].index == 0 for operation in controlled_writes)
    assert circuit.used_symbols == []
    assert all(operation.name not in {"Measure", "Reset"} for operation in operations)


def test_copies_input_and_repeated_builds_are_independent() -> None:
    pixels = [1, 128]
    channels = [pixels]
    component = ImageBasisEncoding((1, 2), channels)
    expected = _expected_brqi_state((1, 2), [1, 128])
    pixels[:] = [0, 0]
    channels.append([255, 255])

    first = component.build_circuit("fixed")
    second = component.build_circuit("fixed")
    np.testing.assert_allclose(first.to_matrix()[:, 0], expected, rtol=0.0, atol=1e-12)
    assert first == second
    assert first is not second
    first.x(0)
    assert first != second


def test_brqi_is_exact_alias_in_feature_and_aggregate_packages() -> None:
    assert BRQI is ImageBasisEncoding
    assert operators.BRQI is ImageBasisEncoding


def test_prefix_is_checked_and_bit_depth_is_not_configurable() -> None:
    component = ImageBasisEncoding((1, 1), [[0]])
    with pytest.raises(TypeError, match="prefix must be a str"):
        component.build_circuit(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bit_depth"):
        ImageBasisEncoding((1, 1), [[0]], bit_depth=8)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("image_shape", "channels", "error", "message"),
    [
        ((0, 1), [[0]], ValueError, "greater than zero"),
        ((1, 0), [[0]], ValueError, "greater than zero"),
        ((True, 1), [[0]], TypeError, "integers"),
        ((3, 1), [[0, 0, 0]], ValueError, "powers of two"),
        ((1, 3), [[0, 0, 0]], ValueError, "powers of two"),
        ((1, 1), [], ValueError, "at least one channel"),
        ((1, 1), [[0], [1]], ValueError, "exactly one"),
        ((1, 2), [[0]], ValueError, "channel length"),
        ((1, 1), [[True]], TypeError, "integers"),
        ((1, 1), [[1.0]], TypeError, "integers"),
        ((1, 1), [[-1]], ValueError, "\\[0, 255\\]"),
        ((1, 1), [[256]], ValueError, "\\[0, 255\\]"),
    ],
)
def test_rejects_invalid_shape_channel_count_length_and_values(
    image_shape: tuple[int, int],
    channels: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        ImageBasisEncoding(image_shape, channels)  # type: ignore[arg-type]


@pytest.mark.parametrize("channels", [None, "0", b"0", ["0"]])
def test_rejects_invalid_channel_containers(channels: object) -> None:
    with pytest.raises(TypeError, match="sequence"):
        ImageBasisEncoding((1, 1), channels)  # type: ignore[arg-type]
