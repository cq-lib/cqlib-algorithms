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

"""Tests for direct-product quantum image angle encoding."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cqlib_algorithms.operators.state_preparation import ImageAngleEncoding
from cqlib_algorithms.protocols import StatePreparation


def _expected_product_state(
    values: list[float], lower: float, upper: float
) -> np.ndarray:
    state = np.array([1.0 + 0.0j])
    for value in reversed(values):
        theta = math.pi * (value - lower) / (upper - lower)
        vector = np.array(
            [math.cos(theta / 2), math.sin(theta / 2)], dtype=np.complex128
        )
        state = np.kron(state, vector)
    return state


def test_multichannel_row_major_layout_matches_independent_state() -> None:
    channels = [[0.0, 32.0, 96.0, 128.0], [255.0, 192.0, 64.0, 16.0]]
    component = ImageAngleEncoding((2, 2), channels)
    expected = _expected_product_state([*channels[0], *channels[1]], 0.0, 255.0)

    assert isinstance(component, StatePreparation)
    component.validate()
    assert component.num_qubits() == 8
    assert component.num_parameters() == 0
    circuit = component.build_circuit("ignored")
    assert circuit.used_symbols == []
    np.testing.assert_allclose(
        circuit.to_matrix()[:, 0], expected, rtol=0.0, atol=1e-12
    )


def test_custom_range_endpoints_midpoint_and_input_copy() -> None:
    channel = [-2.0, 0.0, 2.0]
    channels = [channel]
    component = ImageAngleEncoding((1, 3), channels, value_range=(-2.0, 2.0))
    channel[:] = [2.0, 2.0, 2.0]
    channels.append([0.0, 0.0, 0.0])

    expected = _expected_product_state([-2.0, 0.0, 2.0], -2.0, 2.0)
    np.testing.assert_allclose(
        component.build_circuit("a").to_matrix()[:, 0],
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_overflowing_finite_range_prepares_endpoints_and_midpoint() -> None:
    component = ImageAngleEncoding(
        (1, 3),
        [[-1e308, 0.0, 1e308]],
        value_range=(-1e308, 1e308),
    )
    one = np.array([0.0, 1.0], dtype=np.complex128)
    plus_y = np.array([math.sqrt(0.5), math.sqrt(0.5)], dtype=np.complex128)
    zero = np.array([1.0, 0.0], dtype=np.complex128)
    expected = np.kron(np.kron(one, plus_y), zero)

    actual = component.build_circuit("stable").to_matrix()[:, 0]
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_subnormal_range_prepares_endpoints_and_midpoint() -> None:
    component = ImageAngleEncoding(
        (1, 3),
        [[0.0, 5e-310, 1e-309]],
        value_range=(0.0, 1e-309),
    )
    one = np.array([0.0, 1.0], dtype=np.complex128)
    plus_y = np.array([math.sqrt(0.5), math.sqrt(0.5)], dtype=np.complex128)
    zero = np.array([1.0, 0.0], dtype=np.complex128)
    expected = np.kron(np.kron(one, plus_y), zero)

    actual = component.build_circuit("stable").to_matrix()[:, 0]
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_repeated_builds_are_independent_and_prefix_is_checked() -> None:
    component = ImageAngleEncoding((1, 2), [[0.0, 255.0]])
    first = component.build_circuit("fixed")
    second = component.build_circuit("fixed")

    assert first == second
    assert first is not second
    first.x(0)
    assert first != second
    with pytest.raises(TypeError, match="prefix must be a str"):
        component.build_circuit(None)  # type: ignore[arg-type]


def test_rejects_channel_integer_too_large_for_float_conversion() -> None:
    with pytest.raises(
        ValueError,
        match="image angle values must be representable as finite floats",
    ) as error_info:
        ImageAngleEncoding((1, 1), [[10**1000]])

    assert isinstance(error_info.value.__cause__, OverflowError)


def test_rejects_range_endpoint_too_large_for_float_conversion() -> None:
    with pytest.raises(
        ValueError,
        match="value_range endpoints must be representable as finite floats",
    ) as error_info:
        ImageAngleEncoding(
            (1, 1),
            [[0.0]],
            value_range=(0.0, 10**1000),
        )

    assert isinstance(error_info.value.__cause__, OverflowError)


@pytest.mark.parametrize(
    ("image_shape", "channels", "value_range", "error"),
    [
        ((0, 1), [[0.0]], (0.0, 1.0), ValueError),
        ((True, 1), [[0.0]], (0.0, 1.0), TypeError),
        ((1, 1), [], (0.0, 1.0), ValueError),
        ((1, 2), [[0.0]], (0.0, 1.0), ValueError),
        ((1, 1), [[True]], (0.0, 1.0), TypeError),
        ((1, 1), [["0"]], (0.0, 1.0), TypeError),
        ((1, 1), [[math.nan]], (0.0, 1.0), ValueError),
        ((1, 1), [[math.inf]], (0.0, 1.0), ValueError),
        ((1, 1), [[2.0]], (0.0, 1.0), ValueError),
        ((1, 1), [[0.0]], (1.0, 1.0), ValueError),
        ((1, 1), [[0.0]], (math.nan, 1.0), ValueError),
    ],
)
def test_rejects_invalid_image_data(
    image_shape: tuple[int, int],
    channels: object,
    value_range: tuple[float, float],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        ImageAngleEncoding(
            image_shape,
            channels,  # type: ignore[arg-type]
            value_range=value_range,
        )
