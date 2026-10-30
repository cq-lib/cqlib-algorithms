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

"""Direct-product quantum image angle encoding."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import final

from cqlib.circuit import Circuit

from ...protocols.state_preparation import StatePreparation
from ._image_data import (
    copy_image_channels,
    copy_image_shape,
    validate_image_channels,
    validate_image_shape,
)
from ._state_synthesis import validate_prefix


@final
class ImageAngleEncoding(StatePreparation):
    """Encode channel-major image scalars as independent ``RY`` rotations.

    Channels are flattened in row-major order. For ``N = height * width``,
    channel ``c`` and pixel ``p`` are mapped to qubit ``c * N + p``. Values
    are linearly mapped from ``value_range`` to the inclusive angle range
    ``[0, pi]``. This direct-product representation uses one qubit and one
    local rotation for each scalar.

    Args:
        image_shape: Positive ``(height, width)`` image dimensions.
        channels: Nonempty channel-major, row-major finite scalar data.
        value_range: Strictly increasing finite input range.

    Raises:
        TypeError: If shapes or values have invalid types.
        ValueError: If dimensions, channel lengths, values, or range are invalid.
    """

    __slots__ = ("_channels", "_image_shape", "_value_range")

    def __init__(
        self,
        image_shape: tuple[int, int],
        channels: Sequence[Sequence[float]],
        *,
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> None:
        self._image_shape = copy_image_shape(image_shape)
        copied = copy_image_channels(
            channels,
            pixels_per_channel=self._image_shape[0] * self._image_shape[1],
        )
        self._channels = self._copy_real_channels(copied)
        self._value_range = self._copy_value_range(value_range)
        self.validate()

    @staticmethod
    def _copy_real_channels(
        channels: tuple[tuple[float, ...], ...],
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(
                ImageAngleEncoding._finite_float(
                    value,
                    name="image angle values",
                )
                for value in channel
            )
            for channel in channels
        )

    @staticmethod
    def _finite_float(value: object, *, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be real numbers")
        try:
            number = float(value)
        except OverflowError as error:
            raise ValueError(
                f"{name} must be representable as finite floats"
            ) from error
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        return number

    @staticmethod
    def _validate_real_channels(channels: tuple[tuple[float, ...], ...]) -> None:
        for channel in channels:
            for value in channel:
                ImageAngleEncoding._finite_float(
                    value,
                    name="image angle values",
                )

    @staticmethod
    def _copy_value_range(value_range: tuple[float, float]) -> tuple[float, float]:
        ImageAngleEncoding._validate_value_range_container(value_range)
        lower = ImageAngleEncoding._finite_float(
            value_range[0],
            name="value_range endpoints",
        )
        upper = ImageAngleEncoding._finite_float(
            value_range[1],
            name="value_range endpoints",
        )
        ImageAngleEncoding._validate_range_order(lower, upper)
        return (lower, upper)

    @staticmethod
    def _validate_value_range_container(value_range: tuple[float, float]) -> None:
        if not isinstance(value_range, tuple):
            raise TypeError("value_range must be a tuple of two real numbers")
        if len(value_range) != 2:
            raise ValueError("value_range must contain exactly two endpoints")

    @staticmethod
    def _validate_range_order(lower: float, upper: float) -> None:
        if lower >= upper:
            raise ValueError(
                "value_range lower endpoint must be less than upper endpoint"
            )

    @staticmethod
    def _validate_value_range(value_range: tuple[float, float]) -> None:
        ImageAngleEncoding._validate_value_range_container(value_range)
        lower = ImageAngleEncoding._finite_float(
            value_range[0],
            name="value_range endpoints",
        )
        upper = ImageAngleEncoding._finite_float(
            value_range[1],
            name="value_range endpoints",
        )
        ImageAngleEncoding._validate_range_order(lower, upper)

    def validate(self) -> None:
        """Validate the immutable image snapshot and scaling range."""
        validate_image_shape(self._image_shape)
        height, width = self._image_shape
        validate_image_channels(
            self._channels,
            pixels_per_channel=height * width,
        )
        self._validate_real_channels(self._channels)
        self._validate_value_range(self._value_range)
        lower, upper = self._value_range
        if any(
            value < lower or value > upper
            for channel in self._channels
            for value in channel
        ):
            raise ValueError("image angle values must lie within value_range")

    @staticmethod
    def _normalized_angle(value: float, lower: float, upper: float) -> float:
        if value == lower:
            return 0.0
        if value == upper:
            return math.pi

        span = upper - lower
        if math.isfinite(span):
            return math.pi * ((value - lower) / span)

        magnitude = max(abs(lower), abs(upper))
        scaled_lower = lower / magnitude
        scaled_upper = upper / magnitude
        scaled_value = value / magnitude
        angle = math.pi * (scaled_value - scaled_lower) / (scaled_upper - scaled_lower)
        if not math.isfinite(angle):
            raise ValueError("normalized image angles must be finite")
        return angle

    def build_circuit(self, prefix: str) -> Circuit:
        """Return a fresh fixed ``RY`` product-state circuit."""
        validate_prefix(prefix)
        self.validate()
        circuit = Circuit(self.num_qubits())
        lower, upper = self._value_range
        qubit = 0
        for channel in self._channels:
            for value in channel:
                circuit.ry(qubit, self._normalized_angle(value, lower, upper))
                qubit += 1
        circuit.validate()
        return circuit

    def num_parameters(self) -> int:
        """Return zero because image values are fixed numerical data."""
        return 0

    def num_qubits(self) -> int:
        """Return one qubit per channel and pixel scalar."""
        return len(self._channels) * self._image_shape[0] * self._image_shape[1]
