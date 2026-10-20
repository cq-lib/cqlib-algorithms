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

"""Shared image-layout validation for fixed numerical encodings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

_T = TypeVar("_T")


def copy_image_shape(image_shape: tuple[int, int]) -> tuple[int, int]:
    """Validate and copy a ``(height, width)`` image shape."""
    validate_image_shape(image_shape)
    return (image_shape[0], image_shape[1])


def validate_image_shape(image_shape: tuple[int, int]) -> None:
    """Validate a ``(height, width)`` image shape without modifying it."""
    if not isinstance(image_shape, tuple):
        raise TypeError("image_shape must be a tuple of two integers")
    if len(image_shape) != 2:
        raise ValueError("image_shape must contain exactly height and width")
    height, width = image_shape
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in image_shape
    ):
        raise TypeError("image_shape values must be integers")
    if height <= 0 or width <= 0:
        raise ValueError("image_shape values must be greater than zero")


def copy_image_channels(
    channels: Sequence[Sequence[_T]],
    *,
    pixels_per_channel: int,
) -> tuple[tuple[_T, ...], ...]:
    """Copy channel-major flattened image data into immutable tuples."""
    validate_image_channels(channels, pixels_per_channel=pixels_per_channel)
    return tuple(tuple(channel) for channel in channels)


def validate_image_channels(
    channels: Sequence[Sequence[_T]],
    *,
    pixels_per_channel: int,
) -> None:
    """Validate a channel-major flattened image without modifying it."""
    if isinstance(channels, (str, bytes)) or not isinstance(channels, Sequence):
        raise TypeError("channels must be a sequence of channel sequences")
    if not channels:
        raise ValueError("channels must contain at least one channel")

    for channel in channels:
        if isinstance(channel, (str, bytes)) or not isinstance(channel, Sequence):
            raise TypeError("each channel must be a sequence")
        if len(channel) != pixels_per_channel:
            raise ValueError(
                "each channel length must equal image height times image width"
            )
