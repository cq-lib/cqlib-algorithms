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

"""Bit-plane representation of quantum images (BRQI).

References
----------
[1] H.-S. Li, X. Chen, H. Xia, Y. Liang, and Z. Zhou, "A quantum image
    representation based on bitplanes," IEEE Access, 6, 62396-62404, 2018.
    DOI: 10.1109/ACCESS.2018.2871691.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from cqlib.circuit import Circuit
from cqlib.circuit.gates import StandardGate

from ...protocols.state_preparation import StatePreparation
from ._image_data import (
    copy_image_channels,
    copy_image_shape,
    validate_image_channels,
    validate_image_shape,
)
from ._state_synthesis import validate_prefix


@final
class ImageBasisEncoding(StatePreparation):
    """Prepare the standard 8-bit grayscale BRQI state.

    The value register is qubit 0. Little-endian column and row registers
    follow, then a three-qubit little-endian bit-plane register. The address
    registers are placed in uniform superposition and the value qubit is
    flipped exactly at addresses whose selected grayscale bit is one.

    The implementation follows the published BRQI definition and synthesizes
    the state by explicitly enumerating every pixel and bit-plane address.

    Args:
        image_shape: Power-of-two ``(height, width)`` dimensions.
        channels: Exactly one row-major channel of 8-bit grayscale integers.

    Raises:
        TypeError: If shapes or pixels have invalid types.
        ValueError: If dimensions, channel count, length, or values are invalid.
    """

    __slots__ = ("_channels", "_image_shape")

    def __init__(
        self,
        image_shape: tuple[int, int],
        channels: Sequence[Sequence[int]],
    ) -> None:
        self._image_shape = copy_image_shape(image_shape)
        copied = copy_image_channels(
            channels,
            pixels_per_channel=self._image_shape[0] * self._image_shape[1],
        )
        self._channels = self._copy_integer_channels(copied)
        self.validate()

    @staticmethod
    def _copy_integer_channels(
        channels: tuple[tuple[int, ...], ...],
    ) -> tuple[tuple[int, ...], ...]:
        ImageBasisEncoding._validate_integer_channels(channels)
        return channels

    @staticmethod
    def _validate_integer_channels(channels: tuple[tuple[int, ...], ...]) -> None:
        for channel in channels:
            for value in channel:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError("image basis values must be integers")

    def validate(self) -> None:
        """Validate the immutable 8-bit grayscale image snapshot."""
        validate_image_shape(self._image_shape)
        height, width = self._image_shape
        validate_image_channels(
            self._channels,
            pixels_per_channel=height * width,
        )
        if len(self._channels) != 1:
            raise ValueError("BRQI requires exactly one grayscale channel")
        if height & (height - 1) or width & (width - 1):
            raise ValueError("image height and width must be powers of two")
        self._validate_integer_channels(self._channels)
        if any(
            value < 0 or value > 255 for channel in self._channels for value in channel
        ):
            raise ValueError("BRQI grayscale values must lie in [0, 255]")

    @staticmethod
    def _append_addressed_value_write(
        circuit: Circuit,
        controls: tuple[int, ...],
        address: int,
    ) -> None:
        """Flip value qubit 0 only at one column/row/plane address."""
        open_controls = tuple(
            qubit
            for position, qubit in enumerate(controls)
            if not (address >> position) & 1
        )
        for qubit in open_controls:
            circuit.x(qubit)
        circuit.append_mc_gate(
            StandardGate.X.control(len(controls)),
            [*controls, 0],
        )
        for qubit in reversed(open_controls):
            circuit.x(qubit)

    def build_circuit(self, prefix: str) -> Circuit:
        """Return a fresh, unmeasured BRQI state-preparation circuit."""
        validate_prefix(prefix)
        self.validate()
        circuit = Circuit(self.num_qubits())
        height, width = self._image_shape
        row_width = height.bit_length() - 1
        column_width = width.bit_length() - 1
        controls = tuple(range(1, self.num_qubits()))

        for qubit in controls:
            circuit.h(qubit)

        pixels = self._channels[0]
        for row in range(height):
            for column in range(width):
                value = pixels[row * width + column]
                for plane in range(8):
                    if (value >> plane) & 1:
                        address = (
                            column
                            | (row << column_width)
                            | (plane << (column_width + row_width))
                        )
                        self._append_addressed_value_write(
                            circuit,
                            controls,
                            address,
                        )
        circuit.validate()
        return circuit

    def num_parameters(self) -> int:
        """Return zero because image values are fixed numerical data."""
        return 0

    def num_qubits(self) -> int:
        """Return value, column, row, and three plane-index qubits."""
        height, width = self._image_shape
        return (height.bit_length() - 1) + (width.bit_length() - 1) + 4


BRQI = ImageBasisEncoding
