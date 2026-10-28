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

"""Finite modal spaces and their one-hot qubit encoding."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _copy_modal_counts(num_modals: Sequence[int]) -> tuple[int, ...]:
    if isinstance(num_modals, (str, bytes)) or not isinstance(num_modals, Sequence):
        raise TypeError("num_modals must be a sequence of integers")
    copied = tuple(num_modals)
    if not copied:
        raise ValueError("num_modals must contain at least one mode")
    for mode, count in enumerate(copied):
        if type(count) is not int:
            raise TypeError(f"num_modals[{mode}] must be an int")
        if count < 1:
            raise ValueError(f"num_modals[{mode}] must be at least 1")
    return copied


def _copy_configuration(
    configuration: Sequence[int],
    *,
    num_modals: Sequence[int],
) -> tuple[int, ...]:
    if isinstance(configuration, (str, bytes)) or not isinstance(
        configuration, Sequence
    ):
        raise TypeError("configuration must be a sequence of integers")
    copied = tuple(configuration)
    if len(copied) != len(num_modals):
        raise ValueError("configuration length must match the number of modes")
    for mode, (modal, count) in enumerate(zip(copied, num_modals)):
        if type(modal) is not int:
            raise TypeError(f"configuration[{mode}] must be an int")
        if modal < 0 or modal >= count:
            raise ValueError(f"configuration[{mode}] is outside its modal range")
    return copied


@dataclass(frozen=True, slots=True)
class ModalSpace:
    """Finite direct product of vibrational modal bases.

    Mode ``0`` is the least-significant mixed-radix subsystem. In the one-hot
    encoding, each mode owns a contiguous qubit block and modal ``k`` is the
    single occupied qubit at offset ``k`` in that block.

    Args:
        num_modals: Positive modal count for each vibrational mode.

    Raises:
        TypeError: If the modal counts are not an integer sequence.
        ValueError: If the sequence is empty or a count is not positive.
    """

    num_modals: Sequence[int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "num_modals", _copy_modal_counts(self.num_modals))

    @property
    def num_modes(self) -> int:
        """Return the number of vibrational modes."""
        return len(self.num_modals)

    @property
    def num_qubits(self) -> int:
        """Return the one-hot register width."""
        return sum(self.num_modals)

    @property
    def dimension(self) -> int:
        """Return the physical modal Hilbert-space dimension."""
        return math.prod(self.num_modals)

    @property
    def qubit_offsets(self) -> tuple[int, ...]:
        """Return the first qubit index of every mode block."""
        offsets: list[int] = []
        offset = 0
        for count in self.num_modals:
            offsets.append(offset)
            offset += count
        return tuple(offsets)

    @property
    def physical_basis_indices(self) -> tuple[int, ...]:
        """Return encoded computational-basis indices in modal basis order."""
        return tuple(
            self.encoded_basis_index(self.configuration_from_basis_index(index))
            for index in range(self.dimension)
        )

    def qubit_index(self, mode: int, modal: int) -> int:
        """Return the qubit assigned to one mode/modal pair."""
        self._validate_mode_modal(mode, modal)
        return self.qubit_offsets[mode] + modal

    def modal_basis_index(self, configuration: Sequence[int]) -> int:
        """Encode a modal configuration as a little-endian mixed-radix index."""
        copied = _copy_configuration(configuration, num_modals=self.num_modals)
        index = 0
        stride = 1
        for modal, count in zip(copied, self.num_modals):
            index += modal * stride
            stride *= count
        return index

    def configuration_from_basis_index(self, index: int) -> tuple[int, ...]:
        """Decode a little-endian mixed-radix modal basis index."""
        if type(index) is not int:
            raise TypeError("index must be an int")
        if index < 0 or index >= self.dimension:
            raise ValueError("index is outside the modal basis")
        remainder = index
        configuration: list[int] = []
        for count in self.num_modals:
            configuration.append(remainder % count)
            remainder //= count
        return tuple(configuration)

    def encoded_basis_index(self, configuration: Sequence[int]) -> int:
        """Return the one-hot computational-basis index of a configuration."""
        copied = _copy_configuration(configuration, num_modals=self.num_modals)
        encoded = 0
        for mode, modal in enumerate(copied):
            encoded |= 1 << self.qubit_index(mode, modal)
        return encoded

    def is_physical_basis_index(self, index: int) -> bool:
        """Return whether a computational basis state is one-hot in every mode."""
        if type(index) is not int:
            raise TypeError("index must be an int")
        if index < 0 or index >= 1 << self.num_qubits:
            raise ValueError("index is outside the one-hot qubit register")
        for offset, count in zip(self.qubit_offsets, self.num_modals):
            block = (index >> offset) & ((1 << count) - 1)
            if block.bit_count() != 1:
                return False
        return True

    def one_hot_isometry(self) -> NDArray[np.complex128]:
        """Return the dense modal-to-qubit isometry for verification."""
        isometry = np.zeros(
            (1 << self.num_qubits, self.dimension),
            dtype=np.complex128,
        )
        for column, row in enumerate(self.physical_basis_indices):
            isometry[row, column] = 1.0
        return isometry

    def _validate_mode_modal(self, mode: int, modal: int) -> None:
        if type(mode) is not int:
            raise TypeError("mode must be an int")
        if type(modal) is not int:
            raise TypeError("modal must be an int")
        if mode < 0 or mode >= self.num_modes:
            raise ValueError("mode is outside the modal space")
        if modal < 0 or modal >= self.num_modals[mode]:
            raise ValueError("modal is outside the selected mode")


__all__ = ["ModalSpace"]
