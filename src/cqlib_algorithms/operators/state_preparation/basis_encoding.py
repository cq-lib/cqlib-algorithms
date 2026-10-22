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

"""Computational-basis data encoding."""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from cqlib.circuit import Circuit
from cqlib.circuit.ansatz import BasisEncoding as CoreBasisEncoding

from ...protocols.state_preparation import StatePreparation
from ._state_synthesis import validate_prefix


@final
class BasisEncoding(StatePreparation):
    """Encode Boolean data as a Cqlib computational-basis state.

    ``bits[q]`` is assigned to qubit ``q``; consequently ``bits[0]`` is the
    least-significant basis-index bit. Circuit construction is delegated to
    :class:`cqlib.circuit.ansatz.BasisEncoding`.

    Args:
        bits: Nonempty Boolean sequence in qubit order.

    Raises:
        TypeError: If ``bits`` is not a Boolean sequence.
        ValueError: If ``bits`` is empty.
    """

    __slots__ = ("_bits", "_inner")

    def __init__(self, bits: Sequence[bool]) -> None:
        if isinstance(bits, (str, bytes)) or not isinstance(bits, Sequence):
            raise TypeError("bits must be a sequence of bool values")
        copied = tuple(bits)
        if not copied:
            raise ValueError("bits must not be empty")
        if any(type(bit) is not bool for bit in copied):
            raise TypeError("every item in bits must be a bool")
        self._bits = copied
        self._inner = CoreBasisEncoding(list(copied))
        self.validate()

    def validate(self) -> None:
        """Validate the copied bit string and delegated Core component."""
        self._inner.validate()

    def build_circuit(self, prefix: str) -> Circuit:
        """Return a new basis-encoding circuit."""
        validate_prefix(prefix)
        return self._inner.build_circuit(prefix)

    def num_parameters(self) -> int:
        """Return zero because the prepared state is fixed."""
        return 0

    def num_qubits(self) -> int:
        """Return the number of encoded bits."""
        return len(self._bits)
