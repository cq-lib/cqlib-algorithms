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

"""Computational-basis state initialization."""

from __future__ import annotations

from typing import final

from cqlib.circuit import Circuit

from ...protocols.state_preparation import StatePreparation
from ._state_synthesis import validate_positive_qubit_count
from .basis_encoding import BasisEncoding


@final
class BasisStateInitialization(StatePreparation):
    """Prepare one computational basis state from its integer index.

    Bit ``q`` of ``basis_index`` is placed on qubit ``q``. Thus qubit zero is
    the least-significant basis-index bit.

    Args:
        num_qubits: Positive register width.
        basis_index: Integer in ``[0, 2**num_qubits)``.

    Raises:
        TypeError: If an argument is not an integer.
        ValueError: If the width or index is out of range.
    """

    __slots__ = ("_basis_index", "_encoding", "_num_qubits")

    def __init__(self, num_qubits: int, basis_index: int) -> None:
        validate_positive_qubit_count(num_qubits)
        self._num_qubits = num_qubits
        if type(basis_index) is not int:
            raise TypeError("basis_index must be an int")
        if basis_index < 0 or basis_index >= 1 << self._num_qubits:
            raise ValueError("basis_index must fit within num_qubits bits")
        self._basis_index = basis_index
        self._encoding = BasisEncoding(
            tuple(bool((basis_index >> qubit) & 1) for qubit in range(self._num_qubits))
        )

    def validate(self) -> None:
        """Validate the delegated basis encoding."""
        self._encoding.validate()

    def build_circuit(self, prefix: str) -> Circuit:
        """Return a new basis-state initialization circuit."""
        return self._encoding.build_circuit(prefix)

    def num_parameters(self) -> int:
        """Return zero because the prepared state is fixed."""
        return 0

    def num_qubits(self) -> int:
        """Return the configured register width."""
        return self._num_qubits
