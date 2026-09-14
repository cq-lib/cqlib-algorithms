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

"""Exact sparse-support amplitude encoding."""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from cqlib.circuit import Circuit

from ...protocols.state_preparation import StatePreparation
from ._state_synthesis import (
    build_sparse_two_level_circuit,
    copy_sparse_entries,
    validate_prefix,
)


@final
class SparseAmplitudeEncoding(StatePreparation):
    """Prepare normalized amplitudes supplied only on an explicit support.

    The implementation uses exact support-set two-level synthesis rather than
    allocating a dense length-``2**num_qubits`` vector. For support size ``s``
    and width ``n``, its classical working storage is ``O(s + n)``. It emits
    ``O(s n)`` multi-controlled operations and ``O(s n**2)`` surrounding
    one-qubit operations in the worst case. Cqlib's current exact ancilla-free
    lowering expands each such multi-controlled gate to ``O(n**2)`` elementary
    operations, giving an ``O(s n**3)`` worst-case elementary-gate upper bound.
    This is not a QRAM or polylogarithmic data-loading claim.

    Indices use Cqlib little-endian basis numbering. Entries may be unordered;
    they are copied, normalized, and stored in ascending-index order. Duplicate
    indices and explicit zero amplitudes are rejected.
    """

    __slots__ = ("_entries", "_num_qubits")

    def __init__(self, num_qubits: int, entries: Sequence[tuple[int, complex]]) -> None:
        self._entries = copy_sparse_entries(num_qubits, entries)
        self._num_qubits = num_qubits

    def validate(self) -> None:
        """Validate the immutable sparse support and amplitudes."""
        copy_sparse_entries(self._num_qubits, self._entries)

    def build_circuit(self, prefix: str) -> Circuit:
        """Return a new exact support-set two-level preparation circuit."""
        validate_prefix(prefix)
        return build_sparse_two_level_circuit(self._num_qubits, self._entries)

    def num_parameters(self) -> int:
        """Return zero because the sparse amplitudes are fixed numeric data."""
        return 0

    def num_qubits(self) -> int:
        """Return the configured register width."""
        return self._num_qubits
