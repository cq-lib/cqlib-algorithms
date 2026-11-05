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

"""Circuit construction for the product of one unitary and another's adjoint."""

from __future__ import annotations

from copy import deepcopy

from cqlib import Circuit
from cqlib.circuit import CircuitError

from ..._utils.circuit import (
    validate_static_unitary_structure,
)


class UnitaryOverlap:
    """Build the unmeasured overlap circuit ``U2† U1``.

    Source qubits are associated by insertion position, so concrete qubit
    identifiers need not match. The inputs must have equal positive width and
    contain only statically invertible unitary operations; barriers are
    accepted. Active parameter names must be disjoint between the inputs so
    that the returned circuit never introduces implicit parameter sharing.

    Args:
        unitary1: Unitary circuit applied first, on the ket side.
        unitary2: Unitary circuit whose adjoint is applied second.
        insert_barrier: Whether to separate the two blocks with a barrier.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If circuit widths, symbols, or operation kinds are invalid.
        cqlib.circuit.CircuitError: If direct Core circuit validation fails.
    """

    __slots__ = ("_insert_barrier", "_num_qubits", "_unitary1", "_unitary2_inverse")

    def __init__(
        self,
        unitary1: Circuit,
        unitary2: Circuit,
        *,
        insert_barrier: bool = False,
    ) -> None:
        if not isinstance(unitary1, Circuit):
            raise TypeError("unitary1 must be a cqlib.Circuit")
        if not isinstance(unitary2, Circuit):
            raise TypeError("unitary2 must be a cqlib.Circuit")
        if not isinstance(insert_barrier, bool):
            raise TypeError("insert_barrier must be a boolean")

        unitary1.validate()
        unitary2.validate()
        if unitary1.num_qubits < 1:
            raise ValueError("unitaries must contain at least one qubit")
        if unitary1.num_qubits != unitary2.num_qubits:
            raise ValueError("unitaries must have the same qubit count")

        validate_static_unitary_structure(unitary1, label="unitary1")
        validate_static_unitary_structure(unitary2, label="unitary2")
        shared_symbols = sorted(set(unitary1.used_symbols) & set(unitary2.used_symbols))
        if shared_symbols:
            raise ValueError(
                "unitary parameter names must be disjoint between inputs: "
                f"shared={shared_symbols!r}"
            )

        unitary1_snapshot = deepcopy(unitary1)
        unitary2_snapshot = deepcopy(unitary2)
        try:
            unitary1_snapshot.inverse()
        except CircuitError as exc:
            raise ValueError(
                "unitary1 must contain only statically invertible operations"
            ) from exc
        try:
            unitary2_inverse = unitary2_snapshot.inverse()
        except CircuitError as exc:
            raise ValueError(
                "unitary2 must contain only statically invertible operations"
            ) from exc

        self._unitary1 = unitary1_snapshot
        self._unitary2_inverse = unitary2_inverse
        self._num_qubits = unitary1.num_qubits
        self._insert_barrier = insert_barrier

    @property
    def num_qubits(self) -> int:
        """Return the positive width of the overlap circuit."""
        return self._num_qubits

    @property
    def insert_barrier(self) -> bool:
        """Return whether the two unitary blocks are separated by a barrier."""
        return self._insert_barrier

    def build_circuit(self) -> Circuit:
        """Return a new compact circuit implementing exactly ``U2† U1``."""
        circuit = Circuit(self._num_qubits)
        positions = list(range(self._num_qubits))
        circuit.compose(self._unitary1, positions)
        if self._insert_barrier:
            circuit.barrier(positions)
        circuit.compose(self._unitary2_inverse, positions)
        return circuit
