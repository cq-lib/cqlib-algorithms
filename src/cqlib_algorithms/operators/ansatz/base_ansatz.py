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

"""Shared template implementation for variational ansatz components."""

from __future__ import annotations

from typing import final

from cqlib.circuit import Circuit
from cqlib.circuit.ansatz import PauliEvolutionAnsatz, TwoLocal

from ...protocols.ansatz import Ansatz
from ._validation import validate_prefix

_InnerAnsatz = Ansatz | TwoLocal | PauliEvolutionAnsatz


class BaseAnsatz(Ansatz):
    """Validate and expose a supported Cqlib ansatz through the public protocol.

    Args:
        inner: Supported Cqlib ansatz instance to snapshot and delegate.

    Raises:
        TypeError: If ``inner`` is not a supported ansatz.
        ValueError: If the delegated component is invalid.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: _InnerAnsatz) -> None:
        self._inner = inner

    @final
    def validate(self) -> None:
        """Validate the component configuration."""
        self._validate_configuration()

    @final
    def build_circuit(self, prefix: str) -> Circuit:
        """Build and validate an independent circuit."""
        validate_prefix(prefix)
        self.validate()
        circuit = self._build_circuit(prefix)
        circuit.validate()
        return circuit

    @final
    def num_parameters(self) -> int:
        """Return the logical parameter-slot count."""
        return self._parameter_count()

    @final
    def num_qubits(self) -> int:
        """Return the register width."""
        return self._qubit_count()

    def _validate_configuration(self) -> None:
        self._inner.validate()

    def _build_circuit(self, prefix: str) -> Circuit:
        return self._inner.build_circuit(prefix)

    def _parameter_count(self) -> int:
        return self._inner.num_parameters()

    def _qubit_count(self) -> int:
        return self._inner.num_qubits()
