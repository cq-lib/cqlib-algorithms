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

"""Shared one-parameter mixer template."""

from __future__ import annotations

from abc import abstractmethod
from typing import final

from cqlib.circuit import Circuit, Parameter

from ..ansatz.base_ansatz import BaseAnsatz


class _BaseMixer(BaseAnsatz):
    """Implement the invariant one-parameter mixer contract."""

    __slots__ = ("_num_qubits",)

    def __init__(self, num_qubits: int) -> None:
        self._num_qubits = num_qubits

    @staticmethod
    def _beta_parameter(prefix: str) -> Parameter:
        """Return the standard mixer parameter named ``{prefix}_beta``."""
        return Parameter(f"{prefix}_beta")

    @final
    def _build_circuit(self, prefix: str) -> Circuit:
        """Build and verify a fresh circuit with parameter ``{prefix}_beta``."""
        parameter_name = f"{prefix}_beta"
        circuit = self._build_mixer(prefix)
        if not isinstance(circuit, Circuit):
            raise TypeError("mixer implementation must build a cqlib.Circuit")
        if circuit.num_qubits != self._num_qubits:
            raise ValueError("mixer circuit has an inconsistent width")
        if set(circuit.used_symbols) != {parameter_name}:
            raise ValueError("mixer circuit must use exactly its beta parameter")
        return circuit

    def _parameter_count(self) -> int:
        """Return the single logical beta parameter slot."""
        return 1

    def _qubit_count(self) -> int:
        """Return the encoded register width."""
        return self._num_qubits

    @abstractmethod
    def _validate_configuration(self) -> None:
        """Validate subclass-specific frozen configuration."""
        raise NotImplementedError

    @abstractmethod
    def _build_mixer(self, prefix: str) -> Circuit:
        """Build the subclass-specific mixer circuit."""
        raise NotImplementedError
