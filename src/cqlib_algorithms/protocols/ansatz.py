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

"""Abstract base class for variational ansatz circuit components."""

from __future__ import annotations

from abc import ABC, abstractmethod

from cqlib.circuit import Circuit


class Ansatz(ABC):
    """Nominal base class for variational parameterized circuit families.

    Implementations must inherit from this class and satisfy its four abstract
    methods. Library-provided components do so through the shared
    ``BaseAnsatz`` template implementation. Runtime ``isinstance`` checks
    therefore express actual membership in the ansatz hierarchy.

    ``Ansatz`` and
    :class:`cqlib_algorithms.protocols.state_preparation.StatePreparation`
    have the same
    abstract operations. Their distinct base classes prevent one component
    role from being substituted nominally for the other. Parameters here are
    ordered trainable symbols, rather than state-preparation or input-data
    symbols.

    For a valid component, ``build_circuit`` returns a new circuit whose
    ``num_qubits`` equals ``num_qubits()``. ``num_parameters()`` declares the
    component's logical trainable-parameter slots. A valid degenerate
    configuration need not reference every declared slot in its generated IR,
    so ``Circuit.used_symbols`` may contain fewer names. When present, active
    symbol names must be unique and their count must not exceed
    ``num_parameters()``; they do not by themselves identify a symbol's
    semantic role.

    A prefix may be empty.  It only participates in naming symbols newly
    created by the implementation; explicitly configured names may take
    precedence.  Building repeatedly with the same prefix must be
    deterministic and must return independent circuits.
    """

    __slots__ = ()

    @abstractmethod
    def validate(self) -> None:
        """Validate the ansatz configuration.

        Raises:
            ValueError: If the ansatz configuration is invalid.
        """
        raise NotImplementedError

    @abstractmethod
    def build_circuit(self, prefix: str) -> Circuit:
        """Build and return a new variational circuit.

        Args:
            prefix: Naming input for symbols created while building. It may be
                empty and must not otherwise change the circuit construction.

        Returns:
            A new circuit satisfying the component's declared qubit count and
            logical parameter-slot constraints.

        Raises:
            ValueError: If the ansatz configuration is invalid.
        """
        raise NotImplementedError

    @abstractmethod
    def num_parameters(self) -> int:
        """Return the number of logical trainable-parameter slots."""
        raise NotImplementedError

    @abstractmethod
    def num_qubits(self) -> int:
        """Return the number of qubits in the generated circuit."""
        raise NotImplementedError
