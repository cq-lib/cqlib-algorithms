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
"""Structural protocol for state-preparation circuit components."""

from __future__ import annotations

from typing import Protocol

from cqlib.circuit import Circuit


class StatePreparation(Protocol):
    """A component that builds a state-preparation or data-encoding circuit.

    Implementations are matched structurally and do not need to inherit from
    this protocol.  The protocol deliberately is not runtime-checkable:
    runtime member checks cannot verify signatures or semantic invariants.

    ``StatePreparation`` and
    :class:`cqlib_algorithms.protocols.ansatz.Ansatz` have the
    same structural members.  Static structural typing therefore cannot stop
    one from being passed in place of the other.  Their distinction is a
    semantic contract at the consuming API: parameters here represent state
    preparation or input-data symbols, rather than trainable ansatz symbols.

    For a valid component, ``build_circuit`` returns a new circuit whose
    ``num_qubits`` equals ``num_qubits()``. ``num_parameters()`` declares the
    component's logical input-feature or preparation-parameter slots. A valid
    degenerate configuration need not reference every declared slot in its
    generated IR, so ``Circuit.used_symbols`` may contain fewer names. When
    present, active symbol names must be unique and their count must not exceed
    ``num_parameters()``; they do not by themselves identify a symbol's
    semantic role.

    A prefix may be empty.  It only participates in naming symbols newly
    created by the implementation; explicitly configured names may take
    precedence.  Building repeatedly with the same prefix must be
    deterministic and must return independent circuits.
    """

    def validate(self) -> None:
        """Validate the component configuration.

        Raises:
            ValueError: If the component configuration is invalid.
        """
        ...

    def build_circuit(self, prefix: str) -> Circuit:
        """Build and return a new state-preparation circuit.

        Args:
            prefix: Naming input for symbols created while building. It may be
                empty and must not otherwise change the circuit construction.

        Returns:
            A new circuit satisfying the component's declared qubit count and
            logical parameter-slot constraints.

        Raises:
            ValueError: If the component configuration is invalid.
        """
        ...

    def num_parameters(self) -> int:
        """Return the number of logical preparation or feature slots."""
        ...

    def num_qubits(self) -> int:
        """Return the number of qubits in the generated circuit."""
        ...



