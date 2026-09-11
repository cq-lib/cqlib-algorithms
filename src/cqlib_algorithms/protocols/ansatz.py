"""Structural protocol for variational ansatz circuit components."""

from __future__ import annotations

from typing import Protocol

from cqlib.circuit import Circuit


class Ansatz(Protocol):
    """A component that builds a variational parameterized circuit family.

    Implementations are matched structurally and do not need to inherit from
    this protocol.  The protocol deliberately is not runtime-checkable:
    runtime member checks cannot verify signatures or semantic invariants.

    ``Ansatz`` and
    :class:`cqlib_algorithms.protocols.state_preparation.StatePreparation`
    have the same
    structural members.  Static structural typing therefore cannot stop one
    from being passed in place of the other.  Their distinction is a semantic
    contract at the consuming API: parameters here are ordered trainable
    symbols, rather than state-preparation or input-data symbols.

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

    def validate(self) -> None:
        """Validate the ansatz configuration.

        Raises:
            ValueError: If the ansatz configuration is invalid.
        """
        ...

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
        ...

    def num_parameters(self) -> int:
        """Return the number of logical trainable-parameter slots."""
        ...

    def num_qubits(self) -> int:
        """Return the number of qubits in the generated circuit."""
        ...


__all__ = ["Ansatz"]
