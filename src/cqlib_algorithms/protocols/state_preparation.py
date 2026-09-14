"""Abstract base class for state-preparation circuit components."""

from __future__ import annotations

from abc import ABC, abstractmethod

from cqlib.circuit import Circuit


class StatePreparation(ABC):
    """A component that builds a state-preparation or data-encoding circuit.

    Implementations must inherit from this class and implement its four
    abstract methods. Runtime ``isinstance`` checks therefore express actual
    membership in the state-preparation hierarchy.

    ``StatePreparation`` and
    :class:`cqlib_algorithms.protocols.ansatz.Ansatz` have the
    same abstract operations. Their distinct base classes prevent one
    component role from being substituted nominally for the other. Parameters
    here represent state preparation or input-data symbols, rather than
    trainable ansatz symbols.

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

    __slots__ = ()

    @abstractmethod
    def validate(self) -> None:
        """Validate the component configuration.

        Raises:
            ValueError: If the component configuration is invalid.
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def num_parameters(self) -> int:
        """Return the number of logical preparation or feature slots."""
        raise NotImplementedError

    @abstractmethod
    def num_qubits(self) -> int:
        """Return the number of qubits in the generated circuit."""
        raise NotImplementedError
