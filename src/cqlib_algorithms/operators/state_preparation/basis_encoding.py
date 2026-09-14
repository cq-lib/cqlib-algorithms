"""Computational-basis data encoding."""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from cqlib.circuit import Circuit
from cqlib.circuit.ansatz import BasisEncoding as CoreBasisEncoding

from ._state_synthesis import validate_prefix


@final
class BasisEncoding:
    """Encode Boolean data as a Cqlib computational-basis state.

    ``bits[q]`` is assigned to qubit ``q``; consequently ``bits[0]`` is the
    least-significant basis-index bit. Circuit construction is delegated to
    :class:`cqlib.circuit.ansatz.BasisEncoding`.
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


__all__ = ["BasisEncoding"]
