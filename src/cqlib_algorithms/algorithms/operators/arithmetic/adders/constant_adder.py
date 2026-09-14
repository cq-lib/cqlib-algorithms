"""Draper constant adder circuits.

This module implements the concrete classical constant version of the Draper
adder. A fixed integer value is compiled into phase rotations on a single
Fourier basis register and the result is written back in place under the LSB
first convention used by this package.

References
Thomas G Draper, Addition on a Quantum Computer, 2000,
arXiv quant ph 0008033.
"""

from __future__ import annotations

import math

from cqlib import Circuit

from cqlib_algorithms.transforms import QFT as TransformQFT


class DraperConstantAdder:
    """In-place modular addition by a classical constant."""

    def __init__(
        self,
        num_state_qubits: int,
        value: int,
        *,
        name: str | None = None,
    ) -> None:
        if not isinstance(num_state_qubits, int) or num_state_qubits < 1:
            raise ValueError(
                "num_state_qubits must be an integer >= 1, got "
                f"{num_state_qubits!r}."
            )
        if not isinstance(value, int):
            raise ValueError(
                f"value must be an integer, got {value!r}."
            )

        modulus = 1 << int(num_state_qubits)
        if not 0 <= value < modulus:
            raise ValueError(
                f"value must satisfy 0 <= value < 2**num_state_qubits, got {value!r}."
            )

        self._num_state_qubits = int(num_state_qubits)
        self._value = int(value)
        self._num_qubits = self._num_state_qubits
        self._b_register_indices = list(range(self._num_state_qubits))
        self._name = name or "DraperConstantAdder"

    @property
    def name(self) -> str:
        """Return the configured circuit name."""
        return self._name

    @property
    def num_qubits(self) -> int:
        """Return the total qubit count of the circuit."""
        return self._num_qubits

    @property
    def num_state_qubits(self) -> int:
        """Return the register width n."""
        return self._num_state_qubits

    @property
    def value(self) -> int:
        """Return the compiled classical constant."""
        return self._value

    @property
    def b_register_indices(self) -> list[int]:
        """Return local wire indices of the in-place target register."""
        return list(self._b_register_indices)

    def build_circuit(self) -> Circuit:
        """Build a new circuit implementing |b> -> |b + value mod 2^n>."""
        n = self._num_state_qubits
        circuit = Circuit(self._num_qubits)
        wires = list(self._b_register_indices)

        forward_qft = TransformQFT(
            n,
            approximation_degree=0,
            do_swaps=False,
            inverse=False,
        ).build_circuit()
        circuit.compose(forward_qft, wires)

        for target_bit, qubit in enumerate(wires):
            angle = 0.0
            for const_bit in range(target_bit + 1):
                if (self._value >> const_bit) & 1:
                    angle += (2.0 * math.pi) / (
                        2 ** (target_bit - const_bit + 1)
                    )
            if angle != 0.0:
                circuit.phase(qubit, angle)

        inverse_qft = TransformQFT(
            n,
            approximation_degree=0,
            do_swaps=False,
            inverse=True,
        ).build_circuit()
        circuit.compose(inverse_qft, wires)
        return circuit


__all__ = [
    "DraperConstantAdder",
]
