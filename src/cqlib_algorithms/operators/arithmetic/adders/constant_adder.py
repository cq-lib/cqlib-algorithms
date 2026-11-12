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
"""Draper constant adder circuits.

This module exposes the public constant-adder interface of the arithmetic
package. The wrapper validates the classical value and exposes a fixed local
register layout while delegating the concrete Fourier-basis construction to the
internal implementation module.

References
----------
[1] T. G. Draper, "Addition on a Quantum Computer," 2000.
    arXiv:quant-ph/0008033.
"""

from __future__ import annotations

from cqlib import Circuit

from ._constant_adder import (
    build_draper_constant_adder,
)


class DraperConstantAdder:
    """In-place modular addition by a classical constant."""

    def __init__(
        self,
        num_state_qubits: int,
        value: int,
        *,
        name: str | None = None,
    ) -> None:
        if not isinstance(num_state_qubits, int):
            raise TypeError(
                f"num_state_qubits must be an integer, got {num_state_qubits!r}."
            )
        if num_state_qubits < 1:
            raise ValueError(
                f"num_state_qubits must be >= 1, got {num_state_qubits!r}."
            )
        if not isinstance(value, int):
            raise TypeError(f"value must be an integer, got {value!r}.")
        if name is not None and not isinstance(name, str):
            raise TypeError(f"name must be a string or None, got {name!r}.")

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
        return build_draper_constant_adder(
            self._num_state_qubits,
            self._value,
        )
