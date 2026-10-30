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

"""Random-axis Pauli two-design ansatz.

References
----------
[1] J. R. McClean et al., "Barren plateaus in quantum neural network
    training landscapes," Nature Communications, 9, 4812, 2018.
    DOI: 10.1038/s41467-018-07090-4.
"""

from __future__ import annotations

import math
import random
from typing import final

from cqlib.circuit import Circuit, Parameter
from cqlib.circuit.gates import StandardGate

from ._validation import (
    validate_positive_qubit_count,
    validate_repetitions,
)
from .base_ansatz import BaseAnsatz

_PAULI_ROTATIONS = (StandardGate.RX, StandardGate.RY, StandardGate.RZ)


@final
class PauliTwoDesignAnsatz(BaseAnsatz):
    """Build a random-axis Pauli variational circuit motivated by two-designs.

    Construction begins with ``RY(pi / 4)`` on every qubit. It then applies
    ``reps + 1`` parameterized rotation layers whose independently sampled
    axes are uniformly selected from RX, RY, and RZ. Each of the first
    ``reps`` rotation layers is followed by an even/odd nearest-neighbor CZ
    brickwork layer.

    Axes are sampled once during construction and are immutable thereafter.
    Equal non-``None`` seeds produce equal axes across instances. The exact
    seeded sequence is local to this implementation and is not promised to
    match another framework's random-number generator.

    Args:
        num_qubits: Positive circuit width.
        reps: Nonnegative brickwork repetition count.
        seed: Optional seed for deterministic axis sampling.

    Raises:
        TypeError: If width, repetitions, or seed has the wrong type.
        ValueError: If width or repetitions are invalid.
    """

    __slots__ = ("_num_qubits", "_reps", "_rotation_axes")

    def __init__(
        self,
        num_qubits: int,
        *,
        reps: int = 3,
        seed: int | None = None,
    ) -> None:
        validate_positive_qubit_count(num_qubits)
        validate_repetitions(reps)
        self._num_qubits = num_qubits
        self._reps = reps
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise TypeError("seed must be an integer or None")
            if seed < 0:
                raise ValueError("seed must be non-negative")

        generator = random.Random(seed)
        self._rotation_axes = tuple(
            generator.choice(_PAULI_ROTATIONS)
            for _ in range(self._num_qubits * (self._reps + 1))
        )
        self.validate()

    def _validate_configuration(self) -> None:
        """Validate the frozen Pauli two-design configuration."""
        expected = self._num_qubits * (self._reps + 1)
        if len(self._rotation_axes) != expected:
            raise ValueError("rotation-axis snapshot has an invalid length")
        if any(gate not in _PAULI_ROTATIONS for gate in self._rotation_axes):
            raise ValueError("rotation-axis snapshot contains an invalid gate")

    def _build_circuit(self, prefix: str) -> Circuit:
        """Return a new circuit with parameters ordered by layer and qubit."""
        circuit = Circuit(self._num_qubits)
        for qubit in range(self._num_qubits):
            circuit.ry(qubit, math.pi / 4)

        parameter_index = 0
        for layer in range(self._reps + 1):
            for qubit in range(self._num_qubits):
                gate = self._rotation_axes[parameter_index]
                parameter = Parameter(f"{prefix}_{parameter_index}")
                if gate == StandardGate.RX:
                    circuit.rx(qubit, parameter)
                elif gate == StandardGate.RY:
                    circuit.ry(qubit, parameter)
                else:
                    circuit.rz(qubit, parameter)
                parameter_index += 1

            if layer < self._reps:
                for control in range(0, self._num_qubits - 1, 2):
                    circuit.cz(control, control + 1)
                for control in range(1, self._num_qubits - 1, 2):
                    circuit.cz(control, control + 1)

        return circuit

    def _parameter_count(self) -> int:
        """Return one parameter per qubit in every rotation layer."""
        return self._num_qubits * (self._reps + 1)

    def _qubit_count(self) -> int:
        """Return the configured register width."""
        return self._num_qubits
