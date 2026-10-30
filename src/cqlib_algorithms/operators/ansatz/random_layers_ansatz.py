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

"""Frozen random-layer variational ansatz."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import final

from cqlib.circuit import Circuit, Parameter
from cqlib.circuit.gates import StandardGate

from ._validation import (
    snapshot_rotation_gates,
    validate_entanglement_gate,
    validate_positive_qubit_count,
    validate_repetitions,
)
from .base_ansatz import BaseAnsatz

_FrozenOperation = tuple[StandardGate, tuple[int, ...]]


@final
class RandomLayersAnsatz(BaseAnsatz):
    """Build random rotations interleaved with random two-qubit gates.

    Every layer contains exactly ``rotations_per_layer`` parameterized
    rotations. Before each rotation, at most one entangling gate is sampled
    according to ``entanglement_ratio``. The complete architecture is sampled
    once at construction and then frozen, so repeated circuit builds and
    parameter evaluations are deterministic for one component instance. A
    layer therefore contains at most twice as many gates as rotation slots.

    Args:
        num_qubits: Positive circuit width.
        reps: Positive number of random layers.
        rotations_per_layer: Rotation slots per layer, or ``None`` for width.
        entanglement_ratio: Probability of adding an entangler before a slot.
        rotation_gates: Nonempty supported rotation-gate sequence.
        entanglement_gate: Supported two-qubit entangling gate.
        seed: Optional random seed or ``None`` for nondeterministic sampling.

    Raises:
        TypeError: If an option has the wrong type.
        ValueError: If counts, probability, or gate choices are invalid.
    """

    __slots__ = (
        "_architecture",
        "_entanglement_gate",
        "_entanglement_ratio",
        "_num_qubits",
        "_reps",
        "_rotation_gates",
        "_rotations_per_layer",
    )

    def __init__(
        self,
        num_qubits: int,
        *,
        reps: int = 1,
        rotations_per_layer: int | None = None,
        entanglement_ratio: float = 0.3,
        rotation_gates: Sequence[StandardGate] = (
            StandardGate.RX,
            StandardGate.RY,
            StandardGate.RZ,
        ),
        entanglement_gate: StandardGate = StandardGate.CX,
        seed: int | None = 42,
    ) -> None:
        validate_positive_qubit_count(num_qubits)
        validate_repetitions(reps)
        if rotations_per_layer is None:
            resolved_rotations = num_qubits
        else:
            if isinstance(rotations_per_layer, bool) or not isinstance(
                rotations_per_layer, int
            ):
                raise TypeError("rotations_per_layer must be an integer or None")
            if rotations_per_layer <= 0:
                raise ValueError("rotations_per_layer must be greater than zero")
            resolved_rotations = rotations_per_layer
        if isinstance(entanglement_ratio, bool) or not isinstance(
            entanglement_ratio, (int, float)
        ):
            raise TypeError("entanglement_ratio must be a real number")
        ratio = float(entanglement_ratio)
        if not math.isfinite(ratio):
            raise ValueError("entanglement_ratio must be finite")
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("entanglement_ratio must satisfy 0 <= value <= 1")
        rotations = snapshot_rotation_gates(rotation_gates)
        if not rotations:
            raise ValueError("rotation_gates must not be empty")
        validate_entanglement_gate(entanglement_gate)
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise TypeError("seed must be an integer or None")
            if seed < 0:
                raise ValueError("seed must be non-negative")

        self._num_qubits = num_qubits
        self._reps = reps
        self._rotations_per_layer = resolved_rotations
        self._entanglement_ratio = ratio
        self._rotation_gates = rotations
        self._entanglement_gate = entanglement_gate
        self._architecture = self._sample_architecture(random.Random(seed))
        self.validate()

    def _sample_architecture(
        self,
        generator: random.Random,
    ) -> tuple[_FrozenOperation, ...]:
        operations: list[_FrozenOperation] = []
        for _ in range(self._reps):
            for _ in range(self._rotations_per_layer):
                if (
                    self._num_qubits > 1
                    and generator.random() < self._entanglement_ratio
                ):
                    left = generator.randrange(self._num_qubits)
                    right = generator.randrange(self._num_qubits - 1)
                    if right >= left:
                        right += 1
                    operations.append((self._entanglement_gate, (left, right)))
                gate = generator.choice(self._rotation_gates)
                target = generator.randrange(self._num_qubits)
                operations.append((gate, (target,)))
        return tuple(operations)

    def _validate_configuration(self) -> None:
        """Validate the frozen random architecture."""
        validate_positive_qubit_count(self._num_qubits)
        validate_repetitions(self._reps)
        if self._rotations_per_layer <= 0:
            raise ValueError("rotations_per_layer must be greater than zero")
        if not 0.0 <= self._entanglement_ratio <= 1.0:
            raise ValueError("entanglement_ratio must satisfy 0 <= value <= 1")

    def _build_circuit(self, prefix: str) -> Circuit:
        """Build a new circuit from the frozen random architecture."""
        circuit = Circuit(self._num_qubits)
        parameter_index = 0
        for gate, qubits in self._architecture:
            if len(qubits) == 1:
                circuit.append_gate(
                    gate(Parameter(f"{prefix}_{parameter_index}")),
                    list(qubits),
                )
                parameter_index += 1
            else:
                circuit.append_gate(gate, list(qubits))
        return circuit

    def _parameter_count(self) -> int:
        """Return the fixed number of sampled rotation slots."""
        return self._reps * self._rotations_per_layer

    def _qubit_count(self) -> int:
        """Return the configured register width."""
        return self._num_qubits
