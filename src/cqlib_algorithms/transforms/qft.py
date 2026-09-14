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

"""Quantum Fourier transform circuit construction."""

from __future__ import annotations

import math
from dataclasses import dataclass

from cqlib import Circuit
from cqlib.circuit.gates import StandardGate

from ..protocols.ansatz import Ansatz


@dataclass(frozen=True, slots=True, init=False)
class QFT(Ansatz):
    """Configuration for a quantum Fourier transform circuit.

    The forward transform uses the convention

    ``F[y, x] = exp(2π i x y / 2**n) / sqrt(2**n)``,

    where Core qubit ``0`` is the least-significant basis-state bit. Output
    swaps are included by default, so the resulting circuit implements this
    matrix directly in Core's natural order.

    ``approximation_degree`` removes the smallest controlled-phase rotations.
    A controlled phase between qubits separated by ``distance`` is retained
    exactly when ``distance < num_qubits - approximation_degree``. The inverse
    configuration returns the adjoint of the complete forward circuit built
    with the same approximation and swap settings.

    Args:
        num_qubits: Positive number of circuit qubits.
        approximation_degree: Number of longest interaction distances to omit.
            It must satisfy ``0 <= approximation_degree < num_qubits``.
        do_swaps: Whether to reverse the output qubit order with swaps.
        inverse: Whether to return the adjoint of the configured forward QFT.

    Raises:
        TypeError: If integer fields or Boolean flags have the wrong type.
        ValueError: If a numeric field is outside its valid range.
    """

    _num_qubits: int
    approximation_degree: int
    do_swaps: bool
    inverse: bool

    def __init__(
        self,
        num_qubits: int,
        *,
        approximation_degree: int = 0,
        do_swaps: bool = True,
        inverse: bool = False,
    ) -> None:
        object.__setattr__(self, "_num_qubits", num_qubits)
        object.__setattr__(self, "approximation_degree", approximation_degree)
        object.__setattr__(self, "do_swaps", do_swaps)
        object.__setattr__(self, "inverse", inverse)
        self.validate()

    def validate(self) -> None:
        """Validate the QFT configuration."""
        if isinstance(self._num_qubits, bool) or not isinstance(self._num_qubits, int):
            raise TypeError("num_qubits must be an integer")
        if self._num_qubits <= 0:
            raise ValueError("num_qubits must be greater than zero")

        if isinstance(self.approximation_degree, bool) or not isinstance(
            self.approximation_degree, int
        ):
            raise TypeError("approximation_degree must be an integer")
        if not 0 <= self.approximation_degree < self._num_qubits:
            raise ValueError(
                "approximation_degree must satisfy "
                "0 <= approximation_degree < num_qubits"
            )

        if not isinstance(self.do_swaps, bool):
            raise TypeError("do_swaps must be a boolean")
        if not isinstance(self.inverse, bool):
            raise TypeError("inverse must be a boolean")

    def build_circuit(self, prefix: str = "") -> Circuit:
        """Build and return a new circuit for the configured transform."""
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")

        circuit = Circuit(self._num_qubits)
        distance_cutoff = self._num_qubits - self.approximation_degree

        for target in range(self._num_qubits - 1, -1, -1):
            circuit.h(target)
            for control in range(target):
                distance = target - control
                if distance >= distance_cutoff:
                    continue
                angle = math.pi / (2**distance)
                controlled_phase = StandardGate.Phase(angle).control(1)
                circuit.append_mc_gate(controlled_phase, [control, target])

        if self.do_swaps:
            for low in range(self._num_qubits // 2):
                circuit.swap(low, self._num_qubits - 1 - low)

        return circuit.inverse() if self.inverse else circuit

    def num_parameters(self) -> int:
        """Return zero because QFT contains no logical parameter slots."""
        return 0

    def num_qubits(self) -> int:
        """Return the number of qubits in the generated circuit."""
        return self._num_qubits
