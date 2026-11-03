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

"""Reusable quantum phase-estimation circuit construction.

References
----------
[1] R. Cleve, A. Ekert, C. Macchiavello, and M. Mosca, "Quantum algorithms
    revisited," Proceedings of the Royal Society A, 454, 339-354, 1998.
    DOI: 10.1098/rspa.1998.0164.
"""

from __future__ import annotations

from typing import final

from cqlib import Circuit

from ..._utils.phase_estimation import (
    append_controlled_matrix,
    normalize_evaluation_qubits,
    snapshot_static_unitary_circuit,
    unitary_matrix,
    validate_static_unitary_circuit,
)
from ...transforms import QFT


@final
class PhaseEstimationCircuit:
    """Build the standard multi-qubit phase-estimation circuit.

    For an eigenstate ``|psi>`` satisfying
    ``U|psi> = exp(2*pi*i*phi)|psi>``, evaluation qubit ``j`` controls
    ``U**(2**j)``. An inverse QFT then maps ``phi`` to the evaluation
    register. The returned circuit contains no state preparation or
    measurement operations.

    Arbitrary Core circuits cannot currently be promoted directly to a
    quantum-controlled circuit. This component therefore constructs dense
    controlled matrices, preserving the unitary's global phase. The path has
    exponential memory cost in the system width and enforces a pre-allocation
    memory limit.

    Args:
        unitary: Positive-width, fully bound static unitary circuit.
        num_evaluation_qubits: Positive phase-register width.
        approximation_degree: Small-angle inverse-QFT layers to omit.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the circuit, width, approximation, or memory bound is invalid.
    """

    __slots__ = (
        "_approximation_degree",
        "_num_evaluation_qubits",
        "_unitary",
    )

    def __init__(
        self,
        unitary: Circuit,
        num_evaluation_qubits: int,
        *,
        approximation_degree: int = 0,
    ) -> None:
        self._num_evaluation_qubits = normalize_evaluation_qubits(num_evaluation_qubits)
        if isinstance(approximation_degree, bool) or not isinstance(
            approximation_degree, int
        ):
            raise TypeError("approximation_degree must be an integer")
        if not 0 <= approximation_degree < self._num_evaluation_qubits:
            raise ValueError(
                "approximation_degree must satisfy "
                "0 <= approximation_degree < num_evaluation_qubits"
            )
        self._approximation_degree = approximation_degree
        self._unitary = snapshot_static_unitary_circuit(unitary, "unitary")

    @property
    def num_evaluation_qubits(self) -> int:
        """Return the evaluation-register width."""
        return self._num_evaluation_qubits

    @property
    def num_system_qubits(self) -> int:
        """Return the unitary's system-register width."""
        return int(self._unitary.num_qubits)

    @property
    def num_qubits(self) -> int:
        """Return the total circuit width."""
        return self.num_evaluation_qubits + self.num_system_qubits

    @property
    def approximation_degree(self) -> int:
        """Return the inverse-QFT approximation degree."""
        return self._approximation_degree

    def build_circuit(self) -> Circuit:
        """Return a fresh, unmeasured phase-estimation circuit."""
        validate_static_unitary_circuit(self._unitary, "unitary")
        matrix = unitary_matrix(
            self._unitary,
            self._num_evaluation_qubits,
        )
        circuit = Circuit(self.num_qubits)
        evaluation_targets = tuple(circuit.qubits[: self._num_evaluation_qubits])
        system_targets = tuple(circuit.qubits[self._num_evaluation_qubits :])

        for target in evaluation_targets:
            circuit.h(target)

        powered = matrix
        for index, control in enumerate(evaluation_targets):
            append_controlled_matrix(
                circuit,
                control,
                system_targets,
                powered,
                label=f"controlled-unitary-power-{1 << index}",
            )
            if index + 1 < self._num_evaluation_qubits:
                powered = powered @ powered

        circuit.compose(
            QFT(
                self._num_evaluation_qubits,
                approximation_degree=self._approximation_degree,
                inverse=True,
            ).build_circuit(),
            list(evaluation_targets),
        )
        return circuit


__all__ = ["PhaseEstimationCircuit"]
