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

import numpy as np
from cqlib import Circuit

from ..._utils.phase_estimation import (
    append_controlled_matrix,
    normalize_evaluation_qubits,
    snapshot_static_unitary_circuit,
    unitary_matrix,
    validate_static_unitary_circuit,
)
from .qft import QFT


@final
class PhaseEstimationCircuit:
    """Build the standard multi-qubit phase-estimation circuit.

    For an eigenstate ``|psi>`` satisfying
    ``U|psi> = exp(2*pi*i*phi)|psi>``, evaluation qubit ``j`` controls
    ``U**(2**j)``. An inverse QFT then maps ``phi`` to the evaluation
    register. The returned circuit contains no state preparation or
    measurement operations. The inverse configuration returns the adjoint of
    the complete forward circuit built with the same approximation settings.

    Arbitrary Core circuits cannot currently be promoted directly to a
    quantum-controlled circuit. This component therefore constructs dense
    controlled matrices, preserving the unitary's global phase. The path has
    exponential memory cost in the system width and enforces a pre-allocation
    memory limit.

    Args:
        unitary: Positive-width, fully bound static unitary circuit.
        num_evaluation_qubits: Positive phase-register width.
        approximation_degree: Small-angle inverse-QFT layers to omit.
        inverse: Whether to return the adjoint of the configured forward
            circuit.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the circuit, width, approximation, or memory bound is invalid.
    """

    __slots__ = (
        "_approximation_degree",
        "_inverse",
        "_num_evaluation_qubits",
        "_unitary",
    )

    def __init__(
        self,
        unitary: Circuit,
        num_evaluation_qubits: int,
        *,
        approximation_degree: int = 0,
        inverse: bool = False,
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
        if not isinstance(inverse, bool):
            raise TypeError("inverse must be a boolean")
        self._approximation_degree = approximation_degree
        self._inverse = inverse
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

    @property
    def inverse(self) -> bool:
        """Return whether the adjoint of the forward circuit is built."""
        return self._inverse

    def build_circuit(self) -> Circuit:
        """Return a fresh, unmeasured phase-estimation circuit.

        The adjoint of the forward circuit is returned when ``inverse`` is
        enabled.
        """
        validate_static_unitary_circuit(self._unitary, "unitary")
        matrix = unitary_matrix(
            self._unitary,
            self._num_evaluation_qubits,
        )
        circuit = Circuit(self.num_qubits)
        evaluation_targets = tuple(circuit.qubits[: self._num_evaluation_qubits])
        system_targets = tuple(circuit.qubits[self._num_evaluation_qubits :])

        powers: list[np.ndarray] = []
        powered = matrix
        for index in range(self._num_evaluation_qubits):
            powers.append(powered)
            if index + 1 < self._num_evaluation_qubits:
                powered = powered @ powered

        # The forward block order is Hadamards, controlled powers, inverse QFT.
        # Its adjoint reverses the block order and conjugates every block: the
        # trailing inverse QFT becomes a forward QFT, each dense controlled
        # matrix becomes a conjugate transpose, and the Hadamards stay put
        # because they are self-adjoint. Core's numerical gate cannot be
        # inverted in place, so the reverse order is assembled explicitly.
        if self._inverse:
            controls = tuple(reversed(evaluation_targets))
            matrices = tuple(value.conj().T for value in reversed(powers))
            leading_qft = QFT(
                self._num_evaluation_qubits,
                approximation_degree=self._approximation_degree,
            ).build_circuit()
            circuit.compose(leading_qft, list(evaluation_targets))
        else:
            controls = evaluation_targets
            matrices = tuple(powers)
            for target in evaluation_targets:
                circuit.h(target)

        for control, value in zip(controls, matrices, strict=True):
            exponent = evaluation_targets.index(control)
            append_controlled_matrix(
                circuit,
                control,
                system_targets,
                value,
                label=f"controlled-unitary-power-{1 << exponent}",
            )

        if self._inverse:
            for target in evaluation_targets:
                circuit.h(target)
        else:
            trailing_qft = QFT(
                self._num_evaluation_qubits,
                approximation_degree=self._approximation_degree,
                inverse=True,
            ).build_circuit()
            circuit.compose(trailing_qft, list(evaluation_targets))

        return circuit


__all__ = ["PhaseEstimationCircuit"]
