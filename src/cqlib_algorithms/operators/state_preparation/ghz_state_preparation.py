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

"""Greenberger-Horne-Zeilinger state preparation.

References
----------
[1] D. M. Greenberger, M. A. Horne, A. Shimony, and A. Zeilinger, "Bell's
    theorem without inequalities," American Journal of Physics, 58,
    1131-1143, 1990.
    DOI: 10.1119/1.16243.
"""

from __future__ import annotations

from typing import final

from cqlib.circuit import Circuit

from ...protocols.state_preparation import StatePreparation
from ._state_synthesis import validate_positive_qubit_count, validate_prefix


@final
class GHZStatePreparation(StatePreparation):
    r"""Prepare :math:`(|0\ldots0\rangle+|1\ldots1\rangle)/\sqrt{2}`."""

    __slots__ = ("_num_qubits",)

    def __init__(self, num_qubits: int) -> None:
        validate_positive_qubit_count(num_qubits)
        self._num_qubits = num_qubits

    def validate(self) -> None:
        """Validate the fixed component configuration."""
        validate_positive_qubit_count(self._num_qubits)

    def build_circuit(self, prefix: str) -> Circuit:
        """Return a new H-plus-nearest-neighbor-CX chain."""
        validate_prefix(prefix)
        circuit = Circuit(self._num_qubits)
        circuit.h(0)
        for target in range(1, self._num_qubits):
            circuit.cx(target - 1, target)
        return circuit

    def num_parameters(self) -> int:
        """Return zero because the GHZ state is fixed."""
        return 0

    def num_qubits(self) -> int:
        """Return the configured register width."""
        return self._num_qubits
