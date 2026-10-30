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

"""Exact four-qubit value-selective permutation swap partial mixer.

References
----------
[1] S. Hadfield, Z. Wang, B. O'Gorman, E. G. Rieffel, D. Venturelli, and
    R. Biswas, "From the Quantum Approximate Optimization Algorithm to a
    Quantum Alternating Operator Ansatz," Algorithms, 12(2), 34, 2019.
    DOI: 10.3390/a12020034.
"""

from __future__ import annotations

from typing import final

from cqlib.circuit import Circuit

from ._base import _BaseMixer
from ._exchange import transition_circuit


@final
class ValueSelectivePermutationSwapMixer(_BaseMixer):
    """Mix ``|1100>`` and ``|0011>`` for two values at two positions.

    Local qubits are ordered as ``(u_i, v_j, u_j, v_i)``. The generator is
    the Hermitian four-body transition between the two direct one-hot
    assignments. Its eight Pauli terms commute, so the circuit is exact.

    This component has no construction arguments and exposes one beta parameter.
    """

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(4)
        self.validate()

    def _validate_configuration(self) -> None:
        return None

    def _build_mixer(self, prefix: str) -> Circuit:
        return transition_circuit(prefix)
