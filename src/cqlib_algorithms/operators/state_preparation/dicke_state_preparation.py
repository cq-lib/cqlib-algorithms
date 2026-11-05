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

"""Divide-and-conquer Dicke-state preparation.

This module implements the deterministic divide-and-conquer preparation of
Dicke states introduced by Bärtschi and Eidenbenz. The construction starts
from a unary basis state, recursively applies the Weight Distribution Block to
split the Hamming weight across two subregisters, and finishes each leaf
subproblem with the unary-to-Dicke unitary ``U_{n,k}``.

References
----------
[1] Bärtschi, A., and Eidenbenz, S. (2019). Deterministic Preparation of Dicke States.
    In *Fundamentals of Computation Theory (FCT 2019)*, LNCS 11651, pp. 126-139.
    Springer.
[2] Bärtschi, A., and Eidenbenz, S. (2022). Deterministic Preparation of Dicke States.
    *Quantum Information Processing*, 21, 282.
"""

from __future__ import annotations

from typing import final

from cqlib.circuit import Circuit

from ..._utils.numeric import validate_positive_qubit_count
from ..._utils.parameters import validate_parameter_prefix
from ...protocols.state_preparation import StatePreparation
from ..oracle.dicke_unitary import DickeUnitary
from ..oracle.weight_distribution_block import WeightDistributionBlock
from .basis_state_initialization import BasisStateInitialization


def _validate_weight(weight: int, *, num_qubits: int) -> None:
    """Validate one Dicke Hamming weight."""
    if type(weight) is not int:
        raise TypeError("weight must be an int")
    if weight < 0 or weight > num_qubits:
        raise ValueError("weight must satisfy 0 <= weight <= num_qubits")


def _unary_index(num_qubits: int, ones: int) -> int:
    """Return the basis index of ``|0^(n-ones) 1^ones>``."""
    if ones == 0:
        return 0
    return ((1 << ones) - 1) << (num_qubits - ones)


@final
class DickeStatePreparation(StatePreparation):
    r"""Prepare the Dicke state :math:`|D_k^n\rangle` by divide-and-conquer.

    The component first prepares the unary basis state
    :math:`|0^{n-k}1^k\rangle`. It then recursively applies the Weight
    Distribution Block to split the Hamming weight across two halves until the
    subproblem width is at most ``2 * k``. At that point the implementation
    finishes with the existing unary-to-Dicke unitary on that subregister, as
    described by Bärtschi and Eidenbenz [1]_ [2]_.

    Args:
        num_qubits: Total number of qubits in the prepared state.
        weight: Target Dicke-state Hamming weight.

    Raises:
        TypeError: If either argument is not an integer or is a Boolean.
        ValueError: If ``num_qubits`` is not positive or if ``weight`` falls
            outside ``0 <= weight <= num_qubits``.
    """

    __slots__ = ("_num_qubits", "_weight")

    def __init__(self, num_qubits: int, weight: int) -> None:
        validate_positive_qubit_count(num_qubits)
        _validate_weight(weight, num_qubits=num_qubits)
        self._num_qubits = num_qubits
        self._weight = weight

    def validate(self) -> None:
        """Validate the fixed Dicke-state preparation configuration."""
        validate_positive_qubit_count(self._num_qubits)
        _validate_weight(self._weight, num_qubits=self._num_qubits)

    def build_circuit(self, prefix: str) -> Circuit:
        """Return a new divide-and-conquer Dicke-state preparation circuit."""
        validate_parameter_prefix(prefix)
        circuit = BasisStateInitialization(
            self._num_qubits,
            _unary_index(self._num_qubits, self._weight),
        ).build_circuit(prefix)
        self._append_recursive_preparation(
            circuit,
            positions=list(range(self._num_qubits)),
            max_weight=self._weight,
        )
        return circuit

    def num_parameters(self) -> int:
        """Return zero because the prepared Dicke state is fixed."""
        return 0

    def num_qubits(self) -> int:
        """Return the configured register width."""
        return self._num_qubits

    def _append_recursive_preparation(
        self, circuit: Circuit, *, positions: list[int], max_weight: int
    ) -> None:
        """Append the recursive divide-and-conquer preparation."""
        if max_weight == 0:
            return

        local_qubits = len(positions)
        if local_qubits <= 2 * max_weight:
            circuit.compose(
                DickeUnitary(local_qubits, max_weight).build_circuit(),
                positions,
            )
            return

        left_width = local_qubits // 2
        left_positions = positions[:left_width]
        right_positions = positions[left_width:]
        circuit.compose(
            WeightDistributionBlock(
                local_qubits, left_width, max_weight
            ).build_circuit(),
            positions,
        )
        self._append_recursive_preparation(
            circuit,
            positions=left_positions,
            max_weight=max_weight,
        )
        self._append_recursive_preparation(
            circuit,
            positions=right_positions,
            max_weight=max_weight,
        )
