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

"""Feasible-subspace-exact swap of two one-hot registers."""

from __future__ import annotations

from itertools import combinations
from typing import final

from cqlib.circuit import Circuit, Parameter

from ._base import _BaseMixer
from ._exchange import transition_circuit
from ._validation import validate_positive_integer


@final
class ValueIndependentPermutationSwapMixer(_BaseMixer):
    """Swap two direct one-hot values without selecting their identities.

    The circuit composes one exact four-body transition for every unordered
    value pair. These blocks act on disjoint two-state sectors of the direct
    one-hot feasible subspace, so their product equals the simultaneous
    value-independent evolution on that subspace. No equality is claimed on
    invalid bitstrings outside the encoded subspace.
    """

    __slots__ = ("_num_values",)

    def __init__(
        self,
        num_values: int,
    ) -> None:
        validate_positive_integer(num_values, name="num_values")
        if num_values < 2:
            raise ValueError("num_values must be at least two")
        width = 2 * num_values
        self._num_values = num_values
        super().__init__(width)
        self.validate()

    def _validate_configuration(self) -> None:
        return None

    def _build_mixer(self, prefix: str, beta: Parameter) -> Circuit:
        del beta
        circuit = Circuit(self._num_qubits)
        local = transition_circuit(prefix)
        for first, second in combinations(range(self._num_values), 2):
            circuit.compose(
                local,
                [
                    first,
                    self._num_values + second,
                    self._num_values + first,
                    second,
                ],
            )
        return circuit
