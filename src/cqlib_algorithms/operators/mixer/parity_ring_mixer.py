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

"""Exact parity-partitioned ring mixer."""

from __future__ import annotations

from typing import final

from cqlib.circuit import Circuit, Parameter

from ._base import _BaseMixer
from ._exchange import append_exchange, ring_edge_layers
from ._validation import validate_positive_integer


@final
class ParityRingMixer(_BaseMixer):
    """Apply odd, even and optional closing exchange layers in that order."""

    __slots__ = ("_num_values", "_repetitions")

    def __init__(self, num_values: int, *, repetitions: int = 1) -> None:
        validate_positive_integer(num_values, name="num_values")
        if num_values < 2:
            raise ValueError("num_values must be at least two")
        validate_positive_integer(repetitions, name="repetitions")
        self._num_values = num_values
        self._repetitions = repetitions
        super().__init__(num_values)
        self.validate()

    def _validate_configuration(self) -> None:
        return None

    def _build_mixer(self, prefix: str, beta: Parameter) -> Circuit:
        del prefix
        circuit = Circuit(self._num_values)
        for _ in range(self._repetitions):
            for layer in ring_edge_layers(self._num_values):
                for left, right in layer:
                    append_exchange(circuit, left, right, beta)
        return circuit
