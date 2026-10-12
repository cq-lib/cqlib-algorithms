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

"""Exact color-parity permutation swap mixer."""

from __future__ import annotations

from typing import final

from cqlib.circuit import Circuit, Parameter

from ._base import _BaseMixer
from ._exchange import complete_graph_edge_colors, ring_edge_layers, transition_circuit
from ._validation import validate_positive_integer


@final
class ColorParityPermutationSwapMixer(_BaseMixer):
    """Mix direct one-hot permutations using color and position partitions.

    Qubit ``position * num_items + value`` encodes a value at a tour position.
    Value pairs use a deterministic round-robin edge coloring of the complete
    graph; adjacent cyclic positions use the parity partition. Every local
    four-qubit transition is exact and all applications share one beta.
    """

    __slots__ = ("_num_items", "_repetitions")

    def __init__(self, num_items: int, *, repetitions: int = 1) -> None:
        validate_positive_integer(num_items, name="num_items")
        if num_items < 2:
            raise ValueError("num_items must be at least two")
        validate_positive_integer(repetitions, name="repetitions")
        self._num_items = num_items
        self._repetitions = repetitions
        super().__init__(num_items * num_items)
        self.validate()

    def _validate_configuration(self) -> None:
        return None

    def _build_mixer(self, prefix: str, beta: Parameter) -> Circuit:
        del beta
        circuit = Circuit(self._num_qubits)
        local = transition_circuit(prefix)
        colors = complete_graph_edge_colors(self._num_items)
        position_layers = ring_edge_layers(self._num_items)
        for _ in range(self._repetitions):
            for color in colors:
                for layer in position_layers:
                    for position, next_position in layer:
                        for first, second in color:
                            targets = [
                                position * self._num_items + first,
                                next_position * self._num_items + second,
                                next_position * self._num_items + first,
                                position * self._num_items + second,
                            ]
                            circuit.compose(local, targets)
        return circuit
