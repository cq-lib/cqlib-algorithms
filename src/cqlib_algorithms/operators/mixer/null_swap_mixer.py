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

"""Simultaneous null-swap mixer for a one-hot qudit.

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
from cqlib.circuit.ansatz import EvolutionStrategy
from cqlib.qis import Hamiltonian

from ..._utils.numeric import validate_positive_integer
from ..ansatz import QAOAMixerEvolution
from ._base import _BaseMixer
from ._exchange import (
    add_exchange,
    append_exchange_product_formula,
    validate_exchange_evolution,
)
from ._validation import snapshot_strategy


@final
class NullSwapMixer(_BaseMixer):
    """Evolve under all exchanges between a designated null and other values.

    The full Hamiltonian is ``sum_(a != null) (|null><a| + |a><null|)``.
    Its terms do not commute when more than two values are present, so the
    supplied evolution strategy controls the corresponding approximation.

    Args:
        num_values: Number of one-hot values.
        null_value: Distinguished value exchanged with every other value.
        strategy: Optional Cqlib evolution strategy.

    Raises:
        TypeError: If an option has the wrong type.
        ValueError: If the count, null index, or strategy is invalid.
    """

    __slots__ = ("_evolution", "_pairs")

    def __init__(
        self,
        num_values: int,
        *,
        null_value: int = 0,
        strategy: EvolutionStrategy | None = None,
    ) -> None:
        validate_positive_integer(num_values, name="num_values")
        if num_values < 2:
            raise ValueError("num_values must be at least two")
        if isinstance(null_value, bool) or not isinstance(null_value, int):
            raise TypeError("null_value must be an integer")
        if null_value < 0 or null_value >= num_values:
            raise ValueError("null_value is out of range")
        hamiltonian = Hamiltonian(num_values)
        pairs: list[tuple[int, int]] = []
        for value in range(num_values):
            if value != null_value:
                pair = (null_value, value)
                pairs.append(pair)
                add_exchange(hamiltonian, *pair)
        self._pairs = tuple(pairs)
        self._evolution = QAOAMixerEvolution(
            num_values,
            mixer_operator=hamiltonian,
            strategy=snapshot_strategy(strategy),
        )
        super().__init__(num_values)
        self.validate()

    def _validate_configuration(self) -> None:
        validate_exchange_evolution(self._evolution)

    def _build_mixer(self, prefix: str) -> Circuit:
        beta = self._beta_parameter(prefix)
        circuit = Circuit(self._num_qubits)
        append_exchange_product_formula(circuit, self._pairs, beta, self._evolution)
        return circuit
