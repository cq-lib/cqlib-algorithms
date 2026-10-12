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

"""Simultaneous nearby-values mixer for a one-hot qudit."""

from __future__ import annotations

from typing import final

from cqlib.circuit import Circuit, Parameter
from cqlib.circuit.ansatz import EvolutionStrategy
from cqlib.qis import Hamiltonian

from ..ansatz import QAOAMixerEvolution
from ._base import _BaseMixer
from ._exchange import (
    add_exchange,
    append_exchange_product_formula,
    validate_exchange_evolution,
)
from ._validation import snapshot_strategy, validate_positive_integer


@final
class NearbyValuesMixer(_BaseMixer):
    """Evolve under the cyclic ``radius``-nearby one-hot Hamiltonian.

    For ``d=num_values``, the Hamiltonian is the sum of exchange generators
    ``E_(a,a+s)=(XX+YY)/2`` for every value ``a`` and shift
    ``1 <= s <= radius``, with indices reduced modulo ``d``. Repeated edges
    retain their multiplicity, as required by the cyclic shift definition.
    Noncommuting sums use the supplied Core evolution strategy.
    """

    __slots__ = ("_evolution", "_pairs")

    def __init__(
        self,
        num_values: int,
        radius: int,
        *,
        strategy: EvolutionStrategy | None = None,
    ) -> None:
        validate_positive_integer(num_values, name="num_values")
        if num_values < 2:
            raise ValueError("num_values must be at least two")
        validate_positive_integer(radius, name="radius")
        if radius >= num_values:
            raise ValueError("radius must be less than num_values")
        frozen_strategy = snapshot_strategy(strategy)
        hamiltonian = Hamiltonian(num_values)
        pairs: list[tuple[int, int]] = []
        for shift in range(1, radius + 1):
            for value in range(num_values):
                pair = (value, (value + shift) % num_values)
                pairs.append(pair)
                add_exchange(hamiltonian, *pair)
        self._pairs = tuple(pairs)
        self._evolution = QAOAMixerEvolution(
            num_values,
            mixer_operator=hamiltonian,
            strategy=frozen_strategy,
        )
        super().__init__(num_values)
        self.validate()

    def _validate_configuration(self) -> None:
        validate_exchange_evolution(self._evolution)

    def _build_mixer(self, prefix: str, beta: Parameter) -> Circuit:
        del prefix
        circuit = Circuit(self._num_qubits)
        append_exchange_product_formula(circuit, self._pairs, beta, self._evolution)
        return circuit
