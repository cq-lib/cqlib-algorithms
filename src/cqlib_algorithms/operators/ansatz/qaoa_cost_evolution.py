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

"""Single-layer QAOA cost-Hamiltonian evolution."""

from __future__ import annotations

import copy
from typing import final

from cqlib.circuit import Circuit
from cqlib.circuit.ansatz import EvolutionStrategy, PauliEvolutionAnsatz
from cqlib.qis import Hamiltonian

from ...protocols.ansatz import Ansatz
from ._validation import validate_prefix


@final
class QAOACostEvolution(Ansatz):
    """Build one QAOA cost layer ``exp(-i gamma H_C)``.

    The Hamiltonian and optional evolution strategy are copied at
    construction, so later changes to caller-owned inputs do not affect this
    component. Core selects exact evolution for commuting terms and otherwise
    uses the configured product-formula strategy.
    """

    __slots__ = ("_inner",)

    def __init__(
        self,
        cost_operator: Hamiltonian,
        *,
        strategy: EvolutionStrategy | None = None,
    ) -> None:
        if not isinstance(cost_operator, Hamiltonian):
            raise TypeError("cost_operator must be a Hamiltonian")
        if cost_operator.num_qubits <= 0:
            raise ValueError("cost_operator.num_qubits must be greater than zero")
        if strategy is not None and not isinstance(strategy, EvolutionStrategy):
            raise TypeError("strategy must be an EvolutionStrategy or None")

        frozen_strategy = (
            EvolutionStrategy.auto(1) if strategy is None else copy.copy(strategy)
        )
        inner = PauliEvolutionAnsatz(cost_operator.copy()).with_strategy(
            frozen_strategy
        )
        self._inner = inner
        self.validate()

    def validate(self) -> None:
        """Validate the delegated Hamiltonian-evolution configuration."""
        self._inner.validate()

    def build_circuit(self, prefix: str) -> Circuit:
        """Return a new cost-evolution circuit using ``{prefix}_gamma``."""
        validate_prefix(prefix)
        return self._inner.with_time_param_name(f"{prefix}_gamma").build_circuit(prefix)

    def num_parameters(self) -> int:
        """Return one cost-evolution parameter slot."""
        return 1

    def num_qubits(self) -> int:
        """Return the cost Hamiltonian width."""
        return self._inner.num_qubits()
