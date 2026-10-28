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

"""Single-layer QAOA cost-Hamiltonian evolution.

References
----------
[1] E. Farhi, J. Goldstone, and S. Gutmann, "A quantum approximate
    optimization algorithm," 2014.
    arXiv:1411.4028.
"""

from __future__ import annotations

import copy
from typing import cast, final

from cqlib.circuit import Circuit
from cqlib.circuit.ansatz import EvolutionStrategy, PauliEvolutionAnsatz
from cqlib.qis import Hamiltonian

from ._validation import copy_hermitian_hamiltonian
from .base_ansatz import _DelegatingAnsatz


@final
class QAOACostEvolution(_DelegatingAnsatz):
    """Build one QAOA cost layer ``exp(-i gamma H_C)``.

    The Hamiltonian and optional evolution strategy are copied at
    construction, so later changes to caller-owned inputs do not affect this
    component. Core selects exact evolution for commuting terms and otherwise
    uses the configured product-formula strategy.

    Args:
        cost_operator: Finite Hermitian cost Hamiltonian to snapshot.
        strategy: Optional Cqlib evolution strategy for noncommuting terms.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the Hamiltonian or strategy is invalid.
    """

    __slots__ = ()

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
        cost_snapshot = copy_hermitian_hamiltonian(
            cost_operator,
            name="cost_operator",
        )
        inner = PauliEvolutionAnsatz(cost_snapshot).with_strategy(frozen_strategy)
        super().__init__(inner)
        self.validate()

    def _build_circuit(self, prefix: str) -> Circuit:
        """Return a new cost-evolution circuit using ``{prefix}_gamma``."""
        inner = cast(PauliEvolutionAnsatz, self._inner)
        return inner.with_time_param_name(f"{prefix}_gamma").build_circuit(prefix)

    def _parameter_count(self) -> int:
        """Return one cost-evolution parameter slot."""
        return 1
