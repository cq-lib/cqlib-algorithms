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

"""Single-layer QAOA mixer-Hamiltonian evolution.

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
from cqlib.circuit.ansatz import EvolutionInfo, EvolutionStrategy, PauliEvolutionAnsatz
from cqlib.qis import Hamiltonian, Pauli, PauliString

from ._validation import copy_hermitian_hamiltonian, validate_positive_qubit_count
from .base_ansatz import _DelegatingAnsatz


def _x_mixer(num_qubits: int) -> Hamiltonian:
    mixer = Hamiltonian(num_qubits)
    for qubit in range(num_qubits):
        term = PauliString(num_qubits)
        term.set_pauli(qubit, Pauli.x())
        mixer.add_term(term, 1.0)
    return mixer


@final
class QAOAMixerEvolution(_DelegatingAnsatz):
    """Build one QAOA mixer layer ``exp(-i beta H_M)``.

    By default, ``H_M`` is the transverse-field mixer ``sum_i X_i``. A custom
    Hermitian Pauli Hamiltonian may be supplied, provided its width matches
    ``num_qubits``. Caller-owned Hamiltonians and strategies are copied.

    Args:
        num_qubits: Positive mixer width.
        mixer_operator: Custom Hermitian Hamiltonian, or ``None`` for transverse X.
        strategy: Optional Cqlib evolution strategy for noncommuting terms.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If width, Hamiltonian, or strategy is invalid.
    """

    __slots__ = ()

    def __init__(
        self,
        num_qubits: int,
        *,
        mixer_operator: Hamiltonian | None = None,
        strategy: EvolutionStrategy | None = None,
    ) -> None:
        validate_positive_qubit_count(num_qubits)
        if mixer_operator is not None and not isinstance(mixer_operator, Hamiltonian):
            raise TypeError("mixer_operator must be a Hamiltonian or None")
        if strategy is not None and not isinstance(strategy, EvolutionStrategy):
            raise TypeError("strategy must be an EvolutionStrategy or None")

        if mixer_operator is not None and mixer_operator.num_qubits != num_qubits:
            raise ValueError("mixer_operator must act on num_qubits qubits")

        mixer = _x_mixer(num_qubits) if mixer_operator is None else mixer_operator
        mixer_snapshot = copy_hermitian_hamiltonian(
            mixer,
            name="mixer_operator",
        )
        frozen_strategy = (
            EvolutionStrategy.auto(1) if strategy is None else copy.copy(strategy)
        )
        inner = PauliEvolutionAnsatz(mixer_snapshot).with_strategy(frozen_strategy)
        super().__init__(inner)
        self.validate()

    def _build_circuit(self, prefix: str) -> Circuit:
        """Return a new mixer-evolution circuit using ``{prefix}_beta``."""
        inner = cast(PauliEvolutionAnsatz, self._inner)
        return inner.with_time_param_name(f"{prefix}_beta").build_circuit(prefix)

    def _parameter_count(self) -> int:
        """Return one mixer-evolution parameter slot."""
        return 1

    def evolution_info(self) -> EvolutionInfo:
        """Return Core metadata for the configured Hamiltonian evolution."""
        inner = cast(PauliEvolutionAnsatz, self._inner)
        return inner.evolution_info()
