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

"""Reflection about a state prepared by an invertible circuit.

References
----------
[1] G. Brassard, P. Hoyer, M. Mosca, and A. Tapp, "Quantum amplitude
    amplification and estimation," Contemporary Mathematics, 305, 53-74,
    2002.
    arXiv:quant-ph/0005055.
"""

from __future__ import annotations

import copy

from cqlib import Circuit
from cqlib.circuit import CircuitError

from ..._utils.state_preparation import build_component_circuit
from ...protocols.ansatz import Ansatz
from ...protocols.state_preparation import StatePreparation
from .zero_state_reflection import ZeroStateReflection


class GeneralStateReflection:
    """Construct the exact reflection about a circuit-prepared state.

    For an invertible ``A`` satisfying ``A|0> = |psi>``, the returned circuit
    has matrix ``A R0 A† = 2|psi><psi| - I``. Its time order is therefore
    ``A†``, ``R0``, then ``A``. All preparation qubits participate in the
    reflection; no ancillas are added.

    The preparation or ansatz is built once with ``prefix`` and its circuit
    is defensively copied at construction. Active symbols are preserved. Non-contiguous
    source-qubit IDs are mapped positionally, in ``Circuit.qubits`` order, to
    output qubits ``q0, q1, ...``. Barriers are allowed, but measurement,
    reset, delay, classical control, and any other non-invertible operation,
    including one nested in a circuit gate, are rejected.

    Args:
        state_preparation: Preparation or ansatz with a statically invertible circuit.
        prefix: Naming prefix when building a component input.

    Raises:
        TypeError: If ``state_preparation`` or ``prefix`` has an unsupported type.
        ValueError: If the circuit has no qubits, is invalid, or is not invertible.
    """

    __slots__ = ("_state_preparation",)

    _state_preparation: Circuit

    def __init__(
        self,
        state_preparation: StatePreparation | Ansatz,
        *,
        prefix: str = "reflection",
    ) -> None:
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        circuit = build_component_circuit(state_preparation, prefix=prefix)
        if circuit.num_qubits <= 0:
            raise ValueError("state_preparation must contain at least one qubit")

        candidate = copy.deepcopy(circuit)
        try:
            candidate.validate()
            candidate.inverse()
        except CircuitError as error:
            raise ValueError(
                "state_preparation must contain only valid invertible operations"
            ) from error

        normalized = Circuit(candidate.num_qubits)
        normalized.compose(candidate, normalized.qubits)
        self._state_preparation = normalized

    @classmethod
    def from_state_preparation(
        cls,
        state_preparation: StatePreparation,
        *,
        prefix: str,
    ) -> GeneralStateReflection:
        """Construct from a state-preparation component.

        The component is validated before one circuit is built. Declared qubit
        and parameter-slot counts are checked against the built circuit before
        the reflection constructor enforces static invertibility.
        """
        if not isinstance(state_preparation, StatePreparation):
            raise TypeError("state_preparation must be a StatePreparation")
        return cls(state_preparation, prefix=prefix)

    @property
    def num_qubits(self) -> int:
        """Return the number of preparation and reflection qubits."""
        return int(self._state_preparation.num_qubits)

    @property
    def state_preparation(self) -> Circuit:
        """Return an independent copy of the normalized preparation circuit."""
        return copy.deepcopy(self._state_preparation)

    def build_circuit(self) -> Circuit:
        """Build and return a fresh general-state reflection circuit."""
        preparation = copy.deepcopy(self._state_preparation)
        circuit = Circuit(self.num_qubits)
        circuit.compose(preparation.inverse())
        circuit.compose(ZeroStateReflection(self.num_qubits).build_circuit())
        circuit.compose(preparation)
        return circuit
