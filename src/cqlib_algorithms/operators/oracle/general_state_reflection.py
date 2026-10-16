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

from cqlib.circuit import CircuitError

from cqlib import Circuit

from ...protocols.state_preparation import StatePreparation
from .zero_state_reflection import ZeroStateReflection


class GeneralStateReflection:
    """Construct the exact reflection about a circuit-prepared state.

    For an invertible ``A`` satisfying ``A|0> = |psi>``, the returned circuit
    has matrix ``A R0 A† = 2|psi><psi| - I``. Its time order is therefore
    ``A†``, ``R0``, then ``A``. All preparation qubits participate in the
    reflection; no ancillas are added.

    The input circuit is defensively copied at construction. Non-contiguous
    source-qubit IDs are mapped positionally, in ``Circuit.qubits`` order, to
    output qubits ``q0, q1, ...``. Barriers are allowed, but measurement,
    reset, delay, classical control, and any other non-invertible operation,
    including one nested in a circuit gate, are rejected.

    Args:
        state_preparation: A non-empty, valid, symbolically invertible circuit.

    Raises:
        TypeError: If ``state_preparation`` is not a Cqlib :class:`Circuit`.
        ValueError: If the circuit has no qubits, is invalid, or is not invertible.
    """

    __slots__ = ("_state_preparation",)

    _state_preparation: Circuit

    def __init__(self, state_preparation: Circuit) -> None:
        if not isinstance(state_preparation, Circuit):
            raise TypeError("state_preparation must be a Circuit")
        if state_preparation.num_qubits <= 0:
            raise ValueError("state_preparation must contain at least one qubit")

        candidate = copy.deepcopy(state_preparation)
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
        the ordinary circuit constructor enforces static invertibility.
        """
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        if not isinstance(state_preparation, StatePreparation):
            raise TypeError("state_preparation must be a StatePreparation")

        state_preparation.validate()
        num_qubits = state_preparation.num_qubits()
        num_parameters = state_preparation.num_parameters()
        if isinstance(num_qubits, bool) or not isinstance(num_qubits, int):
            raise TypeError("state_preparation num_qubits must be an integer")
        if num_qubits <= 0:
            raise ValueError("state_preparation num_qubits must be greater than zero")
        if isinstance(num_parameters, bool) or not isinstance(num_parameters, int):
            raise TypeError("state_preparation num_parameters must be an integer")
        if num_parameters < 0:
            raise ValueError("state_preparation num_parameters must be non-negative")

        circuit = state_preparation.build_circuit(prefix)
        if not isinstance(circuit, Circuit):
            raise TypeError("state_preparation build_circuit must return a Circuit")
        if circuit.num_qubits != num_qubits:
            raise ValueError(
                "built circuit qubit count must equal state_preparation num_qubits"
            )
        if len(circuit.used_symbols) > num_parameters:
            raise ValueError(
                "built circuit active symbol count must not exceed num_parameters"
            )
        return cls(circuit)

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
