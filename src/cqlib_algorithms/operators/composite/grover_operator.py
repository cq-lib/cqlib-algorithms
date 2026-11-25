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

"""Grover-iteration composite-operator circuit construction.

References
----------
[1] L. K. Grover, "A fast quantum mechanical algorithm for database search,"
    Proceedings of the 28th Annual ACM Symposium on Theory of Computing,
    212-219, 1996.
    DOI: 10.1145/237814.237866.
[2] G. Brassard, P. Hoyer, M. Mosca, and A. Tapp, "Quantum amplitude
    amplification and estimation," Contemporary Mathematics, 305, 53-74,
    2002.
    arXiv:quant-ph/0005055.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable

from cqlib import Circuit
from cqlib.circuit import CircuitError

from ...protocols.state_preparation import StatePreparation
from ..oracle.general_state_reflection import GeneralStateReflection
from ..oracle.grover_diffusion import GroverDiffusion
from ..oracle.phase_oracle import BitPatternPhaseOracle, PhaseOracle
from ..oracle.zero_state_reflection import ZeroStateReflection


class GroverOperator:
    """Construct one Grover iteration ``Q = A S_(0,R) A† O``.

    ``O`` is the operation supplied by the phase-oracle component and ``S_(0,R) = 2 P_(0,R) - I`` is the
    zero reflection on ``reflection_qubits``. Circuit time order is ``O``,
    ``A†``, ``S_(0,R)``, then ``A``.

    When all qubits are reflected, the diffusion part reflects about the
    state ``A|0...0>``. For a proper subset, it instead reflects about the
    subspace ``A range(P_(0,R))``. If ``state_preparation`` is omitted,
    Hadamard gates on the reflection qubits are used for ``A``.

    Components are built once and their statically invertible circuits are
    defensively copied. Source qubits map by insertion position to ``0..n-1``.
    Symbolic preparations can define an iteration; execution requires full bindings.

    Args:
        oracle: PhaseOracle or BitPatternPhaseOracle defining the marked states.
        state_preparation: Component for preparing ``A|0>``.
            Components are built once at construction with the prefix ``"grover"``.
        reflection_qubits: Optional non-empty reflection-qubit subset.
            ``None`` selects all qubits.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If a circuit or reflection-qubit selection is invalid.
    """

    __slots__ = ("_marked_states", "_oracle", "_reflection_qubits", "_state_reflection")

    _oracle: Circuit
    _state_reflection: GeneralStateReflection | None
    _reflection_qubits: tuple[int, ...]

    def __init__(
        self,
        oracle: PhaseOracle | BitPatternPhaseOracle,
        *,
        state_preparation: StatePreparation | None = None,
        reflection_qubits: Iterable[int] | None = None,
    ) -> None:
        if not isinstance(oracle, (PhaseOracle, BitPatternPhaseOracle)):
            raise TypeError("oracle must be a phase-oracle component")
        oracle_circuit = oracle.build_circuit()
        if not isinstance(oracle_circuit, Circuit):
            raise TypeError("PhaseOracle.build_circuit must return a Circuit")
        if oracle_circuit.num_qubits != oracle.num_qubits:
            raise ValueError("oracle circuit width must match its predicate")
        self._marked_states: tuple[int, ...] | BitPatternPhaseOracle = (
            copy.copy(oracle)
            if isinstance(oracle, BitPatternPhaseOracle)
            else oracle.marked_states
        )
        if oracle_circuit.num_qubits <= 0:
            raise ValueError("oracle must contain at least one qubit")

        self._oracle = self._snapshot_oracle(oracle_circuit)
        if isinstance(state_preparation, StatePreparation):
            self._state_reflection = GeneralStateReflection.from_state_preparation(
                state_preparation, prefix="grover"
            )
        elif state_preparation is None:
            self._state_reflection = None
        else:
            raise TypeError("state_preparation must be a StatePreparation or None")
        if (
            self._state_reflection is not None
            and self._state_reflection.num_qubits != self._oracle.num_qubits
        ):
            raise ValueError(
                "state_preparation must have the same number of qubits as oracle"
            )
        self._reflection_qubits = self._canonicalize_reflection_qubits(
            reflection_qubits, int(self._oracle.num_qubits)
        )

    @staticmethod
    def _snapshot_oracle(oracle: Circuit) -> Circuit:
        copied = copy.deepcopy(oracle)
        try:
            copied.validate()
            copied.inverse()
        except CircuitError as error:
            raise ValueError(
                "oracle must contain only valid invertible operations"
            ) from error
        normalized = Circuit(copied.num_qubits)
        normalized.compose(copied, normalized.qubits)
        return normalized

    @staticmethod
    def _canonicalize_reflection_qubits(
        reflection_qubits: Iterable[int] | None,
        num_qubits: int,
    ) -> tuple[int, ...]:
        if reflection_qubits is None:
            return tuple(range(num_qubits))
        if isinstance(reflection_qubits, (str, bytes)):
            raise TypeError("reflection_qubits must be an iterable of integers")
        try:
            items = tuple(reflection_qubits)
        except TypeError as error:
            raise TypeError(
                "reflection_qubits must be an iterable of integers"
            ) from error
        if not items:
            raise ValueError("reflection_qubits must not be empty")
        for qubit in items:
            if isinstance(qubit, bool) or not isinstance(qubit, int):
                raise TypeError("reflection_qubits must contain only integers")
            if qubit < 0 or qubit >= num_qubits:
                raise ValueError(
                    "reflection_qubits entries must be valid search-qubit indices"
                )
        if len(set(items)) != len(items):
            raise ValueError("reflection_qubits must not contain duplicates")
        return tuple(sorted(items))

    @property
    def num_qubits(self) -> int:
        """Return the number of search-qubits the operator acts on."""
        return int(self._oracle.num_qubits)

    @property
    def oracle(self) -> Circuit:
        """Return an independent copy of the normalized oracle circuit."""
        return copy.deepcopy(self._oracle)

    @property
    def marked_states(self) -> tuple[int, ...]:
        """Return marked indices, expanding a compact bit pattern only on access."""
        if isinstance(self._marked_states, BitPatternPhaseOracle):
            return self._marked_states.marked_states
        return self._marked_states

    @property
    def state_preparation(self) -> Circuit | None:
        """Return an independent copy of the preparation circuit, if any."""
        return (
            None
            if self._state_reflection is None
            else self._state_reflection.state_preparation
        )

    @property
    def reflection_qubits(self) -> tuple[int, ...]:
        """Return the sorted qubit subset acted on by the zero reflection."""
        return self._reflection_qubits

    def build_circuit(self) -> Circuit:
        """Build and return a fresh Grover iteration circuit."""
        circuit = Circuit(self.num_qubits)
        circuit.compose(copy.deepcopy(self._oracle), circuit.qubits)

        if len(self._reflection_qubits) == self.num_qubits:
            reflection = (
                GroverDiffusion(self.num_qubits).build_circuit()
                if self._state_reflection is None
                else self._state_reflection.build_circuit()
            )
            circuit.compose(reflection, circuit.qubits)
            return circuit

        preparation = (
            None
            if self._state_reflection is None
            else self._state_reflection.state_preparation
        )
        if preparation is None:
            for qubit in self._reflection_qubits:
                circuit.h(qubit)
        else:
            circuit.compose(preparation.inverse(), circuit.qubits)

        circuit.compose(
            ZeroStateReflection(len(self._reflection_qubits)).build_circuit(),
            [circuit.qubits[qubit] for qubit in self._reflection_qubits],
        )

        if preparation is None:
            for qubit in self._reflection_qubits:
                circuit.h(qubit)
        else:
            circuit.compose(preparation, circuit.qubits)
        return circuit
