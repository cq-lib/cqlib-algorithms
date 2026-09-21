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

"""Grover-iteration composite-operator circuit construction."""

from __future__ import annotations

import copy
from collections.abc import Iterable

from cqlib import Circuit
from cqlib.circuit import CircuitError

from ..oracle.general_state_reflection import GeneralStateReflection
from ..oracle.grover_diffusion import GroverDiffusion
from ..oracle.zero_state_reflection import ZeroStateReflection


class GroverOperator:
    """Construct one Grover iteration unit ``Q = D O``.

    ``O`` is a unitary phase oracle on ``n`` search qubits (qubit ``0`` is
    the least significant bit). For the selected reflection-qubit set ``R``,
    define ``P_(0,R)`` as the projector onto states whose qubits in ``R`` are
    all zero and ``S_(0,R) = 2 P_(0,R) - I``. The diffusion part is
    ``D_R = A S_(0,R) A†``. Its time order is therefore ``A†``, then
    ``S_(0,R)``, then ``A``. This is the convention of
    ``qiskit.circuit.library.grover_operator`` (used as an interface
    reference only); some textbooks write ``-Q``, which differs by a global
    phase only.

    With no ``state_preparation``, ``A`` applies Hadamard gates on ``R`` only.
    With a custom invertible ``state_preparation``, ``A`` is that circuit on
    the entire search register. When ``R`` contains every search qubit,
    ``P_(0,R) = |0...0><0...0|`` and ``D_R`` is the rank-one reflection
    ``2 A|0...0><0...0|A† - I``. For a proper subset ``R``, ``D_R`` instead
    reflects about the subspace ``A range(P_(0,R))``; it is not generally a
    reflection about the single state ``A|0...0>``.

    The oracle must be a valid, invertible circuit on the search register;
    a typical input is ``PhaseOracle(function).build_circuit()``.
    Non-unitary or dynamic operations are rejected because amplitude
    amplification assumes a unitary oracle. Circuits are defensively copied
    at construction; non-contiguous source-qubit IDs are mapped
    positionally, in ``Circuit.qubits`` order, to ``q0, q1, ...``.
    This component validates only that the input is a valid, statically
    invertible :class:`Circuit`. It does not perform an exponentially sized
    matrix analysis to prove that the circuit is a diagonal phase oracle;
    supplying a circuit with the required oracle semantics is the caller's
    responsibility.

    Args:
        oracle: Valid invertible phase-oracle circuit on the search
            register.
        state_preparation: Optional valid invertible circuit ``A`` defining
            the conjugated reflection, about a state for the full register
            or a subspace for a proper reflection subset. ``None`` uses
            Hadamard gates on the reflection qubits.
        reflection_qubits: Optional non-empty subset of search qubits on
            which the zero reflection acts. ``None`` selects all search
            qubits.

    Raises:
        TypeError: If an input has the wrong type.
        ValueError: If a circuit is empty, invalid, not invertible, or has
            a mismatched qubit count, or if ``reflection_qubits`` is empty,
            duplicated, or out of range.
    """

    __slots__ = ("_oracle", "_reflection_qubits", "_state_reflection")

    _oracle: Circuit
    _state_reflection: GeneralStateReflection | None
    _reflection_qubits: tuple[int, ...]

    def __init__(
        self,
        oracle: Circuit,
        *,
        state_preparation: Circuit | None = None,
        reflection_qubits: Iterable[int] | None = None,
    ) -> None:
        if not isinstance(oracle, Circuit):
            raise TypeError("oracle must be a Circuit")
        if oracle.num_qubits <= 0:
            raise ValueError("oracle must contain at least one qubit")
        if state_preparation is not None and not isinstance(state_preparation, Circuit):
            raise TypeError("state_preparation must be a Circuit")

        self._oracle = self._snapshot_oracle(oracle)
        self._state_reflection = (
            None
            if state_preparation is None
            else GeneralStateReflection(state_preparation)
        )
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
