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

"""Ordered factorized unitary vibrational coupled-cluster ansatz."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import final

from cqlib.circuit import Circuit, CircuitError
from cqlib.circuit.ansatz import EvolutionStrategy
from cqlib.qis import Hamiltonian

from ...extension.many_body import OneHotVibrationalMapper
from ...extension.vibration import VibrationalExcitation
from ...protocols.state_preparation import StatePreparation
from ._validation import copy_hermitian_hamiltonian, snapshot_fixed_initial_state
from .base_ansatz import BaseAnsatz
from .evolved_operator_ansatz import EvolvedOperatorAnsatz


def _snapshot_excitations(
    excitations: Sequence[VibrationalExcitation],
) -> tuple[VibrationalExcitation, ...]:
    """Return a nonempty, equal-space excitation tuple in input order."""
    if isinstance(excitations, (str, bytes)) or not isinstance(excitations, Sequence):
        raise TypeError(
            "excitations must be a sequence of VibrationalExcitation values"
        )
    snapshot = tuple(excitations)
    if not snapshot:
        raise ValueError("excitations must not be empty")
    if any(not isinstance(item, VibrationalExcitation) for item in snapshot):
        raise TypeError("excitations must contain only VibrationalExcitation values")
    modal_space = snapshot[0].modal_space
    if any(item.modal_space != modal_space for item in snapshot[1:]):
        raise ValueError("all excitations must use the same modal space")
    return snapshot


def _mapped_generator_snapshots(
    excitations: tuple[VibrationalExcitation, ...],
) -> tuple[Hamiltonian, ...]:
    """Map ``i(T-T†)`` for every excitation exactly once."""
    mapper = OneHotVibrationalMapper()
    operators: list[Hamiltonian] = []
    for index, excitation in enumerate(excitations):
        transition = excitation.to_vibrational_op()
        generator = (transition - transition.adjoint()).scaled(1j)
        mapped = mapper.map_operator(generator)
        snapshot = copy_hermitian_hamiltonian(
            mapped,
            name=f"mapped generator[{index}]",
        )
        if not snapshot.all_terms_commute():
            raise ValueError(
                "mapped generator Hamiltonian terms must commute for exact evolution"
            )
        operators.append(snapshot)
    return tuple(operators)


@final
class UVCCAnsatz(BaseAnsatz):
    """Build an ordered factorized unitary vibrational coupled-cluster circuit.

    Excitation ``k`` defines ``T_k`` and the Hermitian generator
    ``G_k=i(T_k-T_k†)``. Each repetition appends ``exp(-i theta_k G_k)`` in
    input order. Repeated excitations are preserved as independent ordered
    factors. The optional fixed initial state is prepended once.

    Args:
        excitations: Non-empty ordered excitations in one modal space.
        reps: Non-negative repetition count.
        initial_state: Optional fixed state preparation prepended once.

    Raises:
        TypeError: If an excitation or initial state has the wrong type.
        ValueError: If modal spaces disagree or exact evolution is unavailable.
    """

    __slots__ = ("_initial_circuit",)

    def __init__(
        self,
        excitations: Sequence[VibrationalExcitation],
        *,
        reps: int = 1,
        initial_state: StatePreparation | None = None,
    ) -> None:
        frozen_excitations = _snapshot_excitations(excitations)
        operators = _mapped_generator_snapshots(frozen_excitations)
        inner = EvolvedOperatorAnsatz(
            operators,
            reps=reps,
            strategy=EvolutionStrategy.exact(),
            remove_identity_operators=False,
        )
        self._initial_circuit = snapshot_fixed_initial_state(
            initial_state,
            num_qubits=inner.num_qubits(),
            prefix="uvcc_initial",
        )
        super().__init__(inner)
        self.validate()

    def _validate_configuration(self) -> None:
        """Validate the exact evolution and fixed initial-state snapshots."""
        self._inner.validate()
        if self._initial_circuit is None:
            return
        self._initial_circuit.validate()
        if self._initial_circuit.num_qubits != self._inner.num_qubits():
            raise ValueError("initial_state snapshot has an inconsistent width")
        if self._initial_circuit.used_symbols:
            raise ValueError("initial_state snapshot must not contain active symbols")
        try:
            self._initial_circuit.inverse()
        except CircuitError as error:
            raise ValueError(
                "initial_state snapshot must contain only statically invertible operations"
            ) from error

    def _build_circuit(self, prefix: str) -> Circuit:
        """Return the fixed initial state followed by ordered excitation factors."""
        circuit = (
            Circuit(self._inner.num_qubits())
            if self._initial_circuit is None
            else copy.copy(self._initial_circuit)
        )
        evolution = self._inner.build_circuit(f"{prefix}_theta")
        circuit.compose(evolution, circuit.qubits)
        return circuit


__all__ = ["UVCCAnsatz"]
