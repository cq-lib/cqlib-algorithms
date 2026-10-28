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

"""Shared exact Jordan--Wigner hopping-factor synthesis."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from numbers import Complex

from cqlib.circuit import Circuit
from cqlib.circuit.ansatz import EvolutionStrategy, PauliEvolutionAnsatz
from cqlib.qis import FermionicLadder, FermionicOp, Hamiltonian, JordanWignerMapper

from ._validation import (
    copy_hermitian_hamiltonian,
    validate_hermitian_hamiltonian,
    validate_positive_qubit_count,
)


def snapshot_hopping(value: complex) -> complex:
    """Validate and return a finite nonzero hopping coefficient."""
    if isinstance(value, bool) or not isinstance(value, Complex):
        raise TypeError("hopping must be a complex number")
    hopping = complex(value)
    if not math.isfinite(hopping.real) or not math.isfinite(hopping.imag):
        raise ValueError("hopping must be finite")
    if hopping == 0.0:
        raise ValueError("hopping must be nonzero")
    return hopping


def validate_mode(mode: int, *, num_modes: int, name: str) -> None:
    """Validate a fermionic mode index."""
    if type(mode) is not int:
        raise TypeError(f"{name} must be an int")
    if not 0 <= mode < num_modes:
        raise ValueError(f"{name} is out of range")


def hopping_operator(
    num_modes: int,
    source_mode: int,
    target_mode: int,
    hopping: complex,
) -> FermionicOp:
    """Return one Hermitian directed hopping pair."""
    return FermionicOp.from_terms(
        num_modes,
        (
            (
                (
                    FermionicLadder.create(source_mode),
                    FermionicLadder.annihilate(target_mode),
                ),
                hopping,
            ),
            (
                (
                    FermionicLadder.create(target_mode),
                    FermionicLadder.annihilate(source_mode),
                ),
                hopping.conjugate(),
            ),
        ),
    )


def build_exact_commuting_evolution(
    operator: Hamiltonian,
    *,
    parameter_name: str,
) -> Circuit:
    evolution = (
        PauliEvolutionAnsatz(operator)
        .with_strategy(copy.copy(EvolutionStrategy.exact()))
        .with_time_param_name(parameter_name)
    )
    evolution.validate()
    return evolution.build_circuit(parameter_name)


@dataclass(frozen=True, slots=True)
class _HoppingEvolution:
    """Validated immutable data needed to synthesize one hopping evolution."""

    num_modes: int
    source_mode: int
    target_mode: int
    hopping: complex
    phase: float
    operator: Hamiltonian

    @classmethod
    def create(
        cls,
        num_modes: int,
        source_mode: int,
        target_mode: int,
        hopping: complex,
    ) -> _HoppingEvolution:
        """Validate, map, and snapshot one hopping factor."""
        validate_positive_qubit_count(num_modes)
        validate_mode(source_mode, num_modes=num_modes, name="source_mode")
        validate_mode(target_mode, num_modes=num_modes, name="target_mode")
        if source_mode == target_mode:
            raise ValueError("source_mode and target_mode must be distinct")
        frozen_hopping = snapshot_hopping(hopping)
        real_operator = hopping_operator(
            num_modes,
            source_mode,
            target_mode,
            complex(abs(frozen_hopping)),
        )
        mapped = JordanWignerMapper().map_operator(real_operator)
        operator = copy_hermitian_hamiltonian(mapped, name="mapped hopping")
        if not operator.all_terms_commute():
            raise ValueError("mapped real hopping terms must commute")
        return cls(
            num_modes,
            source_mode,
            target_mode,
            frozen_hopping,
            math.atan2(frozen_hopping.imag, frozen_hopping.real),
            operator,
        )

    def validate(self) -> None:
        """Validate the immutable hopping and mapped-operator snapshot."""
        validate_positive_qubit_count(self.num_modes)
        validate_mode(
            self.source_mode,
            num_modes=self.num_modes,
            name="source_mode",
        )
        validate_mode(
            self.target_mode,
            num_modes=self.num_modes,
            name="target_mode",
        )
        if self.source_mode == self.target_mode:
            raise ValueError("source_mode and target_mode must be distinct")
        snapshot_hopping(self.hopping)
        validate_hermitian_hamiltonian(self.operator, name="mapped hopping")
        if not self.operator.all_terms_commute():
            raise ValueError("mapped real hopping terms must commute")

    def build_circuit(self, parameter_name: str) -> Circuit:
        """Build the exact hopping evolution with one explicit symbol name."""
        circuit = Circuit(self.operator.num_qubits)
        if self.phase != 0.0:
            circuit.rz(self.source_mode, -self.phase / 2.0)
            circuit.rz(self.target_mode, self.phase / 2.0)
        evolution = build_exact_commuting_evolution(
            self.operator,
            parameter_name=parameter_name,
        )
        circuit.compose(evolution, circuit.qubits)
        if self.phase != 0.0:
            circuit.rz(self.source_mode, self.phase / 2.0)
            circuit.rz(self.target_mode, -self.phase / 2.0)
        return circuit
