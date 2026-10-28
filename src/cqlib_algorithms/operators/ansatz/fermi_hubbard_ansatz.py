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

"""Particle-number-preserving Fermi--Hubbard variational circuit."""

from __future__ import annotations

import copy
from typing import final

from cqlib.circuit import Circuit
from cqlib.qis import FermionicOp, Hamiltonian, JordanWignerMapper

from ...extension.lattice import (
    FermiHubbardModel,
    FermionicLatticeConfiguration,
    LatticeSystem,
)
from ..state_preparation import LatticeReferenceStatePreparation
from ._fermionic_hopping import (
    _HoppingEvolution,
    build_exact_commuting_evolution,
    hopping_operator,
)
from ._validation import (
    validate_hermitian_hamiltonian,
    validate_repetitions,
)
from .base_ansatz import BaseAnsatz


def _model_factors(
    model: FermiHubbardModel,
) -> tuple[tuple[_HoppingEvolution, ...], Hamiltonian | None]:
    hopping_sum = FermionicOp(model.num_modes)
    factors: list[_HoppingEvolution] = []
    for source, target, weight in model.lattice.weighted_edges:
        if source == target or weight == 0.0:
            continue
        for spin in ("up", "down"):
            source_mode = model.mode_index(source, spin)
            target_mode = model.mode_index(target, spin)
            hopping = complex(weight)
            factor_operator = hopping_operator(
                model.num_modes,
                source_mode,
                target_mode,
                hopping,
            )
            hopping_sum = hopping_sum + factor_operator
            factors.append(
                _HoppingEvolution.create(
                    model.num_modes,
                    source_mode,
                    target_mode,
                    hopping,
                )
            )

    diagonal = model.fermionic_operator() - hopping_sum
    diagonal.chop(atol=0.0)
    if diagonal.is_zero():
        return tuple(factors), None
    mapped = JordanWignerMapper().map_operator(diagonal).normalized_hermitian()
    validate_hermitian_hamiltonian(mapped, name="mapped diagonal block")
    if not mapped.all_terms_commute():
        raise ValueError("mapped Fermi--Hubbard diagonal terms must commute")
    return tuple(factors), mapped


@final
class FermiHubbardAnsatz(BaseAnsatz):
    r"""Prepare a fixed sector reference and apply exact physical factors.

    Each repetition applies one independent exact hopping evolution for every
    nonzero lattice edge and spin species, followed by one shared exact
    evolution of all onsite-potential and onsite-interaction terms. The order
    is fixed by the lattice edge snapshot and then spin up before spin down.

    The circuit uses the model's interleaved Jordan--Wigner mode order and
    preserves ``N_up`` and ``N_down`` exactly. It is a factorized variational
    family, not an assertion that noncommuting hopping factors equal one exact
    evolution under their sum.

    Args:
        system: Fermi--Hubbard lattice system with a fermionic sector.
        configuration: Reference configuration belonging to that sector.
        reps: Non-negative number of ordered physical-factor layers.

    Raises:
        TypeError: If system or configuration types are incompatible.
        ValueError: If the sector, repetition count, or mapped factors are invalid.
    """

    __slots__ = (
        "_configuration",
        "_diagonal_operator",
        "_hopping_factors",
        "_initial_circuit",
        "_reps",
        "_system",
    )

    def __init__(
        self,
        system: LatticeSystem,
        configuration: FermionicLatticeConfiguration,
        *,
        reps: int = 1,
    ) -> None:
        if not isinstance(system, LatticeSystem):
            raise TypeError("system must be a LatticeSystem")
        if not isinstance(system.model, FermiHubbardModel):
            raise TypeError("system must contain a FermiHubbardModel")
        if not isinstance(configuration, FermionicLatticeConfiguration):
            raise TypeError("configuration must be a FermionicLatticeConfiguration")
        validate_repetitions(reps)
        preparation = LatticeReferenceStatePreparation(system, configuration)
        hopping_factors, diagonal_operator = _model_factors(system.model)

        self._system = system
        self._configuration = configuration
        self._reps = reps
        self._initial_circuit = preparation.build_circuit("lattice_reference")
        self._hopping_factors = hopping_factors
        self._diagonal_operator = diagonal_operator
        self.validate()

    @property
    def system(self) -> LatticeSystem:
        """Return the physical system defining the ansatz."""
        return self._system

    @property
    def configuration(self) -> FermionicLatticeConfiguration:
        """Return the fixed reference configuration."""
        return self._configuration

    def _validate_configuration(self) -> None:
        validate_repetitions(self._reps)
        LatticeReferenceStatePreparation(
            self._system,
            self._configuration,
        ).validate()
        self._initial_circuit.validate()
        if self._initial_circuit.used_symbols:
            raise ValueError("initial lattice reference must be fixed")
        for factor in self._hopping_factors:
            factor.validate()
            if factor.num_modes != self._initial_circuit.num_qubits:
                raise ValueError("hopping factor width is inconsistent")
        if self._diagonal_operator is not None:
            validate_hermitian_hamiltonian(
                self._diagonal_operator,
                name="mapped diagonal block",
            )
            if not self._diagonal_operator.all_terms_commute():
                raise ValueError("mapped Fermi--Hubbard diagonal terms must commute")

    def _build_circuit(self, prefix: str) -> Circuit:
        circuit = copy.copy(self._initial_circuit)
        parameter_index = 0
        for _ in range(self._reps):
            for factor in self._hopping_factors:
                evolution = factor.build_circuit(f"{prefix}_{parameter_index}")
                circuit.compose(evolution, circuit.qubits)
                parameter_index += 1
            if self._diagonal_operator is not None:
                evolution = build_exact_commuting_evolution(
                    self._diagonal_operator,
                    parameter_name=f"{prefix}_{parameter_index}",
                )
                circuit.compose(evolution, circuit.qubits)
                parameter_index += 1
        return circuit

    def _parameter_count(self) -> int:
        factors_per_rep = len(self._hopping_factors) + int(
            self._diagonal_operator is not None
        )
        return self._reps * factors_per_rep

    def _qubit_count(self) -> int:
        return self._initial_circuit.num_qubits


__all__ = ["FermiHubbardAnsatz"]
