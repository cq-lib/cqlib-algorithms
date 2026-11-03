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

"""Fixed-magnetization exchange ansatz on a lattice graph."""

from __future__ import annotations

import copy
from typing import Literal, final

from cqlib.circuit import Circuit
from cqlib.circuit.ansatz import EntanglementTopology

from ..._utils.numeric import validate_repetitions
from ...extension.lattice import (
    FermiHubbardModel,
    LatticeSystem,
    SpinLatticeConfiguration,
    SpinMagnetizationSector,
)
from ..state_preparation import LatticeReferenceStatePreparation
from .base_ansatz import BaseAnsatz
from .excitation_preserving_ansatz import ExcitationPreservingAnsatz

_ExchangeMode = Literal["iswap", "fsim"]


@final
class SpinExchangeAnsatz(BaseAnsatz):
    """Prepare a spin reference and apply number-preserving exchange layers.

    Nonzero lattice edges define the exchange topology. The delegated
    :class:`ExcitationPreservingAnsatz` applies local Z rotations and either
    iSWAP-like or fSim-like exchange blocks. Because computational excitation
    counts spin-down sites, every gate preserves the system's fixed total
    ``S_z`` sector exactly.

    Args:
        system: Spin lattice system with a conserved magnetization sector.
        configuration: Reference configuration belonging to that sector.
        reps: Non-negative number of exchange layers. At zero repetitions,
            only the fixed reference state is returned and no parameter is used.
        mode: ``"iswap"`` or ``"fsim"`` exchange block.

    Raises:
        TypeError: If system, sector, or configuration types are incompatible.
        ValueError: If the configuration, repetition count, or mode is invalid.
    """

    __slots__ = (
        "_configuration",
        "_initial_circuit",
        "_inner_exchange",
        "_system",
    )

    def __init__(
        self,
        system: LatticeSystem,
        configuration: SpinLatticeConfiguration,
        *,
        reps: int = 1,
        mode: _ExchangeMode = "iswap",
    ) -> None:
        if not isinstance(system, LatticeSystem):
            raise TypeError("system must be a LatticeSystem")
        if isinstance(system.model, FermiHubbardModel):
            raise TypeError("system must contain a spin model")
        if not isinstance(system.sector, SpinMagnetizationSector):
            raise TypeError(
                "SpinExchangeAnsatz requires a fixed SpinMagnetizationSector"
            )
        if not isinstance(configuration, SpinLatticeConfiguration):
            raise TypeError("configuration must be a SpinLatticeConfiguration")
        validate_repetitions(reps)
        preparation = LatticeReferenceStatePreparation(system, configuration)
        edges = tuple(
            (source, target)
            for source, target, weight in system.lattice.weighted_edges
            if source != target and weight != 0.0
        )
        exchange = ExcitationPreservingAnsatz(
            system.num_sites,
            reps=reps,
            mode=mode,
            entanglement=EntanglementTopology.custom(list(edges)),
            skip_unentangled_qubits=True,
            skip_final_rotation_layer=reps == 0,
        )

        self._system = system
        self._configuration = configuration
        self._initial_circuit = preparation.build_circuit("lattice_reference")
        self._inner_exchange = exchange
        self.validate()

    @property
    def system(self) -> LatticeSystem:
        """Return the fixed-sector spin system."""
        return self._system

    @property
    def configuration(self) -> SpinLatticeConfiguration:
        """Return the fixed reference spin configuration."""
        return self._configuration

    def _validate_configuration(self) -> None:
        LatticeReferenceStatePreparation(
            self._system,
            self._configuration,
        ).validate()
        self._initial_circuit.validate()
        if self._initial_circuit.used_symbols:
            raise ValueError("initial lattice reference must be fixed")
        self._inner_exchange.validate()
        if self._inner_exchange.num_qubits() != self._initial_circuit.num_qubits:
            raise ValueError("exchange circuit width is inconsistent")

    def _build_circuit(self, prefix: str) -> Circuit:
        circuit = copy.copy(self._initial_circuit)
        exchange = self._inner_exchange.build_circuit(f"{prefix}_theta")
        circuit.compose(exchange, circuit.qubits)
        return circuit

    def _parameter_count(self) -> int:
        return self._inner_exchange.num_parameters()

    def _qubit_count(self) -> int:
        return self._initial_circuit.num_qubits


__all__ = ["SpinExchangeAnsatz"]
