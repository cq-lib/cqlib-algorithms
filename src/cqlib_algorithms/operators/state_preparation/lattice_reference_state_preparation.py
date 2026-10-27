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

"""Computational-basis reference states for lattice systems."""

from __future__ import annotations

from typing import final

from cqlib.circuit import Circuit

from ...extension.lattice import (
    FermiHubbardModel,
    FermionicLatticeConfiguration,
    LatticeConfiguration,
    LatticeSystem,
    SpinLatticeConfiguration,
)
from ...protocols.state_preparation import StatePreparation
from .basis_state_initialization import BasisStateInitialization


def _validate_system_configuration(
    system: LatticeSystem,
    configuration: LatticeConfiguration,
) -> None:
    if not isinstance(system, LatticeSystem):
        raise TypeError("system must be a LatticeSystem")
    if isinstance(system.model, FermiHubbardModel):
        if not isinstance(configuration, FermionicLatticeConfiguration):
            raise TypeError(
                "a FermiHubbardModel requires a FermionicLatticeConfiguration"
            )
    elif not isinstance(configuration, SpinLatticeConfiguration):
        raise TypeError("a spin model requires a SpinLatticeConfiguration")

    if configuration.num_sites != system.num_sites:
        raise ValueError("configuration and system must use the same number of sites")
    if system.sector is not None and configuration.sector != system.sector:
        raise ValueError("configuration must belong to the system sector")


@final
class LatticeReferenceStatePreparation(StatePreparation):
    """Prepare one validated lattice occupation or spin configuration.

    Fermionic systems use the model's interleaved Jordan--Wigner occupation
    layout. Spin systems use ``|0>`` for spin up and ``|1>`` for spin down.
    The component contains no variational parameters.

    Args:
        system: Lattice model and optional conserved sector.
        configuration: Compatible fermionic occupation or spin configuration.

    Raises:
        TypeError: If system and configuration types are incompatible.
        ValueError: If widths or physical sectors disagree.
    """

    __slots__ = ("_basis_state", "_configuration", "_system")

    def __init__(
        self,
        system: LatticeSystem,
        configuration: LatticeConfiguration,
    ) -> None:
        _validate_system_configuration(system, configuration)
        self._system = system
        self._configuration = configuration
        width = (
            system.model.num_modes
            if isinstance(system.model, FermiHubbardModel)
            else system.num_sites
        )
        self._basis_state = BasisStateInitialization(
            width,
            configuration.basis_index,
        )
        self.validate()

    @property
    def system(self) -> LatticeSystem:
        """Return the immutable physical system."""
        return self._system

    @property
    def configuration(self) -> LatticeConfiguration:
        """Return the immutable reference configuration."""
        return self._configuration

    def validate(self) -> None:
        """Validate the system, sector and delegated basis preparation."""
        _validate_system_configuration(self._system, self._configuration)
        self._basis_state.validate()

    def build_circuit(self, prefix: str) -> Circuit:
        """Return an independent computational-basis preparation circuit."""
        return self._basis_state.build_circuit(prefix)

    def num_parameters(self) -> int:
        """Return zero because the reference is fixed."""
        return 0

    def num_qubits(self) -> int:
        """Return the encoded register width."""
        return self._basis_state.num_qubits()


__all__ = ["LatticeReferenceStatePreparation"]
