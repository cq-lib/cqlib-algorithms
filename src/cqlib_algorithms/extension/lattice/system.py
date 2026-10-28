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

"""Validated physical input for lattice algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypeAlias

from ._fermionic_layout import fermionic_mode_layout
from .fermi_hubbard import FermiHubbardModel
from .heisenberg import HeisenbergModel
from .ising import IsingModel
from .sectors import FermionicSector, SpinMagnetizationSector
from .topology import Lattice

LatticeModel: TypeAlias = FermiHubbardModel | HeisenbergModel | IsingModel
LatticeSector: TypeAlias = FermionicSector | SpinMagnetizationSector


class LatticeModelKind(Enum):
    """Supported physical lattice-model families."""

    FERMI_HUBBARD = "fermi_hubbard"
    HEISENBERG = "heisenberg"
    ISING = "ising"


def _spin_model_conserves_z_magnetization(
    model: HeisenbergModel | IsingModel,
) -> bool:
    if isinstance(model, IsingModel):
        return all(field == 0.0 for field in model.transverse_fields)

    jx, jy, _ = model.couplings
    has_nonzero_interaction = any(
        source != target and weight != 0.0
        for source, target, weight in model.lattice.weighted_edges
    )
    exchange_conserves = jx == jy or not has_nonzero_interaction
    fields_conserve = all(
        hx == 0.0 and hy == 0.0 for hx, hy, _ in model.external_fields
    )
    return exchange_conserves and fields_conserve


@dataclass(frozen=True, slots=True)
class LatticeSystem:
    """A lattice model paired with an optional conserved physical sector.

    A Fermi--Hubbard system always requires a spin-resolved particle-number
    sector. Spin models may omit a sector; a fixed magnetization sector is
    accepted only when the model commutes with total :math:`S_z`.

    This value contains physical problem data only. Qubit mappings, mapped
    Hamiltonians, initial states, ansatze, observables and solver results are
    deliberately selected or constructed by later workflow stages.

    Args:
        model: Fermi--Hubbard, Heisenberg, or Ising model.
        sector: Required fermionic sector or optional spin magnetization sector.

    Raises:
        TypeError: If model and sector types are incompatible.
        ValueError: If widths disagree or the selected spin sector is not conserved.
    """

    model: LatticeModel
    sector: LatticeSector | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.model,
            (FermiHubbardModel, HeisenbergModel, IsingModel),
        ):
            raise TypeError(
                "model must be a FermiHubbardModel, HeisenbergModel, or IsingModel"
            )

        if isinstance(self.model, FermiHubbardModel):
            if not isinstance(self.sector, FermionicSector):
                raise TypeError("FermiHubbardModel requires a FermionicSector")
        elif self.sector is not None and not isinstance(
            self.sector,
            SpinMagnetizationSector,
        ):
            raise TypeError("spin models require a SpinMagnetizationSector or None")

        if self.sector is not None and self.sector.num_sites != self.num_sites:
            raise ValueError("sector and model must use the same number of sites")
        if isinstance(self.sector, SpinMagnetizationSector) and not (
            self.conserves_z_magnetization
        ):
            raise ValueError(
                "a spin magnetization sector requires a model that conserves total S_z"
            )

    @property
    def lattice(self) -> Lattice:
        """Return the immutable lattice topology owned by the model."""
        return self.model.lattice

    @property
    def num_sites(self) -> int:
        """Return the number of lattice sites."""
        return self.model.lattice.num_nodes

    @property
    def model_kind(self) -> LatticeModelKind:
        """Return the model family without storing redundant type metadata."""
        if isinstance(self.model, FermiHubbardModel):
            return LatticeModelKind.FERMI_HUBBARD
        if isinstance(self.model, HeisenbergModel):
            return LatticeModelKind.HEISENBERG
        return LatticeModelKind.ISING

    @property
    def conserves_z_magnetization(self) -> bool:
        """Return whether total spin magnetization along z is conserved."""
        if isinstance(self.model, FermiHubbardModel):
            return True
        return _spin_model_conserves_z_magnetization(self.model)

    @property
    def fermionic_mode_layout(
        self,
    ) -> tuple[tuple[int, Literal["up", "down"]], ...] | None:
        """Return ``(site, spin)`` for each interleaved fermionic mode.

        Spin models have one local spin degree of freedom per site and return
        ``None`` because they do not use fermionic modes.
        """
        if not isinstance(self.model, FermiHubbardModel):
            return None
        return fermionic_mode_layout(self.num_sites)


__all__ = [
    "LatticeModel",
    "LatticeModelKind",
    "LatticeSector",
    "LatticeSystem",
]
