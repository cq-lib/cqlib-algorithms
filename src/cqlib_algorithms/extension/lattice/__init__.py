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

"""Reusable lattice models, configurations, sectors, and observables."""

from .configurations import (
    FermionicLatticeConfiguration,
    LatticeConfiguration,
    SpinLatticeConfiguration,
)
from .fermi_hubbard import FermiHubbardModel
from .heisenberg import HeisenbergModel
from .ising import IsingModel
from .observables import (
    DoubleOccupation,
    FermionicMagnetization,
    SiteMagnetization,
    SiteOccupation,
    SpinCorrelation,
    SpinDownParticleNumber,
    SpinInteractionEnergy,
    SpinUpParticleNumber,
    TotalMagnetization,
    TotalParticleNumber,
)
from .sectors import FermionicSector, SpinMagnetizationSector
from .system import LatticeModel, LatticeModelKind, LatticeSector, LatticeSystem
from .topology import (
    BoundaryCondition,
    HyperCubicLattice,
    Lattice,
    LineLattice,
    SquareLattice,
    WeightedEdge,
)

__all__ = [
    "BoundaryCondition",
    "DoubleOccupation",
    "FermiHubbardModel",
    "FermionicLatticeConfiguration",
    "FermionicMagnetization",
    "FermionicSector",
    "HeisenbergModel",
    "HyperCubicLattice",
    "IsingModel",
    "Lattice",
    "LatticeConfiguration",
    "LatticeModel",
    "LatticeModelKind",
    "LatticeSector",
    "LatticeSystem",
    "LineLattice",
    "SiteMagnetization",
    "SiteOccupation",
    "SpinCorrelation",
    "SpinDownParticleNumber",
    "SpinInteractionEnergy",
    "SpinLatticeConfiguration",
    "SpinMagnetizationSector",
    "SpinUpParticleNumber",
    "SquareLattice",
    "TotalMagnetization",
    "TotalParticleNumber",
    "WeightedEdge",
]
