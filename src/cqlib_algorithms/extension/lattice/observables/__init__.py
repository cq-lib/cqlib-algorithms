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

"""Physical observable factories for supported lattice models."""

from .fermionic import (
    DoubleOccupation,
    FermionicMagnetization,
    SiteOccupation,
    SpinDownParticleNumber,
    SpinUpParticleNumber,
    TotalParticleNumber,
)
from .spin import (
    SiteMagnetization,
    SpinCorrelation,
    SpinInteractionEnergy,
    TotalMagnetization,
)

__all__ = [
    "DoubleOccupation",
    "FermionicMagnetization",
    "SiteMagnetization",
    "SiteOccupation",
    "SpinCorrelation",
    "SpinDownParticleNumber",
    "SpinInteractionEnergy",
    "SpinUpParticleNumber",
    "TotalMagnetization",
    "TotalParticleNumber",
]
