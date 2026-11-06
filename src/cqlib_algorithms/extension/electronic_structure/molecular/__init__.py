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

"""Molecular inputs, driver data, and active-space selection."""

from .active_space import (
    ActiveSpace,
    ActiveSpaceResult,
    SpinResolvedActiveSpace,
    SpinResolvedActiveSpaceResult,
    transform_active_space,
    transform_spin_resolved_active_space,
)
from .driver_data import ElectronicStructureData, SpinResolvedElectronicStructureData
from .pyscf_adapter import PySCFDriver, SCFReference
from .specification import Atom, MolecularSpecification

__all__ = [
    "ActiveSpace",
    "ActiveSpaceResult",
    "Atom",
    "ElectronicStructureData",
    "MolecularSpecification",
    "PySCFDriver",
    "SCFReference",
    "SpinResolvedActiveSpace",
    "SpinResolvedActiveSpaceResult",
    "SpinResolvedElectronicStructureData",
    "transform_active_space",
    "transform_spin_resolved_active_space",
]
