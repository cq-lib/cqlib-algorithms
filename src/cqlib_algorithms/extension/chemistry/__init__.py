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

"""Chemistry-domain data, models, and fermionic conversion utilities."""

from .electronic import (
    AngularMomentum,
    ElectronicEnergy,
    ElectronicIntegrals,
    ElectronicOccupation,
    ElectronicSystem,
    Magnetization,
    ParticleNumber,
    SpinResolvedElectronicIntegrals,
    hartree_fock_initial_point,
    mp2_initial_point,
)
from .fermionic import (
    BogoliubovTransform,
    FermionicExcitation,
    GroupedFermionicGenerator,
    QuadraticFermionicHamiltonian,
    generate_double_excitations,
    generate_paired_double_excitations,
    generate_puccsd_generators,
    generate_single_excitations,
    generate_uccsd_excitations,
)
from .molecular import (
    ActiveSpace,
    ActiveSpaceResult,
    Atom,
    ElectronicStructureData,
    MolecularSpecification,
    PySCFDriver,
    SCFReference,
    SpinResolvedActiveSpace,
    SpinResolvedActiveSpaceResult,
    SpinResolvedElectronicStructureData,
    transform_active_space,
    transform_spin_resolved_active_space,
)

__all__ = [
    "ActiveSpace",
    "ActiveSpaceResult",
    "AngularMomentum",
    "Atom",
    "BogoliubovTransform",
    "ElectronicEnergy",
    "ElectronicIntegrals",
    "ElectronicOccupation",
    "ElectronicStructureData",
    "ElectronicSystem",
    "FermionicExcitation",
    "GroupedFermionicGenerator",
    "Magnetization",
    "MolecularSpecification",
    "ParticleNumber",
    "PySCFDriver",
    "QuadraticFermionicHamiltonian",
    "SCFReference",
    "SpinResolvedActiveSpace",
    "SpinResolvedActiveSpaceResult",
    "SpinResolvedElectronicIntegrals",
    "SpinResolvedElectronicStructureData",
    "generate_double_excitations",
    "generate_paired_double_excitations",
    "generate_puccsd_generators",
    "generate_single_excitations",
    "generate_uccsd_excitations",
    "hartree_fock_initial_point",
    "mp2_initial_point",
    "transform_active_space",
    "transform_spin_resolved_active_space",
]
