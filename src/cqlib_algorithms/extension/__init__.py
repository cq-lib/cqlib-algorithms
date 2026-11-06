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

"""Domain extensions supporting quantum algorithm construction."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import electronic_structure, lattice, many_body, vibration
    from .electronic_structure import (
        ActiveSpace,
        ActiveSpaceResult,
        AngularMomentum,
        Atom,
        BogoliubovTransform,
        ElectronicEnergy,
        ElectronicIntegrals,
        ElectronicOccupation,
        ElectronicStructureData,
        ElectronicSystem,
        FermionicExcitation,
        GroupedFermionicGenerator,
        Magnetization,
        MolecularSpecification,
        ParticleNumber,
        PySCFDriver,
        QuadraticFermionicHamiltonian,
        SCFReference,
        SpinResolvedActiveSpace,
        SpinResolvedActiveSpaceResult,
        SpinResolvedElectronicIntegrals,
        SpinResolvedElectronicStructureData,
        generate_double_excitations,
        generate_paired_double_excitations,
        generate_puccsd_generators,
        generate_single_excitations,
        generate_uccsd_excitations,
        hartree_fock_initial_point,
        mp2_initial_point,
        transform_active_space,
        transform_spin_resolved_active_space,
    )

_EXPORTS: dict[str, tuple[str, str | None]] = {
    "electronic_structure": (".electronic_structure", None),
    "lattice": (".lattice", None),
    "many_body": (".many_body", None),
    "vibration": (".vibration", None),
    "ActiveSpace": (".electronic_structure", "ActiveSpace"),
    "ActiveSpaceResult": (".electronic_structure", "ActiveSpaceResult"),
    "AngularMomentum": (".electronic_structure", "AngularMomentum"),
    "Atom": (".electronic_structure", "Atom"),
    "BogoliubovTransform": (".electronic_structure", "BogoliubovTransform"),
    "ElectronicEnergy": (".electronic_structure", "ElectronicEnergy"),
    "ElectronicIntegrals": (".electronic_structure", "ElectronicIntegrals"),
    "ElectronicOccupation": (".electronic_structure", "ElectronicOccupation"),
    "ElectronicStructureData": (".electronic_structure", "ElectronicStructureData"),
    "ElectronicSystem": (".electronic_structure", "ElectronicSystem"),
    "FermionicExcitation": (".electronic_structure", "FermionicExcitation"),
    "GroupedFermionicGenerator": (".electronic_structure", "GroupedFermionicGenerator"),
    "Magnetization": (".electronic_structure", "Magnetization"),
    "MolecularSpecification": (".electronic_structure", "MolecularSpecification"),
    "ParticleNumber": (".electronic_structure", "ParticleNumber"),
    "PySCFDriver": (".electronic_structure", "PySCFDriver"),
    "QuadraticFermionicHamiltonian": (
        ".electronic_structure",
        "QuadraticFermionicHamiltonian",
    ),
    "SCFReference": (".electronic_structure", "SCFReference"),
    "SpinResolvedActiveSpace": (".electronic_structure", "SpinResolvedActiveSpace"),
    "SpinResolvedActiveSpaceResult": (
        ".electronic_structure",
        "SpinResolvedActiveSpaceResult",
    ),
    "SpinResolvedElectronicIntegrals": (
        ".electronic_structure",
        "SpinResolvedElectronicIntegrals",
    ),
    "SpinResolvedElectronicStructureData": (
        ".electronic_structure",
        "SpinResolvedElectronicStructureData",
    ),
    "generate_double_excitations": (
        ".electronic_structure",
        "generate_double_excitations",
    ),
    "generate_paired_double_excitations": (
        ".electronic_structure",
        "generate_paired_double_excitations",
    ),
    "generate_puccsd_generators": (
        ".electronic_structure",
        "generate_puccsd_generators",
    ),
    "generate_single_excitations": (
        ".electronic_structure",
        "generate_single_excitations",
    ),
    "generate_uccsd_excitations": (
        ".electronic_structure",
        "generate_uccsd_excitations",
    ),
    "hartree_fock_initial_point": (
        ".electronic_structure",
        "hartree_fock_initial_point",
    ),
    "mp2_initial_point": (".electronic_structure", "mp2_initial_point"),
    "transform_active_space": (".electronic_structure", "transform_active_space"),
    "transform_spin_resolved_active_space": (
        ".electronic_structure",
        "transform_spin_resolved_active_space",
    ),
}

# Domain namespaces intentionally precede compatibility aliases, whose prior
# relative order remains stable for callers inspecting the public facade.
__all__ = [  # noqa: RUF022
    "electronic_structure",
    "lattice",
    "many_body",
    "vibration",
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


def __getattr__(name: str) -> object:
    """Resolve and cache a public extension namespace or compatibility alias."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value: object = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module globals together with lazily exported names."""
    return sorted(set(globals()) | set(__all__))
