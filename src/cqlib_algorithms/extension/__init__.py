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
    from . import chemistry, lattice, many_body, vibration
    from .chemistry import (
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
    "chemistry": (".chemistry", None),
    "lattice": (".lattice", None),
    "many_body": (".many_body", None),
    "vibration": (".vibration", None),
    "ActiveSpace": (".chemistry", "ActiveSpace"),
    "ActiveSpaceResult": (".chemistry", "ActiveSpaceResult"),
    "AngularMomentum": (".chemistry", "AngularMomentum"),
    "Atom": (".chemistry", "Atom"),
    "BogoliubovTransform": (".chemistry", "BogoliubovTransform"),
    "ElectronicEnergy": (".chemistry", "ElectronicEnergy"),
    "ElectronicIntegrals": (".chemistry", "ElectronicIntegrals"),
    "ElectronicOccupation": (".chemistry", "ElectronicOccupation"),
    "ElectronicStructureData": (".chemistry", "ElectronicStructureData"),
    "ElectronicSystem": (".chemistry", "ElectronicSystem"),
    "FermionicExcitation": (".chemistry", "FermionicExcitation"),
    "GroupedFermionicGenerator": (".chemistry", "GroupedFermionicGenerator"),
    "Magnetization": (".chemistry", "Magnetization"),
    "MolecularSpecification": (".chemistry", "MolecularSpecification"),
    "ParticleNumber": (".chemistry", "ParticleNumber"),
    "PySCFDriver": (".chemistry", "PySCFDriver"),
    "QuadraticFermionicHamiltonian": (
        ".chemistry",
        "QuadraticFermionicHamiltonian",
    ),
    "SCFReference": (".chemistry", "SCFReference"),
    "SpinResolvedActiveSpace": (".chemistry", "SpinResolvedActiveSpace"),
    "SpinResolvedActiveSpaceResult": (
        ".chemistry",
        "SpinResolvedActiveSpaceResult",
    ),
    "SpinResolvedElectronicIntegrals": (
        ".chemistry",
        "SpinResolvedElectronicIntegrals",
    ),
    "SpinResolvedElectronicStructureData": (
        ".chemistry",
        "SpinResolvedElectronicStructureData",
    ),
    "generate_double_excitations": (".chemistry", "generate_double_excitations"),
    "generate_paired_double_excitations": (
        ".chemistry",
        "generate_paired_double_excitations",
    ),
    "generate_puccsd_generators": (".chemistry", "generate_puccsd_generators"),
    "generate_single_excitations": (".chemistry", "generate_single_excitations"),
    "generate_uccsd_excitations": (".chemistry", "generate_uccsd_excitations"),
    "hartree_fock_initial_point": (".chemistry", "hartree_fock_initial_point"),
    "mp2_initial_point": (".chemistry", "mp2_initial_point"),
    "transform_active_space": (".chemistry", "transform_active_space"),
    "transform_spin_resolved_active_space": (
        ".chemistry",
        "transform_spin_resolved_active_space",
    ),
}

# Domain namespaces intentionally precede compatibility aliases, whose prior
# relative order remains stable for callers inspecting the public facade.
__all__ = [  # noqa: RUF022
    "chemistry",
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
