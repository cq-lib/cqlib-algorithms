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

"""Public API checks for the electronic-structure extension."""

from importlib import import_module


def test_electronic_structure_components_are_exported_from_extension_packages():
    extension = import_module("cqlib_algorithms.extension")
    electronic_structure = import_module(
        "cqlib_algorithms.extension.electronic_structure"
    )
    expected = {
        "ActiveSpace",
        "ActiveSpaceResult",
        "AngularMomentum",
        "Atom",
        "BogoliubovTransform",
        "ElectronicEnergy",
        "ElectronicIntegrals",
        "ElectronicOccupation",
        "ElectronicSystem",
        "ElectronicStructureData",
        "FermionicExcitation",
        "GroupedFermionicGenerator",
        "MolecularSpecification",
        "Magnetization",
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
    }
    assert extension.__all__[:4] == [
        "electronic_structure",
        "lattice",
        "many_body",
        "vibration",
    ]
    assert set(extension.__all__[4:]) == expected
    assert set(electronic_structure.__all__) == expected
    for name in expected:
        assert getattr(extension, name) is getattr(electronic_structure, name)
