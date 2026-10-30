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

"""Public API checks for the chemistry extension."""

from importlib import import_module


def test_chemistry_components_are_exported_from_extension_packages():
    extension = import_module("cqlib_algorithms.extension")
    chemistry = import_module("cqlib_algorithms.extension.chemistry")
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
    assert set(extension.__all__) == expected
    assert set(chemistry.__all__) == expected
    for name in expected:
        assert getattr(extension, name) is getattr(chemistry, name)
