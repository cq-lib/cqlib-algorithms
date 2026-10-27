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

"""Public API checks for lattice topology components."""

from importlib import import_module


def test_lattice_public_api_is_explicit_and_excludes_deferred_topologies():
    lattice = import_module("cqlib_algorithms.extension.lattice")
    expected = {
        "BoundaryCondition",
        "DoubleOccupation",
        "FermionicLatticeConfiguration",
        "FermionicMagnetization",
        "FermionicSector",
        "FermiHubbardModel",
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
        "SquareLattice",
        "SpinCorrelation",
        "SpinDownParticleNumber",
        "SpinInteractionEnergy",
        "SpinLatticeConfiguration",
        "SpinMagnetizationSector",
        "SpinUpParticleNumber",
        "TotalMagnetization",
        "TotalParticleNumber",
        "WeightedEdge",
    }

    assert set(lattice.__all__) == expected
    for name in expected:
        assert getattr(lattice, name) is not None
    assert not hasattr(lattice, "TriangularLattice")
