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

"""Public namespace tests for lattice algorithms."""

from cqlib_algorithms import LatticeGroundState, LatticeGroundStateResult
from cqlib_algorithms.algorithms import (
    LatticeGroundState as AlgorithmLatticeGroundState,
)
from cqlib_algorithms.algorithms import (
    LatticeGroundStateResult as AlgorithmLatticeGroundStateResult,
)
from cqlib_algorithms.algorithms.lattice import (
    FermionicLatticeMeasurements,
    SpinCorrelationMeasurement,
    SpinLatticeMeasurements,
)
from cqlib_algorithms.algorithms.lattice import (
    LatticeGroundState as DomainLatticeGroundState,
)
from cqlib_algorithms.algorithms.lattice import (
    LatticeGroundStateResult as DomainLatticeGroundStateResult,
)


def test_lattice_algorithm_is_exported_from_public_namespaces():
    assert LatticeGroundState is AlgorithmLatticeGroundState is DomainLatticeGroundState
    assert (
        LatticeGroundStateResult
        is AlgorithmLatticeGroundStateResult
        is DomainLatticeGroundStateResult
    )


def test_domain_measurement_results_are_exported_from_lattice_namespace():
    assert FermionicLatticeMeasurements.__module__.endswith("lattice.results")
    assert SpinLatticeMeasurements.__module__.endswith("lattice.results")
    assert SpinCorrelationMeasurement.__module__.endswith("lattice.results")
