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

"""Validation tests for immutable lattice algorithm results."""

from dataclasses import FrozenInstanceError

import pytest
from cqlib import Circuit

from cqlib_algorithms.algorithms.lattice import (
    FermionicLatticeMeasurements,
    LatticeGroundStateResult,
    SpinCorrelationMeasurement,
    SpinLatticeMeasurements,
)
from cqlib_algorithms.algorithms.variational import VQEResult
from cqlib_algorithms.extension.lattice import (
    FermiHubbardModel,
    FermionicSector,
    HeisenbergModel,
    LatticeSystem,
    LineLattice,
)
from cqlib_algorithms.protocols import (
    EstimatorResult,
    OptimizerResult,
    OptimizerStatus,
)


def _vqe_result(num_qubits: int) -> VQEResult:
    optimizer_result = OptimizerResult(
        (),
        (),
        -0.5,
        0,
        OptimizerStatus.NO_PARAMETERS,
    )
    return VQEResult(-0.5, {}, Circuit(num_qubits), optimizer_result)


def _fermionic_measurements(num_sites: int = 2) -> FermionicLatticeMeasurements:
    zero = EstimatorResult(0.0, 0.0)
    return FermionicLatticeMeasurements(
        zero,
        zero,
        zero,
        zero,
        (zero,) * num_sites,
        (zero,) * num_sites,
    )


def _spin_measurements(num_sites: int = 2) -> SpinLatticeMeasurements:
    zero = EstimatorResult(0.0, 0.0)
    correlations = (
        (SpinCorrelationMeasurement(0, 1, "Z", zero),) if num_sites > 1 else ()
    )
    return SpinLatticeMeasurements(
        zero,
        (zero,) * num_sites,
        zero,
        correlations,
    )


def test_result_exposes_variational_energy_and_is_frozen():
    system = LatticeSystem(
        FermiHubbardModel(LineLattice(2)),
        FermionicSector(2, 1, 1),
    )
    result = LatticeGroundStateResult(
        system,
        _vqe_result(4),
        _fermionic_measurements(),
    )

    assert result.variational_energy == -0.5
    with pytest.raises(FrozenInstanceError):
        result.system = system  # type: ignore[misc]


def test_result_requires_family_and_width_consistency():
    fermionic_system = LatticeSystem(
        FermiHubbardModel(LineLattice(2)),
        FermionicSector(2, 1, 1),
    )
    spin_system = LatticeSystem(HeisenbergModel(LineLattice(2)))

    with pytest.raises(TypeError, match="FermionicLatticeMeasurements"):
        LatticeGroundStateResult(
            fermionic_system,
            _vqe_result(4),
            _spin_measurements(),
        )
    with pytest.raises(TypeError, match="SpinLatticeMeasurements"):
        LatticeGroundStateResult(
            spin_system,
            _vqe_result(2),
            _fermionic_measurements(),
        )
    with pytest.raises(ValueError, match="width"):
        LatticeGroundStateResult(
            fermionic_system,
            _vqe_result(2),
            _fermionic_measurements(),
        )
    with pytest.raises(ValueError, match="one result per site"):
        LatticeGroundStateResult(
            fermionic_system,
            _vqe_result(4),
            _fermionic_measurements(1),
        )


def test_measurement_values_validate_shape_type_and_correlation_keys():
    zero = EstimatorResult(0.0)
    with pytest.raises(TypeError, match="total_particle_number"):
        FermionicLatticeMeasurements(
            object(),  # type: ignore[arg-type]
            zero,
            zero,
            zero,
            (zero,),
            (zero,),
        )
    with pytest.raises(TypeError, match="tuple"):
        FermionicLatticeMeasurements(
            zero,
            zero,
            zero,
            zero,
            [zero],  # type: ignore[arg-type]
            (zero,),
        )
    with pytest.raises(ValueError, match="must match"):
        FermionicLatticeMeasurements(zero, zero, zero, zero, (zero,), ())
    with pytest.raises(ValueError, match="must not be empty"):
        SpinLatticeMeasurements(zero, (), zero, ())
    with pytest.raises(TypeError, match="integers"):
        SpinCorrelationMeasurement(True, 1, "Z", zero)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        SpinCorrelationMeasurement(-1, 1, "Z", zero)
    with pytest.raises(ValueError, match="source < target"):
        SpinCorrelationMeasurement(1, 0, "Z", zero)
    with pytest.raises(TypeError, match="axis"):
        SpinCorrelationMeasurement(0, 1, 1, zero)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="axis"):
        SpinCorrelationMeasurement(0, 1, "A", zero)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="correlation result"):
        SpinCorrelationMeasurement(0, 1, "Z", object())  # type: ignore[arg-type]

    duplicate = SpinCorrelationMeasurement(0, 1, "Z", zero)
    with pytest.raises(ValueError, match="duplicate"):
        SpinLatticeMeasurements(zero, (zero, zero), zero, (duplicate, duplicate))
    with pytest.raises(ValueError, match="endpoint"):
        SpinLatticeMeasurements(
            zero,
            (zero, zero),
            zero,
            (SpinCorrelationMeasurement(0, 2, "Z", zero),),
        )
    with pytest.raises(TypeError, match="correlations must be a tuple"):
        SpinLatticeMeasurements(zero, (zero, zero), zero, [])  # type: ignore[arg-type]
