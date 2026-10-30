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

"""Validation tests for immutable vibrational algorithm results."""

from dataclasses import FrozenInstanceError

import pytest
from cqlib import Circuit

from cqlib_algorithms.algorithms.variational import VQEResult
from cqlib_algorithms.algorithms.vibration import (
    VibrationalGroundStateResult,
    VibrationalMeasurements,
)
from cqlib_algorithms.extension.vibration import (
    ModalIntegrals,
    ModalSpace,
    VibrationalEnergy,
    VibrationalSystem,
)
from cqlib_algorithms.protocols import (
    EstimatorResult,
    OptimizerResult,
    OptimizerStatus,
)


def _system() -> VibrationalSystem:
    space = ModalSpace((2, 3))
    return VibrationalSystem(
        VibrationalEnergy(ModalIntegrals.from_terms(space, (), constant=0.25))
    )


def _vqe_result(num_qubits: int) -> VQEResult:
    optimizer = OptimizerResult((), (), -0.5, 0, OptimizerStatus.NO_PARAMETERS)
    return VQEResult(-0.5, {}, Circuit(num_qubits), optimizer)


def _measurements(counts: tuple[int, ...] = (2, 3)) -> VibrationalMeasurements:
    zero = EstimatorResult(0.0, 0.0)
    return VibrationalMeasurements(
        tuple((zero,) * count for count in counts),
        (zero,) * len(counts),
        (zero,) * len(counts),
    )


def test_result_exposes_energy_decomposition_and_is_frozen():
    system = _system()
    result = VibrationalGroundStateResult(
        system,
        _vqe_result(system.num_qubits),
        _measurements(),
    )

    assert result.variational_energy == -0.5
    assert result.hamiltonian_constant == 0.25
    assert result.nonconstant_energy == -0.75
    with pytest.raises(FrozenInstanceError):
        result.system = system  # type: ignore[misc]


def test_measurements_validate_container_types_elements_and_dimensions():
    zero = EstimatorResult(0.0)
    with pytest.raises(TypeError, match="modal_occupations must be a tuple"):
        VibrationalMeasurements([(zero,)], (zero,), (zero,))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"modal_occupations\[0\] must be a tuple"):
        VibrationalMeasurements(([zero],), (zero,), (zero,))  # type: ignore[list-item]
    with pytest.raises(TypeError, match="EstimatorResult"):
        VibrationalMeasurements(((object(),),), (zero,), (zero,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rows must not be empty"):
        VibrationalMeasurements(((),), (zero,), (zero,))
    with pytest.raises(ValueError, match="must not be empty"):
        VibrationalMeasurements((), (), ())
    with pytest.raises(TypeError, match="mode_occupations must be a tuple"):
        VibrationalMeasurements(((zero,),), [zero], (zero,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one result"):
        VibrationalMeasurements(((zero,),), (), (zero,))
    with pytest.raises(TypeError, match="mode_leakages must be a tuple"):
        VibrationalMeasurements(((zero,),), (zero,), [zero])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mode_leakages must contain one result"):
        VibrationalMeasurements(((zero,),), (zero,), ())


def test_result_validates_system_vqe_measurements_width_and_shape():
    system = _system()
    measurements = _measurements()
    with pytest.raises(TypeError, match="system"):
        VibrationalGroundStateResult(  # type: ignore[arg-type]
            object(), _vqe_result(system.num_qubits), measurements
        )
    with pytest.raises(TypeError, match="vqe_result"):
        VibrationalGroundStateResult(system, object(), measurements)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="measurements"):
        VibrationalGroundStateResult(  # type: ignore[arg-type]
            system, _vqe_result(system.num_qubits), object()
        )
    with pytest.raises(ValueError, match="width"):
        VibrationalGroundStateResult(system, _vqe_result(1), measurements)
    with pytest.raises(ValueError, match="modal count"):
        VibrationalGroundStateResult(
            system,
            _vqe_result(system.num_qubits),
            _measurements((2, 2)),
        )
