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

"""Immutable results for variational vibrational ground-state workflows."""

from __future__ import annotations

from dataclasses import dataclass

from ...extension.vibration import VibrationalSystem
from ...protocols.estimator import EstimatorResult
from ..variational import VQEResult


def _estimator_results(
    values: tuple[EstimatorResult, ...],
    *,
    name: str,
) -> tuple[EstimatorResult, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    if any(not isinstance(value, EstimatorResult) for value in values):
        raise TypeError(f"{name} must contain only EstimatorResult values")
    return tuple(values)


@dataclass(frozen=True, slots=True)
class VibrationalMeasurements:
    """Terminal modal populations, mode occupations, and strict leakages.

    Args:
        modal_occupations: Per-mode rows of modal projector estimates.
        mode_occupations: One total-occupation estimate per mode.
        mode_leakages: One strict squared one-hot constraint per mode.

    Raises:
        TypeError: If a result or container has the wrong type.
        ValueError: If rows are empty or mode counts disagree.
    """

    modal_occupations: tuple[tuple[EstimatorResult, ...], ...]
    mode_occupations: tuple[EstimatorResult, ...]
    mode_leakages: tuple[EstimatorResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.modal_occupations, tuple):
            raise TypeError("modal_occupations must be a tuple")
        rows: list[tuple[EstimatorResult, ...]] = []
        for mode, row in enumerate(self.modal_occupations):
            copied = _estimator_results(
                row,
                name=f"modal_occupations[{mode}]",
            )
            if not copied:
                raise ValueError("modal occupation rows must not be empty")
            rows.append(copied)
        if not rows:
            raise ValueError("modal_occupations must not be empty")
        totals = _estimator_results(
            self.mode_occupations,
            name="mode_occupations",
        )
        if len(totals) != len(rows):
            raise ValueError(
                "mode_occupations must contain one result per modal-occupation row"
            )
        leakages = _estimator_results(
            self.mode_leakages,
            name="mode_leakages",
        )
        if len(leakages) != len(rows):
            raise ValueError(
                "mode_leakages must contain one result per modal-occupation row"
            )
        object.__setattr__(self, "modal_occupations", tuple(rows))
        object.__setattr__(self, "mode_occupations", totals)
        object.__setattr__(self, "mode_leakages", leakages)


@dataclass(frozen=True, slots=True)
class VibrationalGroundStateResult:
    """A variational vibrational energy with physical-sector diagnostics.

    Args:
        system: Vibrational system solved by the workflow.
        vqe_result: Underlying variational eigensolver result.
        measurements: Modal occupation and one-hot leakage diagnostics.

    Raises:
        TypeError: If a result component has the wrong type.
        ValueError: If circuit width or modal result dimensions disagree.
    """

    system: VibrationalSystem
    vqe_result: VQEResult
    measurements: VibrationalMeasurements

    def __post_init__(self) -> None:
        if not isinstance(self.system, VibrationalSystem):
            raise TypeError("system must be a VibrationalSystem")
        if not isinstance(self.vqe_result, VQEResult):
            raise TypeError("vqe_result must be a VQEResult")
        if not isinstance(self.measurements, VibrationalMeasurements):
            raise TypeError("measurements must be VibrationalMeasurements")
        if self.vqe_result.optimal_circuit.num_qubits != self.system.num_qubits:
            raise ValueError("optimal circuit width must match the vibrational system")
        expected = self.system.modal_space.num_modals
        if tuple(len(row) for row in self.measurements.modal_occupations) != expected:
            raise ValueError(
                "modal_occupations must match the modal count of every mode"
            )
        if len(self.measurements.mode_occupations) != self.system.num_modes:
            raise ValueError("mode_occupations must contain one result per mode")
        if len(self.measurements.mode_leakages) != self.system.num_modes:
            raise ValueError("mode_leakages must contain one result per mode")

    @property
    def variational_energy(self) -> float:
        """Return the final variational expectation including the constant."""
        return self.vqe_result.eigenvalue

    @property
    def hamiltonian_constant(self) -> float:
        """Return the scalar term stored in the vibrational integrals."""
        return self.system.energy.integrals.constant

    @property
    def nonconstant_energy(self) -> float:
        """Return the variational expectation after removing the scalar term."""
        return self.variational_energy - self.hamiltonian_constant


__all__ = ["VibrationalGroundStateResult", "VibrationalMeasurements"]
