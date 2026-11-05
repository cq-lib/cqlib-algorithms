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

"""Immutable results for variational lattice ground-state workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from ...extension.lattice import FermiHubbardModel, LatticeSystem
from ...protocols.estimator import EstimatorResult
from .._ground_state_support import snapshot_estimator_results
from ..variational import VQEResult

SpinAxis: TypeAlias = Literal["X", "Y", "Z"]


def _estimator_result(value: EstimatorResult, *, name: str) -> EstimatorResult:
    if not isinstance(value, EstimatorResult):
        raise TypeError(f"{name} must be an EstimatorResult")
    return value


@dataclass(frozen=True, slots=True)
class FermionicLatticeMeasurements:
    """Terminal Fermi--Hubbard symmetry and local-density estimates.

    Args:
        total_particle_number: Total particle-number estimate.
        spin_up_particle_number: Spin-up particle-number estimate.
        spin_down_particle_number: Spin-down particle-number estimate.
        magnetization: Fermionic magnetization estimate.
        site_occupations: One total-occupation estimate per site.
        double_occupations: One double-occupation estimate per site.

    Raises:
        TypeError: If a result or container has the wrong type.
        ValueError: If site-resolved result counts disagree.
    """

    total_particle_number: EstimatorResult
    spin_up_particle_number: EstimatorResult
    spin_down_particle_number: EstimatorResult
    magnetization: EstimatorResult
    site_occupations: tuple[EstimatorResult, ...]
    double_occupations: tuple[EstimatorResult, ...]

    def __post_init__(self) -> None:
        for name in (
            "total_particle_number",
            "spin_up_particle_number",
            "spin_down_particle_number",
            "magnetization",
        ):
            _estimator_result(getattr(self, name), name=name)
        sites = snapshot_estimator_results(
            self.site_occupations,
            name="site_occupations",
        )
        doubles = snapshot_estimator_results(
            self.double_occupations,
            name="double_occupations",
        )
        if not sites:
            raise ValueError("site_occupations must not be empty")
        if len(doubles) != len(sites):
            raise ValueError(
                "double_occupations must match the number of site occupations"
            )
        object.__setattr__(self, "site_occupations", sites)
        object.__setattr__(self, "double_occupations", doubles)


@dataclass(frozen=True, slots=True)
class SpinCorrelationMeasurement:
    """One unconnected Pauli-axis correlation estimate on a lattice edge.

    Args:
        source: Lower endpoint index.
        target: Higher endpoint index.
        axis: Correlation axis ``X``, ``Y``, or ``Z``.
        result: Estimator result for the correlation observable.

    Raises:
        TypeError: If an endpoint, axis, or result has the wrong type.
        ValueError: If endpoints or the axis are invalid.
    """

    source: int
    target: int
    axis: SpinAxis
    result: EstimatorResult

    def __post_init__(self) -> None:
        if type(self.source) is not int or type(self.target) is not int:
            raise TypeError("correlation endpoints must be integers")
        if self.source < 0 or self.target < 0:
            raise ValueError("correlation endpoints must be non-negative")
        if self.source >= self.target:
            raise ValueError("correlation endpoints must satisfy source < target")
        if not isinstance(self.axis, str):
            raise TypeError("correlation axis must be a str")
        if self.axis not in ("X", "Y", "Z"):
            raise ValueError("correlation axis must be X, Y, or Z")
        _estimator_result(self.result, name="correlation result")


@dataclass(frozen=True, slots=True)
class SpinLatticeMeasurements:
    """Terminal magnetization, interaction-energy and edge correlations.

    Args:
        total_magnetization: Total spin-magnetization estimate.
        site_magnetizations: One magnetization estimate per site.
        interaction_energy: Two-site interaction-energy estimate.
        correlations: Unique edge- and axis-resolved correlation estimates.

    Raises:
        TypeError: If a result or container has the wrong type.
        ValueError: If site counts, endpoints, or correlation keys are invalid.
    """

    total_magnetization: EstimatorResult
    site_magnetizations: tuple[EstimatorResult, ...]
    interaction_energy: EstimatorResult
    correlations: tuple[SpinCorrelationMeasurement, ...]

    def __post_init__(self) -> None:
        _estimator_result(
            self.total_magnetization,
            name="total_magnetization",
        )
        _estimator_result(self.interaction_energy, name="interaction_energy")
        sites = snapshot_estimator_results(
            self.site_magnetizations,
            name="site_magnetizations",
        )
        if not sites:
            raise ValueError("site_magnetizations must not be empty")
        if not isinstance(self.correlations, tuple):
            raise TypeError("correlations must be a tuple")
        correlations = tuple(self.correlations)
        if any(
            not isinstance(item, SpinCorrelationMeasurement) for item in correlations
        ):
            raise TypeError(
                "correlations must contain only SpinCorrelationMeasurement values"
            )
        keys = {(item.source, item.target, item.axis) for item in correlations}
        if len(keys) != len(correlations):
            raise ValueError("correlations must not contain duplicate entries")
        if any(item.target >= len(sites) for item in correlations):
            raise ValueError("correlation endpoint exceeds the number of sites")
        object.__setattr__(self, "site_magnetizations", sites)
        object.__setattr__(self, "correlations", correlations)


LatticeMeasurements: TypeAlias = FermionicLatticeMeasurements | SpinLatticeMeasurements


@dataclass(frozen=True, slots=True)
class LatticeGroundStateResult:
    """A variational sector result with terminal lattice measurements.

    ``variational_energy`` is the final VQE objective in the declared physical
    sector or custom ansatz manifold. It is not an exact-ground-state or global
    optimizer guarantee.

    Args:
        system: Lattice model and sector solved by the workflow.
        vqe_result: Underlying variational eigensolver result.
        measurements: Model-specific terminal physical measurements.

    Raises:
        TypeError: If result and model types are incompatible.
        ValueError: If circuit widths or site-resolved result counts disagree.
    """

    system: LatticeSystem
    vqe_result: VQEResult
    measurements: LatticeMeasurements

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.system, LatticeSystem):
            raise TypeError("system must be a LatticeSystem")
        if not isinstance(self.vqe_result, VQEResult):
            raise TypeError("vqe_result must be a VQEResult")
        expected_width = (
            self.system.model.num_modes
            if isinstance(self.system.model, FermiHubbardModel)
            else self.system.num_sites
        )
        if self.vqe_result.optimal_circuit.num_qubits != expected_width:
            raise ValueError("optimal circuit width is inconsistent with the system")
        if isinstance(self.system.model, FermiHubbardModel):
            if not isinstance(self.measurements, FermionicLatticeMeasurements):
                raise TypeError(
                    "a FermiHubbardModel requires FermionicLatticeMeasurements"
                )
            if len(self.measurements.site_occupations) != self.system.num_sites:
                raise ValueError(
                    "fermionic measurements must contain one result per site"
                )
        else:
            if not isinstance(self.measurements, SpinLatticeMeasurements):
                raise TypeError("a spin model requires SpinLatticeMeasurements")
            if len(self.measurements.site_magnetizations) != self.system.num_sites:
                raise ValueError("spin measurements must contain one result per site")

    @property
    def variational_energy(self) -> float:
        """Return the final VQE energy without an exactness guarantee."""
        return self.vqe_result.eigenvalue


__all__ = [
    "FermionicLatticeMeasurements",
    "LatticeGroundStateResult",
    "LatticeMeasurements",
    "SpinCorrelationMeasurement",
    "SpinLatticeMeasurements",
]
