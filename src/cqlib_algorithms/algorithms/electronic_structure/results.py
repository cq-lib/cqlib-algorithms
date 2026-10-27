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

"""Result values for electronic ground-state workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType

import numpy as np
from cqlib.qis import FermionToQubitMapper
from numpy.typing import NDArray

from ...extension.chemistry.electronic.system import ElectronicSystem
from ...extension.chemistry.fermionic import FermionicExcitation
from ...protocols.estimator import EstimatorResult
from ..variational import VQEResult

_AUXILIARY_NAMES = frozenset({"angular_momentum", "magnetization", "particle_number"})
_QEOM_STRUCTURE_NAMES = frozenset({"M", "Q", "V", "W"})


@dataclass(frozen=True, slots=True)
class ElectronicGroundStateResult:
    """A VQE result with electronic-system and symmetry-observable metadata.

    Args:
        system: Electronic system solved by the workflow.
        vqe_result: Underlying variational eigensolver result.
        auxiliary_results: Particle number, magnetization, and total-spin results.
        mapper: Mapper provenance used for every fermionic operator.

    Raises:
        TypeError: If a result component has the wrong type.
        ValueError: If required auxiliary-observable results are missing.
    """

    system: ElectronicSystem
    vqe_result: VQEResult
    auxiliary_results: Mapping[str, EstimatorResult]
    mapper: FermionToQubitMapper | None = field(
        default=None,
        repr=False,
        kw_only=True,
    )

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.system, ElectronicSystem):
            raise TypeError("system must be an ElectronicSystem")
        if not isinstance(self.vqe_result, VQEResult):
            raise TypeError("vqe_result must be a VQEResult")
        if self.mapper is not None and not isinstance(
            self.mapper, FermionToQubitMapper
        ):
            raise TypeError("mapper must implement FermionToQubitMapper or be None")
        if not isinstance(self.auxiliary_results, Mapping):
            raise TypeError("auxiliary_results must be a mapping")
        if set(self.auxiliary_results) != _AUXILIARY_NAMES:
            raise ValueError(
                "auxiliary_results must contain particle_number, magnetization, "
                "and angular_momentum"
            )
        snapshot: dict[str, EstimatorResult] = {}
        for name, result in self.auxiliary_results.items():
            if not isinstance(name, str):
                raise TypeError("auxiliary result names must be strings")
            if not isinstance(result, EstimatorResult):
                raise TypeError("auxiliary results must be EstimatorResult values")
            snapshot[name] = result
        object.__setattr__(self, "auxiliary_results", MappingProxyType(snapshot))

    @property
    def total_energy(self) -> float:
        """Return the final variational expectation including integral constants."""
        return self.vqe_result.eigenvalue

    @property
    def hamiltonian_constant(self) -> float:
        """Return the scalar term stored in the active Hamiltonian."""
        return self.system.integrals.constant

    @property
    def active_space_energy(self) -> float:
        """Return the expectation after removing the Hamiltonian scalar."""
        return self.total_energy - self.hamiltonian_constant

    @property
    def nuclear_repulsion_energy(self) -> float | None:
        """Return the nuclear repulsion when the system came from driver data."""
        return self.system.nuclear_repulsion_energy

    @property
    def inactive_energy(self) -> float | None:
        """Return the frozen-orbital electronic contribution, when known."""
        return self.system.inactive_energy

    @property
    def electronic_energy(self) -> float | None:
        """Return total energy without nuclear repulsion, when known."""
        nuclear_repulsion = self.nuclear_repulsion_energy
        return (
            None if nuclear_repulsion is None else self.total_energy - nuclear_repulsion
        )

    @property
    def correlation_energy(self) -> float | None:
        """Return energy relative to the stored Hartree--Fock reference, if any."""
        reference = self.system.hartree_fock_energy
        return None if reference is None else self.total_energy - reference

    @property
    def particle_number(self) -> EstimatorResult:
        """Return the final particle-number estimate."""
        return self.auxiliary_results["particle_number"]

    @property
    def magnetization(self) -> EstimatorResult:
        """Return the final spin-magnetization estimate."""
        return self.auxiliary_results["magnetization"]

    @property
    def angular_momentum(self) -> EstimatorResult:
        """Return the final total-spin-squared estimate."""
        return self.auxiliary_results["angular_momentum"]


def _immutable_real_vector(
    values: NDArray[np.float64],
    *,
    name: str,
) -> NDArray[np.float64]:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "fiu":
        raise TypeError(f"{name} must be a rank-1 real array")
    snapshot = np.array(array, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(snapshot)):
        raise ValueError(f"{name} must contain finite values")
    snapshot.flags.writeable = False
    return snapshot


def _immutable_complex_array(
    values: NDArray[np.complex128],
    *,
    name: str,
    ndim: int,
) -> NDArray[np.complex128]:
    array = np.asarray(values)
    if array.ndim != ndim or array.dtype.kind not in "fciu":
        raise TypeError(f"{name} must be a rank-{ndim} numeric array")
    snapshot = np.array(array, dtype=np.complex128, copy=True)
    if not np.all(np.isfinite(snapshot)):
        raise ValueError(f"{name} must contain finite values")
    snapshot.flags.writeable = False
    return snapshot


@dataclass(frozen=True, slots=True, eq=False)
class ElectronicQEOMResult:
    """Neutral qEOM excitation energies, amplitudes, and measured matrices.

    Args:
        ground_state_result: Electronic reference used for qEOM measurements.
        excitations: Ordered neutral excitation pool.
        excitation_energies: Positive retained qEOM roots.
        amplitudes: Generalized eigenvectors stored by column.
        residuals: Generalized eigenpair residual norms.
        structure_residuals: Pre-projection ``M``, ``Q``, ``V``, and ``W`` defects.
        metric_cutoff: Numerical metric-eigenvalue cutoff.
        retained_rank: Rank retained after metric truncation.
        positive_inertia: Positive retained metric inertia.
        negative_inertia: Negative retained metric inertia.
        discarded_metric_eigenvalues: Metric values removed by truncation.
        m_matrix: Hermitian qEOM ``M`` block.
        q_matrix: Symmetric qEOM ``Q`` block.
        v_matrix: Hermitian metric ``V`` block.
        w_matrix: Antisymmetric metric ``W`` block.

    Raises:
        TypeError: If a result component has the wrong type.
        ValueError: If dimensions, finiteness, positivity, or structure disagree.
    """

    ground_state_result: ElectronicGroundStateResult
    excitations: tuple[FermionicExcitation, ...]
    excitation_energies: NDArray[np.float64]
    amplitudes: NDArray[np.complex128]
    residuals: NDArray[np.float64]
    structure_residuals: Mapping[str, float]
    metric_cutoff: float
    retained_rank: int
    positive_inertia: int
    negative_inertia: int
    discarded_metric_eigenvalues: NDArray[np.float64] = field(repr=False)
    m_matrix: NDArray[np.complex128] = field(repr=False)
    q_matrix: NDArray[np.complex128] = field(repr=False)
    v_matrix: NDArray[np.complex128] = field(repr=False)
    w_matrix: NDArray[np.complex128] = field(repr=False)

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.ground_state_result, ElectronicGroundStateResult):
            raise TypeError(
                "ground_state_result must be an ElectronicGroundStateResult"
            )
        if not isinstance(self.excitations, tuple):
            raise TypeError("excitations must be a tuple")
        if not self.excitations:
            raise ValueError("excitations must not be empty")
        if any(
            not isinstance(excitation, FermionicExcitation)
            for excitation in self.excitations
        ):
            raise TypeError("excitations must contain FermionicExcitation values")
        if len(set(self.excitations)) != len(self.excitations):
            raise ValueError("excitations must not contain duplicates")
        if any(
            excitation.num_spin_orbitals
            != self.ground_state_result.system.num_spin_orbitals
            for excitation in self.excitations
        ):
            raise ValueError("excitations must match the electronic-system mode count")
        energies = _immutable_real_vector(
            self.excitation_energies,
            name="excitation_energies",
        )
        residuals = _immutable_real_vector(self.residuals, name="residuals")
        amplitudes = _immutable_complex_array(
            self.amplitudes,
            name="amplitudes",
            ndim=2,
        )
        if energies.size < 1:
            raise ValueError("excitation_energies must not be empty")
        if np.any(energies <= 0.0):
            raise ValueError("excitation_energies must be positive")
        if residuals.shape != energies.shape:
            raise ValueError("residuals must match excitation_energies")
        if np.any(residuals < 0.0):
            raise ValueError("residuals must be non-negative")
        if amplitudes.shape[1] != energies.size:
            raise ValueError("amplitude columns must match excitation_energies")

        pool_size = len(self.excitations)
        if amplitudes.shape[0] != 2 * pool_size:
            raise ValueError("amplitude rows must equal twice the excitation count")

        if not isinstance(self.structure_residuals, Mapping):
            raise TypeError("structure_residuals must be a mapping")
        if set(self.structure_residuals) != _QEOM_STRUCTURE_NAMES:
            raise ValueError("structure_residuals must contain M, Q, V, and W")
        structure_snapshot: dict[str, float] = {}
        for name, value in self.structure_residuals.items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("structure residuals must be real numbers")
            normalized = float(value)
            if not np.isfinite(normalized) or normalized < 0.0:
                raise ValueError("structure residuals must be finite and non-negative")
            structure_snapshot[name] = normalized

        if isinstance(self.metric_cutoff, bool) or not isinstance(
            self.metric_cutoff, Real
        ):
            raise TypeError("metric_cutoff must be a real number")
        metric_cutoff = float(self.metric_cutoff)
        if not np.isfinite(metric_cutoff) or metric_cutoff <= 0.0:
            raise ValueError("metric_cutoff must be finite and positive")
        for name in ("retained_rank", "positive_inertia", "negative_inertia"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
        if self.retained_rank < 1 or self.retained_rank > 2 * pool_size:
            raise ValueError("retained_rank must be within the qEOM pencil dimension")
        if self.positive_inertia < 1 or self.negative_inertia < 1:
            raise ValueError("metric inertia counts must be positive")
        if self.positive_inertia != self.negative_inertia:
            raise ValueError("metric inertia counts must be balanced")
        if self.positive_inertia + self.negative_inertia != self.retained_rank:
            raise ValueError("metric inertia counts must sum to retained_rank")
        discarded = _immutable_real_vector(
            self.discarded_metric_eigenvalues,
            name="discarded_metric_eigenvalues",
        )
        if discarded.size != 2 * pool_size - self.retained_rank:
            raise ValueError(
                "discarded_metric_eigenvalues size must match the removed metric rank"
            )
        if np.any(np.abs(discarded) > metric_cutoff):
            raise ValueError(
                "discarded metric eigenvalues must not exceed metric_cutoff"
            )
        matrices = []
        for name in ("m_matrix", "q_matrix", "v_matrix", "w_matrix"):
            matrix = _immutable_complex_array(
                getattr(self, name),
                name=name,
                ndim=2,
            )
            if matrix.shape != (pool_size, pool_size):
                raise ValueError(f"{name} shape must match the excitation pool")
            matrices.append(matrix)

        object.__setattr__(self, "excitation_energies", energies)
        object.__setattr__(self, "residuals", residuals)
        object.__setattr__(self, "amplitudes", amplitudes)
        object.__setattr__(
            self,
            "structure_residuals",
            MappingProxyType(structure_snapshot),
        )
        object.__setattr__(self, "metric_cutoff", metric_cutoff)
        object.__setattr__(self, "discarded_metric_eigenvalues", discarded)
        for name, matrix in zip(
            ("m_matrix", "q_matrix", "v_matrix", "w_matrix"),
            matrices,
            strict=True,
        ):
            object.__setattr__(self, name, matrix)

        if not np.allclose(matrices[0], matrices[0].conj().T, atol=1e-12, rtol=0.0):
            raise ValueError("m_matrix must be Hermitian")
        if not np.allclose(matrices[1], matrices[1].T, atol=1e-12, rtol=0.0):
            raise ValueError("q_matrix must be symmetric")
        if not np.allclose(matrices[2], matrices[2].conj().T, atol=1e-12, rtol=0.0):
            raise ValueError("v_matrix must be Hermitian")
        if not np.allclose(matrices[3], -matrices[3].T, atol=1e-12, rtol=0.0):
            raise ValueError("w_matrix must be antisymmetric")

    @property
    def total_energies(self) -> NDArray[np.float64]:
        """Return independent total excited-state energy snapshots."""
        values = self.ground_state_result.total_energy + self.excitation_energies
        values.flags.writeable = False
        return values


__all__ = ["ElectronicGroundStateResult", "ElectronicQEOMResult"]
