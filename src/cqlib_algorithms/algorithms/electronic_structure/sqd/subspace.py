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

"""Determinant-subspace construction and Ritz solving for SQD."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from cqlib.qis import Hamiltonian
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix  # type: ignore[import-untyped]
from scipy.sparse.linalg import eigsh  # type: ignore[import-untyped]

from ....extension.chemistry.electronic import ElectronicEnergy, ElectronicSystem
from .recovery import (
    _combine_spin_strings,
    _configuration_spin_strings,
    _has_target_particles,
    _spin_string,
)
from .results import SQDSubspaceResult, _validate_configuration

_HERMITIAN_TOLERANCE = 1e-11
_DENSE_SOLVER_CUTOFF = 64

SQDSubspaceSolver = Callable[
    [ElectronicSystem, tuple[int, ...]],
    SQDSubspaceResult,
]


def _sample_subspaces(
    distribution: dict[int, float],
    *,
    samples_per_batch: int,
    num_batches: int,
    required_configurations: tuple[int, ...],
    symmetrize_spin: bool,
    num_spatial_orbitals: int,
    rng: np.random.Generator,
) -> tuple[tuple[int, ...], ...]:
    """Sample ordinary or spin-inversion-closed determinant subspaces."""
    if symmetrize_spin:
        return _sample_spin_symmetric_subspaces(
            distribution,
            samples_per_batch=samples_per_batch,
            num_batches=num_batches,
            required_configurations=required_configurations,
            num_spatial_orbitals=num_spatial_orbitals,
            rng=rng,
        )

    required = set(required_configurations)
    configurations = np.asarray(
        sorted(
            configuration
            for configuration in distribution
            if configuration not in required
        ),
        dtype=object,
    )
    probabilities = np.asarray(
        [distribution[int(value)] for value in configurations],
        dtype=np.float64,
    )
    available_slots = samples_per_batch - len(required_configurations)
    sample_size = min(available_slots, configurations.size)
    if sample_size == configurations.size:
        complete = tuple(
            sorted(
                (*required_configurations, *(int(value) for value in configurations))
            )
        )
        return tuple(complete for _ in range(num_batches))

    probabilities /= probabilities.sum()
    batches: list[tuple[int, ...]] = []
    for _ in range(num_batches):
        selected = rng.choice(
            configurations.size,
            size=sample_size,
            replace=False,
            p=probabilities,
        )
        batches.append(
            tuple(
                sorted(
                    (
                        *required_configurations,
                        *(int(configurations[index]) for index in selected),
                    )
                )
            )
        )
    return tuple(batches)


def _sample_spin_symmetric_subspaces(
    distribution: dict[int, float],
    *,
    samples_per_batch: int,
    num_batches: int,
    required_configurations: tuple[int, ...],
    num_spatial_orbitals: int,
    rng: np.random.Generator,
) -> tuple[tuple[int, ...], ...]:
    """Build Eq. (11) spin-string Cartesian-product determinant batches."""
    maximum_spin_strings = math.isqrt(samples_per_batch)
    draws_per_batch = maximum_spin_strings // 2
    required_spin_strings = _configuration_spin_strings(
        required_configurations,
        num_spatial_orbitals,
    )
    configurations = np.asarray(sorted(distribution), dtype=object)
    probabilities = np.asarray(
        [distribution[int(value)] for value in configurations],
        dtype=np.float64,
    )
    probabilities /= probabilities.sum()

    batches: list[tuple[int, ...]] = []
    for _ in range(num_batches):
        spin_strings = set(required_spin_strings)
        selected = rng.choice(
            configurations.size,
            size=draws_per_batch,
            replace=True,
            p=probabilities,
        )
        for index in selected:
            candidate = int(configurations[index])
            candidate_strings = {
                _spin_string(candidate, num_spatial_orbitals, spin) for spin in (0, 1)
            }
            if len(spin_strings | candidate_strings) <= maximum_spin_strings:
                spin_strings.update(candidate_strings)

        if not spin_strings:
            raise ValueError("spin-symmetric sampling produced an empty spin pool")
        batches.append(
            tuple(
                sorted(
                    _combine_spin_strings(alpha, beta, num_spatial_orbitals)
                    for alpha in spin_strings
                    for beta in spin_strings
                )
            )
        )
    return tuple(batches)


def _select_required_configurations(
    include_configurations: tuple[int, ...],
    result: SQDSubspaceResult,
    *,
    carryover_threshold: float | None,
    samples_per_batch: int,
    symmetrize_spin: bool,
    num_spatial_orbitals: int,
) -> tuple[int, ...]:
    """Select amplitude-ranked cross-iteration determinant carryover."""
    if carryover_threshold is None:
        return include_configurations
    included = set(include_configurations)
    ranked = sorted(
        (
            (abs(amplitude), configuration)
            for configuration, amplitude in zip(
                result.configurations,
                result.amplitudes,
                strict=True,
            )
            if configuration not in included and abs(amplitude) > carryover_threshold
        ),
        key=lambda item: (-item[0], item[1]),
    )
    if symmetrize_spin:
        selected_list: list[int] = []
        spin_strings = _configuration_spin_strings(
            include_configurations,
            num_spatial_orbitals,
        )
        for _, configuration in ranked:
            candidate_strings = _configuration_spin_strings(
                (configuration,),
                num_spatial_orbitals,
            )
            combined = spin_strings | candidate_strings
            if len(combined) ** 2 <= samples_per_batch:
                selected_list.append(configuration)
                spin_strings = combined
        selected = tuple(selected_list)
    else:
        remaining = samples_per_batch - len(include_configurations)
        selected = tuple(configuration for _, configuration in ranked[:remaining])
    return tuple(sorted((*include_configurations, *selected)))


def _project_hamiltonian(
    hamiltonian: Hamiltonian,
    configurations: tuple[int, ...],
) -> csr_matrix:
    """Project a Pauli Hamiltonian into a determinant configuration basis."""
    lookup = {
        configuration: index for index, configuration in enumerate(configurations)
    }
    rows: list[int] = []
    columns: list[int] = []
    data: list[complex] = []
    for column, configuration in enumerate(configurations):
        for pauli, coefficient in hamiltonian.terms:
            target = configuration ^ pauli.x_mask
            row = lookup.get(target)
            if row is None:
                continue
            sign = -1.0 if (configuration & pauli.z_mask).bit_count() & 1 else 1.0
            value = coefficient * pauli.phase.to_complex() * pauli.y_phase() * sign
            rows.append(row)
            columns.append(column)
            data.append(value)
    matrix = coo_matrix(
        (np.asarray(data, dtype=np.complex128), (rows, columns)),
        shape=(len(configurations), len(configurations)),
        dtype=np.complex128,
    ).tocsr()
    matrix.eliminate_zeros()
    difference = matrix - matrix.getH()
    if difference.nnz and float(np.max(np.abs(difference.data))) > _HERMITIAN_TOLERANCE:
        raise ValueError("projected Hamiltonian is not Hermitian")
    return (0.5 * (matrix + matrix.getH())).tocsr()


def _canonicalize_phase(
    vector: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Normalize an eigenvector and fix its arbitrary global phase."""
    pivot = int(np.argmax(np.abs(vector)))
    if abs(vector[pivot]) == 0.0:
        raise ValueError("eigensolver returned a zero eigenvector")
    result = vector * np.exp(-1j * np.angle(vector[pivot]))
    result[pivot] = complex(abs(result[pivot]), 0.0)
    return np.asarray(result / np.linalg.norm(result), dtype=np.complex128)


def _solve_subspace(
    hamiltonian: Hamiltonian,
    configurations: tuple[int, ...],
) -> SQDSubspaceResult:
    """Solve the lowest Ritz pair in a selected determinant subspace."""
    matrix = _project_hamiltonian(hamiltonian, configurations)
    dimension = len(configurations)
    if matrix.nnz == 0:
        energy = 0.0
        amplitudes = np.zeros(dimension, dtype=np.complex128)
        amplitudes[0] = 1.0
    elif dimension <= _DENSE_SOLVER_CUTOFF:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix.toarray())
        energy = float(eigenvalues[0])
        amplitudes = eigenvectors[:, 0]
    else:
        eigenvalues, eigenvectors = eigsh(
            matrix,
            k=1,
            which="SA",
            v0=np.full(dimension, 1.0 / math.sqrt(dimension)),
            tol=1e-10,
            maxiter=max(1000, 20 * dimension),
        )
        energy = float(eigenvalues[0])
        amplitudes = eigenvectors[:, 0]
    amplitudes = _canonicalize_phase(np.asarray(amplitudes, dtype=np.complex128))
    return SQDSubspaceResult(
        energy,
        configurations,
        tuple(complex(value) for value in amplitudes),
        hamiltonian.num_qubits,
    )


def solve_sqd_subspace(
    system: ElectronicSystem,
    configurations: Sequence[int],
) -> SQDSubspaceResult:
    """Solve the Hamiltonian projected into specified determinants.

    This reference solver maps ``system`` with the repository's Jordan--Wigner
    convention and builds only the selected sparse matrix. Applications with
    larger selected spaces can inject another callable into :class:`SQD` while
    preserving the same result contract.

    Args:
        system: Electronic Hamiltonian and spin-orbital register definition.
        configurations: Non-empty target-sector determinant indices.

    Returns:
        Lowest Ritz pair in the requested determinant subspace.

    Raises:
        TypeError: If an input has the wrong type.
        ValueError: If configurations are empty, duplicated, outside the
            register, or outside the target particle sector.
        scipy.sparse.linalg.ArpackError: If a large sparse solve fails.
    """
    if not isinstance(system, ElectronicSystem):
        raise TypeError("system must be an ElectronicSystem")
    if isinstance(configurations, (str, bytes)) or not isinstance(
        configurations,
        Sequence,
    ):
        raise TypeError("configurations must be a sequence of integers")
    normalized: list[int] = []
    for configuration in configurations:
        _validate_configuration(
            configuration,
            num_spin_orbitals=system.num_spin_orbitals,
        )
        if not _has_target_particles(configuration, system):
            raise ValueError("configurations must satisfy the target particle sector")
        normalized.append(configuration)
    if not normalized:
        raise ValueError("configurations must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("configurations must be unique")
    hamiltonian = ElectronicEnergy(system.integrals).to_qubit_hamiltonian()
    if hamiltonian.num_qubits != system.num_spin_orbitals:
        raise ValueError("mapped Hamiltonian width must match the electronic system")
    return _solve_subspace(hamiltonian, tuple(sorted(normalized)))


__all__ = ["SQDSubspaceSolver", "solve_sqd_subspace"]
