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

"""Electronic sample-based quantum diagonalization with recovery.

References
----------
[1] J. Robledo-Moreno et al., "Chemistry beyond the scale of exact
    diagonalization on a quantum-centric supercomputer," arXiv:2405.05068,
    2025. DOI: 10.48550/arXiv.2405.05068.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np
from cqlib import Circuit
from cqlib.qis import Hamiltonian
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix  # type: ignore[import-untyped]
from scipy.sparse.linalg import eigsh  # type: ignore[import-untyped]

from ...extension.chemistry.electronic import ElectronicEnergy, ElectronicSystem
from ...operators.estimator._circuit_validation import (
    validate_unitary_state_preparation,
)
from ...protocols.sampler import (
    Sampler,
    SamplerRequest,
    spawn_seeds,
    validate_sampler_results,
)

_FLIP_EPSILON = 0.01
_HERMITIAN_TOLERANCE = 1e-11
_NORMALIZATION_TOLERANCE = 1e-10
_DENSE_SOLVER_CUTOFF = 64


def _validate_strict_positive_integer(value: int, name: str) -> None:
    """Validate a strictly positive built-in integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def _normalize_nonnegative_real(value: float, name: str) -> float:
    """Return a finite non-negative floating-point value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if normalized < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _normalize_optional_nonnegative_real(
    value: float | None,
    name: str,
) -> float | None:
    """Return a normalized optional non-negative real value."""
    if value is None:
        return None
    return _normalize_nonnegative_real(value, name)


def _normalize_occupancies(
    occupancies: Sequence[float],
    system: ElectronicSystem,
) -> tuple[float, ...]:
    """Validate and snapshot interleaved spin-orbital occupancies."""
    if isinstance(occupancies, (str, bytes)) or not isinstance(occupancies, Sequence):
        raise TypeError("initial_occupancies must be a sequence of real numbers")
    if len(occupancies) != system.num_spin_orbitals:
        raise ValueError(
            "initial_occupancies length must equal the number of spin orbitals"
        )

    normalized: list[float] = []
    for value in occupancies:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("initial_occupancies must contain real numbers")
        item = float(value)
        if not math.isfinite(item):
            raise ValueError("initial_occupancies must contain finite values")
        if not 0.0 <= item <= 1.0:
            raise ValueError("initial_occupancies must lie in the interval [0, 1]")
        normalized.append(item)

    alpha_total = math.fsum(normalized[0::2])
    beta_total = math.fsum(normalized[1::2])
    if not math.isclose(
        alpha_total,
        system.occupation.num_alpha,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise ValueError("alpha initial occupancies must sum to num_alpha")
    if not math.isclose(
        beta_total,
        system.occupation.num_beta,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise ValueError("beta initial occupancies must sum to num_beta")
    return tuple(normalized)


def _validate_configuration(
    configuration: int,
    *,
    num_spin_orbitals: int,
) -> None:
    """Validate a determinant index against the register width."""
    if isinstance(configuration, bool) or not isinstance(configuration, int):
        raise TypeError("configurations must contain integers")
    if not 0 <= configuration < 1 << num_spin_orbitals:
        raise ValueError("configuration is outside the spin-orbital register")


def _normalize_amplitude(value: complex) -> complex:
    """Return a finite complex determinant amplitude."""
    if isinstance(value, bool) or not isinstance(value, (Real, complex)):
        raise TypeError("amplitudes must contain complex numbers")
    normalized = complex(value)
    if not math.isfinite(normalized.real) or not math.isfinite(normalized.imag):
        raise ValueError("amplitudes must contain finite values")
    return normalized


@dataclass(frozen=True, slots=True, repr=False)
class SQDSubspaceResult:
    """Ground Ritz pair obtained in one sampled determinant subspace.

    ``configurations[i]`` is the little-endian interleaved alpha/beta
    determinant associated with ``amplitudes[i]``. The state is normalized,
    and ``energy`` is a variational upper bound within that subspace.

    Args:
        energy: Finite projected ground-state energy.
        configurations: Sorted unique determinant indices.
        amplitudes: Normalized coefficients in configuration order.
        num_spin_orbitals: Width of every determinant.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If values, dimensions, ordering, or normalization fail.
    """

    energy: float
    configurations: tuple[int, ...]
    amplitudes: tuple[complex, ...]
    num_spin_orbitals: int

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if isinstance(self.energy, bool) or not isinstance(self.energy, Real):
            raise TypeError("energy must be a real number")
        energy = float(self.energy)
        if not math.isfinite(energy):
            raise ValueError("energy must be finite")
        object.__setattr__(self, "energy", energy)

        _validate_strict_positive_integer(
            self.num_spin_orbitals,
            "num_spin_orbitals",
        )
        if not isinstance(self.configurations, tuple):
            raise TypeError("configurations must be a tuple")
        if not self.configurations:
            raise ValueError("configurations must not be empty")
        for configuration in self.configurations:
            _validate_configuration(
                configuration,
                num_spin_orbitals=self.num_spin_orbitals,
            )
        if tuple(sorted(set(self.configurations))) != self.configurations:
            raise ValueError("configurations must be sorted and unique")

        if not isinstance(self.amplitudes, tuple):
            raise TypeError("amplitudes must be a tuple")
        amplitudes = tuple(_normalize_amplitude(value) for value in self.amplitudes)
        if len(amplitudes) != len(self.configurations):
            raise ValueError("amplitudes must match configurations")
        norm = math.fsum(abs(value) ** 2 for value in amplitudes)
        if not math.isclose(
            norm,
            1.0,
            rel_tol=0.0,
            abs_tol=_NORMALIZATION_TOLERANCE,
        ):
            raise ValueError("amplitudes must define a normalized state")
        object.__setattr__(self, "amplitudes", amplitudes)

    @property
    def dimension(self) -> int:
        """Return the selected determinant-subspace dimension."""
        return len(self.configurations)

    @property
    def orbital_occupancies(self) -> tuple[float, ...]:
        """Return occupancies evaluated from this Ritz eigenstate."""
        return _subspace_occupancies(self)

    def __repr__(self) -> str:
        """Summarize the Ritz pair without printing all coefficients."""
        return (
            f"SQDSubspaceResult(energy={self.energy!r}, "
            f"dimension={self.dimension}, "
            f"num_spin_orbitals={self.num_spin_orbitals})"
        )


SQDSubspaceSolver = Callable[
    [ElectronicSystem, tuple[int, ...]],
    SQDSubspaceResult,
]


@dataclass(frozen=True, slots=True, repr=False)
class SQDIteration:
    """One setup or self-consistent configuration-recovery round.

    Args:
        index: Zero-based iteration index.
        subspaces: Ritz results for all independently sampled batches.
        orbital_occupancies: Batch-averaged interleaved spin-orbital occupancies.
        num_recovered_configurations: Number of unique postselected or recovered
            determinants supplied to batch construction.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If an index, collection, or occupancy is invalid.
    """

    index: int
    subspaces: tuple[SQDSubspaceResult, ...]
    orbital_occupancies: tuple[float, ...]
    num_recovered_configurations: int

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise TypeError("index must be an integer")
        if self.index < 0:
            raise ValueError("index must be non-negative")
        if not isinstance(self.subspaces, tuple):
            raise TypeError("subspaces must be a tuple")
        if not self.subspaces:
            raise ValueError("subspaces must not be empty")
        if any(not isinstance(item, SQDSubspaceResult) for item in self.subspaces):
            raise TypeError("subspaces must contain SQDSubspaceResult values")
        width = self.subspaces[0].num_spin_orbitals
        if any(item.num_spin_orbitals != width for item in self.subspaces):
            raise ValueError("subspaces must use one spin-orbital width")

        if not isinstance(self.orbital_occupancies, tuple):
            raise TypeError("orbital_occupancies must be a tuple")
        if len(self.orbital_occupancies) != width:
            raise ValueError("orbital_occupancies must match the register width")
        normalized: list[float] = []
        for value in self.orbital_occupancies:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("orbital_occupancies must contain real numbers")
            occupancy = float(value)
            if not math.isfinite(occupancy) or not 0.0 <= occupancy <= 1.0:
                raise ValueError("orbital_occupancies must be finite values in [0, 1]")
            normalized.append(occupancy)
        object.__setattr__(self, "orbital_occupancies", tuple(normalized))

        _validate_strict_positive_integer(
            self.num_recovered_configurations,
            "num_recovered_configurations",
        )

    @property
    def best_subspace(self) -> SQDSubspaceResult:
        """Return the lowest-energy batch result in this iteration."""
        return min(self.subspaces, key=lambda item: item.energy)

    @property
    def energy(self) -> float:
        """Return the iteration's lowest projected energy."""
        return self.best_subspace.energy

    def __repr__(self) -> str:
        """Summarize one iteration without printing determinant data."""
        return (
            f"SQDIteration(index={self.index}, energy={self.energy!r}, "
            f"num_subspaces={len(self.subspaces)}, "
            f"num_recovered_configurations={self.num_recovered_configurations})"
        )


SQDCallback = Callable[[SQDIteration], None]


@dataclass(frozen=True, slots=True, repr=False)
class SQDResult:
    """Final sample-based diagonalization result and convergence history.

    ``ground_state`` is the lowest Ritz result seen in any iteration, while
    ``orbital_occupancies`` comes from the final completed iteration. A finite
    result is returned when ``max_iterations`` is reached without convergence;
    callers must inspect ``converged`` when convergence is required.

    Args:
        ground_state: Lowest projected-energy result observed.
        orbital_occupancies: Final interleaved spin-orbital occupancies.
        iterations: Complete ordered setup and recovery history.
        converged: Whether both requested convergence tests were satisfied.
        shots: Number of raw circuit measurements.
        valid_raw_shots: Raw shots already in the requested particle sector.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If fields are inconsistent or outside their valid ranges.
    """

    ground_state: SQDSubspaceResult
    orbital_occupancies: tuple[float, ...]
    iterations: tuple[SQDIteration, ...]
    converged: bool
    shots: int
    valid_raw_shots: int

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.ground_state, SQDSubspaceResult):
            raise TypeError("ground_state must be an SQDSubspaceResult")
        if not isinstance(self.iterations, tuple):
            raise TypeError("iterations must be a tuple")
        if not self.iterations:
            raise ValueError("iterations must not be empty")
        if any(not isinstance(item, SQDIteration) for item in self.iterations):
            raise TypeError("iterations must contain SQDIteration values")
        if tuple(item.index for item in self.iterations) != tuple(
            range(len(self.iterations))
        ):
            raise ValueError("iterations must have consecutive indices from zero")
        if self.ground_state not in tuple(
            subspace for item in self.iterations for subspace in item.subspaces
        ):
            raise ValueError("ground_state must come from the iteration history")
        if self.ground_state.energy != min(
            subspace.energy for item in self.iterations for subspace in item.subspaces
        ):
            raise ValueError("ground_state must have the lowest observed energy")
        if self.orbital_occupancies != self.iterations[-1].orbital_occupancies:
            raise ValueError("orbital_occupancies must match the final iteration")
        if not isinstance(self.converged, bool):
            raise TypeError("converged must be a bool")
        _validate_strict_positive_integer(self.shots, "shots")
        if isinstance(self.valid_raw_shots, bool) or not isinstance(
            self.valid_raw_shots, int
        ):
            raise TypeError("valid_raw_shots must be an integer")
        if not 0 <= self.valid_raw_shots <= self.shots:
            raise ValueError("valid_raw_shots must lie between zero and shots")

    @property
    def energy(self) -> float:
        """Return the lowest variational energy observed."""
        return self.ground_state.energy

    @property
    def num_iterations(self) -> int:
        """Return the number of completed diagonalization rounds."""
        return len(self.iterations)

    @property
    def ground_state_occupancies(self) -> tuple[float, ...]:
        """Return occupancies belonging to the reported lowest Ritz state."""
        return self.ground_state.orbital_occupancies

    @property
    def best_iteration_index(self) -> int:
        """Return the iteration that produced ``ground_state``."""
        return next(
            iteration.index
            for iteration in self.iterations
            if self.ground_state in iteration.subspaces
        )

    def __repr__(self) -> str:
        """Summarize the result without printing determinant amplitudes."""
        return (
            f"SQDResult(energy={self.energy!r}, converged={self.converged!r}, "
            f"num_iterations={self.num_iterations}, "
            f"subspace_dimension={self.ground_state.dimension})"
        )


def _sector_positions(num_spatial_orbitals: int, spin: int) -> tuple[int, ...]:
    """Return interleaved register positions for one spin species."""
    return tuple(2 * orbital + spin for orbital in range(num_spatial_orbitals))


def _sector_weight(configuration: int, positions: tuple[int, ...]) -> int:
    """Return the occupation count over selected register positions."""
    return sum((configuration >> position) & 1 for position in positions)


def _has_target_particles(configuration: int, system: ElectronicSystem) -> bool:
    """Return whether a determinant belongs to the requested spin sector."""
    alpha = _sector_positions(system.num_spatial_orbitals, 0)
    beta = _sector_positions(system.num_spatial_orbitals, 1)
    return (
        _sector_weight(configuration, alpha) == system.occupation.num_alpha
        and _sector_weight(configuration, beta) == system.occupation.num_beta
    )


def _spin_string(
    configuration: int,
    num_spatial_orbitals: int,
    spin: int,
) -> int:
    """Extract one compact spatial-orbital spin string from a determinant."""
    return sum(
        ((configuration >> (2 * orbital + spin)) & 1) << orbital
        for orbital in range(num_spatial_orbitals)
    )


def _combine_spin_strings(
    alpha: int,
    beta: int,
    num_spatial_orbitals: int,
) -> int:
    """Combine compact alpha and beta strings into interleaved ordering."""
    return sum(
        ((alpha >> orbital) & 1) << (2 * orbital)
        | ((beta >> orbital) & 1) << (2 * orbital + 1)
        for orbital in range(num_spatial_orbitals)
    )


def _configuration_spin_strings(
    configurations: Sequence[int],
    num_spatial_orbitals: int,
) -> set[int]:
    """Collect unique alpha and beta spatial occupation strings."""
    return {
        _spin_string(configuration, num_spatial_orbitals, spin)
        for configuration in configurations
        for spin in (0, 1)
    }


def _normalize_required_configurations(
    configurations: Sequence[int],
    system: ElectronicSystem,
) -> tuple[int, ...]:
    """Validate and snapshot user-required determinant configurations."""
    if isinstance(configurations, (str, bytes)) or not isinstance(
        configurations,
        Sequence,
    ):
        raise TypeError("include_configurations must be a sequence of integers")
    normalized: list[int] = []
    for configuration in configurations:
        _validate_configuration(
            configuration,
            num_spin_orbitals=system.num_spin_orbitals,
        )
        if not _has_target_particles(configuration, system):
            raise ValueError(
                "include_configurations must satisfy the target particle sector"
            )
        normalized.append(configuration)
    if len(set(normalized)) != len(normalized):
        raise ValueError("include_configurations must be unique")
    return tuple(sorted(normalized))


def _normalize_run_options(
    system: ElectronicSystem,
    *,
    samples_per_batch: int,
    num_batches: int,
    max_iterations: int,
    energy_tolerance: float,
    occupancy_tolerance: float,
    initial_occupancies: Sequence[float] | None,
    include_configurations: Sequence[int],
    carryover_threshold: float | None,
    symmetrize_spin: bool,
    callback: SQDCallback | None,
) -> tuple[
    float,
    float,
    tuple[float, ...] | None,
    tuple[int, ...],
    float | None,
]:
    """Validate and normalize options shared by both SQD entry points."""
    _validate_strict_positive_integer(samples_per_batch, "samples_per_batch")
    _validate_strict_positive_integer(num_batches, "num_batches")
    _validate_strict_positive_integer(max_iterations, "max_iterations")
    normalized_energy_tolerance = _normalize_nonnegative_real(
        energy_tolerance,
        "energy_tolerance",
    )
    normalized_occupancy_tolerance = _normalize_nonnegative_real(
        occupancy_tolerance,
        "occupancy_tolerance",
    )
    normalized_carryover_threshold = _normalize_optional_nonnegative_real(
        carryover_threshold,
        "carryover_threshold",
    )
    if not isinstance(symmetrize_spin, bool):
        raise TypeError("symmetrize_spin must be a bool")
    if symmetrize_spin and system.occupation.num_alpha != system.occupation.num_beta:
        raise ValueError("symmetrize_spin requires equal alpha and beta populations")
    if symmetrize_spin and samples_per_batch < 4:
        raise ValueError("symmetrize_spin requires samples_per_batch to be at least 4")
    if callback is not None and not callable(callback):
        raise TypeError("callback must be callable or None")
    required_by_user = _normalize_required_configurations(
        include_configurations,
        system,
    )
    if symmetrize_spin:
        required_spin_strings = _configuration_spin_strings(
            required_by_user,
            system.num_spatial_orbitals,
        )
        if len(required_spin_strings) ** 2 > samples_per_batch:
            raise ValueError(
                "the spin closure of include_configurations exceeds samples_per_batch"
            )
    elif len(required_by_user) > samples_per_batch:
        raise ValueError("include_configurations cannot exceed samples_per_batch")
    normalized_initial = (
        None
        if initial_occupancies is None
        else _normalize_occupancies(initial_occupancies, system)
    )
    return (
        normalized_energy_tolerance,
        normalized_occupancy_tolerance,
        normalized_initial,
        required_by_user,
        normalized_carryover_threshold,
    )


def _normalize_distribution(
    distribution: Mapping[int, float],
    system: ElectronicSystem,
) -> dict[int, float]:
    """Validate and normalize a determinant-weight distribution."""
    if not isinstance(distribution, Mapping):
        raise TypeError("distribution must be a mapping")
    normalized: dict[int, float] = {}
    for configuration, probability in distribution.items():
        _validate_configuration(
            configuration,
            num_spin_orbitals=system.num_spin_orbitals,
        )
        if isinstance(probability, bool) or not isinstance(probability, Real):
            raise TypeError("distribution values must be real numbers")
        value = float(probability)
        if not math.isfinite(value):
            raise ValueError("distribution values must be finite")
        if value < 0.0:
            raise ValueError("distribution values must be non-negative")
        if value > 0.0:
            normalized[configuration] = value
    total = math.fsum(normalized.values())
    if total <= 0.0:
        raise ValueError("distribution must contain positive total weight")
    return {
        configuration: probability / total
        for configuration, probability in normalized.items()
    }


def _flip_zero_to_one_weight(expected_ratio: float, occupancy: float) -> float:
    """Return the recovery weight for changing an empty mode to occupied."""
    if expected_ratio <= 0.0:
        return max(_FLIP_EPSILON, occupancy)
    if occupancy < expected_ratio:
        return occupancy * _FLIP_EPSILON / expected_ratio
    if expected_ratio >= 1.0:
        return _FLIP_EPSILON
    slope = (1.0 - _FLIP_EPSILON) / (1.0 - expected_ratio)
    return occupancy * slope + 1.0 - slope


def _flip_one_to_zero_weight(expected_ratio: float, occupancy: float) -> float:
    """Return the recovery weight for changing an occupied mode to empty."""
    return _flip_zero_to_one_weight(1.0 - expected_ratio, 1.0 - occupancy)


def _recover_sector(
    configuration: int,
    positions: tuple[int, ...],
    target: int,
    occupancies: tuple[float, ...],
    rng: np.random.Generator,
) -> int:
    """Repair one spin sector to its requested particle count."""
    current = _sector_weight(configuration, positions)
    difference = current - target
    if difference == 0:
        return configuration

    expected_ratio = target / len(positions)
    if difference > 0:
        candidates = tuple(
            position for position in positions if configuration & (1 << position)
        )
        weights = np.asarray(
            [
                _flip_one_to_zero_weight(expected_ratio, occupancies[position])
                for position in candidates
            ],
            dtype=np.float64,
        )
        new_value = 0
    else:
        candidates = tuple(
            position for position in positions if not configuration & (1 << position)
        )
        weights = np.asarray(
            [
                _flip_zero_to_one_weight(expected_ratio, occupancies[position])
                for position in candidates
            ],
            dtype=np.float64,
        )
        new_value = 1

    weights = np.clip(weights, 0.0, 1.0)
    if not np.any(weights):
        weights = np.ones_like(weights)
    weights /= weights.sum()
    chosen = rng.choice(
        np.asarray(candidates, dtype=np.int64),
        size=abs(difference),
        replace=False,
        p=weights,
    )
    recovered = configuration
    for value in chosen:
        mask = 1 << int(value)
        recovered = recovered | mask if new_value else recovered & ~mask
    return recovered


def _recover_distribution(
    distribution: dict[int, float],
    system: ElectronicSystem,
    occupancies: tuple[float, ...],
    rng: np.random.Generator,
) -> dict[int, float]:
    """Recover every determinant into the target alpha/beta particle sector."""
    alpha = _sector_positions(system.num_spatial_orbitals, 0)
    beta = _sector_positions(system.num_spatial_orbitals, 1)
    recovered: defaultdict[int, float] = defaultdict(float)
    for configuration, probability in distribution.items():
        corrected = _recover_sector(
            configuration,
            alpha,
            system.occupation.num_alpha,
            occupancies,
            rng,
        )
        corrected = _recover_sector(
            corrected,
            beta,
            system.occupation.num_beta,
            occupancies,
            rng,
        )
        recovered[corrected] += probability
    total = math.fsum(recovered.values())
    return {
        configuration: probability / total
        for configuration, probability in recovered.items()
    }


def recover_sqd_configurations(
    distribution: Mapping[int, float],
    system: ElectronicSystem,
    occupancies: Sequence[float],
    *,
    seed: int | None = None,
) -> dict[int, float]:
    """Recover target-particle configurations from a weighted distribution.

    This is the independently reusable configuration-recovery step of SQD.
    Configuration integers use the repository's little-endian interleaved
    spin-orbital order. The returned distribution is normalized and contains
    only determinants with the target alpha and beta particle numbers.

    Args:
        distribution: Non-negative weights keyed by determinant integer.
        system: Electronic system defining register width and particle sector.
        occupancies: Interleaved spin-orbital occupation estimates.
        seed: Optional non-negative random seed.

    Returns:
        Recovered normalized determinant distribution.

    Raises:
        TypeError: If an input has the wrong type.
        ValueError: If dimensions, values, or occupancies are invalid.
    """
    if not isinstance(system, ElectronicSystem):
        raise TypeError("system must be an ElectronicSystem")
    normalized_distribution = _normalize_distribution(distribution, system)
    normalized_occupancies = _normalize_occupancies(occupancies, system)
    recovery_seed = spawn_seeds(seed, 1)[0]
    return _recover_distribution(
        normalized_distribution,
        system,
        normalized_occupancies,
        np.random.default_rng(recovery_seed),
    )


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


def _subspace_occupancies(result: SQDSubspaceResult) -> tuple[float, ...]:
    """Compute spin-orbital occupancies from one selected-subspace state."""
    weights = tuple(abs(amplitude) ** 2 for amplitude in result.amplitudes)
    return tuple(
        math.fsum(
            weight
            for configuration, weight in zip(
                result.configurations,
                weights,
                strict=True,
            )
            if configuration & (1 << mode)
        )
        for mode in range(result.num_spin_orbitals)
    )


def _average_occupancies(
    results: tuple[SQDSubspaceResult, ...],
) -> tuple[float, ...]:
    """Return the unweighted batch average of orbital occupancies."""
    all_occupancies = tuple(_subspace_occupancies(result) for result in results)
    return tuple(
        math.fsum(values) / len(all_occupancies)
        for values in zip(*all_occupancies, strict=True)
    )


def _normalize_counts_distribution(
    counts: Mapping[str, int],
    *,
    num_bits: int,
) -> tuple[dict[int, float], dict[int, int], int]:
    """Validate counts and return normalized integer-keyed statistics."""
    if not isinstance(counts, Mapping):
        raise TypeError("counts must be a mapping")
    compressed: defaultdict[int, int] = defaultdict(int)
    for bitstring, count in counts.items():
        if not isinstance(bitstring, str):
            raise TypeError("counts keys must be bitstrings")
        if len(bitstring) != num_bits or any(bit not in "01" for bit in bitstring):
            raise ValueError("counts bitstrings must match the spin-orbital width")
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("counts values must be integers")
        if count < 0:
            raise ValueError("counts values must be non-negative")
        if count:
            compressed[int(bitstring, 2)] += count
    total = sum(compressed.values())
    if total <= 0:
        raise ValueError("sampler counts must contain positive total weight")
    distribution = {
        configuration: count / total for configuration, count in compressed.items()
    }
    return distribution, dict(compressed), total


class SQD:
    """Run electronic sample-based quantum diagonalization.

    The algorithm samples a fully bound state-preparation circuit in the
    Jordan--Wigner occupation basis, repairs incorrect alpha and beta particle
    numbers using self-consistent orbital occupancies, and independently
    diagonalizes the Hamiltonian projected into several sampled determinant
    subspaces. Each projected energy is variational. New occupancies are the
    unweighted batch average specified by reference [1]. Optional spin
    symmetrization implements its spatial-spin-string Cartesian closure to
    facilitate total-spin conservation without claiming exact ``S²`` projection.

    The implementation uses the repository's interleaved ordering: mode
    ``2*p`` is spatial orbital ``p`` with alpha spin and ``2*p + 1`` is beta.
    Circuit position ``i`` must therefore represent spin-orbital mode ``i``.
    The built-in sparse Ritz solver does not materialize the full Hilbert-space
    Hamiltonian, but its classical cost still grows with the selected subspace.

    Args:
        sampler: Optional sampler used to obtain a raw determinant distribution.
            It is required by :meth:`compute` but not by
            :meth:`compute_from_counts`.
        subspace_solver: Optional solver receiving an electronic system and a
            sorted determinant tuple. It must return the lowest Ritz pair for
            exactly that tuple. The built-in sparse solver is used by default.

    Raises:
        TypeError: If a supplied component violates its callable contract.
    """

    __slots__ = ("_sampler", "_subspace_solver")

    def __init__(
        self,
        sampler: Sampler | None = None,
        *,
        subspace_solver: SQDSubspaceSolver | None = None,
    ) -> None:
        if sampler is not None and not isinstance(sampler, Sampler):
            raise TypeError("sampler must be a Sampler or None")
        if subspace_solver is not None and not callable(subspace_solver):
            raise TypeError("subspace_solver must be callable or None")
        self._sampler = sampler
        self._subspace_solver = subspace_solver

    @property
    def sampler(self) -> Sampler | None:
        """Return the injected determinant sampler, if present."""
        return self._sampler

    @property
    def subspace_solver(self) -> SQDSubspaceSolver | None:
        """Return the custom selected-subspace solver, if present."""
        return self._subspace_solver

    def compute(
        self,
        system: ElectronicSystem,
        state_preparation: Circuit,
        *,
        shots: int,
        samples_per_batch: int,
        num_batches: int = 1,
        max_iterations: int = 5,
        energy_tolerance: float = 1e-8,
        occupancy_tolerance: float = 1e-5,
        initial_occupancies: Sequence[float] | None = None,
        include_configurations: Sequence[int] = (),
        carryover_threshold: float | None = 1e-4,
        symmetrize_spin: bool = False,
        callback: SQDCallback | None = None,
        seed: int | None = None,
    ) -> SQDResult:
        """Estimate the electronic ground state from sampled determinants.

        When ``initial_occupancies`` is omitted, the setup round postselects raw
        shots with the requested alpha and beta particle numbers, solves their
        sampled subspaces, and obtains the first recovery vector from the Ritz
        states, as in reference [1]. At least one such raw shot is required.
        Supplying a physically normalized initial vector skips this setup and
        permits immediate recovery when no raw shot is initially valid.

        Args:
            system: Electronic Hamiltonian and target spin populations.
            state_preparation: Fully bound unitary circuit in interleaved
                Jordan--Wigner occupation order.
            shots: Positive number of raw determinant samples.
            samples_per_batch: Maximum unique determinants in each subspace.
            num_batches: Number of independently sampled subspaces per round.
            max_iterations: Maximum total diagonalization rounds. The setup round
                counts toward this limit when initial occupancies are omitted.
            energy_tolerance: Absolute iteration-energy convergence tolerance.
            occupancy_tolerance: Maximum orbital-occupancy change tolerance.
            initial_occupancies: Optional interleaved spin-orbital occupancies.
            include_configurations: Target-sector determinants included in every
                subspace, regardless of whether they were sampled.
            carryover_threshold: Minimum absolute Ritz amplitude retained in all
                subspaces of the next iteration, or ``None`` to disable carryover.
            symmetrize_spin: Whether to form each batch from the Cartesian product
                of a shared alpha/beta spatial-occupation-string set. This
                facilitates total-spin conservation and requires equal alpha and
                beta populations, but does not itself enforce an ``S²`` eigenstate.
            callback: Optional observer called with each immutable iteration.
            seed: Optional master seed for sampling and classical recovery.

        Returns:
            Lowest observed Ritz pair, final occupancies, and full history.

        Raises:
            TypeError: If an input or component response has the wrong type.
            ValueError: If dimensions, values, sampling metadata, particle
                sectors, or projected Hamiltonians violate the contract.
            scipy.sparse.linalg.ArpackError: If a large sparse solve fails.
        """
        if not isinstance(system, ElectronicSystem):
            raise TypeError("system must be an ElectronicSystem")
        if not isinstance(state_preparation, Circuit):
            raise TypeError("state_preparation must be a cqlib.Circuit")
        if self._sampler is None:
            raise ValueError("compute requires an SQD sampler")
        _validate_strict_positive_integer(shots, "shots")
        (
            energy_tolerance,
            occupancy_tolerance,
            normalized_initial,
            required_by_user,
            carryover_threshold,
        ) = _normalize_run_options(
            system,
            samples_per_batch=samples_per_batch,
            num_batches=num_batches,
            max_iterations=max_iterations,
            energy_tolerance=energy_tolerance,
            occupancy_tolerance=occupancy_tolerance,
            initial_occupancies=initial_occupancies,
            include_configurations=include_configurations,
            carryover_threshold=carryover_threshold,
            symmetrize_spin=symmetrize_spin,
            callback=callback,
        )

        state_preparation.validate()
        if state_preparation.used_symbols:
            raise ValueError("state_preparation must be fully bound")
        if state_preparation.num_qubits != system.num_spin_orbitals:
            raise ValueError(
                "state_preparation width must equal the number of spin orbitals"
            )
        validate_unitary_state_preparation(state_preparation)

        sampler_seed, recovery_seed = spawn_seeds(seed, 2)
        request = SamplerRequest(
            state_preparation,
            tuple(state_preparation.qubits),
            shots,
            sampler_seed,
        )
        results = validate_sampler_results(self._sampler.run([request]), [request])
        raw_distribution, raw_counts, observed_shots = _normalize_counts_distribution(
            results[0].counts,
            num_bits=system.num_spin_orbitals,
        )
        if observed_shots != shots:
            raise ValueError("sampler counts total must equal shots")
        return self._run(
            system,
            raw_distribution,
            raw_counts,
            shots=shots,
            samples_per_batch=samples_per_batch,
            num_batches=num_batches,
            max_iterations=max_iterations,
            energy_tolerance=energy_tolerance,
            occupancy_tolerance=occupancy_tolerance,
            initial_occupancies=normalized_initial,
            include_configurations=required_by_user,
            carryover_threshold=carryover_threshold,
            symmetrize_spin=symmetrize_spin,
            callback=callback,
            recovery_seed=recovery_seed,
        )

    def compute_from_counts(
        self,
        system: ElectronicSystem,
        counts: Mapping[str, int],
        *,
        samples_per_batch: int,
        num_batches: int = 1,
        max_iterations: int = 5,
        energy_tolerance: float = 1e-8,
        occupancy_tolerance: float = 1e-5,
        initial_occupancies: Sequence[float] | None = None,
        include_configurations: Sequence[int] = (),
        carryover_threshold: float | None = 1e-4,
        symmetrize_spin: bool = False,
        callback: SQDCallback | None = None,
        seed: int | None = None,
    ) -> SQDResult:
        """Estimate the electronic ground state from existing sample counts.

        This entry point avoids rerunning a circuit when counts were collected
        earlier or on external hardware. Bitstrings must have exactly one bit
        per interleaved Jordan--Wigner spin orbital. Their integer values use
        the same rightmost-bit-as-mode-zero convention as sampler results.

        Args:
            system: Electronic Hamiltonian and target spin populations.
            counts: Non-negative integer counts keyed by fixed-width bitstrings.
            samples_per_batch: Maximum determinant dimension of each subspace.
            num_batches: Number of independently sampled subspaces per round.
            max_iterations: Maximum total diagonalization rounds. The setup round
                counts toward this limit when initial occupancies are omitted.
            energy_tolerance: Absolute iteration-energy convergence tolerance.
            occupancy_tolerance: Maximum orbital-occupancy change tolerance.
            initial_occupancies: Optional interleaved spin-orbital occupancies.
            include_configurations: Target-sector determinants included in every
                subspace, regardless of whether they were sampled.
            carryover_threshold: Minimum absolute Ritz amplitude retained in all
                subspaces of the next iteration, or ``None`` to disable carryover.
            symmetrize_spin: Whether to close each selected subspace under
                alpha/beta spin-string recombination. Equal spin populations and
                at least four determinants of batch capacity are required.
            callback: Optional observer called with each immutable iteration.
            seed: Optional seed controlling classical recovery and subsampling.

        Returns:
            Lowest observed Ritz pair, final occupancies, and full history.

        Raises:
            TypeError: If an input or solver response has the wrong type.
            ValueError: If dimensions, values, particle sectors, or projected
                Hamiltonians violate the contract.
            scipy.sparse.linalg.ArpackError: If a large sparse solve fails.
        """
        if not isinstance(system, ElectronicSystem):
            raise TypeError("system must be an ElectronicSystem")
        (
            energy_tolerance,
            occupancy_tolerance,
            normalized_initial,
            required_by_user,
            carryover_threshold,
        ) = _normalize_run_options(
            system,
            samples_per_batch=samples_per_batch,
            num_batches=num_batches,
            max_iterations=max_iterations,
            energy_tolerance=energy_tolerance,
            occupancy_tolerance=occupancy_tolerance,
            initial_occupancies=initial_occupancies,
            include_configurations=include_configurations,
            carryover_threshold=carryover_threshold,
            symmetrize_spin=symmetrize_spin,
            callback=callback,
        )
        raw_distribution, raw_counts, shots = _normalize_counts_distribution(
            counts,
            num_bits=system.num_spin_orbitals,
        )
        recovery_seed = spawn_seeds(seed, 1)[0]
        return self._run(
            system,
            raw_distribution,
            raw_counts,
            shots=shots,
            samples_per_batch=samples_per_batch,
            num_batches=num_batches,
            max_iterations=max_iterations,
            energy_tolerance=energy_tolerance,
            occupancy_tolerance=occupancy_tolerance,
            initial_occupancies=normalized_initial,
            include_configurations=required_by_user,
            carryover_threshold=carryover_threshold,
            symmetrize_spin=symmetrize_spin,
            callback=callback,
            recovery_seed=recovery_seed,
        )

    def _run(
        self,
        system: ElectronicSystem,
        raw_distribution: dict[int, float],
        raw_counts: dict[int, int],
        *,
        shots: int,
        samples_per_batch: int,
        num_batches: int,
        max_iterations: int,
        energy_tolerance: float,
        occupancy_tolerance: float,
        initial_occupancies: tuple[float, ...] | None,
        include_configurations: tuple[int, ...],
        carryover_threshold: float | None,
        symmetrize_spin: bool,
        callback: SQDCallback | None,
        recovery_seed: int,
    ) -> SQDResult:
        """Execute the self-consistent recovery and diagonalization loop."""
        required_by_user = include_configurations

        valid_distribution = {
            configuration: probability
            for configuration, probability in raw_distribution.items()
            if _has_target_particles(configuration, system)
        }
        valid_raw_shots = sum(
            count
            for configuration, count in raw_counts.items()
            if _has_target_particles(configuration, system)
        )
        if initial_occupancies is None and not valid_distribution:
            raise ValueError(
                "raw samples contain no target-sector configuration; "
                "provide initial_occupancies"
            )
        current_occupancies = initial_occupancies

        hamiltonian: Hamiltonian | None = None
        if self._subspace_solver is None:
            hamiltonian = ElectronicEnergy(system.integrals).to_qubit_hamiltonian()
            if hamiltonian.num_qubits != system.num_spin_orbitals:
                raise ValueError(
                    "mapped Hamiltonian width must match the electronic system"
                )

        rng = np.random.default_rng(recovery_seed)
        history: list[SQDIteration] = []
        best_result: SQDSubspaceResult | None = None
        previous_energy: float | None = None
        converged = False
        required_configurations = required_by_user
        for index in range(max_iterations):
            selected_distribution = (
                valid_distribution
                if current_occupancies is None
                else _recover_distribution(
                    raw_distribution,
                    system,
                    current_occupancies,
                    rng,
                )
            )
            batches = _sample_subspaces(
                selected_distribution,
                samples_per_batch=samples_per_batch,
                num_batches=num_batches,
                required_configurations=required_configurations,
                symmetrize_spin=symmetrize_spin,
                num_spatial_orbitals=system.num_spatial_orbitals,
                rng=rng,
            )
            if self._subspace_solver is None:
                assert hamiltonian is not None
                subspaces = tuple(
                    _solve_subspace(hamiltonian, configurations)
                    for configurations in batches
                )
            else:
                subspaces_list: list[SQDSubspaceResult] = []
                for configurations in batches:
                    result = self._subspace_solver(system, configurations)
                    if not isinstance(result, SQDSubspaceResult):
                        raise TypeError(
                            "subspace_solver must return an SQDSubspaceResult"
                        )
                    if result.configurations != configurations:
                        raise ValueError(
                            "subspace_solver result must match requested configurations"
                        )
                    if result.num_spin_orbitals != system.num_spin_orbitals:
                        raise ValueError(
                            "subspace_solver result width must match the system"
                        )
                    subspaces_list.append(result)
                subspaces = tuple(subspaces_list)
            next_occupancies = _average_occupancies(subspaces)
            iteration = SQDIteration(
                index,
                subspaces,
                next_occupancies,
                len(selected_distribution),
            )
            history.append(iteration)
            if callback is not None:
                callback(iteration)
            candidate = iteration.best_subspace
            if best_result is None or candidate.energy < best_result.energy:
                best_result = candidate
            required_configurations = _select_required_configurations(
                required_by_user,
                candidate,
                carryover_threshold=carryover_threshold,
                samples_per_batch=samples_per_batch,
                symmetrize_spin=symmetrize_spin,
                num_spatial_orbitals=system.num_spatial_orbitals,
            )

            occupancy_change = (
                math.inf
                if current_occupancies is None
                else max(
                    abs(after - before)
                    for after, before in zip(
                        next_occupancies,
                        current_occupancies,
                        strict=True,
                    )
                )
            )
            if (
                previous_energy is not None
                and abs(iteration.energy - previous_energy) <= energy_tolerance
                and occupancy_change <= occupancy_tolerance
            ):
                converged = True
                current_occupancies = next_occupancies
                break
            previous_energy = iteration.energy
            current_occupancies = next_occupancies

        assert best_result is not None
        assert current_occupancies is not None
        return SQDResult(
            best_result,
            current_occupancies,
            tuple(history),
            converged,
            shots,
            valid_raw_shots,
        )
