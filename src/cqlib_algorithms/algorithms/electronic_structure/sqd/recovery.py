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

"""Particle-sector recovery for sample-based quantum diagonalization."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from numbers import Real

import numpy as np

from ....extension.chemistry.electronic import ElectronicSystem
from ....protocols.sampler import spawn_seeds
from .results import _validate_configuration

_FLIP_EPSILON = 0.01


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


__all__ = ["recover_sqd_configurations"]
