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

"""Immutable results for sample-based quantum diagonalization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from ...._utils.numeric import validate_positive_integer

_NORMALIZATION_TOLERANCE = 1e-10


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

        validate_positive_integer(
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

        validate_positive_integer(
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
        validate_positive_integer(self.shots, "shots")
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


__all__ = ["SQDIteration", "SQDResult", "SQDSubspaceResult"]
