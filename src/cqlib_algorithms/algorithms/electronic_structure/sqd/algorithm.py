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

import copy
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from cqlib.qis import Hamiltonian

from ...._utils.circuit import validate_bound_static_unitary_circuit
from ...._utils.numeric import (
    normalize_nonnegative_real,
    validate_positive_integer,
)
from ...._utils.state_preparation import build_state_preparation_circuit
from ....extension.electronic_structure.electronic import (
    ElectronicEnergy,
    ElectronicSystem,
)
from ....protocols.sampler import (
    Sampler,
    SamplerRequest,
    require_sampler_results,
    spawn_seeds,
)
from ....protocols.state_preparation import StatePreparation
from .recovery import (
    _configuration_spin_strings,
    _has_target_particles,
    _normalize_occupancies,
    _recover_distribution,
)
from .results import (
    SQDIteration,
    SQDResult,
    SQDSubspaceResult,
    _subspace_occupancies,
    _validate_configuration,
)
from .subspace import (
    SQDSubspaceSolver,
    _sample_subspaces,
    _select_required_configurations,
    _solve_subspace,
)

SQDCallback = Callable[[SQDIteration], None]


def _normalize_optional_nonnegative_real(
    value: float | None,
    name: str,
) -> float | None:
    """Return a normalized optional non-negative real value."""
    if value is None:
        return None
    return normalize_nonnegative_real(value, name)


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
    validate_positive_integer(samples_per_batch, "samples_per_batch")
    validate_positive_integer(num_batches, "num_batches")
    validate_positive_integer(max_iterations, "max_iterations")
    normalized_energy_tolerance = normalize_nonnegative_real(
        energy_tolerance,
        "energy_tolerance",
    )
    normalized_occupancy_tolerance = normalize_nonnegative_real(
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
            It is required by :meth:`run` but not by
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

    def run(
        self,
        system: ElectronicSystem,
        state_preparation: StatePreparation,
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
            state_preparation: Fully bound preparation component in
                interleaved Jordan--Wigner occupation order.
                Components are built with prefix ``"sqd"``. The execution
                circuit is copied independently of the caller's input.
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
        preparation = build_state_preparation_circuit(state_preparation, prefix="sqd")
        if self._sampler is None:
            raise ValueError("run requires an SQD sampler")
        validate_positive_integer(shots, "shots")
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

        preparation.validate()
        if preparation.used_symbols:
            raise ValueError("state_preparation must be fully bound")
        if preparation.num_qubits != system.num_spin_orbitals:
            raise ValueError(
                "state_preparation width must equal the number of spin orbitals"
            )
        # Occupation modes follow positions; preserve the backend qubit identifiers.
        circuit = copy.deepcopy(preparation)
        validate_bound_static_unitary_circuit(circuit, "state_preparation")

        sampler_seed, recovery_seed = spawn_seeds(seed, 2)
        request = SamplerRequest(
            circuit,
            tuple(circuit.qubits),
            shots,
            sampler_seed,
        )
        results = require_sampler_results(self._sampler.run([request]), [request])
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

    def compute(
        self,
        system: ElectronicSystem,
        state_preparation: StatePreparation,
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
        """Run SQD with the same inputs and result as :meth:`run`."""
        return self.run(
            system,
            state_preparation,
            shots=shots,
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
            seed=seed,
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


__all__ = ["SQD", "SQDCallback"]
