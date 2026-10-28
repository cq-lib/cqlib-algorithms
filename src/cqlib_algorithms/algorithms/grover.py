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

"""Grover search over finite Boolean functions.

References
----------
[1] L. K. Grover, "A fast quantum mechanical algorithm for database search,"
    Proceedings of the 28th Annual ACM Symposium on Theory of Computing,
    212-219, 1996. DOI: 10.1145/237814.237866.
[2] G. Brassard, P. Hoyer, M. Mosca, and A. Tapp, "Quantum amplitude
    amplification and estimation," Contemporary Mathematics, 305, 53-74,
    2002. DOI: 10.1090/conm/305/05215.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import final

from cqlib import Circuit

from ..operators.composite import GroverOperator
from ..operators.oracle import BooleanFunction, BooleanVectorFunction, PhaseOracle
from ..operators.state_preparation import UniformSuperposition
from ..protocols.sampler import Sampler, SamplerRequest, validate_sampler_results


def _selected_marked_states(
    function: BooleanFunction | BooleanVectorFunction,
    output_mask: int | None,
) -> tuple[int, ...]:
    """Return states selected by the phase-oracle scalar predicate."""
    oracle = PhaseOracle(function, output_mask=output_mask)
    if isinstance(function, BooleanFunction):
        return function.marked_states
    return tuple(
        state
        for state, values in enumerate(function.truth_table)
        if sum(
            int(value)
            for output, value in enumerate(values)
            if (oracle.output_mask >> output) & 1
        )
        % 2
    )


def _validate_iterations(iterations: int | None) -> None:
    """Validate an optional non-negative Grover iteration count."""
    if iterations is None:
        return
    if isinstance(iterations, bool) or not isinstance(iterations, int):
        raise TypeError("iterations must be an integer or None")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")


@dataclass(frozen=True, slots=True)
class GroverRequest:
    """One finite Boolean search request.

    The selected scalar predicate is ``function`` itself for a scalar
    function. For a vector function it is the parity of the outputs selected
    by ``output_mask``. The default iteration count is chosen from the exact
    number of marked inputs. Supplying ``iterations`` overrides that choice
    and is useful for experiments or externally selected schedules.

    Args:
        function: Immutable scalar or vector Boolean marking function.
        shots: Positive number of search-circuit samples.
        output_mask: Selected output bits for a vector function, as in
            :class:`PhaseOracle`.
        iterations: Non-negative Grover iteration count, or ``None`` to use
            the optimal first-maximum count for the known marked fraction.
        seed: Non-negative finite-shot seed, or ``None`` for fresh draws.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a field is out of range or no state is marked.
    """

    function: BooleanFunction | BooleanVectorFunction
    shots: int
    output_mask: int | None = None
    iterations: int | None = None
    seed: int | None = None

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Revalidate all request fields before execution.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If a field is out of range or no state is marked.
        """
        if isinstance(self.shots, bool) or not isinstance(self.shots, int):
            raise TypeError("shots must be an integer")
        if self.shots <= 0:
            raise ValueError("shots must be greater than zero")
        _validate_iterations(self.iterations)
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise TypeError("seed must be an integer or None")
            if self.seed < 0:
                raise ValueError("seed must be non-negative")
        marked_states = _selected_marked_states(self.function, self.output_mask)
        if not marked_states:
            raise ValueError("function must mark at least one state")


@dataclass(frozen=True, slots=True)
class GroverResult:
    """Complete basis counts and success statistics for one search.

    ``counts[state]`` is the observed count of computational-basis integer
    ``state``. Integer bit ``q`` and ``modal_bitstrings[*][q]`` both refer to
    qubit ``q``; these are little-endian coordinate tuples even though raw
    sampler strings print the highest requested qubit first.

    Args:
        marked_states: Sorted states selected by the search predicate.
        counts: Complete basis-count tuple indexed by state integer.
        iterations: Grover iterations contained in each sampled circuit.
        shots: Total number of samples.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If dimensions, ordering, or counts are inconsistent.
    """

    marked_states: tuple[int, ...]
    counts: tuple[int, ...]
    iterations: int
    shots: int

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.marked_states, tuple):
            raise TypeError("marked_states must be a tuple")
        if any(
            isinstance(state, bool) or not isinstance(state, int)
            for state in self.marked_states
        ):
            raise TypeError("marked_states entries must be integers")
        if not self.marked_states:
            raise ValueError("marked_states must not be empty")
        if self.marked_states != tuple(sorted(set(self.marked_states))):
            raise ValueError("marked_states must be unique and sorted")
        if not isinstance(self.counts, tuple):
            raise TypeError("counts must be a tuple")
        if len(self.counts) <= 1 or len(self.counts) & (len(self.counts) - 1):
            raise ValueError("counts length must be a power of two greater than one")
        if any(
            isinstance(count, bool) or not isinstance(count, int)
            for count in self.counts
        ):
            raise TypeError("counts entries must be integers")
        if any(count < 0 for count in self.counts):
            raise ValueError("counts entries must be non-negative")
        if any(state < 0 or state >= len(self.counts) for state in self.marked_states):
            raise ValueError("marked states must be valid count indices")
        _validate_iterations(self.iterations)
        if self.iterations is None:
            raise TypeError("iterations must be an integer")
        if isinstance(self.shots, bool) or not isinstance(self.shots, int):
            raise TypeError("shots must be an integer")
        if self.shots <= 0:
            raise ValueError("shots must be greater than zero")
        if sum(self.counts) != self.shots:
            raise ValueError("counts must sum to shots")

    @property
    def num_qubits(self) -> int:
        """Return the search-register width."""
        return len(self.counts).bit_length() - 1

    @property
    def num_solutions(self) -> int:
        """Return the number of marked states."""
        return len(self.marked_states)

    @property
    def success_count(self) -> int:
        """Return the number of observed marked outcomes."""
        return sum(self.counts[state] for state in self.marked_states)

    @property
    def success_probability(self) -> float:
        """Return the empirical marked-outcome probability."""
        return self.success_count / self.shots

    @property
    def modal_states(self) -> tuple[int, ...]:
        """Return all states tied for the largest observed count."""
        support = max(self.counts)
        return tuple(
            state for state, count in enumerate(self.counts) if count == support
        )

    @property
    def modal_bitstrings(self) -> tuple[tuple[bool, ...], ...]:
        """Return modal states as little-endian qubit-coordinate tuples."""
        return tuple(
            tuple(bool((state >> qubit) & 1) for qubit in range(self.num_qubits))
            for state in self.modal_states
        )

    @property
    def marked_modal_states(self) -> tuple[int, ...]:
        """Return modal outcomes that satisfy the marking predicate."""
        marked = set(self.marked_states)
        return tuple(state for state in self.modal_states if state in marked)


@final
class Grover:
    """Search a finite Boolean domain by amplitude amplification.

    For ``M`` marked values among ``N = 2**n`` inputs, uniform preparation
    gives ``sin(theta)**2 = M/N``. After ``k`` applications of
    :class:`GroverOperator`, the total marked probability is
    ``sin((2*k + 1)*theta)**2``. With no explicit iteration count this class
    chooses the non-negative integer nearest the first continuous maximum,
    breaking equal-probability ties toward fewer oracle queries.

    The implementation composes :class:`UniformSuperposition`,
    :class:`PhaseOracle`, and :class:`GroverOperator`; it does not duplicate
    their gate synthesis. It accepts an injected :class:`Sampler`, validates
    the complete batch before one sampler call, and derives all reported
    success statistics from returned counts. Explicit truth-table oracle
    synthesis is exponential classical preprocessing and is outside the
    abstract query-complexity claim. Each circuit contains ``iterations``
    oracle applications, while finite-shot execution repeats that circuit.

    Args:
        sampler: Concrete Sampler used for final-register measurements.

    Raises:
        TypeError: If ``sampler`` or a request has the wrong type.
        ValueError: If request or result contracts are violated.
    """

    __slots__ = ("_sampler",)

    def __init__(self, sampler: Sampler) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError("sampler must be a Sampler instance")
        self._sampler = sampler

    @property
    def sampler(self) -> Sampler:
        """Return the injected Sampler implementation."""
        return self._sampler

    @staticmethod
    def optimal_iterations(num_qubits: int, num_solutions: int) -> int:
        """Return the shortest integer nearest the first success maximum.

        Args:
            num_qubits: Positive search-register width.
            num_solutions: Number of marked inputs in ``1..2**num_qubits``.

        Returns:
            Non-negative number of Grover iterations.

        Raises:
            TypeError: If an argument is not an integer.
            ValueError: If an argument is outside its valid range.
        """
        if isinstance(num_qubits, bool) or not isinstance(num_qubits, int):
            raise TypeError("num_qubits must be an integer")
        if num_qubits <= 0:
            raise ValueError("num_qubits must be greater than zero")
        if isinstance(num_solutions, bool) or not isinstance(num_solutions, int):
            raise TypeError("num_solutions must be an integer")
        dimension = 1 << num_qubits
        if not 1 <= num_solutions <= dimension:
            raise ValueError("num_solutions must satisfy 1 <= value <= 2**num_qubits")

        theta = math.asin(math.sqrt(num_solutions / dimension))
        continuous = math.pi / (4.0 * theta) - 0.5
        lower = max(0, math.floor(continuous))
        upper = max(0, math.ceil(continuous))
        candidates = {lower, upper}
        return min(
            candidates,
            key=lambda value: (-(math.sin((2 * value + 1) * theta) ** 2), value),
        )

    @staticmethod
    def build_circuit(
        function: BooleanFunction | BooleanVectorFunction,
        *,
        output_mask: int | None = None,
        iterations: int | None = None,
    ) -> Circuit:
        """Build one unmeasured standard Grover search circuit.

        ``iterations=None`` selects the optimal count from the marked fraction.

        Args:
            function: Immutable scalar or vector Boolean marking function.
            output_mask: Selected output bits for a vector function, as in
                :class:`PhaseOracle`.
            iterations: Non-negative Grover iteration count, or ``None`` to
                select the first discrete success maximum.

        Returns:
            Fresh unmeasured circuit containing uniform preparation followed
            by the selected number of Grover iterations.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If an argument is out of range, a vector mask is
                invalid, or the selected predicate marks no states.
        """
        _validate_iterations(iterations)
        marked_states = _selected_marked_states(function, output_mask)
        if not marked_states:
            raise ValueError("function must mark at least one state")
        num_qubits = function.num_variables
        count = (
            Grover.optimal_iterations(num_qubits, len(marked_states))
            if iterations is None
            else iterations
        )
        circuit = UniformSuperposition(num_qubits).build_circuit("grover")
        oracle = PhaseOracle(function, output_mask=output_mask).build_circuit()
        grover_iteration = GroverOperator(oracle).build_circuit()
        for _ in range(count):
            circuit.compose(grover_iteration, circuit.qubits)
        return circuit

    def run(self, requests: Sequence[GroverRequest], /) -> list[GroverResult]:
        """Execute a complete request batch and return results in input order.

        The complete batch is validated and transformed before the single
        Sampler call. An empty request sequence returns an empty list without
        invoking the Sampler.

        Args:
            requests: Grover search requests in output order.

        Returns:
            One immutable search result per request, in input order.

        Raises:
            TypeError: If a request or Sampler result has the wrong type.
            ValueError: If a request invariant or Sampler-result contract is
                violated.
        """
        sampler_requests: list[SamplerRequest] = []
        marked_states_by_request: list[tuple[int, ...]] = []
        iteration_counts: list[int] = []
        for request in requests:
            if not isinstance(request, GroverRequest):
                raise TypeError("requests must contain GroverRequest instances")
            request.validate()
            marked_states = _selected_marked_states(
                request.function, request.output_mask
            )
            iterations = (
                self.optimal_iterations(
                    request.function.num_variables, len(marked_states)
                )
                if request.iterations is None
                else request.iterations
            )
            circuit = self.build_circuit(
                request.function,
                output_mask=request.output_mask,
                iterations=iterations,
            )
            sampler_requests.append(
                SamplerRequest(
                    circuit,
                    tuple(circuit.qubits),
                    request.shots,
                    request.seed,
                )
            )
            marked_states_by_request.append(marked_states)
            iteration_counts.append(iterations)

        if not sampler_requests:
            return []

        sampled = validate_sampler_results(
            self._sampler.run(sampler_requests), sampler_requests
        )
        results: list[GroverResult] = []
        for result, sampler_request, marked_states, iterations in zip(
            sampled, sampler_requests, marked_states_by_request, iteration_counts
        ):
            counts = [0] * (1 << len(sampler_request.qubits))
            for outcome, count in result.counts.items():
                counts[int(outcome, 2)] = count
            results.append(
                GroverResult(
                    marked_states=marked_states,
                    counts=tuple(counts),
                    iterations=iterations,
                    shots=sampler_request.shots,
                )
            )
        return results
