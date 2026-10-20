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

"""Bernstein-Vazirani hidden-string recovery.

References
----------
[1] E. Bernstein and U. Vazirani, "Quantum complexity theory," SIAM
    Journal on Computing, 26(5), 1411-1473, 1997.
    DOI: 10.1137/S0097539796300921.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import final

from cqlib import Circuit

from ..operators.composite import HadamardPhaseQuery
from ..operators.oracle import (
    BooleanFunction,
    BooleanVectorFunction,
)
from ..protocols.sampler import Sampler, SamplerRequest, validate_sampler_results


@dataclass(frozen=True, slots=True)
class BernsteinVaziraniRequest:
    """One hidden-string recovery query.

    The oracle is the immutable Boolean marking function; multi-output
    functions are reduced to a scalar predicate by output-mask parity, so
    the linearity promise applies to that parity function.

    Args:
        function: Immutable scalar or vector Boolean function, promised to
            be linear, ``f(x) = s . x (mod 2)``, for a hidden string ``s``.
        shots: Positive number of recovery shots.
        output_mask: Selected output bits for vector functions, as in
            :class:`PhaseOracle`.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If ``shots`` is not positive or the mask is invalid.
    """

    function: BooleanFunction | BooleanVectorFunction
    shots: int
    output_mask: int | None = None

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Revalidate the request fields before execution."""
        if isinstance(self.shots, bool) or not isinstance(self.shots, int):
            raise TypeError("shots must be an integer")
        if self.shots <= 0:
            raise ValueError("shots must be greater than zero")
        HadamardPhaseQuery(self.function, output_mask=self.output_mask)


@dataclass(frozen=True, slots=True)
class BernsteinVaziraniResult:
    """Empirical modal candidates and their observed support.

    ``modal_strings`` contains every bitstring tied for the largest observed
    count, sorted by its little-endian integer value. Bit ``q`` of each entry
    corresponds to qubit ``q``. Under ideal promised execution there is one
    candidate, it is the hidden string, and ``support == shots``. With noisy
    observations, the candidates are empirical modes rather than a claim of
    successful error correction.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If the statistics are inconsistent.
    """

    modal_strings: tuple[tuple[bool, ...], ...]
    support: int
    shots: int

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.modal_strings, tuple):
            raise TypeError("modal_strings must be a tuple")
        if not self.modal_strings:
            raise ValueError("modal_strings must not be empty")
        if any(not isinstance(bits, tuple) for bits in self.modal_strings):
            raise TypeError("modal_strings entries must be tuples")
        if any(not bits for bits in self.modal_strings):
            raise ValueError("modal strings must not be empty")
        if any(
            not isinstance(bit, bool) for bits in self.modal_strings for bit in bits
        ):
            raise TypeError("modal string entries must be bool")
        width = len(self.modal_strings[0])
        if any(len(bits) != width for bits in self.modal_strings):
            raise ValueError("modal strings must have equal width")
        canonical = tuple(
            sorted(
                set(self.modal_strings),
                key=lambda bits: sum(int(bit) << q for q, bit in enumerate(bits)),
            )
        )
        if self.modal_strings != canonical:
            raise ValueError("modal_strings must be unique and sorted")
        if isinstance(self.support, bool) or not isinstance(self.support, int):
            raise TypeError("support must be an integer")
        if isinstance(self.shots, bool) or not isinstance(self.shots, int):
            raise TypeError("shots must be an integer")
        if self.shots <= 0:
            raise ValueError("shots must be greater than zero")
        if not 0 < self.support <= self.shots:
            raise ValueError("support must be positive and not exceed shots")
        num_modes = len(self.modal_strings)
        num_outcomes = 1 << width
        minimum_shots = num_modes * self.support
        maximum_shots = minimum_shots + (num_outcomes - num_modes) * (self.support - 1)
        if not minimum_shots <= self.shots <= maximum_shots:
            raise ValueError("modal statistics are inconsistent with shots")

    @property
    def is_ambiguous(self) -> bool:
        """Return whether multiple outcomes share the largest count."""
        return len(self.modal_strings) != 1

    @property
    def hidden_string(self) -> tuple[bool, ...] | None:
        """Return the unique empirical mode, or ``None`` for a tie."""
        return None if self.is_ambiguous else self.modal_strings[0]

    @property
    def hidden_integer(self) -> int | None:
        """Return the unique mode as an integer, or ``None`` for a tie."""
        if self.hidden_string is None:
            return None
        return sum(int(bit) << q for q, bit in enumerate(self.hidden_string))


@final
class BernsteinVazirani:
    """Recover the hidden string of a linear Boolean function.

    For ``f(x) = s . x (mod 2)``, one phase-oracle query
    ``S_f |x> = (-1)**f(x) |x>`` between two Hadamard layers satisfies
    ``H**n S_f H**n |0...0> = |s>`` exactly: the all-qubit measurement (added
    by the Sampler on a private execution circuit) returns the hidden string
    with probability one under ideal execution. The component therefore reads
    every modal (most frequent) observed bitstring as an empirical candidate
    and reports how many shots supported each mode. A unique mode is exposed
    as ``hidden_string``; a tie is reported as ambiguous rather than resolved
    without evidence.

    The oracle circuit is built from the immutable Boolean function by
    :class:`PhaseOracle`; multi-output functions define the derived scalar
    function ``g(x)`` as the parity selected by ``output_mask``, and the
    linearity promise applies to ``g``. The algorithm layer does not
    classically inspect the truth table to decide the promise, although
    explicit-table oracle synthesis necessarily reads it. An input violating
    the promise produces a formally computed but semantically unspecified
    recovered string.

    The complete input batch is validated and transformed before one
    flattened Sampler call. The component performs no classical pre-check,
    iteration management, or backend administration. Each constructed circuit
    contains one phase-oracle application; requesting multiple shots repeats
    the circuit-execution experiment and must not be described as one total
    physical oracle evaluation. References: Bernstein and Vazirani, *Quantum
    complexity theory*, SIAM J. Comput. 26(5), 1411 (1997); Cleve et al.,
    *Quantum Algorithms Revisited*, Proc. R. Soc. Lond. A 454, 339 (1998).

    Args:
        sampler: Concrete Sampler implementation used for all measurements.

    Raises:
        TypeError: If ``sampler``, a request, or a result has the wrong type.
        ValueError: If request invariants or sampler-result contracts are
            violated.
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
    def build_circuit(
        function: BooleanFunction | BooleanVectorFunction,
        *,
        output_mask: int | None = None,
    ) -> Circuit:
        """Build the unmeasured ``H**n -> S_f -> H**n`` query circuit.

        The returned circuit contains no terminal measurement; the Sampler
        appends it on a private execution circuit.
        """
        return HadamardPhaseQuery(
            function,
            output_mask=output_mask,
        ).build_circuit()

    def run(
        self,
        requests: Sequence[BernsteinVaziraniRequest],
        /,
    ) -> list[BernsteinVaziraniResult]:
        """Evaluate a complete batch in input order with one Sampler call."""
        sampler_requests: list[SamplerRequest] = []
        for request in requests:
            if not isinstance(request, BernsteinVaziraniRequest):
                raise TypeError(
                    "requests must contain BernsteinVaziraniRequest instances"
                )
            request.validate()
            circuit = self.build_circuit(
                request.function, output_mask=request.output_mask
            )
            sampler_requests.append(
                SamplerRequest(circuit, tuple(circuit.qubits), request.shots)
            )

        if not sampler_requests:
            return []

        sample_results = validate_sampler_results(
            self._sampler.run(sampler_requests), sampler_requests
        )

        results: list[BernsteinVaziraniResult] = []
        for result, sampler_request in zip(sample_results, sampler_requests):
            support = max(result.counts.values())
            outcomes = sorted(
                outcome for outcome, count in result.counts.items() if count == support
            )
            # Little-endian: the rightmost bitstring character is qubit 0.
            modal_strings = tuple(
                tuple(bit == "1" for bit in reversed(outcome)) for outcome in outcomes
            )
            results.append(
                BernsteinVaziraniResult(
                    modal_strings=modal_strings,
                    support=support,
                    shots=sampler_request.shots,
                )
            )
        return results
