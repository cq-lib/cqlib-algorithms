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

"""Finite-shot Pauli expectation estimation through an injected Sampler."""

from __future__ import annotations

from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
from typing import final

from cqlib import Circuit
from cqlib.device import ExecutionResult
from cqlib.qis import Hamiltonian, Pauli, PauliString, Phase

from ..._utils.circuit import validate_unitary_state_preparation
from ..._utils.hamiltonian import aggregate_finite_pauli_terms
from ...protocols.estimator import Estimator, EstimatorRequest, EstimatorResult
from ...protocols.sampler import (
    Sampler,
    SamplerRequest,
    require_sampler_results,
    spawn_seeds,
)


@dataclass(frozen=True, slots=True)
class _ObservableTerm:
    pauli: PauliString
    coefficient: float
    support_mask: int


@dataclass(frozen=True, slots=True)
class _SampleJob:
    request_index: int
    coefficient: float
    support_mask: int
    sampler_request: SamplerRequest


@dataclass(slots=True)
class _Accumulator:
    expectation: float
    variance: float = 0.0
    variance_known: bool = True


def _normalized_terms(
    observable: Hamiltonian | PauliString,
) -> tuple[_ObservableTerm, ...]:
    """Absorb Pauli phases and stably combine equal phase-free strings."""
    raw_terms: Sequence[tuple[PauliString, complex]]
    if isinstance(observable, PauliString):
        representative = observable.copy()
        effective = representative.phase.to_complex()
        representative.phase = Phase.plus()
        raw_terms = ((representative, effective),)
    else:
        raw_terms = aggregate_finite_pauli_terms(observable, name="observable")

    terms: list[_ObservableTerm] = []
    for representative, effective in raw_terms:
        if effective == 0.0j:
            continue
        if effective.imag != 0.0:
            raise ValueError("observable must be Hermitian after term aggregation")
        terms.append(
            _ObservableTerm(
                representative,
                float(effective.real),
                representative.x_mask | representative.z_mask,
            )
        )
    return tuple(terms)


def _basis_circuit(circuit: Circuit, pauli: PauliString) -> Circuit:
    """Return an unmeasured circuit rotated into one Pauli product basis."""
    rotated = copy(circuit)
    for position, qubit in enumerate(rotated.qubits):
        axis = pauli.get_pauli(position)
        if axis == Pauli.x():
            rotated.h(qubit)
        elif axis == Pauli.y():
            rotated.sdg(qubit)
            rotated.h(qubit)
        elif axis not in (Pauli.i(), Pauli.z()):
            raise ValueError("observable contains an unsupported Pauli axis")
    return rotated


def _sample_moments(
    result: ExecutionResult,
    support_mask: int,
) -> tuple[float, float | None]:
    """Return a parity mean and unbiased variance estimate of that mean."""
    signed_total = 0
    for bitstring, count in result.counts.items():
        outcome = int(bitstring, 2)
        eigenvalue = -1 if ((outcome & support_mask).bit_count() & 1) else 1
        signed_total += eigenvalue * count
    mean = signed_total / result.shots
    if result.shots == 1:
        return mean, None
    return mean, (1.0 - mean * mean) / (result.shots - 1)


@final
class SamplingEstimator(Estimator):
    """Estimate Hermitian Pauli observables with finite-shot sampling.

    A concrete :class:`~cqlib_algorithms.protocols.sampler.Sampler` is injected
    at construction. Each distinct non-identity phase-free Pauli term is
    measured independently in its local product basis. Equal terms are first
    combined and global Pauli phases are absorbed into real coefficients.

    ``EstimatorRequest.shots`` is the total shot budget for that estimator
    request. It is distributed as evenly as possible over active terms, in
    stable first-occurrence order; therefore the budget must be at least the
    number of active terms. Identity-only observables require no sampler call.
    Independent term measurements make the total variance the sum of term
    variances. For two or more shots per term, each term uses the unbiased
    sample estimate ``(1 - mean**2) / (shots - 1)`` for the variance of its
    sample mean. If any active term receives one shot, result variance is
    ``None`` because it cannot be estimated from that sample.

    ``EstimatorRequest.seed`` reaches the sampler through one child seed per
    active term, so an explicit seed reproduces the whole estimate while the
    term measurements stay independent. The identity term consumes no
    randomness.

    This component deliberately does not duplicate QWC grouping. The existing
    measurement-planner component owns that optimization; integration requires
    its plan to expose an unmeasured basis circuit compatible with the Sampler
    contract. This implementation is the independent, correctness-first
    product-basis path.

    Args:
        sampler: Explicit Sampler implementation used for all measurements.

    Raises:
        TypeError: If ``sampler`` or a request has the wrong nominal type.
        ValueError: If shots are absent/insufficient, the circuit is not a
            unitary preparation, or sampler results violate their contract.
    """

    __slots__ = ("_sampler",)

    def __init__(self, sampler: Sampler) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError("sampler must be a Sampler instance")
        self._sampler = sampler

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        """Estimate a complete batch in input order using one Sampler call."""
        accumulators: list[_Accumulator] = []
        jobs: list[_SampleJob] = []

        for request_index, request in enumerate(requests):
            if not isinstance(request, EstimatorRequest):
                raise TypeError("requests must contain EstimatorRequest instances")
            request.validate()
            if request.shots is None:
                raise ValueError("sampling estimation requires a positive shot count")
            validate_unitary_state_preparation(request.circuit)

            bound = request.circuit.assign_parameters(dict(request.parameter_bindings))
            terms = _normalized_terms(request.observable)
            identity = sum(
                (term.coefficient for term in terms if term.support_mask == 0),
                0.0,
            )
            active = tuple(term for term in terms if term.support_mask != 0)
            accumulators.append(_Accumulator(identity))
            if not active:
                continue
            if request.shots < len(active):
                raise ValueError(
                    "shots must be at least the number of sampled Pauli terms"
                )

            quotient, remainder = divmod(request.shots, len(active))
            # Each term draws from its own child seed so that the term
            # measurements stay independent, while one estimator request with
            # an explicit seed still reproduces the whole batch.
            term_seeds = spawn_seeds(request.seed, len(active))
            for term_index, term in enumerate(active):
                term_shots = quotient + (1 if term_index < remainder else 0)
                rotated = _basis_circuit(bound, term.pauli)
                sampler_request = SamplerRequest(
                    rotated,
                    tuple(rotated.qubits),
                    term_shots,
                    term_seeds[term_index],
                )
                jobs.append(
                    _SampleJob(
                        request_index,
                        term.coefficient,
                        term.support_mask,
                        sampler_request,
                    )
                )

        if jobs:
            sampler_requests = [job.sampler_request for job in jobs]
            sample_results = require_sampler_results(
                self._sampler.run(sampler_requests),
                sampler_requests,
            )
            for result, job in zip(sample_results, jobs, strict=True):
                mean, mean_variance = _sample_moments(result, job.support_mask)
                accumulator = accumulators[job.request_index]
                accumulator.expectation += job.coefficient * mean
                if mean_variance is None:
                    accumulator.variance_known = False
                else:
                    accumulator.variance += job.coefficient**2 * mean_variance

        return [
            EstimatorResult(
                accumulator.expectation,
                accumulator.variance if accumulator.variance_known else None,
            )
            for accumulator in accumulators
        ]
