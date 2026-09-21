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

from cqlib.device import ExecutionResult
from cqlib.qis import Hamiltonian, Pauli, PauliString, Phase

from cqlib import Circuit

from ...protocols.estimator import Estimator, EstimatorRequest, EstimatorResult
from ...protocols.sampler import Sampler, SamplerRequest
from ._circuit_validation import validate_unitary_state_preparation


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
        raw_terms = ((observable, 1.0 + 0.0j),)
    else:
        raw_terms = tuple(observable.terms)

    order: list[tuple[int, int, int]] = []
    representatives: dict[tuple[int, int, int], PauliString] = {}
    coefficients: dict[tuple[int, int, int], complex] = {}
    for pauli, coefficient in raw_terms:
        key = (pauli.num_qubits, pauli.x_mask, pauli.z_mask)
        if key not in coefficients:
            order.append(key)
            representatives[key] = pauli.copy()
            coefficients[key] = 0.0j
        coefficients[key] += complex(coefficient) * pauli.phase.to_complex()

    terms: list[_ObservableTerm] = []
    for key in order:
        effective = coefficients[key]
        if effective == 0.0j:
            continue
        if effective.imag != 0.0:
            raise ValueError("observable must be Hermitian after term aggregation")
        representative = representatives[key]
        representative.phase = Phase.plus()
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


def _validate_sample_result(
    result: ExecutionResult,
    request: SamplerRequest,
) -> None:
    """Validate the counts needed for expectation and variance calculation."""
    if not result.status.is_terminal() or not result.status.is_success():
        raise ValueError("sampler must return successful terminal results")
    if result.shots != request.shots:
        raise ValueError("sampler result shots must match the request")
    if result.num_qubits != len(request.qubits):
        raise ValueError("sampler result width must match the requested qubits")
    if result.qubits != list(request.qubits):
        raise ValueError("sampler result qubits must preserve request order")

    total = 0
    width = len(request.qubits)
    for bitstring, count in result.counts.items():
        if not isinstance(bitstring, str) or len(bitstring) != width:
            raise ValueError("sampler count keys must match the requested width")
        if any(bit not in "01" for bit in bitstring):
            raise ValueError("sampler count keys must be binary strings")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("sampler counts must be non-negative integers")
        total += count
    if total != request.shots:
        raise ValueError("sampler counts must sum to the requested shots")


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
            for term_index, term in enumerate(active):
                term_shots = quotient + (1 if term_index < remainder else 0)
                rotated = _basis_circuit(bound, term.pauli)
                sampler_request = SamplerRequest(
                    rotated,
                    tuple(rotated.qubits),
                    term_shots,
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
            sample_results = self._sampler.run([job.sampler_request for job in jobs])
            if len(sample_results) != len(jobs):
                raise ValueError("sampler must return one result per request")
            for result, job in zip(sample_results, jobs):
                if not isinstance(result, ExecutionResult):
                    raise TypeError(
                        "sampler results must be cqlib.device.ExecutionResult"
                    )
                _validate_sample_result(result, job.sampler_request)
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
