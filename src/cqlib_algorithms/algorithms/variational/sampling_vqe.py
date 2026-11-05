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

"""Sampling variational eigensolving for diagonal objectives.

References
----------
[1] A. Peruzzo et al., "A variational eigenvalue solver on a photonic quantum
    processor," Nature Communications 5, 4213, 2014.
    DOI: 10.1038/ncomms5213.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import final

import numpy as np
from cqlib.device import ExecutionResult
from cqlib.qis import Hamiltonian, PauliString

from ..._utils.hamiltonian import (
    snapshot_nonzero_hermitian_hamiltonian,
)
from ..._utils.numeric import normalize_finite_real
from ...protocols import (
    Ansatz,
    Estimator,
    EstimatorRequest,
    EstimatorResult,
    Optimizer,
    OptimizerRequest,
    OptimizerResult,
    Sampler,
    SamplerRequest,
)
from ...protocols.sampler import require_sampler_results, spawn_seeds
from .result import VQAResult


@dataclass(frozen=True, slots=True)
class SamplingVQEMeasurement:
    """The best computational-basis outcome observed during optimization.

    Args:
        bitstring: Full-register outcome with qubit zero as the least
            significant bit.
        value: Diagonal objective value of the outcome.
        probability: Empirical probability in the objective sample where the
            outcome was selected.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a field is malformed or non-finite.
    """

    bitstring: str
    value: float
    probability: float

    def __post_init__(self) -> None:
        if not isinstance(self.bitstring, str):
            raise TypeError("bitstring must be a string")
        if not self.bitstring or any(bit not in "01" for bit in self.bitstring):
            raise ValueError("bitstring must be a non-empty binary string")
        object.__setattr__(self, "value", normalize_finite_real(self.value, "value"))
        probability = normalize_finite_real(self.probability, "probability")
        if probability <= 0.0 or probability > 1.0:
            raise ValueError("probability must be in (0, 1]")
        object.__setattr__(self, "probability", probability)


@dataclass(frozen=True, slots=True, repr=False)
class SamplingVQEResult(VQAResult):
    """Optimizer result plus an independent final joint-basis sample.

    ``optimal_value`` is the optimizer's final objective estimate.
    ``final_sample_value`` is the ordinary sample mean from a separate final
    execution, so the two values need not be equal at finite shot count.
    ``best_measurement`` is accumulated only from optimizer objective samples;
    it is intentionally independent of ``final_distribution``.

    Args:
        optimal_value: Optimizer's final sampled objective estimate.
        optimal_parameters: Final bindings in optimizer order.
        optimal_circuit: Fully bound circuit at the optimizer's final point.
        optimizer_result: Metadata returned by the optimizer.
        final_distribution: Independent final empirical bitstring distribution.
        final_sample_value: Mean objective value of the final distribution.
        best_measurement: Best outcome seen in any optimizer objective sample.
        shots: Shot count used by every objective and final sample.
        objective_sense: Objective direction inherited from ``VQAResult``.

    Raises:
        TypeError: If sampling metadata has an invalid type.
        ValueError: If a distribution or value is inconsistent.
    """

    final_distribution: Mapping[str, float]
    final_sample_value: float
    best_measurement: SamplingVQEMeasurement
    shots: int

    __hash__ = None

    def __post_init__(self) -> None:
        VQAResult.__post_init__(self)
        if isinstance(self.shots, bool) or not isinstance(self.shots, int):
            raise TypeError("shots must be an integer")
        if self.shots <= 0:
            raise ValueError("shots must be positive")
        if not isinstance(self.final_distribution, Mapping):
            raise TypeError("final_distribution must be a mapping")
        distribution: dict[str, float] = {}
        width = self.optimal_circuit.num_qubits
        for bitstring, probability in self.final_distribution.items():
            if not isinstance(bitstring, str):
                raise TypeError("distribution keys must be strings")
            if len(bitstring) != width or any(bit not in "01" for bit in bitstring):
                raise ValueError("distribution keys must match the circuit width")
            numeric = normalize_finite_real(probability, "distribution probability")
            if numeric <= 0.0 or numeric > 1.0:
                raise ValueError("distribution probabilities must be in (0, 1]")
            if not math.isclose(
                numeric * self.shots,
                round(numeric * self.shots),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "distribution probabilities must correspond to integer counts"
                )
            distribution[bitstring] = numeric
        if not distribution or not math.isclose(
            sum(distribution.values()), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("final_distribution probabilities must sum to one")
        object.__setattr__(self, "final_distribution", MappingProxyType(distribution))
        object.__setattr__(
            self,
            "final_sample_value",
            normalize_finite_real(self.final_sample_value, "final_sample_value"),
        )
        if not isinstance(self.best_measurement, SamplingVQEMeasurement):
            raise TypeError("best_measurement must be a SamplingVQEMeasurement")
        if len(self.best_measurement.bitstring) != width:
            raise ValueError("best_measurement bitstring must match the circuit width")
        if not math.isclose(
            self.best_measurement.probability * self.shots,
            round(self.best_measurement.probability * self.shots),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "best_measurement probability must correspond to an integer count"
            )


@dataclass(frozen=True, slots=True)
class _DiagonalTerm:
    z_mask: int
    coefficient: float


@dataclass(frozen=True, slots=True)
class _SampleSummary:
    distribution: Mapping[str, float]
    mean: float
    variance: float | None
    best: SamplingVQEMeasurement


def _diagonal_terms(
    operator: Hamiltonian | PauliString,
) -> tuple[Hamiltonian, tuple[_DiagonalTerm, ...]]:
    """Return a diagonal Hermitian snapshot and its real phase-free terms."""
    if isinstance(operator, PauliString):
        operator = Hamiltonian.from_pauli(operator.copy())
    if not isinstance(operator, Hamiltonian):
        raise TypeError("operator must be a Hamiltonian or PauliString")
    snapshot = snapshot_nonzero_hermitian_hamiltonian(operator, name="operator")
    if snapshot.num_qubits <= 0:
        raise ValueError("operator must act on at least one qubit")
    terms: list[_DiagonalTerm] = []
    for pauli, coefficient in snapshot.terms:
        if pauli.x_mask != 0:
            raise ValueError("operator must be diagonal in the computational basis")
        effective = complex(coefficient) * pauli.phase.to_complex()
        if effective.imag != 0.0:
            raise ValueError("operator must be Hermitian")
        terms.append(_DiagonalTerm(pauli.z_mask, float(effective.real)))
    return snapshot, tuple(terms)


def _basis_value(bitstring: str, terms: Sequence[_DiagonalTerm]) -> float:
    """Return the diagonal objective value for one little-endian outcome."""
    outcome = int(bitstring, 2)
    return sum(
        term.coefficient * (-1.0 if (outcome & term.z_mask).bit_count() & 1 else 1.0)
        for term in terms
    )


def _summarize_counts(
    result: ExecutionResult,
    terms: Sequence[_DiagonalTerm],
) -> _SampleSummary:
    """Compute an ordinary sample mean from one joint count distribution."""
    weighted_sum = 0.0
    weighted_square_sum = 0.0
    probabilities: dict[str, float] = {}
    values: dict[str, float] = {}
    for bitstring, count in sorted(result.counts.items()):
        if count == 0:
            continue
        probability = count / result.shots
        value = _basis_value(bitstring, terms)
        probabilities[bitstring] = probability
        values[bitstring] = value
        weighted_sum += count * value
        weighted_square_sum += count * value * value
    mean = weighted_sum / result.shots
    variance = None
    if result.shots > 1:
        variance = max(
            0.0,
            (weighted_square_sum / result.shots - mean * mean) / (result.shots - 1),
        )
    best_bitstring = min(
        values,
        key=lambda item: (values[item], -probabilities[item], item),
    )
    best = SamplingVQEMeasurement(
        best_bitstring,
        values[best_bitstring],
        probabilities[best_bitstring],
    )
    return _SampleSummary(
        MappingProxyType(probabilities),
        mean,
        variance,
        best,
    )


@final
class _JointSamplingEstimator(Estimator):
    """Adapt joint computational-basis samples to a mean-value estimator."""

    __slots__ = ("_best_measurement", "_rng", "_sampler", "_shots")

    def __init__(self, sampler: Sampler, seed: int, shots: int) -> None:
        self._sampler = sampler
        self._rng = np.random.default_rng(seed)
        self._shots = shots
        self._best_measurement: SamplingVQEMeasurement | None = None

    @property
    def best_measurement(self) -> SamplingVQEMeasurement:
        """Return the best outcome observed by any objective evaluation."""
        if self._best_measurement is None:
            raise ValueError("optimizer did not evaluate the sampled objective")
        return self._best_measurement

    def _record_best(self, candidate: SamplingVQEMeasurement) -> None:
        """Retain the best sampled outcome with deterministic tie breaking."""
        current = self._best_measurement
        if current is None or (
            candidate.value,
            -candidate.probability,
            candidate.bitstring,
        ) < (
            current.value,
            -current.probability,
            current.bitstring,
        ):
            self._best_measurement = candidate

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        """Evaluate each objective point with one full-register sample."""
        sampler_requests: list[SamplerRequest] = []
        term_batches: list[tuple[_DiagonalTerm, ...]] = []
        for request in requests:
            if not isinstance(request, EstimatorRequest):
                raise TypeError("requests must contain EstimatorRequest values")
            request.validate()
            if request.shots is None:
                raise ValueError("sampling optimization requires positive shots")
            if request.shots != self._shots:
                raise ValueError("optimizer objective requests must preserve shots")
            _, terms = _diagonal_terms(request.observable)
            bound = request.circuit.assign_parameters(dict(request.parameter_bindings))
            bound.validate()
            sampler_requests.append(
                SamplerRequest(
                    bound,
                    tuple(bound.qubits),
                    request.shots,
                    int(self._rng.integers(0, 2**32, dtype=np.uint64)),
                )
            )
            term_batches.append(terms)
        results = require_sampler_results(
            self._sampler.run(sampler_requests), sampler_requests
        )
        summaries = [
            _summarize_counts(result, terms)
            for result, terms in zip(results, term_batches, strict=True)
        ]
        for summary in summaries:
            self._record_best(summary.best)
        return [
            EstimatorResult(summary.mean, summary.variance) for summary in summaries
        ]


@final
class SamplingVQE:
    """Minimize a diagonal objective from joint computational-basis samples.

    Every objective point uses exactly one full-register ``SamplerRequest``.
    All Pauli-Z terms are evaluated from that same joint count distribution,
    preserving correlations and the caller's total shot budget. The optimizer
    receives the ordinary sample mean; nonlinear aggregations are outside this
    component's contract. The lowest-value outcome observed anywhere in the
    optimization is retained; equal values prefer higher observed probability,
    then lexicographically smaller bitstrings. Outcome integers use qubit zero
    as their least-significant bit. This is the sampled diagonal specialization
    of the variational principle in Ref. [1].

    Args:
        sampler: Sampler used for all objective and final measurements.
        ansatz: Parameterized trial-state circuit.
        optimizer: Classical optimizer for the sampled mean objective.
        prefix: Parameter prefix passed to the ansatz.

    Raises:
        TypeError: If a dependency or prefix has the wrong type.
    """

    __slots__ = ("_ansatz", "_optimizer", "_prefix", "_sampler")

    def __init__(
        self,
        sampler: Sampler,
        ansatz: Ansatz,
        optimizer: Optimizer,
        *,
        prefix: str = "theta",
    ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError("sampler must be a Sampler")
        if not isinstance(ansatz, Ansatz):
            raise TypeError("ansatz must be an Ansatz")
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be an Optimizer")
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        self._sampler = sampler
        self._ansatz = ansatz
        self._optimizer = optimizer
        self._prefix = prefix

    def compute_minimum_eigenvalue(
        self,
        operator: Hamiltonian | PauliString,
        initial_point: Mapping[str, float],
        *,
        shots: int,
        seed: int | None = None,
    ) -> SamplingVQEResult:
        """Optimize the sample mean and independently sample the final state.

        Args:
            operator: Nonempty computational-basis diagonal Hermitian target.
            initial_point: Exact finite bindings for active ansatz parameters.
            shots: Positive shot count used for every objective evaluation and
                the independent final sample.
            seed: Optional master seed for the complete run.

        Returns:
            Optimizer metadata and the independent final sample summary.

        Raises:
            TypeError: If inputs or component results have invalid types.
            ValueError: If the objective, bindings, shots, seed, widths, or
                component results violate their contracts.
        """
        snapshot, terms = _diagonal_terms(operator)
        circuit = self._ansatz.build_circuit(self._prefix)
        parameter_names = tuple(circuit.used_symbols)
        objective_seed, final_seed = spawn_seeds(seed, 2)
        estimator_request = EstimatorRequest(
            circuit,
            snapshot,
            initial_point,
            shots,
        )
        optimizer_request = OptimizerRequest(estimator_request, parameter_names)
        estimator = _JointSamplingEstimator(self._sampler, objective_seed, shots)
        optimizer_results = self._optimizer.run(estimator, [optimizer_request])
        if not isinstance(optimizer_results, list):
            raise TypeError("optimizer must return a list of OptimizerResult values")
        if len(optimizer_results) != 1:
            raise ValueError("optimizer must return exactly one result")
        optimizer_result = optimizer_results[0]
        if not isinstance(optimizer_result, OptimizerResult):
            raise TypeError("optimizer must return OptimizerResult values")
        if optimizer_result.parameter_names != parameter_names:
            raise ValueError(
                "optimizer result parameter names must match the ansatz circuit"
            )
        parameters = dict(
            zip(
                optimizer_result.parameter_names,
                optimizer_result.optimal_parameters,
                strict=True,
            )
        )
        optimal_circuit = circuit.assign_parameters(parameters)
        optimal_circuit.validate()
        if optimal_circuit.used_symbols:
            raise ValueError("optimizer result did not fully bind the ansatz circuit")
        final_request = SamplerRequest(
            optimal_circuit,
            tuple(optimal_circuit.qubits),
            shots,
            final_seed,
        )
        final_result = require_sampler_results(
            self._sampler.run([final_request]), [final_request]
        )[0]
        summary = _summarize_counts(final_result, terms)
        return SamplingVQEResult(
            optimizer_result.optimal_value,
            parameters,
            optimal_circuit,
            optimizer_result,
            final_distribution=summary.distribution,
            final_sample_value=summary.mean,
            best_measurement=estimator.best_measurement,
            shots=shots,
        )


__all__ = ["SamplingVQE", "SamplingVQEMeasurement", "SamplingVQEResult"]
