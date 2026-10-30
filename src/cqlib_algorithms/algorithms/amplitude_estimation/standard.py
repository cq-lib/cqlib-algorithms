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

"""Phase-estimation-based amplitude estimation.

References
----------
[1] G. Brassard, P. Hoyer, M. Mosca, and A. Tapp, "Quantum amplitude
    amplification and estimation," Contemporary Mathematics, 305, 53-74,
    2002.
    arXiv:quant-ph/0005055.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import final

from cqlib import Circuit

from ...operators.composite import PhaseEstimationCircuit
from ...protocols.sampler import Sampler, SamplerRequest, validate_sampler_results
from ._common import _validate_positive_integer
from .problem import AmplitudeEstimationProblem, _validate_problem


@dataclass(frozen=True, slots=True)
class StandardAmplitudeEstimationRequest:
    """Configure one phase-estimation-based amplitude-estimation run.

    Args:
        problem: State preparation, Grover iterate, and good-state predicate.
        num_evaluation_qubits: Width ``m`` of the phase register.
        shots: Positive number of circuit samples.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a numeric value or problem invariant is invalid.
    """

    problem: AmplitudeEstimationProblem
    num_evaluation_qubits: int
    shots: int

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Revalidate the shallowly frozen request before execution."""
        _validate_problem(self.problem)
        _validate_positive_integer(self.num_evaluation_qubits, "num_evaluation_qubits")
        _validate_positive_integer(self.shots, "shots")


@dataclass(frozen=True, slots=True)
class StandardAmplitudeEstimationResult:
    """Observed phase grid, distribution, and amplitude estimate.

    ``evaluation_probabilities`` contains every raw integer phase grid point,
    including zero-probability points. ``grid_point`` is the canonical
    amplitude bin ``min(y, 2**m-y)``. Selection first aggregates both phase
    representatives into canonical bins; ties select the smallest canonical
    bin. This finite-shot result does not claim a rigorous confidence interval.

    Attributes:
        estimation: Selected estimate in ``[0, 1]``.
        confidence_interval: Always ``None`` for this implementation.
        shots: Samples used for the circuit.
        num_grover_queries: Grover powers present in one circuit execution.
        num_oracle_queries: Shot-weighted Grover-query count.
        grid_point: Selected canonical amplitude bin.
        evaluation_probabilities: Probability at every raw phase grid point.
        num_evaluation_qubits: Phase-register width.
    """

    estimation: float
    confidence_interval: tuple[float, float] | None
    shots: int
    num_grover_queries: int
    num_oracle_queries: int
    grid_point: int
    evaluation_probabilities: Mapping[int, float]
    num_evaluation_qubits: int

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.estimation, float) or not math.isfinite(self.estimation):
            raise TypeError("estimation must be a finite float")
        if not 0.0 <= self.estimation <= 1.0:
            raise ValueError("estimation must lie in [0, 1]")
        if self.confidence_interval is not None:
            raise ValueError("standard amplitude estimation has no confidence interval")
        _validate_positive_integer(self.shots, "shots")
        _validate_positive_integer(self.num_evaluation_qubits, "num_evaluation_qubits")
        dimension = 1 << self.num_evaluation_qubits
        if (
            isinstance(self.grid_point, bool)
            or not isinstance(self.grid_point, int)
            or not 0 <= self.grid_point <= dimension // 2
        ):
            raise ValueError("grid_point must be a valid canonical amplitude bin")
        expected_queries = dimension - 1
        if isinstance(self.num_grover_queries, bool) or not isinstance(
            self.num_grover_queries, int
        ):
            raise TypeError("num_grover_queries must be an integer")
        if isinstance(self.num_oracle_queries, bool) or not isinstance(
            self.num_oracle_queries, int
        ):
            raise TypeError("num_oracle_queries must be an integer")
        if self.num_grover_queries != expected_queries:
            raise ValueError("num_grover_queries must equal 2**m - 1")
        if self.num_oracle_queries != expected_queries * self.shots:
            raise ValueError("num_oracle_queries must be shot-weighted")
        expected_estimation = math.sin(math.pi * self.grid_point / dimension) ** 2
        if not math.isclose(
            self.estimation,
            expected_estimation,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("estimation must match grid_point")
        if not isinstance(self.evaluation_probabilities, Mapping):
            raise TypeError("evaluation_probabilities must be a mapping")
        probabilities = dict(self.evaluation_probabilities)
        if set(probabilities) != set(range(dimension)):
            raise ValueError("evaluation_probabilities must contain every grid point")
        if any(
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(float(probability))
            or not 0.0 <= float(probability) <= 1.0
            for probability in probabilities.values()
        ):
            raise ValueError("evaluation probabilities must lie in [0, 1]")
        if not math.isclose(
            sum(float(value) for value in probabilities.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("evaluation probabilities must sum to one")
        object.__setattr__(
            self,
            "evaluation_probabilities",
            MappingProxyType(
                {key: float(probabilities[key]) for key in range(dimension)}
            ),
        )


@final
class StandardAmplitudeEstimation:
    """Estimate amplitudes with dense controlled Grover powers and inverse QFT.

    The phase-estimation register and controlled ``Q**(2**j)`` powers are
    delegated to :class:`PhaseEstimationCircuit`, so standard amplitude
    estimation and QPE share one implementation of dense quantum control,
    global-phase preservation, target ordering, inverse QFT, and allocation
    limits.

    A measured integer ``y`` maps to ``sin(pi*y/M)**2``. The original
    single-run accuracy guarantee applies to ideal phase-estimation samples;
    finite-shot empirical modes are returned without an additional claimed
    confidence interval. The dense construction has exponential memory cost
    and may not be accepted by every execution backend.

    Reference: G. Brassard, P. Hoyer, M. Mosca, and A. Tapp, *Quantum
    Amplitude Amplification and Estimation*, arXiv:quant-ph/0005055.

    Args:
        sampler: Sampling implementation used to execute all circuits.

    Raises:
        TypeError: If the sampler or a request has the wrong type.
        ValueError: If a request or sampler result violates its contract.
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
        problem: AmplitudeEstimationProblem,
        num_evaluation_qubits: int,
    ) -> Circuit:
        """Build an unmeasured standard amplitude-estimation circuit.

        Args:
            problem: Validated amplitude-estimation problem.
            num_evaluation_qubits: Positive phase-register width.

        Returns:
            A fresh circuit with phase qubits followed by system qubits.
        """
        _validate_problem(problem)
        _validate_positive_integer(num_evaluation_qubits, "num_evaluation_qubits")
        component = PhaseEstimationCircuit(
            problem.grover_operator,
            num_evaluation_qubits,
        )
        circuit = Circuit(component.num_qubits)
        system_targets = tuple(circuit.qubits[num_evaluation_qubits:])

        circuit.compose(problem.state_preparation, list(system_targets))
        circuit.compose(component.build_circuit(), circuit.qubits)
        return circuit

    def run(
        self,
        requests: Sequence[StandardAmplitudeEstimationRequest],
        /,
    ) -> list[StandardAmplitudeEstimationResult]:
        """Execute a request batch with one Sampler call.

        Args:
            requests: Requests in the desired output order.

        Returns:
            One result per request in the same order.
        """
        sampler_requests: list[SamplerRequest] = []
        for request in requests:
            if not isinstance(request, StandardAmplitudeEstimationRequest):
                raise TypeError(
                    "requests must contain StandardAmplitudeEstimationRequest instances"
                )
            request.validate()
            circuit = self.build_circuit(request.problem, request.num_evaluation_qubits)
            sampler_requests.append(
                SamplerRequest(
                    circuit,
                    tuple(circuit.qubits[: request.num_evaluation_qubits]),
                    request.shots,
                )
            )
        if not sampler_requests:
            return []

        sampled = validate_sampler_results(
            self._sampler.run(sampler_requests), sampler_requests
        )
        results: list[StandardAmplitudeEstimationResult] = []
        for request, sampler_request, sample in zip(
            requests, sampler_requests, sampled
        ):
            dimension = 1 << request.num_evaluation_qubits
            counts = {point: 0 for point in range(dimension)}
            for bitstring, count in sample.counts.items():
                counts[int(bitstring, 2)] += count
            probabilities = {
                point: count / sampler_request.shots for point, count in counts.items()
            }
            canonical_counts = {
                point: counts[point]
                + (0 if point in {0, dimension // 2} else counts[dimension - point])
                for point in range(dimension // 2 + 1)
            }
            maximum = max(canonical_counts.values())
            grid_point = min(
                point for point, count in canonical_counts.items() if count == maximum
            )
            estimate = math.sin(math.pi * grid_point / dimension) ** 2
            grover_queries = dimension - 1
            results.append(
                StandardAmplitudeEstimationResult(
                    estimation=float(estimate),
                    confidence_interval=None,
                    shots=sampler_request.shots,
                    num_grover_queries=grover_queries,
                    num_oracle_queries=grover_queries * sampler_request.shots,
                    grid_point=grid_point,
                    evaluation_probabilities=probabilities,
                    num_evaluation_qubits=request.num_evaluation_qubits,
                )
            )
        return results


__all__ = [
    "StandardAmplitudeEstimation",
    "StandardAmplitudeEstimationRequest",
    "StandardAmplitudeEstimationResult",
]
