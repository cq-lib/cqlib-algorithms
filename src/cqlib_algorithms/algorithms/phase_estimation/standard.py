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

"""Standard multi-qubit quantum phase estimation.

References
----------
[1] R. Cleve, A. Ekert, C. Macchiavello, and M. Mosca, "Quantum algorithms
    revisited," Proceedings of the Royal Society A, 454, 339-354, 1998.
    DOI: 10.1098/rspa.1998.0164.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import final

from cqlib import Circuit

from ...operators import PhaseEstimationCircuit
from ...operators.composite.phase_estimation_circuit import (
    _validate_evaluation_qubits,
)
from ...protocols import Sampler, SamplerRequest
from ...protocols.sampler import validate_sampler_results
from .problem import PhaseEstimationProblem, _validate_problem

_MAX_EXACT_BINARY64_INTEGER = (1 << 53) - 1


def _validate_shots(value: object, name: str = "shots") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    if value > _MAX_EXACT_BINARY64_INTEGER:
        raise ValueError(f"{name} must not exceed 2**53 - 1")
    return value


@dataclass(frozen=True, slots=True)
class PhaseEstimationRequest:
    """Configure one standard quantum phase-estimation run.

    Args:
        problem: Unitary and state preparation to evaluate.
        num_evaluation_qubits: Width ``m`` of the phase register.
        shots: Positive number of circuit samples.
        approximation_degree: Small-angle inverse-QFT layers to omit.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a numeric setting or problem invariant is invalid.
    """

    problem: PhaseEstimationProblem
    num_evaluation_qubits: int
    shots: int
    approximation_degree: int = 0

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Revalidate the shallowly frozen request before execution."""
        _validate_problem(self.problem)
        width = _validate_evaluation_qubits(self.num_evaluation_qubits)
        _validate_shots(self.shots)
        if isinstance(self.approximation_degree, bool) or not isinstance(
            self.approximation_degree, int
        ):
            raise TypeError("approximation_degree must be an integer")
        if not 0 <= self.approximation_degree < width:
            raise ValueError(
                "approximation_degree must satisfy "
                "0 <= approximation_degree < num_evaluation_qubits"
            )


@dataclass(frozen=True, slots=True)
class PhaseEstimationResult:
    """Empirical phase distribution and its stable modal estimate.

    ``phase_probabilities`` is sparse: it contains the observed integer grid
    points only. Grid point ``y`` represents normalized phase ``y / 2**m``.
    Equal-frequency modes select the smallest grid point.

    Attributes:
        phase: Selected normalized phase in ``[0, 1)``.
        grid_point: Selected integer phase-grid point.
        phase_probabilities: Observed sparse grid distribution.
        shots: Samples used for the circuit.
        num_evaluation_qubits: Phase-register width.
        num_unitary_applications: Controlled-unitary powers in one circuit.
        num_unitary_queries: Shot-weighted unitary-query count.
    """

    phase: float
    grid_point: int
    phase_probabilities: Mapping[int, float]
    shots: int
    num_evaluation_qubits: int
    num_unitary_applications: int
    num_unitary_queries: int

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        width = _validate_evaluation_qubits(self.num_evaluation_qubits)
        shots = _validate_shots(self.shots)
        dimension = 1 << width
        if (
            isinstance(self.grid_point, bool)
            or not isinstance(self.grid_point, int)
            or not 0 <= self.grid_point < dimension
        ):
            raise ValueError("grid_point must lie on the phase grid")
        expected_phase = self.grid_point / dimension
        if (
            isinstance(self.phase, bool)
            or not isinstance(self.phase, (int, float))
            or not math.isfinite(float(self.phase))
            or not math.isclose(
                float(self.phase), expected_phase, rel_tol=0.0, abs_tol=0.0
            )
        ):
            raise ValueError("phase must match grid_point")
        expected_applications = dimension - 1
        if self.num_unitary_applications != expected_applications:
            raise ValueError("num_unitary_applications must equal 2**m - 1")
        if self.num_unitary_queries != shots * expected_applications:
            raise ValueError("num_unitary_queries must be shot-weighted")
        if not isinstance(self.phase_probabilities, Mapping):
            raise TypeError("phase_probabilities must be a mapping")
        probabilities = dict(self.phase_probabilities)
        if not probabilities:
            raise ValueError("phase_probabilities must not be empty")
        for point, probability in probabilities.items():
            if (
                isinstance(point, bool)
                or not isinstance(point, int)
                or not 0 <= point < dimension
            ):
                raise ValueError("phase grid points must be valid integers")
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(float(probability))
                or not 0.0 < float(probability) <= 1.0
            ):
                raise ValueError("observed phase probabilities must lie in (0, 1]")
        if not math.isclose(
            sum(float(value) for value in probabilities.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("phase probabilities must sum to one")
        maximum = max(probabilities.values())
        expected_grid_point = min(
            point
            for point, probability in probabilities.items()
            if probability == maximum
        )
        if self.grid_point != expected_grid_point:
            raise ValueError("grid_point must be the stable empirical mode")
        object.__setattr__(self, "phase", float(self.phase))
        object.__setattr__(
            self,
            "phase_probabilities",
            MappingProxyType(
                {point: float(probabilities[point]) for point in sorted(probabilities)}
            ),
        )


@final
class PhaseEstimation:
    """Estimate eigenphases with controlled unitary powers and inverse QFT.

    If ``U|psi> = exp(2*pi*i*phi)|psi>``, an ``m``-qubit phase register
    resolves ``phi`` on the grid ``y / 2**m``. Superposed eigenstates produce
    the corresponding mixture of phase outcomes. Dense controlled matrices
    impose exponential classical memory cost.

    Args:
        sampler: Sampling implementation used to execute phase circuits.

    Raises:
        TypeError: If the sampler or a request has the wrong type.
        ValueError: If validation or sampler-result checks fail.
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
        problem: PhaseEstimationProblem,
        num_evaluation_qubits: int,
        *,
        approximation_degree: int = 0,
    ) -> Circuit:
        """Build a fresh, unmeasured standard-QPE circuit."""
        _validate_problem(problem)
        component = PhaseEstimationCircuit(
            problem.unitary,
            num_evaluation_qubits,
            approximation_degree=approximation_degree,
        )
        circuit = Circuit(component.num_qubits)
        system_targets = circuit.qubits[num_evaluation_qubits:]
        circuit.compose(problem.state_preparation, list(system_targets))
        circuit.compose(component.build_circuit(), circuit.qubits)
        return circuit

    def run(
        self,
        requests: Sequence[PhaseEstimationRequest],
        /,
    ) -> list[PhaseEstimationResult]:
        """Execute a validated request batch with one Sampler call."""
        sampler_requests: list[SamplerRequest] = []
        for request in requests:
            if not isinstance(request, PhaseEstimationRequest):
                raise TypeError(
                    "requests must contain PhaseEstimationRequest instances"
                )
            request.validate()
            circuit = self.build_circuit(
                request.problem,
                request.num_evaluation_qubits,
                approximation_degree=request.approximation_degree,
            )
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
            self._sampler.run(sampler_requests),
            sampler_requests,
        )
        results: list[PhaseEstimationResult] = []
        for request, sample in zip(requests, sampled):
            counts = {
                int(bitstring, 2): count
                for bitstring, count in sample.counts.items()
                if count > 0
            }
            maximum = max(counts.values())
            grid_point = min(
                point for point, count in counts.items() if count == maximum
            )
            dimension = 1 << request.num_evaluation_qubits
            applications = dimension - 1
            results.append(
                PhaseEstimationResult(
                    phase=grid_point / dimension,
                    grid_point=grid_point,
                    phase_probabilities={
                        point: count / request.shots for point, count in counts.items()
                    },
                    shots=request.shots,
                    num_evaluation_qubits=request.num_evaluation_qubits,
                    num_unitary_applications=applications,
                    num_unitary_queries=applications * request.shots,
                )
            )
        return results


__all__ = [
    "PhaseEstimation",
    "PhaseEstimationRequest",
    "PhaseEstimationResult",
]
