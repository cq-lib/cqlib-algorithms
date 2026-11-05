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

"""Single-ancilla iterative quantum phase estimation.

References
----------
[1] M. Dobšíček, G. Johansson, V. Shumeiko, and G. Wendin, "Arbitrary
    accuracy iterative phase estimation algorithm as a two qubit benchmark,"
    Physical Review A, 76, 030306(R), 2007.
    DOI: 10.1103/PhysRevA.76.030306.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import final

import numpy as np
from cqlib import Circuit

from ..._utils.phase_estimation import (
    append_controlled_matrix,
    normalize_evaluation_qubits,
    normalize_shots,
    unitary_matrix,
)
from ...protocols import Sampler, SamplerRequest
from ...protocols.sampler import require_sampler_results
from .problem import PhaseEstimationProblem, _validate_problem


@dataclass(frozen=True, slots=True)
class IterativePhaseEstimationRequest:
    """One adaptive iterative phase-estimation request.

    ``shots_per_iteration`` samples every extracted bit before a majority
    decision. A tie deterministically selects zero.

    Args:
        problem: Unitary and promised eigenstate preparation.
        num_iterations: Number of phase bits to extract.
        shots_per_iteration: Samples used for each bit decision.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a count or problem invariant is invalid.
    """

    problem: PhaseEstimationProblem
    num_iterations: int
    shots_per_iteration: int = 1

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Revalidate the shallowly frozen request before execution."""
        _validate_problem(self.problem)
        normalize_evaluation_qubits(self.num_iterations)
        normalize_shots(self.shots_per_iteration, "shots_per_iteration")


@dataclass(frozen=True, slots=True)
class IterativePhaseEstimationResult:
    """Store a bitwise phase estimate and per-round statistics.

    Attributes:
        phase: Estimated normalized phase.
        bitstring: Extracted phase bits, most significant first.
        one_probabilities: Observed one probability for each reported bit.
        shots_per_iteration: Samples used for every bit decision.
        num_iterations: Number of extracted bits.
        num_unitary_applications: Controlled powers across one pass.
        num_unitary_queries: Shot-weighted unitary-query count.
    """

    phase: float
    bitstring: str
    one_probabilities: tuple[float, ...]
    shots_per_iteration: int
    num_iterations: int
    num_unitary_applications: int
    num_unitary_queries: int

    def __post_init__(self) -> None:
        width = normalize_evaluation_qubits(self.num_iterations)
        shots = normalize_shots(
            self.shots_per_iteration,
            "shots_per_iteration",
        )
        if (
            not isinstance(self.bitstring, str)
            or len(self.bitstring) != width
            or any(bit not in "01" for bit in self.bitstring)
        ):
            raise ValueError("bitstring must contain one binary digit per iteration")
        if (
            not isinstance(self.one_probabilities, tuple)
            or len(self.one_probabilities) != width
        ):
            raise ValueError("one_probabilities must contain one value per iteration")
        for probability in self.one_probabilities:
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(float(probability))
                or not 0.0 <= float(probability) <= 1.0
            ):
                raise ValueError("one probabilities must lie in [0, 1]")
        grid_point = int(self.bitstring, 2)
        expected_phase = grid_point / (1 << width)
        if (
            isinstance(self.phase, bool)
            or not isinstance(self.phase, (int, float))
            or float(self.phase) != expected_phase
        ):
            raise ValueError("phase must match bitstring")
        applications = (1 << width) - 1
        if self.num_unitary_applications != applications:
            raise ValueError("num_unitary_applications must equal 2**m - 1")
        if self.num_unitary_queries != shots * applications:
            raise ValueError("num_unitary_queries must be shot-weighted")
        object.__setattr__(self, "phase", float(self.phase))
        object.__setattr__(
            self,
            "one_probabilities",
            tuple(float(value) for value in self.one_probabilities),
        )


def _build_iteration_circuit(
    problem: PhaseEstimationProblem,
    powered_unitary: np.ndarray,
    known_fraction: float,
) -> Circuit:
    circuit = Circuit(problem.num_qubits + 1)
    control = circuit.qubits[0]
    system_targets = tuple(circuit.qubits[1:])
    circuit.compose(problem.state_preparation, list(system_targets))
    circuit.h(control)
    append_controlled_matrix(
        circuit,
        control,
        system_targets,
        powered_unitary,
        label="controlled-iterative-unitary-power",
    )
    if known_fraction != 0.0:
        circuit.phase(control, -math.pi * known_fraction)
    circuit.h(control)
    return circuit


@final
class IterativePhaseEstimation:
    """Estimate one eigenphase using one recycled logical ancilla.

    Rounds proceed from the least-significant requested bit to the most
    significant bit. Each round uses a fresh preparation of the same promised
    eigenstate, applies a controlled power of the unitary, subtracts the phase
    implied by already extracted lower-significance bits, and measures after a
    Hadamard. This static-circuit implementation performs the classical
    feed-forward between Sampler calls.

    Args:
        sampler: Sampling implementation used in each adaptive round.

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

    def run(
        self,
        requests: Sequence[IterativePhaseEstimationRequest],
        /,
    ) -> list[IterativePhaseEstimationResult]:
        """Execute adaptive requests in input order."""
        results: list[IterativePhaseEstimationResult] = []
        for request in requests:
            if not isinstance(request, IterativePhaseEstimationRequest):
                raise TypeError(
                    "requests must contain IterativePhaseEstimationRequest instances"
                )
            request.validate()
            matrix = unitary_matrix(request.problem.unitary, 1)
            known_fraction = 0.0
            extracted_bits: list[int] = []
            extracted_probabilities: list[float] = []

            for bit_position in range(request.num_iterations - 1, -1, -1):
                power = 1 << bit_position
                powered = np.linalg.matrix_power(matrix, power)
                circuit = _build_iteration_circuit(
                    request.problem,
                    powered,
                    known_fraction,
                )
                sampler_request = SamplerRequest(
                    circuit,
                    (circuit.qubits[0],),
                    request.shots_per_iteration,
                )
                sampled = require_sampler_results(
                    self._sampler.run([sampler_request]),
                    [sampler_request],
                )[0]
                ones = sampled.counts.get("1", 0)
                one_probability = ones / request.shots_per_iteration
                bit = int(ones * 2 > request.shots_per_iteration)
                extracted_bits.append(bit)
                extracted_probabilities.append(one_probability)
                known_fraction = bit / 2.0 + known_fraction / 2.0

            bitstring = "".join(str(bit) for bit in reversed(extracted_bits))
            one_probabilities = tuple(reversed(extracted_probabilities))
            applications = (1 << request.num_iterations) - 1
            results.append(
                IterativePhaseEstimationResult(
                    phase=int(bitstring, 2) / (1 << request.num_iterations),
                    bitstring=bitstring,
                    one_probabilities=one_probabilities,
                    shots_per_iteration=request.shots_per_iteration,
                    num_iterations=request.num_iterations,
                    num_unitary_applications=applications,
                    num_unitary_queries=(applications * request.shots_per_iteration),
                )
            )
        return results


__all__ = [
    "IterativePhaseEstimation",
    "IterativePhaseEstimationRequest",
    "IterativePhaseEstimationResult",
]
