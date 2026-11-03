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

"""Chernoff--Hoeffding iterative amplitude estimation.

References
----------
[1] D. Grinko, J. Gacon, C. Zoufal, and S. Woerner, "Iterative quantum
    amplitude estimation," npj Quantum Information, 7, 52, 2021.
    DOI: 10.1038/s41534-021-00379-1.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import final

from ..._utils.numeric import normalize_positive_integer
from ...protocols.sampler import Sampler, SamplerRequest, require_sampler_results
from ._common import (
    _build_grover_power_circuit,
    _count_good_outcomes,
    _normalize_open_probability,
)
from .problem import AmplitudeEstimationProblem, _validate_problem

_MIN_BINARY64_EPSILON = math.pi * math.ulp(0.25)
_MAX_BINARY64_CHERNOFF_SHOTS = (1 << 53) - 1


def _theoretical_rounds(epsilon: float) -> int:
    """Return Algorithm 1's confidence-budget round bound."""
    return math.ceil(math.log2(math.pi / 8.0) - math.log2(epsilon))


def _chernoff_lmax(epsilon: float, alpha: float, shots: int) -> float:
    if shots > _MAX_BINARY64_CHERNOFF_SHOTS:
        raise ValueError(
            "shots exceeds the Chernoff--Hoeffding numeric limit of 2**53 - 1"
        )
    rounds = _theoretical_rounds(epsilon)
    argument = 2.0 * math.log(2.0 * rounds / alpha) / shots
    if argument > 1.0:
        raise ValueError(
            "shots is too small for the Chernoff--Hoeffding Lmax construction"
        )
    return math.asin(argument**0.25)


@dataclass(frozen=True, slots=True)
class IterativeAmplitudeEstimationRequest:
    """Configure one iterative amplitude-estimation run.

    Args:
        problem: State preparation, Grover iterate, and good-state predicate.
        epsilon: Target half-width of the final amplitude interval.
        alpha: Total failure-probability budget.
        shots: Maximum samples allocated to an adaptive round.
        max_rounds: Hard safety limit on adaptive rounds.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a probability, shot count, or round limit is invalid.
    """

    problem: AmplitudeEstimationProblem
    epsilon: float
    alpha: float
    shots: int
    max_rounds: int = 50

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Revalidate the shallowly frozen request before execution."""
        _validate_problem(self.problem)
        epsilon = _normalize_open_probability(
            self.epsilon, "epsilon", upper=math.pi / 8.0
        )
        if epsilon < _MIN_BINARY64_EPSILON:
            raise ValueError(
                "epsilon must be at least pi * ulp(0.25) for binary64 geometry"
            )
        alpha = _normalize_open_probability(self.alpha, "alpha", upper=1.0)
        normalize_positive_integer(self.shots, "shots")
        normalize_positive_integer(self.max_rounds, "max_rounds")
        rounds = _theoretical_rounds(epsilon)
        if self.max_rounds < rounds:
            raise ValueError("max_rounds must be at least ceil(log2(pi/(8*epsilon)))")
        _chernoff_lmax(epsilon, alpha, self.shots)


@dataclass(frozen=True, slots=True)
class IterativeAmplitudeEstimationResult:
    """Store an iterative estimate and its adaptive execution record.

    Attributes:
        estimation: Midpoint of the final amplitude interval.
        confidence_interval: Final Chernoff--Hoeffding amplitude interval.
        shots: Total samples across all rounds.
        num_grover_queries: Grover powers summed across sampled circuits.
        num_oracle_queries: Shot-weighted Grover-query count.
        powers: Grover exponents selected by the adaptive schedule.
        round_shots: Samples used at each reported round.
        round_intervals: Amplitude interval after each round.
    """

    estimation: float
    confidence_interval: tuple[float, float]
    shots: int
    num_grover_queries: int
    num_oracle_queries: int
    powers: tuple[int, ...]
    round_shots: tuple[int, ...]
    round_intervals: tuple[tuple[float, float], ...]

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.estimation, float) or not math.isfinite(self.estimation):
            raise TypeError("estimation must be a finite float")
        if not 0.0 <= self.estimation <= 1.0:
            raise ValueError("estimation must lie in [0, 1]")
        if (
            not isinstance(self.confidence_interval, tuple)
            or len(self.confidence_interval) != 2
        ):
            raise TypeError("confidence_interval must be a pair")
        low, high = self.confidence_interval
        if not 0.0 <= low <= self.estimation <= high <= 1.0:
            raise ValueError("confidence_interval must contain estimation in [0, 1]")
        for name, value in (
            ("shots", self.shots),
            ("num_grover_queries", self.num_grover_queries),
            ("num_oracle_queries", self.num_oracle_queries),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not isinstance(self.powers, tuple) or any(
            isinstance(power, bool) or not isinstance(power, int) or power < 0
            for power in self.powers
        ):
            raise TypeError("powers must be a tuple of non-negative integers")
        if not isinstance(self.round_shots, tuple) or any(
            isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0
            for shots in self.round_shots
        ):
            raise TypeError("round_shots must be a tuple of positive integers")
        if len(self.powers) != len(self.round_shots):
            raise ValueError("powers and round_shots must have equal length")
        if len(self.round_intervals) != len(self.powers):
            raise ValueError("round_intervals must have one entry per power")
        if self.shots != sum(self.round_shots):
            raise ValueError("shots must equal the sum of round_shots")
        if self.num_grover_queries != sum(self.powers):
            raise ValueError("num_grover_queries must equal the sum of powers")
        if self.num_oracle_queries != sum(
            power * shots for power, shots in zip(self.powers, self.round_shots)
        ):
            raise ValueError("num_oracle_queries must be shot-weighted")
        previous_width = 1.0 + 1e-15
        for interval in self.round_intervals:
            if not isinstance(interval, tuple) or len(interval) != 2:
                raise TypeError("round_intervals entries must be pairs")
            interval_low, interval_high = interval
            if not 0.0 <= interval_low <= interval_high <= 1.0:
                raise ValueError("round_intervals entries must lie in [0, 1]")
            width = interval_high - interval_low
            if width > previous_width + 1e-12:
                raise ValueError("round_intervals must not widen")
            previous_width = width
        if (
            self.round_intervals
            and self.confidence_interval != self.round_intervals[-1]
        ):
            raise ValueError("confidence_interval must equal the final round interval")


@dataclass(slots=True)
class _IterativeState:
    request: IterativeAmplitudeEstimationRequest
    theta_interval: tuple[float, float] = (0.0, 0.25)
    amplitude_interval: tuple[float, float] = (0.0, 1.0)
    powers: list[int] = field(default_factory=list)
    round_shots: list[int] = field(default_factory=list)
    round_intervals: list[tuple[float, float]] = field(default_factory=list)
    good_by_power: dict[int, int] = field(default_factory=dict)
    shots_by_power: dict[int, int] = field(default_factory=dict)
    upper_half_circle: bool = True

    @property
    def converged(self) -> bool:
        epsilon = float(self.request.epsilon)
        return (
            self.theta_interval[1] - self.theta_interval[0] <= epsilon / math.pi
            and self.amplitude_interval[1] - self.amplitude_interval[0] <= 2.0 * epsilon
        )


def _round_shots(request: IterativeAmplitudeEstimationRequest, power: int) -> int:
    """Apply Algorithm 1's no-overshooting shot reduction."""
    epsilon = float(request.epsilon)
    lmax = _chernoff_lmax(epsilon, float(request.alpha), request.shots)
    scale = 4 * power + 2
    if scale > math.ceil(lmax / epsilon):
        return max(
            1,
            math.ceil(request.shots * lmax / epsilon / scale / 10.0),
        )
    return request.shots


def _find_next_power(
    previous_power: int,
    theta_interval: tuple[float, float],
    previous_upper_half_circle: bool,
) -> tuple[int, bool]:
    """Implement the paper's same-half-circle power-selection rule."""
    theta_low, theta_high = theta_interval
    rational_low = Fraction.from_float(theta_low)
    rational_high = Fraction.from_float(theta_high)
    width = rational_high - rational_low
    previous_scale = 4 * previous_power + 2
    maximum_scale = max(2, math.floor(Fraction(1, 2) / width))
    scale = maximum_scale - ((maximum_scale - 2) % 4)

    while scale >= 2 * previous_scale:
        scaled_low = scale * rational_low
        scaled_high = scale * rational_high
        low_floor = math.floor(scaled_low)
        high_floor = math.floor(scaled_high)
        if low_floor == high_floor:
            fractional_low = scaled_low - low_floor
            fractional_high = scaled_high - high_floor
            if fractional_high <= Fraction(1, 2):
                return (scale - 2) // 4, True
            if fractional_low >= Fraction(1, 2):
                return (scale - 2) // 4, False
        scale -= 4
    return previous_power, previous_upper_half_circle


def _measurement_phase_interval(
    probability_low: float,
    probability_high: float,
    *,
    upper_half_circle: bool,
) -> tuple[float, float]:
    angle_low = math.acos(1.0 - 2.0 * probability_low) / (2.0 * math.pi)
    angle_high = math.acos(1.0 - 2.0 * probability_high) / (2.0 * math.pi)
    if upper_half_circle:
        return angle_low, angle_high
    return 1.0 - angle_high, 1.0 - angle_low


def _update_state(
    state: _IterativeState,
    power: int,
    upper_half_circle: bool,
    good_count: int,
    round_shots: int,
) -> None:
    request = state.request
    state.good_by_power[power] = state.good_by_power.get(power, 0) + good_count
    state.shots_by_power[power] = state.shots_by_power.get(power, 0) + round_shots
    total_good = state.good_by_power[power]
    total_shots = state.shots_by_power[power]
    estimate = total_good / total_shots
    rounds = _theoretical_rounds(float(request.epsilon))
    radius = math.sqrt(
        math.log(2.0 * rounds / float(request.alpha)) / (2.0 * total_shots)
    )
    probability_low = max(0.0, estimate - radius)
    probability_high = min(1.0, estimate + radius)
    fractional_low, fractional_high = _measurement_phase_interval(
        probability_low,
        probability_high,
        upper_half_circle=upper_half_circle,
    )

    scale = 4 * power + 2
    old_low, old_high = state.theta_interval
    base = math.floor(scale * Fraction.from_float(old_low))
    candidate_low = (base + fractional_low) / scale
    candidate_high = (base + fractional_high) / scale
    new_low = max(old_low, candidate_low)
    new_high = min(old_high, candidate_high)
    if new_low > new_high:
        # This is possible only on the allocated confidence-failure event.
        # Retaining the previous interval keeps the adaptive scheduler valid.
        new_low, new_high = old_low, old_high
    state.theta_interval = (new_low, new_high)
    amplitude_low = math.sin(2.0 * math.pi * new_low) ** 2
    amplitude_high = math.sin(2.0 * math.pi * new_high) ** 2
    state.amplitude_interval = (float(amplitude_low), float(amplitude_high))
    state.powers.append(power)
    state.round_shots.append(round_shots)
    state.round_intervals.append(state.amplitude_interval)
    state.upper_half_circle = upper_half_circle


@final
class IterativeAmplitudeEstimation:
    """Estimate amplitudes with adaptive Grover powers and no phase register.

    Each round samples ``Q**k A`` and estimates
    ``sin((2*k+1)*theta)**2``. Algorithm 1's theoretical round bound
    ``T=ceil(log2(pi/(8*epsilon)))`` determines the Chernoff--Hoeffding
    confidence budget. Repeated powers aggregate all their samples, and the
    paper's ``Lmax`` rule reduces shots when a power would otherwise
    overshoot. Active requests are batched once per adaptive round. The
    returned interval has width at most ``2*epsilon`` or the independent hard
    safety limit ``max_rounds`` raises.

    Reference: D. Grinko, J. Gacon, C. Zoufal, and S. Woerner, *Iterative
    Quantum Amplitude Estimation*, arXiv:1912.05559v3.

    Args:
        sampler: Sampling implementation used for adaptive experiments.

    Raises:
        TypeError: If the sampler or a request has the wrong type.
        ValueError: If validation fails or the target interval is not reached.
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
        requests: Sequence[IterativeAmplitudeEstimationRequest],
        /,
    ) -> list[IterativeAmplitudeEstimationResult]:
        """Execute adaptive request rounds, batching all active problems."""
        states: list[_IterativeState] = []
        for request in requests:
            if not isinstance(request, IterativeAmplitudeEstimationRequest):
                raise TypeError(
                    "requests must contain IterativeAmplitudeEstimationRequest instances"
                )
            request.validate()
            states.append(_IterativeState(request))
        if not states:
            return []

        while any(not state.converged for state in states):
            active = [state for state in states if not state.converged]
            if any(len(state.powers) >= state.request.max_rounds for state in active):
                raise RuntimeError(
                    "iterative amplitude estimation did not converge within max_rounds"
                )

            sampler_requests: list[SamplerRequest] = []
            choices: list[tuple[_IterativeState, int, bool, int]] = []
            for state in active:
                previous_power = state.powers[-1] if state.powers else 0
                power, upper_half_circle = _find_next_power(
                    previous_power,
                    state.theta_interval,
                    state.upper_half_circle,
                )
                circuit = _build_grover_power_circuit(state.request.problem, power)
                round_shots = _round_shots(state.request, power)
                measured = tuple(
                    circuit.qubits[position]
                    for position in state.request.problem.objective_qubits
                )
                sampler_requests.append(SamplerRequest(circuit, measured, round_shots))
                choices.append((state, power, upper_half_circle, round_shots))

            sampled = require_sampler_results(
                self._sampler.run(sampler_requests), sampler_requests
            )
            for (state, power, upper_half_circle, round_shots), sample in zip(
                choices, sampled
            ):
                good_count = _count_good_outcomes(sample, state.request.problem)
                _update_state(
                    state,
                    power,
                    upper_half_circle,
                    good_count,
                    round_shots,
                )

        results: list[IterativeAmplitudeEstimationResult] = []
        for state in states:
            low, high = state.amplitude_interval
            if high - low > 2.0 * float(state.request.epsilon):
                raise RuntimeError(
                    "iterative amplitude estimation did not meet the requested "
                    "amplitude width"
                )
            results.append(
                IterativeAmplitudeEstimationResult(
                    estimation=float((low + high) / 2.0),
                    confidence_interval=(low, high),
                    shots=sum(state.round_shots),
                    num_grover_queries=sum(state.powers),
                    num_oracle_queries=sum(
                        power * shots
                        for power, shots in zip(state.powers, state.round_shots)
                    ),
                    powers=tuple(state.powers),
                    round_shots=tuple(state.round_shots),
                    round_intervals=tuple(state.round_intervals),
                )
            )
        return results


__all__ = [
    "IterativeAmplitudeEstimation",
    "IterativeAmplitudeEstimationRequest",
    "IterativeAmplitudeEstimationResult",
]
