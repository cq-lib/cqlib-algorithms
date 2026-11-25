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

"""Nakaji's faster amplitude-estimation algorithm.

References
----------
[1] K. Nakaji, "Faster Amplitude Estimation," Quantum Information &
    Computation, 20(13&14), 1109--1123, 2020. arXiv:2003.02417.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import final, overload

from ..._utils.numeric import normalize_positive_integer
from ...protocols.sampler import Sampler, SamplerRequest, require_sampler_results
from ._common import (
    _build_grover_power_circuit,
    _count_good_outcomes,
    _normalize_open_probability,
)
from .problem import AmplitudeEstimationProblem, _validate_problem

_DEFAULT_RESCALING_FACTOR = 1.0
_SUPPORTED_RESCALING_FACTORS = (_DEFAULT_RESCALING_FACTOR, 0.25)
_MAX_EXPLICIT_GROVER_POWER = 1_000_000
_MAX_SAFE_POWER_ITERATIONS = 21


def _normalize_rescaling_factor(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("amplitude_rescaling_factor must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("amplitude_rescaling_factor must be finite")
    if converted not in _SUPPORTED_RESCALING_FACTORS:
        raise ValueError("amplitude_rescaling_factor must be either 1.0 or 0.25")
    return converted


def _shot_schedule(alpha: float, max_iterations: int) -> tuple[int, int]:
    logarithm = math.log(4.0 * max_iterations) - math.log(alpha)
    return math.ceil(1944.0 * logarithm), math.ceil(972.0 * logarithm)


@dataclass(frozen=True, slots=True)
class FasterAmplitudeEstimationRequest:
    """Configure one faster amplitude-estimation run.

    The supplied problem must already have good-state amplitude at most
    ``0.25``. A rescaling factor of ``0.25`` declares that the supplied
    problem represents an externally rescaled target; the result is divided
    by ``0.25**2``. The state preparation, Grover iterate, and predicate
    must describe that same scaled problem.

    Args:
        problem: Consistent state preparation, Grover iterate, and predicate.
        alpha: Total upper bound on the confidence-failure probability.
        max_iterations: Number of exponential-power update rounds.
        amplitude_rescaling_factor: Amplitude scale already applied by the
            caller; supported values are ``1.0`` and ``0.25``.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a probability, iteration count, or scale is invalid.
    """

    problem: AmplitudeEstimationProblem
    alpha: float
    max_iterations: int
    amplitude_rescaling_factor: float = _DEFAULT_RESCALING_FACTOR

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Revalidate the shallowly frozen request before execution."""
        _validate_problem(self.problem)
        alpha = _normalize_open_probability(self.alpha, "alpha")
        if 1.0 - alpha == 1.0:
            raise ValueError("alpha is below binary64 probability resolution")
        max_iterations = normalize_positive_integer(
            self.max_iterations, "max_iterations"
        )
        _normalize_rescaling_factor(self.amplitude_rescaling_factor)
        if max_iterations > _MAX_SAFE_POWER_ITERATIONS:
            raise ValueError(
                "max_iterations exceeds the safe explicit-power validation limit"
            )
        maximum_power = (
            1
            if max_iterations == 1
            else (1 << (max_iterations - 1)) + (1 << (max_iterations - 2))
        )
        if maximum_power > _MAX_EXPLICIT_GROVER_POWER:
            raise ValueError(
                "max_iterations requires more than 1,000,000 explicit Grover "
                "repetitions in one circuit"
            )
        shots = _shot_schedule(alpha, max_iterations)
        if max(shots) > (1 << 53) - 1:
            raise ValueError("the confidence schedule exceeds the binary64 shot limit")


@dataclass(frozen=True, slots=True)
class FasterAmplitudeEstimationResult:
    """Store a faster estimate and its complete adaptive execution record.

    Attributes:
        estimation: Good-state probability estimate obtained from the final
            angle midpoint after undoing the declared amplitude scale.
        confidence_interval: Final good-state probability interval after
            undoing the declared amplitude scale.
        shots: Total samples across all executed circuits.
        num_grover_queries: Sum of Grover exponents across circuit instances.
        num_oracle_queries: Shot-weighted sum of Grover exponents.
        powers: Grover exponent for every executed circuit.
        circuit_shots: Samples used by every executed circuit.
        good_counts: Good-state samples observed from every circuit.
        num_steps: Number of adaptive update rounds.
        num_first_stage_steps: Rounds completed before phase disambiguation.
        theta_intervals: Initial and per-round scaled-angle intervals.
        success_probability_lower_bound: Joint theoretical success lower bound.
        amplitude_rescaling_factor: Caller-declared amplitude scale.
    """

    estimation: float
    confidence_interval: tuple[float, float]
    shots: int
    num_grover_queries: int
    num_oracle_queries: int
    powers: tuple[int, ...]
    circuit_shots: tuple[int, ...]
    good_counts: tuple[int, ...]
    num_steps: int
    num_first_stage_steps: int
    theta_intervals: tuple[tuple[float, float], ...]
    success_probability_lower_bound: float
    amplitude_rescaling_factor: float

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        factor = _normalize_rescaling_factor(self.amplitude_rescaling_factor)
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
            ("num_steps", self.num_steps),
            ("num_first_stage_steps", self.num_first_stage_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.num_first_stage_steps > self.num_steps:
            raise ValueError("num_first_stage_steps must not exceed num_steps")
        if not isinstance(self.powers, tuple) or any(
            isinstance(power, bool) or not isinstance(power, int) or power < 0
            for power in self.powers
        ):
            raise TypeError("powers must contain only non-negative integers")
        if not isinstance(self.circuit_shots, tuple) or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.circuit_shots
        ):
            raise TypeError("circuit_shots must contain only positive integers")
        if not isinstance(self.good_counts, tuple) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.good_counts
        ):
            raise TypeError("good_counts must contain only non-negative integers")
        if not (len(self.powers) == len(self.circuit_shots) == len(self.good_counts)):
            raise ValueError("powers, circuit_shots, and good_counts must align")
        if any(good > used for good, used in zip(self.good_counts, self.circuit_shots)):
            raise ValueError("good_counts must not exceed circuit_shots")
        if self.shots != sum(self.circuit_shots):
            raise ValueError("shots must equal the sum of circuit_shots")
        if self.num_grover_queries != sum(self.powers):
            raise ValueError("num_grover_queries must equal the sum of powers")
        if self.num_oracle_queries != sum(
            power * used for power, used in zip(self.powers, self.circuit_shots)
        ):
            raise ValueError("num_oracle_queries must be shot-weighted")
        if (
            not isinstance(self.theta_intervals, tuple)
            or len(self.theta_intervals) != self.num_steps + 1
        ):
            raise ValueError("theta_intervals must contain the initial and every step")
        theta_limit = math.asin(0.25)
        for interval in self.theta_intervals:
            if not isinstance(interval, tuple) or len(interval) != 2:
                raise TypeError("theta_intervals entries must be pairs")
            if not 0.0 <= interval[0] <= interval[1] <= theta_limit:
                raise ValueError("theta_intervals entries exceed the promised domain")
        lower_bound = _normalize_open_probability(
            self.success_probability_lower_bound,
            "success_probability_lower_bound",
        )
        if factor != self.amplitude_rescaling_factor:
            raise ValueError("amplitude_rescaling_factor is not canonical")
        if lower_bound != self.success_probability_lower_bound:
            raise ValueError("success_probability_lower_bound is not canonical")


@dataclass(slots=True)
class _FasterState:
    request: FasterAmplitudeEstimationRequest
    theta_interval: tuple[float, float] = (0.0, math.asin(0.25))
    theta_intervals: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.0, math.asin(0.25))]
    )
    first_stage: bool = True
    first_stage_steps: int = 0
    transition_iteration: int = 0
    phase_offset: float = 0.0
    powers: list[int] = field(default_factory=list)
    circuit_shots: list[int] = field(default_factory=list)
    good_counts: list[int] = field(default_factory=list)


def _cosine_estimate(good_count: int, shots: int) -> float:
    return 1.0 - 2.0 * good_count / shots


def _chernoff_interval(
    cosine: float, alpha: float, iterations: int, shots: int
) -> tuple[float, float]:
    logarithm = math.log(4.0 * iterations) - math.log(alpha)
    width = math.sqrt(12.0 * logarithm / shots)
    return max(-1.0, cosine - width), min(1.0, cosine + width)


def _clip_theta_interval(low: float, high: float) -> tuple[float, float]:
    limit = math.asin(0.25)
    clipped = max(0.0, low), min(limit, high)
    if clipped[0] > clipped[1]:
        raise RuntimeError("faster amplitude estimation produced an empty interval")
    return clipped


def _record_sample(state: _FasterState, power: int, shots: int, good: int) -> float:
    state.powers.append(power)
    state.circuit_shots.append(shots)
    state.good_counts.append(good)
    return _cosine_estimate(good, shots)


@final
class FasterAmplitudeEstimation:
    """Estimate a good-state probability with Nakaji's two-stage algorithm.

    Every round samples one or two circuits of the form ``Q**k A``. The first
    stage obtains a narrow angle interval from one cosine estimate; the second
    stage resolves successive doubled angles using two cosine estimates and
    the transition-stage phase offset. Requests in the same round are sent in
    one Sampler batch, while adaptive dependencies remain sequential.

    The supplied problem must already satisfy
    ``sqrt(a_problem) <= 0.25``. When ``amplitude_rescaling_factor=0.25``, the
    caller must also supply the matching externally rescaled ``A``, ``Q``, and
    predicate; the reported target is ``a_problem / 0.25**2``.

    Args:
        sampler: Sampling implementation used for every experiment.

    Raises:
        TypeError: If the sampler has the wrong type.
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

    @overload
    def run(
        self,
        problem: AmplitudeEstimationProblem,
        *,
        alpha: float,
        max_iterations: int,
        amplitude_rescaling_factor: float | None = None,
    ) -> FasterAmplitudeEstimationResult: ...

    @overload
    def run(
        self,
        requests: Sequence[FasterAmplitudeEstimationRequest],
        /,
    ) -> list[FasterAmplitudeEstimationResult]: ...

    def run(
        self,
        problem: AmplitudeEstimationProblem
        | Sequence[FasterAmplitudeEstimationRequest],
        *,
        alpha: float | None = None,
        max_iterations: int | None = None,
        amplitude_rescaling_factor: float | None = None,
    ) -> FasterAmplitudeEstimationResult | list[FasterAmplitudeEstimationResult]:
        """Estimate one probability or run an ordered batch of small-amplitude problems.

        Args:
            problem: An amplitude-estimation problem, or a positional batch of requests.
            alpha: Total failure-probability budget for a single problem.
            max_iterations: Number of adaptive rounds. Together with ``alpha``, this
                sets the sample count per circuit in each stage.
            amplitude_rescaling_factor: Amplitude scale already applied to the
                supplied problem. Defaults to 1.0. Use 0.25 for external rescaling.

        Returns:
            One result or a list in request order. Each result includes the
            probability estimate, confidence interval, and per-circuit counts.

        Raises:
            TypeError: If required options are missing, argument types are invalid,
                or single-problem options are supplied with a batch.
            ValueError: If a request or sampling result violates its contract.
            RuntimeError: If sampled values produce invalid phase geometry.
        """
        if isinstance(problem, AmplitudeEstimationProblem):
            if alpha is None or max_iterations is None:
                raise TypeError("a single problem requires alpha and max_iterations")
            request = FasterAmplitudeEstimationRequest(
                problem,
                alpha,
                max_iterations,
                _DEFAULT_RESCALING_FACTOR
                if amplitude_rescaling_factor is None
                else amplitude_rescaling_factor,
            )
            return self._run_batch((request,))[0]
        if any(
            value is not None
            for value in (alpha, max_iterations, amplitude_rescaling_factor)
        ):
            raise TypeError("batch requests specify their own estimation options")
        return self._run_batch(problem)

    def _run_batch(
        self, requests: Sequence[FasterAmplitudeEstimationRequest]
    ) -> list[FasterAmplitudeEstimationResult]:
        """Sample all active problems together and update each angle interval per round."""
        states: list[_FasterState] = []
        for request in requests:
            if not isinstance(request, FasterAmplitudeEstimationRequest):
                raise TypeError(
                    "requests must contain FasterAmplitudeEstimationRequest instances"
                )
            request.validate()
            states.append(_FasterState(request))
        if not states:
            return []

        for iteration in range(
            1, max(state.request.max_iterations for state in states) + 1
        ):
            sampler_requests: list[SamplerRequest] = []
            experiments: list[tuple[_FasterState, int, int]] = []
            for state in states:
                if iteration > state.request.max_iterations:
                    continue
                first_shots, second_shots = _shot_schedule(
                    float(state.request.alpha), state.request.max_iterations
                )
                base_power = 1 << (iteration - 1)
                powers = (
                    (base_power,)
                    if state.first_stage
                    else (
                        base_power,
                        base_power + (1 << (state.transition_iteration - 1)),
                    )
                )
                shots = first_shots if state.first_stage else second_shots
                for power in powers:
                    circuit = _build_grover_power_circuit(state.request.problem, power)
                    measured = tuple(
                        circuit.qubits[position]
                        for position in state.request.problem.objective_qubits
                    )
                    sampler_requests.append(SamplerRequest(circuit, measured, shots))
                    experiments.append((state, power, shots))

            sampled = require_sampler_results(
                self._sampler.run(sampler_requests), sampler_requests
            )
            observations: dict[int, list[float]] = {id(state): [] for state in states}
            for (state, power, shots), sample in zip(experiments, sampled):
                good = _count_good_outcomes(sample, state.request.problem)
                observations[id(state)].append(
                    _record_sample(state, power, shots, good)
                )

            for state in states:
                if iteration > state.request.max_iterations:
                    continue
                cosines = observations[id(state)]
                denominator = (1 << (iteration + 1)) + 2
                if state.first_stage:
                    state.first_stage_steps += 1
                    shots = state.circuit_shots[-1]
                    cosine_low, cosine_high = _chernoff_interval(
                        cosines[0],
                        float(state.request.alpha),
                        state.request.max_iterations,
                        shots,
                    )
                    state.theta_interval = _clip_theta_interval(
                        math.acos(cosine_high) / denominator,
                        math.acos(cosine_low) / denominator,
                    )
                    if (
                        (1 << (iteration + 1)) * state.theta_interval[1]
                        >= 3.0 * math.pi / 8.0
                        and iteration < state.request.max_iterations
                    ):
                        state.transition_iteration = iteration
                        state.phase_offset = (1 << iteration) * sum(
                            state.theta_interval
                        )
                        state.first_stage = False
                else:
                    sine_denominator = math.sin(state.phase_offset)
                    if abs(sine_denominator) <= 16.0 * math.ulp(1.0):
                        raise RuntimeError(
                            "faster amplitude estimation phase offset is singular"
                        )
                    sine = (
                        cosines[0] * math.cos(state.phase_offset) - cosines[1]
                    ) / sine_denominator
                    rho = math.atan2(sine, cosines[0])
                    turns = math.floor(
                        (denominator * state.theta_interval[1] - rho + math.pi / 3.0)
                        / (2.0 * math.pi)
                    )
                    state.theta_interval = _clip_theta_interval(
                        (2.0 * math.pi * turns + rho - math.pi / 3.0) / denominator,
                        (2.0 * math.pi * turns + rho + math.pi / 3.0) / denominator,
                    )
                state.theta_intervals.append(state.theta_interval)

        results: list[FasterAmplitudeEstimationResult] = []
        for state in states:
            theta_low, theta_high = state.theta_interval
            theta = (theta_low + theta_high) / 2.0
            factor = float(state.request.amplitude_rescaling_factor)
            estimation = min(1.0, max(0.0, (math.sin(theta) / factor) ** 2))
            confidence_interval = (
                min(1.0, max(0.0, (math.sin(theta_low) / factor) ** 2)),
                min(1.0, max(0.0, (math.sin(theta_high) / factor) ** 2)),
            )
            results.append(
                FasterAmplitudeEstimationResult(
                    estimation=float(estimation),
                    confidence_interval=confidence_interval,
                    shots=sum(state.circuit_shots),
                    num_grover_queries=sum(state.powers),
                    num_oracle_queries=sum(
                        power * shots
                        for power, shots in zip(state.powers, state.circuit_shots)
                    ),
                    powers=tuple(state.powers),
                    circuit_shots=tuple(state.circuit_shots),
                    good_counts=tuple(state.good_counts),
                    num_steps=state.request.max_iterations,
                    num_first_stage_steps=state.first_stage_steps,
                    theta_intervals=tuple(state.theta_intervals),
                    success_probability_lower_bound=1.0 - float(state.request.alpha),
                    amplitude_rescaling_factor=factor,
                )
            )
        return results
