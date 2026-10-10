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

"""Maximum-likelihood amplitude estimation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal, localcontext
from fractions import Fraction
from functools import cache
from itertools import pairwise
from numbers import Real
from typing import final

from ...protocols.sampler import Sampler, SamplerRequest, validate_sampler_results
from ._common import (
    _build_grover_power_circuit,
    _count_good_outcomes,
    _validate_positive_integer,
)
from .problem import AmplitudeEstimationProblem, _validate_problem

# Core transports ``shots`` as Rust ``usize`` and can therefore carry larger
# values on 64-bit hosts. MLAE's candidate geometry is nevertheless binary64;
# this stricter bound keeps every observed integer count exactly representable.
_MAX_BINARY64_LIKELIHOOD_SHOTS = (1 << 53) - 1


@dataclass(frozen=True, slots=True)
class MaximumLikelihoodAmplitudeEstimationRequest:
    """One fixed-schedule maximum-likelihood estimation request."""

    problem: AmplitudeEstimationProblem
    powers: tuple[int, ...]
    shots: int
    search_precision: float = 1e-8

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Revalidate the shallowly frozen request before execution."""
        _validate_problem(self.problem)
        if not isinstance(self.powers, tuple):
            raise TypeError("powers must be a tuple")
        if not self.powers:
            raise ValueError("powers must not be empty")
        if any(
            isinstance(power, bool) or not isinstance(power, int)
            for power in self.powers
        ):
            raise TypeError("powers must contain only integers")
        if any(power < 0 for power in self.powers):
            raise ValueError("powers must be non-negative")
        if self.powers[0] != 0:
            raise ValueError("powers must start with zero")
        if any(left >= right for left, right in zip(self.powers, self.powers[1:])):
            raise ValueError("powers must be strictly increasing")
        _validate_positive_integer(self.shots, "shots")
        if self.shots > _MAX_BINARY64_LIKELIHOOD_SHOTS:
            raise ValueError(
                "shots exceeds the likelihood candidate geometry numeric limit "
                "of 2**53 - 1"
            )
        if isinstance(self.search_precision, bool) or not isinstance(
            self.search_precision, Real
        ):
            raise TypeError("search_precision must be a real number")
        precision = float(self.search_precision)
        if not math.isfinite(precision):
            raise ValueError("search_precision must be finite")
        if precision <= 0.0:
            raise ValueError("search_precision must be greater than zero")
        if precision < math.ulp(math.pi / 2.0):
            raise ValueError(
                "search_precision is below floating-point resolution on [0, pi/2]"
            )


@dataclass(frozen=True, slots=True)
class MaximumLikelihoodAmplitudeEstimationResult:
    """Global likelihood optimum and its mapped asymptotic interval."""

    estimation: float
    confidence_interval: tuple[float, float]
    shots: int
    num_grover_queries: int
    num_oracle_queries: int
    powers: tuple[int, ...]
    good_counts: tuple[int, ...]
    log_likelihood: float
    asymptotic_interval: tuple[float, float]

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.estimation, float) or not math.isfinite(self.estimation):
            raise TypeError("estimation must be a finite float")
        if not 0.0 <= self.estimation <= 1.0:
            raise ValueError("estimation must lie in [0, 1]")
        if self.confidence_interval != self.asymptotic_interval:
            raise ValueError("confidence_interval must equal asymptotic_interval")
        if (
            not isinstance(self.asymptotic_interval, tuple)
            or len(self.asymptotic_interval) != 2
        ):
            raise TypeError("asymptotic_interval must be a pair")
        low, high = self.asymptotic_interval
        if not 0.0 <= low <= self.estimation <= high <= 1.0:
            raise ValueError("asymptotic_interval must contain estimation in [0, 1]")
        for name, value in (
            ("shots", self.shots),
            ("num_grover_queries", self.num_grover_queries),
            ("num_oracle_queries", self.num_oracle_queries),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not isinstance(self.powers, tuple) or not self.powers:
            raise TypeError("powers must be a non-empty tuple")
        if not isinstance(self.good_counts, tuple) or len(self.good_counts) != len(
            self.powers
        ):
            raise TypeError("good_counts must have one tuple entry per power")
        per_power_shots = self.shots // len(self.powers)
        if per_power_shots * len(self.powers) != self.shots:
            raise ValueError("shots must be divisible by the number of powers")
        if any(
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 0 <= count <= per_power_shots
            for count in self.good_counts
        ):
            raise ValueError("good_counts entries must be valid sampled counts")
        if self.num_grover_queries != sum(self.powers):
            raise ValueError("num_grover_queries must equal the sum of powers")
        if self.num_oracle_queries != per_power_shots * sum(self.powers):
            raise ValueError("num_oracle_queries must be shot-weighted")
        if not isinstance(self.log_likelihood, float) or not math.isfinite(
            self.log_likelihood
        ):
            raise TypeError("log_likelihood must be a finite float")


def _log_likelihood(
    theta: float,
    powers: Sequence[int],
    good_counts: Sequence[int],
    shots: int,
) -> float:
    total = 0.0
    for power, good in zip(powers, good_counts):
        angle = (2 * power + 1) * theta
        sine = abs(math.sin(angle))
        cosine = abs(math.cos(angle))
        failures = shots - good
        if good:
            if sine == 0.0:
                return -math.inf
            total += 2.0 * good * math.log(sine)
        if failures:
            if cosine == 0.0:
                return -math.inf
            total += 2.0 * failures * math.log(cosine)
    return total


def _boundary_log_likelihood(
    normalized_theta: Fraction,
    powers: Sequence[int],
    good_counts: Sequence[int],
    shots: int,
) -> float:
    """Evaluate a rational multiple of pi with algebraic endpoint handling."""
    total = 0.0
    for power, good in zip(powers, good_counts):
        phase = (2 * power + 1) * normalized_theta
        failures = shots - good
        if phase.denominator == 1:
            if good:
                return -math.inf
            continue
        if (phase - Fraction(1, 2)).denominator == 1:
            if failures:
                return -math.inf
            continue

        fractional_phase = phase - math.floor(phase)
        angle = float(fractional_phase) * math.pi
        if good:
            total += 2.0 * good * math.log(abs(math.sin(angle)))
        if failures:
            total += 2.0 * failures * math.log(abs(math.cos(angle)))
    return total


def _score(
    theta: float,
    powers: Sequence[int],
    good_counts: Sequence[int],
    shots: int,
) -> float:
    """Evaluate the first derivative of the log likelihood."""
    contributions: list[float] = []
    for power, good in zip(powers, good_counts):
        frequency = 2 * power + 1
        angle = frequency * theta
        sine = math.sin(angle)
        cosine = math.cos(angle)
        failures = shots - good
        contributions.append(
            2.0
            * frequency
            * (good * cosine * cosine - failures * sine * sine)
            / (sine * cosine)
        )
    return math.fsum(contributions)


def _boundary_score_limit(
    normalized_theta: Fraction,
    direction: int,
    powers: Sequence[int],
    good_counts: Sequence[int],
    shots: int,
) -> float:
    """Evaluate the one-sided score limit at an exact singularity boundary."""
    contributions: list[float] = []
    for power, good in zip(powers, good_counts):
        frequency = 2 * power + 1
        phase = frequency * normalized_theta
        failures = shots - good
        if phase.denominator == 1:
            if good:
                return math.copysign(math.inf, direction)
            continue
        if (phase - Fraction(1, 2)).denominator == 1:
            if failures:
                return math.copysign(math.inf, direction)
            continue

        fractional_phase = phase - math.floor(phase)
        angle = float(fractional_phase) * math.pi
        sine = math.sin(angle)
        cosine = math.cos(angle)
        contributions.append(
            2.0
            * frequency
            * (good * cosine * cosine - failures * sine * sine)
            / (sine * cosine)
        )
    return math.fsum(contributions)


@dataclass(frozen=True, slots=True)
class _LikelihoodCandidate:
    theta: float
    value: float
    boundary: Fraction | None
    interval: tuple[Fraction, Fraction] | None = None


def _decimal_arctangent_inverse(inverse: int) -> Decimal:
    value = Decimal(1) / Decimal(inverse)
    squared = value * value
    term = value
    total = value
    index = 1
    while True:
        term *= -squared
        updated = total + term / Decimal(2 * index + 1)
        if updated == total:
            return total
        total = updated
        index += 1


@cache
def _decimal_pi(precision: int) -> Decimal:
    with localcontext() as context:
        context.prec = precision + 12
        value = 16 * _decimal_arctangent_inverse(5) - 4 * _decimal_arctangent_inverse(
            239
        )
        context.prec = precision
        return +value


def _decimal_sine_cosine(angle: Decimal) -> tuple[Decimal, Decimal]:
    squared = angle * angle

    sine_term = angle
    sine = angle
    index = 1
    while True:
        sine_term *= -squared / Decimal((2 * index) * (2 * index + 1))
        updated = sine + sine_term
        if updated == sine:
            break
        sine = updated
        index += 1

    cosine_term = Decimal(1)
    cosine = Decimal(1)
    index = 1
    while True:
        cosine_term *= -squared / Decimal((2 * index - 1) * (2 * index))
        updated = cosine + cosine_term
        if updated == cosine:
            break
        cosine = updated
        index += 1
    return abs(sine), abs(cosine)


def _decimal_phase_sine_cosine(
    candidate: _LikelihoodCandidate,
    frequency: int,
    pi: Decimal,
    refined_theta: Decimal | None,
) -> tuple[Decimal, Decimal, bool, bool]:
    if candidate.boundary is not None:
        rational_phase = frequency * candidate.boundary
        if rational_phase.denominator == 1:
            return Decimal(0), Decimal(1), True, False
        if (rational_phase - Fraction(1, 2)).denominator == 1:
            return Decimal(1), Decimal(0), False, True
        fractional_phase = rational_phase - math.floor(rational_phase)
        if fractional_phase > Fraction(1, 2):
            fractional_phase = 1 - fractional_phase
        angle = (
            Decimal(fractional_phase.numerator)
            / Decimal(fractional_phase.denominator)
            * pi
        )
    else:
        if refined_theta is None:
            raise RuntimeError("interior candidate is missing its refined theta")
        decimal_phase = refined_theta * Decimal(frequency)
        turns = (decimal_phase / pi).to_integral_value(rounding=ROUND_FLOOR)
        angle = decimal_phase - turns * pi
        half_pi = pi / 2
        if angle > half_pi:
            angle = pi - angle
    sine, cosine = _decimal_sine_cosine(angle)
    return sine, cosine, False, False


def _decimal_score(
    theta: Decimal,
    powers: Sequence[int],
    good_counts: Sequence[int],
    shots: int,
    pi: Decimal,
) -> Decimal:
    contributions: list[Decimal] = []
    half_pi = pi / 2
    for power, good in zip(powers, good_counts):
        frequency = 2 * power + 1
        phase = theta * Decimal(frequency)
        turns = (phase / pi).to_integral_value(rounding=ROUND_FLOOR)
        angle = phase - turns * pi
        cosine_sign = Decimal(1)
        if angle > half_pi:
            angle = pi - angle
            cosine_sign = Decimal(-1)
        sine, cosine = _decimal_sine_cosine(angle)
        failures = shots - good
        contributions.append(
            Decimal(2 * frequency)
            * (Decimal(good) * cosine * cosine - Decimal(failures) * sine * sine)
            / (sine * cosine * cosine_sign)
        )
    return sum(contributions, Decimal(0))


def _refine_decimal_root(
    candidate: _LikelihoodCandidate,
    powers: Sequence[int],
    good_counts: Sequence[int],
    shots: int,
    pi: Decimal,
    precision: int,
) -> Decimal:
    if candidate.interval is None:
        return Decimal.from_float(candidate.theta)
    normalized_low, normalized_high = candidate.interval
    low = Decimal(normalized_low.numerator) / Decimal(normalized_low.denominator) * pi
    high = (
        Decimal(normalized_high.numerator) / Decimal(normalized_high.denominator) * pi
    )
    target_width = Decimal(1).scaleb(-precision)
    for _ in range(4 * precision + 64):
        if high - low <= target_width:
            return (low + high) / 2
        midpoint = (low + high) / 2
        score = _decimal_score(midpoint, powers, good_counts, shots, pi)
        if score > 0:
            low = midpoint
        elif score < 0:
            high = midpoint
        else:
            return midpoint
    raise RuntimeError("high-precision maximum-likelihood root search is unstable")


def _decimal_log_likelihood(
    candidate: _LikelihoodCandidate,
    powers: Sequence[int],
    good_counts: Sequence[int],
    shots: int,
    precision: int,
) -> Decimal:
    with localcontext() as context:
        context.prec = precision + 12
        pi = _decimal_pi(context.prec)
        refined_theta = (
            _refine_decimal_root(
                candidate,
                powers,
                good_counts,
                shots,
                pi,
                precision,
            )
            if candidate.boundary is None
            else None
        )
        total = Decimal(0)
        for power, good in zip(powers, good_counts):
            failures = shots - good
            sine, cosine, exact_zero, exact_one = _decimal_phase_sine_cosine(
                candidate,
                2 * power + 1,
                pi,
                refined_theta,
            )
            if good:
                if exact_zero or sine.is_zero():
                    return Decimal("-Infinity")
                total += Decimal(2 * good) * sine.ln()
            if failures:
                if exact_one or cosine.is_zero():
                    return Decimal("-Infinity")
                total += Decimal(2 * failures) * cosine.ln()
        context.prec = precision
        return +total


def _select_with_decimal_scores(
    candidates: Sequence[_LikelihoodCandidate],
    powers: Sequence[int],
    good_counts: Sequence[int],
    shots: int,
    precision: int,
) -> int:
    scores = [
        _decimal_log_likelihood(
            candidate,
            powers,
            good_counts,
            shots,
            precision,
        )
        for candidate in candidates
    ]
    maximum = max(scores)
    return min(
        (index for index, score in enumerate(scores) if score == maximum),
        key=lambda index: candidates[index].theta,
    )


def _select_global_candidate(
    candidates: Sequence[_LikelihoodCandidate],
    powers: Sequence[int],
    good_counts: Sequence[int],
    shots: int,
) -> _LikelihoodCandidate:
    maximum_frequency = 2 * max(powers) + 1
    base_precision = max(
        80,
        len(str(shots)) + 60,
        len(str(maximum_frequency)) + 60,
    )
    first = _select_with_decimal_scores(
        candidates, powers, good_counts, shots, base_precision
    )
    second = _select_with_decimal_scores(
        candidates, powers, good_counts, shots, base_precision + 32
    )
    if first == second:
        selected_index = second
        selected_precision = base_precision + 32
    else:
        third = _select_with_decimal_scores(
            candidates, powers, good_counts, shots, base_precision + 64
        )
        if second != third:
            raise RuntimeError(
                "high-precision maximum-likelihood candidate ranking is unstable"
            )
        selected_index = third
        selected_precision = base_precision + 64

    selected = candidates[selected_index]
    if selected.interval is None:
        return selected
    with localcontext() as context:
        context.prec = selected_precision + 12
        refined_theta = _refine_decimal_root(
            selected,
            powers,
            good_counts,
            shots,
            _decimal_pi(context.prec),
            selected_precision,
        )
    theta = float(refined_theta)
    return _LikelihoodCandidate(
        theta,
        _log_likelihood(theta, powers, good_counts, shots),
        None,
        selected.interval,
    )


def _boundary_candidate(
    normalized_theta: Fraction,
    powers: Sequence[int],
    good_counts: Sequence[int],
    shots: int,
) -> _LikelihoodCandidate:
    return _LikelihoodCandidate(
        float(normalized_theta) * math.pi,
        _boundary_log_likelihood(
            normalized_theta,
            powers,
            good_counts,
            shots,
        ),
        normalized_theta,
    )


def _interval_maximum(
    normalized_low: Fraction,
    normalized_high: Fraction,
    powers: Sequence[int],
    good_counts: Sequence[int],
    shots: int,
    precision: float,
) -> tuple[_LikelihoodCandidate, bool]:
    """Maximize one strictly concave open interval using its monotone score."""
    low_candidate = _boundary_candidate(normalized_low, powers, good_counts, shots)
    high_candidate = _boundary_candidate(normalized_high, powers, good_counts, shots)
    score_low = _boundary_score_limit(normalized_low, 1, powers, good_counts, shots)
    score_high = _boundary_score_limit(normalized_high, -1, powers, good_counts, shots)
    if score_low <= 0.0:
        return low_candidate, False
    if score_high >= 0.0:
        return high_candidate, False

    low = low_candidate.theta
    high = high_candidate.theta
    for _ in range(256):
        if high - low <= precision:
            break
        midpoint = (low + high) / 2.0
        if midpoint in {low, high}:
            break
        if _score(midpoint, powers, good_counts, shots) > 0.0:
            low = midpoint
        else:
            high = midpoint
    else:
        raise RuntimeError("maximum-likelihood score search failed to terminate")

    theta = (low + high) / 2.0
    snap_candidates: list[_LikelihoodCandidate] = []
    if math.isfinite(low_candidate.value) and high - low_candidate.theta <= precision:
        snap_candidates.append(low_candidate)
    if math.isfinite(high_candidate.value) and high_candidate.theta - low <= precision:
        snap_candidates.append(high_candidate)
    if snap_candidates:
        return min(
            snap_candidates,
            key=lambda candidate: (abs(theta - candidate.theta), candidate.theta),
        ), False
    return (
        _LikelihoodCandidate(
            theta,
            _log_likelihood(theta, powers, good_counts, shots),
            None,
            (normalized_low, normalized_high),
        ),
        True,
    )


def _global_maximum_likelihood(
    powers: tuple[int, ...],
    good_counts: tuple[int, ...],
    shots: int,
    precision: float,
) -> tuple[float, float]:
    """Partition the complete domain at every trigonometric singularity."""
    normalized_boundaries = {
        Fraction(index, 2 * (2 * power + 1))
        for power in powers
        for index in range(2 * power + 2)
    }
    ordered_boundaries = sorted(normalized_boundaries)
    candidates: list[_LikelihoodCandidate] = []
    dominated_boundaries: set[Fraction] = set()
    for low, high in pairwise(ordered_boundaries):
        candidate, interior = _interval_maximum(
            low,
            high,
            powers,
            good_counts,
            shots,
            precision,
        )
        candidates.append(candidate)
        if interior:
            dominated_boundaries.update((low, high))

    finite = [
        candidate
        for candidate in candidates
        if math.isfinite(candidate.value)
        and candidate.boundary not in dominated_boundaries
    ]
    if not finite:
        raise ValueError("sample counts define no finite likelihood point")
    selected = _select_global_candidate(finite, powers, good_counts, shots)
    return selected.theta, selected.value


@final
class MaximumLikelihoodAmplitudeEstimation:
    """Estimate an amplitude from a fixed set of Grover-power experiments.

    Every configured ``Q**k A`` circuit is included in one flattened Sampler
    call. The likelihood is searched over the complete physical domain
    ``theta in [0, pi/2]`` by partitioning it at every sine/cosine singularity.
    The log likelihood is concave inside each resulting interval, so every
    interval and boundary can be optimized deterministically. This avoids
    relying on a single initial point for the multimodal likelihood.

    ``asymptotic_interval`` is a mapped 95% normal interval in ``theta`` based
    on Fisher information. It is an asymptotic diagnostic, not a finite-shot
    coverage guarantee.

    Reference: Y. Suzuki et al., *Amplitude Estimation without Phase
    Estimation*, arXiv:1904.10246v2.
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
        requests: Sequence[MaximumLikelihoodAmplitudeEstimationRequest],
        /,
    ) -> list[MaximumLikelihoodAmplitudeEstimationResult]:
        """Execute every fixed power for a complete batch in one Sampler call."""
        sampler_requests: list[SamplerRequest] = []
        spans: list[tuple[int, int]] = []
        for request in requests:
            if not isinstance(request, MaximumLikelihoodAmplitudeEstimationRequest):
                raise TypeError(
                    "requests must contain "
                    "MaximumLikelihoodAmplitudeEstimationRequest instances"
                )
            request.validate()
            start = len(sampler_requests)
            for power in request.powers:
                circuit = _build_grover_power_circuit(request.problem, power)
                measured = tuple(
                    circuit.qubits[position]
                    for position in request.problem.objective_qubits
                )
                sampler_requests.append(
                    SamplerRequest(circuit, measured, request.shots)
                )
            spans.append((start, len(sampler_requests)))
        if not sampler_requests:
            return []

        sampled = validate_sampler_results(
            self._sampler.run(sampler_requests), sampler_requests
        )
        results: list[MaximumLikelihoodAmplitudeEstimationResult] = []
        for request, (start, stop) in zip(requests, spans):
            good_counts = tuple(
                _count_good_outcomes(sample, request.problem)
                for sample in sampled[start:stop]
            )
            theta, log_likelihood = _global_maximum_likelihood(
                request.powers,
                good_counts,
                request.shots,
                float(request.search_precision),
            )
            estimation = float(math.sin(theta) ** 2)
            fisher_information = (
                4.0
                * request.shots
                * sum((2 * power + 1) ** 2 for power in request.powers)
            )
            radius = 1.959963984540054 / math.sqrt(fisher_information)
            theta_low = max(0.0, theta - radius)
            theta_high = min(math.pi / 2.0, theta + radius)
            interval = (
                float(math.sin(theta_low) ** 2),
                float(math.sin(theta_high) ** 2),
            )
            results.append(
                MaximumLikelihoodAmplitudeEstimationResult(
                    estimation=estimation,
                    confidence_interval=interval,
                    shots=request.shots * len(request.powers),
                    num_grover_queries=sum(request.powers),
                    num_oracle_queries=request.shots * sum(request.powers),
                    powers=request.powers,
                    good_counts=good_counts,
                    log_likelihood=float(log_likelihood),
                    asymptotic_interval=interval,
                )
            )
        return results


__all__ = [
    "MaximumLikelihoodAmplitudeEstimation",
    "MaximumLikelihoodAmplitudeEstimationRequest",
    "MaximumLikelihoodAmplitudeEstimationResult",
]
