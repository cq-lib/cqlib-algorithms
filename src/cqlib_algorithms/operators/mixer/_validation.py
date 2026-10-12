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

"""Validation helpers shared by mixer components."""

from __future__ import annotations

import copy
from collections.abc import Sequence

from cqlib.circuit.ansatz import EvolutionStrategy


def validate_positive_integer(value: int, *, name: str) -> None:
    """Validate a strictly positive integer without coercion."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def validate_nonnegative_integer(value: int, *, name: str) -> None:
    """Validate a non-negative integer without coercion."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def snapshot_strategy(strategy: EvolutionStrategy | None) -> EvolutionStrategy:
    """Return an isolated evolution strategy, defaulting to one auto step."""
    if strategy is not None and not isinstance(strategy, EvolutionStrategy):
        raise TypeError("strategy must be an EvolutionStrategy or None")
    return EvolutionStrategy.auto(1) if strategy is None else copy.copy(strategy)


def resolve_control_state(num_controls: int, control_state: int | None) -> int:
    """Validate and resolve a little-endian computational control state."""
    validate_nonnegative_integer(num_controls, name="num_controls")
    if control_state is None:
        return 0
    if isinstance(control_state, bool) or not isinstance(control_state, int):
        raise TypeError("control_state must be an integer or None")
    if control_state < 0 or control_state >= 1 << num_controls:
        raise ValueError("control_state does not fit num_controls bits")
    return control_state


def snapshot_processing_times(processing_times: Sequence[int]) -> tuple[int, ...]:
    """Validate positive job durations and freeze their order."""
    if isinstance(processing_times, (str, bytes)) or not isinstance(
        processing_times, Sequence
    ):
        raise TypeError("processing_times must be a sequence of integers")
    frozen = tuple(processing_times)
    if len(frozen) < 2:
        raise ValueError("processing_times must contain at least two jobs")
    for duration in frozen:
        validate_positive_integer(duration, name="processing time")
    return frozen


def resolve_horizon(processing_times: tuple[int, ...], horizon: int | None) -> int:
    """Resolve the paper's common absolute-time horizon."""
    minimum = sum(processing_times)
    if horizon is None:
        return minimum
    validate_positive_integer(horizon, name="horizon")
    if horizon < minimum:
        raise ValueError("horizon must be at least the sum of processing_times")
    return horizon


def validate_job(job: int, *, num_jobs: int, name: str) -> None:
    """Validate a zero-based job index."""
    if isinstance(job, bool) or not isinstance(job, int):
        raise TypeError(f"{name} must be an integer")
    if job < 0 or job >= num_jobs:
        raise ValueError(f"{name} is out of range")
