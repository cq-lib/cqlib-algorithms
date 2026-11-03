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

"""Shared validation for domain ground-state workflows and results."""

from __future__ import annotations

from ..protocols.estimator import EstimatorResult


def snapshot_estimator_results(
    values: tuple[EstimatorResult, ...],
    *,
    name: str,
) -> tuple[EstimatorResult, ...]:
    """Return an immutable validated estimator-result sequence."""
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    if any(not isinstance(value, EstimatorResult) for value in values):
        raise TypeError(f"{name} must contain only EstimatorResult values")
    return tuple(values)


__all__: list[str] = []
