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

"""Exact parameter-shift gradients of estimator expectation values."""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from ...protocols.estimator import Estimator, EstimatorRequest
from ...protocols.gradient import (
    Gradient,
    GradientRequest,
    GradientResult,
)
from ._parameter_analysis import analyze_symbol, shift_plan, validate_circuit

_IMAGINARY_TOLERANCE = 1e-12


@final
class ParameterShiftGradient(Gradient):
    """Batched gradient of expectation values via exact shift rules.

    For every requested parameter the component analyzes the circuit to
    derive the parameter's expectation frequencies, then evaluates the
    expectation at exact shift points whose weights reproduce the
    derivative.  Classical single-occurrence rotations and phase gates
    reduce to the textbook two-point rules; symbols shared by several
    gates use the generalized shift rules implied by their frequency
    sets.  Global-phase-only parameters yield a zero gradient without
    any estimator evaluation.

    All shifted evaluations of all requests are flattened into a single
    :meth:`Estimator.run` batch.  Unsupported configurations (non-affine
    parameter expressions, unsupported parametric gates, circuit gates,
    non-unitary instructions) raise ``ValueError`` before any estimator
    call.  Gate support is checked only for parameter slots containing a
    requested gradient symbol; parameters outside that subset may occur in
    other gates as long as the bound circuit is valid for the estimator.
    Shot-based estimator requests are passed through unchanged; the
    statistical quality of a shot-based gradient is that of the underlying
    estimator evaluations.

    Raises:
        TypeError: If the estimator or a request has the wrong type.
        ValueError: If a request is invalid, a gradient parameter is not
            exactly analyzable, or an estimator returns a mismatched
            result count.
    """

    __slots__ = ()

    def run(
        self,
        estimator: Estimator,
        requests: Sequence[GradientRequest],
        /,
    ) -> list[GradientResult]:
        """Differentiate a batch of estimator expectation objectives."""
        if not isinstance(estimator, Estimator):
            raise TypeError("estimator must be an Estimator")
        request_list = list(requests)
        if any(not isinstance(request, GradientRequest) for request in request_list):
            raise TypeError("requests must contain only GradientRequest values")

        plans: list[tuple[int, str, tuple[float, ...], tuple[complex, ...]]] = []
        for position, request in enumerate(request_list):
            request.validate()
            circuit = request.estimator_request.circuit
            validate_circuit(circuit)
            for name in request.parameter_names:
                occurrences = analyze_symbol(circuit, name)
                plans.append((position, name, *shift_plan(occurrences)))

        evaluations: list[EstimatorRequest] = []
        owners: list[tuple[int, int]] = []
        for plan_index, (position, name, shifts, _) in enumerate(plans):
            base = request_list[position].estimator_request
            bindings = dict(base.parameter_bindings)
            for shift_index, shift in enumerate(shifts):
                shifted = dict(bindings)
                shifted[name] = bindings[name] + shift
                evaluations.append(
                    type(base)(
                        base.circuit,
                        base.observable,
                        shifted,
                        base.shots,
                    )
                )
                owners.append((plan_index, shift_index))

        results = list(estimator.run(evaluations)) if evaluations else []
        if len(results) != len(evaluations):
            raise ValueError(
                "estimator must return exactly one result per evaluation, "
                f"expected {len(evaluations)}, got {len(results)}"
            )

        totals = {plan_index: 0.0 + 0.0j for plan_index in range(len(plans))}
        weights_sums = {
            plan_index: sum(abs(weight) for weight in weights)
            for plan_index, (_, _, _, weights) in enumerate(plans)
        }
        for (plan_index, shift_index), result in zip(owners, results):
            totals[plan_index] += plans[plan_index][3][shift_index] * result.expectation

        values: dict[tuple[int, str], float] = {}
        for plan_index, (position, name, shifts, weights) in enumerate(plans):
            total = totals[plan_index]
            tolerance = _IMAGINARY_TOLERANCE * (1.0 + weights_sums[plan_index])
            if abs(total.imag) > tolerance:
                raise ValueError(
                    "parameter-shift gradient has an unexpected imaginary "
                    f"component {total.imag!r} for parameter {name!r}"
                )
            values[(position, name)] = float(total.real)

        gradient_results: list[GradientResult] = []
        for position, request in enumerate(request_list):
            names = request.parameter_names
            gradient_results.append(
                GradientResult(names, tuple(values[(position, name)] for name in names))
            )
        return gradient_results
