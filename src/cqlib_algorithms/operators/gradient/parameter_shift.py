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
from typing import TypeAlias, final

from cqlib.circuit import Circuit, Parameter, ValueOperation
from cqlib.circuit.gates import StandardGate

from ...protocols.estimator import Estimator, EstimatorRequest
from ...protocols.gradient import (
    Gradient,
    GradientRequest,
    GradientResult,
)
from ._parameter_analysis import (
    Occurrence,
    analyze_symbol,
    local_shift_plan,
    validate_circuit,
)

_Plan: TypeAlias = tuple[int, int, Occurrence, tuple[float, ...], tuple[float, ...]]


def _numeric_angle(value: float | Parameter) -> float:
    """Return a bound gate angle or reject an incomplete binding."""
    if isinstance(value, Parameter):
        raise TypeError("gate parameters must be fully bound before evaluation")
    return float(value)


def _shifted_operation(
    operation: ValueOperation,
    slot: int,
    shift: float,
) -> ValueOperation:
    """Return ``operation`` with one numeric parameter slot shifted."""
    params = list(operation.params)
    if slot >= len(params):
        raise ValueError(f"parameter slot {slot} is missing from {operation.name!r}")
    params[slot] = _numeric_angle(params[slot]) + shift

    if operation.instruction_type == "standard":
        gate_factory = getattr(StandardGate, operation.name, None)
        if gate_factory is None:
            raise ValueError(f"standard gate {operation.name!r} is not supported")
        gate = gate_factory(*params)
        return ValueOperation.from_standard_gate(
            gate,
            operation.qubits,
            operation.label,
        )
    if operation.instruction_type == "mcgate":
        control_part, base_name = operation.name.split("-", 1)
        num_controls = int(control_part[1:])
        gate_factory = getattr(StandardGate, base_name, None)
        if gate_factory is None:
            raise ValueError(f"controlled gate {operation.name!r} is not supported")
        gate = gate_factory(*params).control(num_controls)
        return ValueOperation.from_mc_gate(
            gate,
            operation.qubits,
            operation.label,
        )
    raise ValueError(
        "parameter occurrence does not belong to a supported gate operation: "
        f"{operation.name!r}"
    )


def _shifted_occurrence_circuit(
    bound: Circuit,
    occurrence: Occurrence,
    shift: float,
) -> Circuit:
    """Copy a bound circuit while shifting exactly one parameter slot."""
    operations: list[ValueOperation] = []
    for index, operation in enumerate(bound.operations):
        if index == occurrence.op_index:
            operations.append(_shifted_operation(operation, occurrence.slot, shift))
        else:
            operations.append(operation)
    shifted = Circuit.from_operations(
        list(bound.qubits),
        operations,
        classical_vars=list(bound.classical_vars),
        classical_values=list(bound.classical_values),
    )
    shifted.set_global_phase(bound.global_phase)
    return shifted


@final
class ParameterShiftGradient(Gradient):
    """Batched gradient of expectation values via exact shift rules.

    For every requested parameter the component analyzes each gate-slot
    occurrence separately, evaluates that slot at its exact local shift
    points, applies the affine chain coefficient, and sums the occurrence
    derivatives.  Ordinary rotations and phase gates use two evaluations
    per occurrence.  Controlled rotations use four.  Consequently symbols
    shared by many gates have linear rather than combinatorial evaluation
    cost.  Global-phase-only parameters yield a zero gradient without any
    estimator evaluation.

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

        plans: list[_Plan] = []
        for position, request in enumerate(request_list):
            request.validate()
            circuit = request.estimator_request.circuit
            validate_circuit(circuit)
            for name_index, name in enumerate(request.parameter_names):
                occurrences = analyze_symbol(circuit, name)
                for occurrence in occurrences:
                    shifts, weights = local_shift_plan(occurrence)
                    plans.append((position, name_index, occurrence, shifts, weights))

        evaluations: list[EstimatorRequest] = []
        owners: list[tuple[int, int]] = []
        bound_circuits = [
            request.estimator_request.circuit.assign_parameters(
                dict(request.estimator_request.parameter_bindings)
            )
            for request in request_list
        ]
        for plan_index, (position, _, occurrence, shifts, _) in enumerate(plans):
            base = request_list[position].estimator_request
            for shift_index, shift in enumerate(shifts):
                evaluations.append(
                    type(base)(
                        _shifted_occurrence_circuit(
                            bound_circuits[position], occurrence, shift
                        ),
                        base.observable,
                        {},
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

        totals = {plan_index: 0.0 for plan_index in range(len(plans))}
        for (plan_index, shift_index), result in zip(owners, results):
            occurrence = plans[plan_index][2]
            weights = plans[plan_index][4]
            totals[plan_index] += (
                occurrence.coefficient * weights[shift_index] * result.expectation
            )

        values: dict[tuple[int, int], float] = {}
        for plan_index, (position, name_index, _, _, _) in enumerate(plans):
            key = (position, name_index)
            values[key] = values.get(key, 0.0) + totals[plan_index]

        gradient_results: list[GradientResult] = []
        for position, request in enumerate(request_list):
            names = request.parameter_names
            gradient_results.append(
                GradientResult(
                    names,
                    tuple(
                        values.get((position, name_index), 0.0)
                        for name_index in range(len(names))
                    ),
                )
            )
        return gradient_results
