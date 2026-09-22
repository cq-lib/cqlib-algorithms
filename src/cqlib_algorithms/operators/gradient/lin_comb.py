# This code is part of Cqlib.
#
# (C) Copyright China Telecom Quantum Group 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# use this file except in compliance with the License, obtained from the
# LICENSE.txt file in the root directory of this source tree or
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Linear-combination gradients via auxiliary-qubit Hadamard tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias, TypeVar, final

import numpy as np
from cqlib.circuit import Circuit, Parameter, ValueOperation
from cqlib.circuit.gates import StandardGate
from cqlib.qis import Hamiltonian, Pauli, PauliString

from cqlib import Qubit

from ...protocols.estimator import Estimator, EstimatorRequest
from ...protocols.gradient import (
    Gradient,
    GradientRequest,
    GradientResult,
)
from ._parameter_analysis import (
    Occurrence,
    analyze_symbol,
    validate_circuit,
)

_HALF_PI = np.pi / 2

Axis: TypeAlias = Literal["X", "Y", "Z"]

_AXIS_GATES: dict[Axis, StandardGate] = {
    "X": StandardGate.X,
    "Y": StandardGate.Y,
    "Z": StandardGate.Z,
}

# One unitary term of a generator decomposition: a weight plus the
# Pauli factors ``(axis, circuit qubit position)`` realized under one
# auxiliary-qubit control.  Identity terms contribute
# ``Im<psi|O|psi> = 0`` for Hermitian observables and are omitted.
Term: TypeAlias = tuple[float, tuple[tuple[Axis, int], ...]]

_T = TypeVar("_T")


def _powerset(items: list[_T]) -> list[list[_T]]:
    """Return all subsets of ``items`` in deterministic binary expansion order."""
    subsets: list[list[_T]] = [[]]
    for item in items:
        subsets.extend(subset + [item] for subset in list(subsets))
    return subsets


def _validated_axis(value: object) -> Axis:
    """Return a supported Pauli axis from an internal generator recipe."""
    if value == "X":
        return "X"
    if value == "Y":
        return "Y"
    if value == "Z":
        return "Z"
    raise ValueError(f"invalid Pauli axis {value!r}")


def _decompose_generator(
    generator: tuple[object, ...], qubits: tuple[int, ...]
) -> tuple[Term, ...]:
    """Decompose one generator recipe into weighted unitary terms.

    The recipes come from ``_parameter_analysis`` with a built-in
    factor ``1/2`` on rotation generators.  Expanding every control
    projector ``|1><1| = (I - Z)/2`` turns the generator into a real
    weighted sum of Pauli products; each product becomes one
    Hadamard-test term whose factors are applied under the auxiliary
    control.  Verified numerically against parameter-shift gradients
    on every supported gate family.
    """
    kind = generator[0]
    if kind == "pauli":
        if len(generator) != 2 or generator[1] not in _AXIS_GATES:
            raise ValueError(f"invalid Pauli generator recipe {generator!r}")
        if len(qubits) != 1:
            raise ValueError("a Pauli generator must act on one qubit")
        pauli_axis = _validated_axis(generator[1])
        return ((0.5, ((pauli_axis, qubits[0]),)),)
    if kind == "pauli2":
        if len(generator) != 2 or not isinstance(generator[1], str):
            raise ValueError(f"invalid two-Pauli generator recipe {generator!r}")
        axes = generator[1]
        if len(axes) != len(qubits):
            raise ValueError(f"invalid two-Pauli generator recipe {generator!r}")
        validated_axes = tuple(_validated_axis(axis) for axis in axes)
        factors: tuple[tuple[Axis, int], ...] = tuple(
            (axis, position)
            for axis, position in zip(validated_axes, qubits, strict=True)
        )
        return ((0.5, factors),)
    if kind == "neg_projector":
        if len(qubits) != 1:
            raise ValueError("a projector generator must act on one qubit")
        return ((0.5, (("Z", qubits[0]),)),)
    if kind == "ctrl_pauli":
        if len(generator) != 2 or generator[1] not in _AXIS_GATES:
            raise ValueError(f"invalid controlled-Pauli generator recipe {generator!r}")
        if len(qubits) < 2:
            raise ValueError("a controlled-Pauli generator needs a control and target")
        controlled_axis = _validated_axis(generator[1])
        controls, target = qubits[:-1], qubits[-1]
        # Pi_controls P / 2 = sum_S (-1)^|S| Z_S P / 2^(k+1).
        denominator = float(2 ** (len(controls) + 1))
        controlled_terms: list[Term] = []
        for subset in _powerset(list(controls)):
            weight = (-1.0) ** len(subset) / denominator
            factors = tuple(("Z", position) for position in subset) + (
                (controlled_axis, target),
            )
            controlled_terms.append((weight, factors))
        return tuple(controlled_terms)
    if kind == "ctrl_neg_projector":
        if len(qubits) < 2:
            raise ValueError("a controlled projector needs a control and target")
        # -Pi_controls |1><1|_target = -(1/2^(k+1)) prod(I - Z); the
        # empty subset is the identity term and is dropped.
        all_positions = list(qubits)
        denominator = float(2 ** len(all_positions))
        projector_terms: list[Term] = []
        for subset in _powerset(all_positions):
            if not subset:
                continue
            weight = -((-1.0) ** len(subset)) / denominator
            factors = tuple(("Z", position) for position in subset)
            projector_terms.append((weight, factors))
        return tuple(projector_terms)
    raise ValueError(f"unknown generator kind {kind!r}")


def _extend_pauli_string(pauli_string: PauliString, count: int) -> PauliString:
    """Return the Pauli string with ``Y`` added on qubit ``count``.

    ``PauliString`` indices follow little-endian bit order (index 0 is
    qubit 0), matching the ``from_str`` label order where the last
    character acts on qubit 0.
    """
    extended = PauliString(count + 1)
    for position in range(pauli_string.num_qubits):
        extended.set_pauli(position, pauli_string.get_pauli(position))
    extended.set_pauli(count, Pauli.y())
    extended.phase = pauli_string.phase
    return extended


def _extend_observable(
    observable: Hamiltonian | PauliString, count: int
) -> Hamiltonian | PauliString:
    """Extend every observable term with ``Y`` on the auxiliary qubit."""
    if isinstance(observable, PauliString):
        return _extend_pauli_string(observable, count)
    extended = Hamiltonian(observable.num_qubits + 1)
    for term in observable.terms:
        pauli_string, coefficient = term
        extended.add_term(_extend_pauli_string(pauli_string, count), coefficient)
    return extended


def _insert_controlled_term(
    circuit: Circuit, factors: tuple[tuple[Axis, int], ...], aux: int
) -> None:
    """Append one unitary term as consecutive single-controlled gates.

    The factors of one Pauli product commute, so applying each under
    the same auxiliary control realizes the controlled product.
    """
    aux_qubit = circuit.qubits[aux]
    for axis, position in factors:
        circuit.append_mc_gate(
            _AXIS_GATES[axis].control(1),
            [aux_qubit, circuit.qubits[position]],
        )


def _bound_angle(value: float | Parameter) -> float:
    """Return one fully bound gate angle or reject an incomplete binding."""
    if isinstance(value, Parameter):
        raise TypeError("U gate parameters must be fully bound before evaluation")
    return float(value)


def _fresh_qubit(qubits: Sequence[Qubit]) -> Qubit:
    """Return a qubit whose identifier is unused by ``qubits``."""
    used = {qubit.index for qubit in qubits}
    index = 0
    while index in used:
        index += 1
    return Qubit(index)


def _expanded_stream(bound: Circuit) -> list[tuple[ValueOperation, int, int]]:
    """Expand the bound circuit into single-parameter stream factors.

    Mirrors the QGT expansion: ``U`` gates become
    ``[RZ(lambda), RY(theta), RZ(phi)]`` and every other operation
    passes through with its slot (or ``-1`` when non-parametric).
    Barriers are dropped.
    """
    stream: list[tuple[ValueOperation, int, int]] = []
    for index, operation in enumerate(bound.operations):
        if operation.instruction_type == "directive":
            continue
        if operation.instruction_type == "standard" and operation.name == "U":
            target = operation.qubits[0]
            theta = _bound_angle(operation.params[0])
            phi = _bound_angle(operation.params[1])
            lam = _bound_angle(operation.params[2])
            for slot, (gate_name, angle) in (
                (2, ("RZ", lam)),
                (0, ("RY", theta)),
                (1, ("RZ", phi)),
            ):
                helper = Circuit(list(bound.qubits))
                if gate_name == "RZ":
                    helper.rz(target, angle)
                else:
                    helper.ry(target, angle)
                stream.append((helper.operations[0], index, slot))
            continue
        slot = 0 if operation.params else -1
        stream.append((operation, index, slot))
    return stream


@final
class LinCombEstimatorGradient(Gradient):
    """Gradients via linear-combination Hadamard-test circuits.

    For every occurrence of a requested parameter the component expands
    the occurrence's generator into a real weighted sum of Pauli
    products and builds one auxiliary-qubit Hadamard-test circuit per
    product (Schuld et al., arXiv:1811.11184).  The derivative of
    parameter ``t`` entering a slot through ``angle = a * t + b`` is

    ``dE/dt = 2 * a * sum_k w_k * <O tensor Y_aux>_k``

    where ``w_k`` is the classical weight of product ``k`` and
    ``<O tensor Y_aux>_k`` is the expectation of the observable
    extended with ``Y`` on the fresh auxiliary qubit of that test
    circuit.  Global-phase parameters contribute zero without any
    estimator evaluation.

    All test circuits of all requests are fully bound and flattened
    into a single :meth:`Estimator.run` batch.  The batch contains one
    circuit per unitary term per non-zero occurrence: rotation and
    phase generators keep one term, while ``k``-controlled generators
    expand to ``2^k`` terms for rotations and ``2^(k+1) - 1`` terms for
    controlled phases, so callers control their batch volume through
    the request set.
    Requested parameter occurrences are supported in rotation and phase
    gates, their controlled variants, and ``U`` gates; parameters
    outside the requested subset may occur in other gates once bound.
    Circuit gates and non-unitary instructions are rejected with
    ``ValueError`` before any estimator call.

    Raises:
        TypeError: If the estimator or a request has the wrong type.
        ValueError: If a request is invalid, a gradient parameter is
            not exactly analyzable, or the estimator returns a
            mismatched result count.
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

        # Analyze first: all validation happens before any estimator call.
        analyzed: list[dict[str, tuple[Occurrence, ...]]] = []
        for request in request_list:
            request.validate()
            circuit = request.estimator_request.circuit
            validate_circuit(circuit)
            analyzed.append(
                {
                    name: analyze_symbol(circuit, name)
                    for name in request.parameter_names
                }
            )

        # One evaluation per unitary term of each occurrence, flattened
        # across requests into a single batch.  (owner, prefactor)
        # records the aggregation site per evaluation.
        evaluations: list[EstimatorRequest] = []
        owners: list[tuple[tuple[int, int], float]] = []
        for position, request in enumerate(request_list):
            base = request.estimator_request
            bound = base.circuit.assign_parameters(dict(base.parameter_bindings))
            num_qubits = bound.num_qubits
            auxiliary = _fresh_qubit(bound.qubits)
            extended = _extend_observable(base.observable, num_qubits)
            stream = _expanded_stream(bound)
            stream_by_slot: dict[tuple[int, int], int] = {
                (source, slot): stream_position
                for stream_position, (_, source, slot) in enumerate(stream)
                if slot >= 0
            }
            for name_index, name in enumerate(request.parameter_names):
                for occurrence in analyzed[position][name]:
                    if occurrence.zero:
                        continue
                    stream_position = stream_by_slot.get(
                        (occurrence.op_index, occurrence.slot)
                    )
                    if stream_position is None:
                        raise ValueError(
                            f"parameter occurrence not found in stream for {name!r}"
                        )
                    for weight, factors in _decompose_generator(
                        occurrence.generator, occurrence.qubits
                    ):
                        test_circuit = Circuit(list(bound.qubits) + [auxiliary])
                        test_circuit.ry(auxiliary, _HALF_PI)
                        for replay_index, (operation, _, _) in enumerate(stream):
                            if replay_index == stream_position:
                                _insert_controlled_term(
                                    test_circuit, factors, num_qubits
                                )
                            test_circuit.append(operation)
                        evaluations.append(
                            EstimatorRequest(test_circuit, extended, {}, base.shots)
                        )
                        owners.append(
                            (
                                (position, name_index),
                                2.0 * occurrence.coefficient * weight,
                            )
                        )

        results = list(estimator.run(evaluations)) if evaluations else []
        if len(results) != len(evaluations):
            raise ValueError(
                "estimator must return exactly one result per evaluation, "
                f"expected {len(evaluations)}, got {len(results)}"
            )

        derivatives: dict[tuple[int, int], float] = {}
        for (key, prefactor), result in zip(owners, results):
            derivatives[key] = (
                derivatives.get(key, 0.0) + prefactor * result.expectation
            )

        response: list[GradientResult] = []
        for position, request in enumerate(request_list):
            names = request.parameter_names
            if not names:
                response.append(GradientResult((), ()))
                continue
            values = tuple(
                derivatives.get((position, name_index), 0.0)
                for name_index in range(len(names))
            )
            response.append(GradientResult(names, values))
        return response
