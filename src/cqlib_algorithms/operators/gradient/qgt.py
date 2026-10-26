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

"""Exact quantum geometric tensor from generator insertion.

References
----------
[1] J. Stokes, J. Izaac, N. Killoran, and G. Carleo, "Quantum natural
    gradient," Quantum, 4, 269, 2020.
    DOI: 10.22331/q-2020-05-25-269.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, final

import numpy as np
from cqlib.circuit import Circuit, Qubit, ValueOperation
from cqlib.qis import Statevector

from ...protocols.gradient import GradientRequest
from ._parameter_analysis import Occurrence, analyze_symbol, validate_circuit

_HERMITIAN_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class QGTResult:
    """The quantum geometric tensor and its real metric of one request.

    ``tensor`` is the complex Hermitian matrix
    ``F_ij = <d_i psi | (1 - |psi><psi|) | d_j psi>`` and ``metric`` is
    its real symmetric part ``g_ij = Re F_ij`` (Stokes et al.,
    arXiv:1909.02108).  Rows and columns follow ``parameter_names``.
    ``tensor`` and ``metric`` accept array-like values; validated inputs are
    copied into complex and real NumPy arrays, respectively.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If names are duplicated, shapes disagree, or the tensor
            is not Hermitian, the metric is not symmetric, or the metric is
            not the real part of the tensor (all within 1e-9).
    """

    parameter_names: tuple[str, ...]
    tensor: np.ndarray
    metric: np.ndarray

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.parameter_names, tuple):
            raise TypeError("parameter_names must be a tuple")
        if any(not isinstance(name, str) for name in self.parameter_names):
            raise TypeError("parameter names must be strings")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("parameter names must be unique")

        count = len(self.parameter_names)
        tensor = np.array(self.tensor, dtype=np.complex128, copy=True)
        metric_input = np.asarray(self.metric)
        if np.iscomplexobj(metric_input):
            if np.any(metric_input.imag != 0.0):
                raise ValueError("metric must be real-valued")
            metric_input = metric_input.real
        metric = np.array(metric_input, dtype=np.float64, copy=True)
        if tensor.shape != (count, count):
            raise ValueError("tensor must be a square matrix matching the names")
        if metric.shape != (count, count):
            raise ValueError("metric must be a square matrix matching the names")
        if not np.all(np.isfinite(tensor)) or not np.all(np.isfinite(metric)):
            raise ValueError("tensor and metric must be finite")
        if not np.allclose(
            tensor,
            tensor.conj().T,
            atol=_HERMITIAN_TOLERANCE,
            rtol=0.0,
        ):
            raise ValueError("tensor must be Hermitian")
        if not np.allclose(
            metric,
            metric.T,
            atol=_HERMITIAN_TOLERANCE,
            rtol=0.0,
        ):
            raise ValueError("metric must be symmetric")
        if not np.allclose(
            metric,
            tensor.real,
            atol=_HERMITIAN_TOLERANCE,
            rtol=0.0,
        ):
            raise ValueError("metric must be the real part of the tensor")
        object.__setattr__(self, "tensor", tensor)
        object.__setattr__(self, "metric", metric)


def _apply_axis(
    amplitudes: np.ndarray, axis: str, qubit: int, half: bool
) -> np.ndarray:
    """Apply a single-qubit Pauli (optionally halved) to amplitudes."""
    indices = np.arange(amplitudes.size)
    mask = 1 << qubit
    if axis == "X":
        result = amplitudes[indices ^ mask]
        if half:
            result = result * 0.5
    elif axis == "Z":
        signs = np.where(indices & mask, -1.0, 1.0)
        result = amplitudes * signs
        if half:
            result = result * 0.5
    else:  # Y = [[0, -i], [i, 0]]
        swapped = amplitudes[indices ^ mask]
        bit = np.where(indices & mask, 1.0, -1.0)
        factor = 0.5j * bit if half else 1j * bit
        result = swapped * factor
    return np.asarray(result, dtype=np.complex128)


def _projector_one(amplitudes: np.ndarray, qubits: tuple[int, ...]) -> np.ndarray:
    """Return the component with every listed qubit in state ``1``."""
    indices = np.arange(amplitudes.size)
    keep = np.ones(amplitudes.size, dtype=bool)
    for qubit in qubits:
        keep &= (indices & (1 << qubit)) != 0
    result = np.where(keep, amplitudes, 0.0 + 0.0j)
    return result


def apply_generator(
    generator: tuple[Any, ...], qubits: tuple[int, ...], amplitudes: np.ndarray
) -> np.ndarray:
    """Apply one occurrence's effective generator to amplitudes.

    The generator recipes come from ``_parameter_analysis`` and cover
    Pauli generators of one- and two-qubit rotations, diagonal phase
    generators, and controlled variants.  The caller scales the result
    by the occurrence's affine coefficient.
    """
    kind = generator[0]
    if kind == "pauli":
        _, axis = generator
        return _apply_axis(amplitudes, axis, qubits[0], half=True)
    if kind == "pauli2":
        _, axes = generator
        first, second = qubits
        result = amplitudes
        # For "ZX" the leading Z acts on the first listed qubit and the
        # trailing X on the second; each factor keeps its own half.
        result = _apply_axis(result, axes[1], second, half=True)
        result = _apply_axis(result, axes[0], first, half=False)
        return result
    if kind == "neg_projector":
        projected = _projector_one(amplitudes, (qubits[0],))
        return -projected
    if kind == "ctrl_pauli":
        _, axis = generator
        controls, target = qubits[:-1], qubits[-1]
        projected = _projector_one(amplitudes, controls)
        return _apply_axis(projected, axis, target, half=True)
    if kind == "ctrl_neg_projector":
        projected = _projector_one(amplitudes, qubits)
        return -projected
    raise ValueError(f"unknown generator kind {kind!r}")


def _numeric_value(expression: object) -> float:
    """Return the numeric value of a bound parameter expression.

    Bound ``U`` gates may expose plain floats instead of ``Parameter``
    objects for constant slots.
    """
    if isinstance(expression, (int, float)):
        return float(expression)
    evaluate = getattr(expression, "evaluate", None)
    if evaluate is None:
        raise TypeError("expression must be a number or Parameter")
    return float(evaluate())


def _expanded_stream(bound: Circuit) -> list[tuple[ValueOperation, int, int]]:
    """Expand the bound circuit into single-parameter stream factors.

    Each entry is ``(operation, source_op_index, slot)`` where ``slot``
    identifies the parameter slot the factor realizes.  ``U`` gates are
    expanded into ``[RZ(lambda), RY(theta), RZ(phi)]`` following
    ``U(theta, phi, lambda) = exp(i(phi+lambda)/2) RZ(phi) RY(theta)
    RZ(lambda)``; the global-phase factor cannot influence the projected
    QGT inner products and is dropped.  Every other unitary operation is
    passed through unchanged with ``slot=-1``.  Barriers are dropped.

    This expansion is required for exactness: the generators of a
    ``U`` gate's phi and lambda slots do not commute with the full gate,
    so generator insertion is only position-exact against the expanded
    single-parameter factors.
    """
    qubits = list(bound.qubits)
    stream: list[tuple[ValueOperation, int, int]] = []
    for index, operation in enumerate(bound.operations):
        if operation.instruction_type == "directive":
            continue
        if operation.instruction_type == "standard" and operation.name == "U":
            target = operation.qubits[0]
            theta = _numeric_value(operation.params[0])
            phi = _numeric_value(operation.params[1])
            lam = _numeric_value(operation.params[2])
            for slot, (gate_name, angle) in (
                (2, ("RZ", lam)),
                (0, ("RY", theta)),
                (1, ("RZ", phi)),
            ):
                helper = Circuit(qubits)
                if gate_name == "RZ":
                    helper.rz(target, angle)
                else:
                    helper.ry(target, angle)
                stream.append((helper.operations[0], index, slot))
            continue
        # Every supported parametric operation other than ``U`` is
        # single-parameter, so its pass-through entry realizes slot 0.
        slot = 0 if operation.params else -1
        stream.append((operation, index, slot))
    return stream


def _suffix_circuits(
    qubits: Sequence[Qubit],
    stream: Sequence[tuple[ValueOperation, int, int]],
) -> list[Circuit]:
    """Build suffix circuits for every position of the expanded stream."""
    suffixes: list[Circuit] = []
    for start in range(len(stream) + 1):
        suffix = Circuit(list(qubits))
        for operation, _, _ in stream[start:]:
            suffix.append(operation)
        suffixes.append(suffix)
    return suffixes


@final
class StatevectorQGT:
    """Exact quantum geometric tensor via generator insertion.

    ``compute`` evaluates
    ``F_ij = <d_i psi | (1 - |psi><psi|) | d_j psi>`` exactly on
    statevectors.  Each derivative state is assembled by inserting the
    scaled generator of every occurrence of one parameter directly into
    the bound circuit's instruction stream; no finite differences or
    dense unitaries are used.  Global phases (circuit ``set_global_phase``
    and ``GPhase`` gates) cannot influence ``F`` because their derivative
    contribution is parallel to ``|psi>`` and is projected out; this also
    neutralizes the known Core gap where ``Statevector.from_circuit``
    ignores the circuit global phase.

    Requested parameter occurrences are supported in rotation and phase
    gates, their controlled variants, and ``U`` gates.  Parameters outside
    the requested subset may occur in other gates once bound.  Barriers are
    skipped; measurements, resets, delays, classical control and circuit
    gates are rejected with ``ValueError`` regardless of the requested
    subset.

    Raises:
        TypeError: If the request has the wrong type.
        ValueError: If the request is invalid or the circuit contains
            unsupported instructions.
    """

    __slots__ = ()

    def compute(self, request: GradientRequest) -> QGTResult:
        """Return the QGT of one estimator objective's parameters."""
        if not isinstance(request, GradientRequest):
            raise TypeError("request must be a GradientRequest")
        request.validate()

        estimator_request = request.estimator_request
        circuit = estimator_request.circuit
        names = request.parameter_names
        validate_circuit(circuit)
        if not names:
            empty = np.zeros((0, 0), dtype=np.complex128)
            return QGTResult((), empty, empty.real)
        occurrences = {name: analyze_symbol(circuit, name) for name in names}

        bound = circuit.assign_parameters(dict(estimator_request.parameter_bindings))
        num_qubits = bound.num_qubits
        dimension = 1 << num_qubits

        psi = Statevector.from_circuit(bound).data
        derivative_states = {
            name: np.zeros(dimension, dtype=np.complex128) for name in names
        }

        stream = _expanded_stream(bound)
        suffixes = _suffix_circuits(list(bound.qubits), stream)
        by_slot: dict[tuple[int, int], list[tuple[str, Occurrence]]] = {}
        for name in names:
            for occurrence in occurrences[name]:
                if occurrence.zero:
                    continue
                by_slot.setdefault((occurrence.op_index, occurrence.slot), []).append(
                    (name, occurrence)
                )

        current = Statevector.from_state(
            num_qubits, [1.0 + 0.0j] + [0.0 + 0.0j] * (dimension - 1)
        )
        for position, (operation, source, slot) in enumerate(stream):
            one_step = Circuit(list(bound.qubits))
            one_step.append(operation)
            current.apply_circuit(one_step)

            entries = by_slot.get((source, slot), ())
            if not entries:
                continue
            for name, occurrence in entries:
                generated = apply_generator(
                    occurrence.generator,
                    occurrence.qubits,
                    current.data,
                )
                generated = generated * (-1j * occurrence.coefficient)
                # ``from_state`` requires a normalized vector, while
                # generator images generally are not.  Normalize for
                # construction and undo the scale after the unitary
                # suffix evolution, which preserves the norm.
                norm = float(np.linalg.norm(generated))
                if norm == 0.0:
                    continue
                evolved = Statevector.from_state(
                    num_qubits, (generated / norm).tolist()
                )
                evolved.apply_circuit(suffixes[position + 1])
                derivative_states[name] += evolved.data * norm

        count = len(names)
        tensor = np.zeros((count, count), dtype=np.complex128)
        for row, first in enumerate(names):
            d_first = derivative_states[first]
            for column, second in enumerate(names):
                d_second = derivative_states[second]
                overlap = np.vdot(d_first, d_second)
                bra = np.vdot(d_first, psi)
                ket = np.vdot(psi, d_second)
                tensor[row, column] = overlap - bra * ket

        return QGTResult(
            tuple(names), tensor, np.real(tensor).astype(np.float64, copy=False)
        )
