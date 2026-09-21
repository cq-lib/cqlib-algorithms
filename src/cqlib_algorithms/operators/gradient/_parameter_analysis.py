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

"""Private circuit analysis for gradient and QGT components.

This module walks a Core circuit's instruction list and, for one symbol,
records every parameter-slot occurrence together with the signed spectrum
of the gate's effective generator and a recipe for applying that generator
to a state amplitude array.  It also derives exact parameter-shift
evaluation points and weights for expectation-value gradients.

A parametric gate ``G(angle)`` is written as a spectral sum
``G(angle) = sum(exp(-i * lambda * angle) * P_lambda)`` over real branch
eigenvalues ``lambda``.  For a symbol ``t`` entering a slot through an
affine expression ``angle = a * t + b`` the branch phase becomes
``exp(-i * (a * lambda) * t)``, so the occurrence contributes the scaled
spectrum ``{a * lambda}``.  An expectation value that depends on ``t``
through occurrences ``k`` is a trigonometric polynomial whose frequencies
are differences of sums of one scaled eigenvalue per occurrence, and the
gradient is an exact linear combination of shifted evaluations whose
weights solve ``sum_j w_j * exp(i * nu * s_j) == i * nu`` for every
frequency ``nu``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from cqlib.circuit import Circuit, Parameter

_HALF = (0.5, -0.5)
_DIAG01 = (0.0, -1.0)
_CTRL_HALF = (0.0, 0.5, -0.5)

# One slot's spectrum and generator recipe.  Generators are expressed as
# one of:
#   ("pauli", axis)                axis in "X"/"Y"/"Z", factor 1/2
#   ("pauli2", axes)               two-qubit products like "ZX"
#   ("neg_projector",)             -|1><1| on the slot's qubit
#   ("ctrl_pauli", axis)           controlled Pauli with factor 1/2
#   ("ctrl_neg_projector",)        controlled -|1><1|
# Every rotation-type generator carries the overall factor 1/2.
SlotSpec: TypeAlias = tuple[tuple[float, ...], tuple[Any, ...] | None]
GateSpec: TypeAlias = tuple[SlotSpec, ...]

_SLOT_SPECS: dict[str, GateSpec] = {
    "RX": ((_HALF, ("pauli", "X")),),
    "RY": ((_HALF, ("pauli", "Y")),),
    "RZ": ((_HALF, ("pauli", "Z")),),
    "RXX": ((_HALF, ("pauli2", "XX")),),
    "RYY": ((_HALF, ("pauli2", "YY")),),
    "RZZ": ((_HALF, ("pauli2", "ZZ")),),
    "RZX": ((_HALF, ("pauli2", "ZX")),),
    "Phase": ((_DIAG01, ("neg_projector",)),),
    "CRX": ((_CTRL_HALF, ("ctrl_pauli", "X")),),
    "CRY": ((_CTRL_HALF, ("ctrl_pauli", "Y")),),
    "CRZ": ((_CTRL_HALF, ("ctrl_pauli", "Z")),),
    "U": (
        (_HALF, ("pauli", "Y")),
        (_DIAG01, ("neg_projector",)),
        (_DIAG01, ("neg_projector",)),
    ),
    # GPhase contributes a parameter-dependent global phase only, which
    # cancels from expectation values and projected QGT inner products.
    "GPhase": (((), None),),
}

_MC_ROTATION_BASES = ("RX", "RY", "RZ")

_UNSUPPORTED_GATES = ("XY", "XY2P", "XY2M", "RXY", "FSIM")

_WEIGHT_TOLERANCE = 1e-10


@dataclass(frozen=True, slots=True)
class Occurrence:
    """One parameter-slot use of a symbol inside a circuit operation.

    ``coefficient`` is the constant ``d(angle)/d(symbol)`` of the affine
    slot expression.  ``spectrum`` holds the coefficient-scaled branch
    eigenvalues.  ``generator`` is the recipe for the scaled effective
    generator.  ``zero`` marks occurrences that cannot influence
    expectation values or QGT inner products (global-phase slots).
    """

    op_index: int
    slot: int
    qubits: tuple[int, ...]
    coefficient: float
    spectrum: tuple[float, ...]
    generator: tuple[Any, ...]
    zero: bool = False


def _slot_expression_symbols(expression: object) -> list[str]:
    """Return the symbols of a parameter slot expression.

    Fully bound numeric slots may expose plain floats instead of
    ``Parameter`` objects; they contain no symbols.
    """
    symbols = getattr(expression, "symbols", None)
    return list(symbols) if symbols is not None else []


def _parse_mcgate_name(name: str) -> tuple[int, str]:
    """Split an mcgate operation name such as ``"C2-RZ"``."""
    control_part, base = name.split("-", 1)
    return int(control_part[1:]), base


def _slot_specs_for_operation(name: str, instruction_type: str) -> GateSpec | None:
    """Return the supported parameter slots of one operation."""
    if instruction_type == "mcgate":
        _, base = _parse_mcgate_name(name)
        if base in _MC_ROTATION_BASES:
            return ((_CTRL_HALF, ("ctrl_pauli", base[1])),)
        if base == "Phase":
            return ((_DIAG01, ("ctrl_neg_projector",)),)
        return None
    if instruction_type == "standard":
        return _SLOT_SPECS.get(name)
    return None


def validate_circuit(circuit: Circuit) -> None:
    """Validate circuit-wide requirements shared by gradient components.

    The validation is independent of the requested gradient symbols.  This
    matters for empty ``parameter_names`` requests, which must not turn a
    non-unitary circuit or an unsupported structural instruction into an
    accepted no-op.  Parameterized-gate support remains symbol-specific and
    is checked by :func:`analyze_symbol`, so parameters outside a requested
    differentiation subset do not make an otherwise valid request fail.
    """
    for index, operation in enumerate(circuit.operations):
        name = operation.name
        instruction_type = operation.instruction_type

        if instruction_type == "directive":
            if name.lower() != "barrier":
                raise ValueError(
                    "circuit must not contain non-unitary directives, "
                    f"found {name!r} at operation {index}"
                )
            continue
        if instruction_type in ("classical_data", "classical_control", "delay"):
            raise ValueError(
                "circuit must be a static unitary circuit, "
                f"found {name!r} at operation {index}"
            )
        if instruction_type in ("circuit", "circuit_gate"):
            raise ValueError(
                "circuit gates are not supported by the gradient "
                f"components, found {name!r} at operation {index}"
            )


def analyze_symbol(circuit: Circuit, symbol: str) -> tuple[Occurrence, ...]:
    """Return all gradient-relevant occurrences of ``symbol``.

    The walk rejects, with ``ValueError`` naming the offending operation,
    any instruction that cannot be analyzed exactly: non-affine parameter
    expressions, unsupported parametric standard gates, symbolic unitary
    gates, circuit gates, and non-unitary or classical instructions.
    Barriers are skipped.  Global-phase dependence (``GPhase`` gates and
    the circuit global phase) cannot change expectation values or the
    projected QGT inner products and therefore yields no occurrences.

    Raises:
        TypeError: If ``symbol`` is not a string.
        ValueError: If an operation containing the symbol is not exactly
            analyzable, or the circuit contains non-unitary instructions.
    """
    if not isinstance(symbol, str):
        raise TypeError("symbol must be a string")
    validate_circuit(circuit)

    # Qubit positions follow the circuit's insertion order, matching the
    # Estimator contract, so non-contiguous qubit identifiers stay valid.
    positions = {qubit: position for position, qubit in enumerate(circuit.qubits)}

    occurrences: list[Occurrence] = []
    for index, operation in enumerate(circuit.operations):
        name = operation.name
        itype = operation.instruction_type

        if itype == "directive":
            continue

        slot_specs = _slot_specs_for_operation(name, itype)

        params = operation.params
        qubits = tuple(positions[qubit] for qubit in operation.qubits)

        used_slots: list[tuple[int, SlotSpec]] = []
        for slot, expression in enumerate(params):
            if symbol not in _slot_expression_symbols(expression):
                continue
            if slot_specs is None:
                raise ValueError(
                    "gate is not supported by the gradient components: "
                    f"{name!r} (symbol {symbol!r} at operation {index})"
                )
            spec = slot_specs[slot] if slot < len(slot_specs) else None
            if spec is None:
                raise ValueError(
                    "gate is not supported by the gradient components: "
                    f"{name!r} slot {slot} (symbol {symbol!r} at "
                    f"operation {index})"
                )
            used_slots.append((slot, spec))

        for slot, (spectrum, generator) in used_slots:
            expression = params[slot]
            if not isinstance(expression, Parameter):
                raise TypeError(
                    "symbolic parameter slots must contain Parameter values"
                )
            derivative = expression.derivative(symbol)
            if not derivative.is_constant():
                raise ValueError(
                    "parameter expressions must be affine in the gradient "
                    f"symbol {symbol!r} at operation {index}, slot {slot}"
                )
            coefficient = float(derivative.evaluate())
            if coefficient == 0.0:
                continue
            if not spectrum:
                occurrences.append(
                    Occurrence(index, slot, qubits, coefficient, (), (), True)
                )
                continue
            if generator is None:
                raise ValueError(
                    "gate is not supported by the gradient components: "
                    f"{name!r} (symbol {symbol!r} at operation {index})"
                )
            scaled = tuple(coefficient * value for value in spectrum)
            occurrences.append(
                Occurrence(index, slot, qubits, coefficient, scaled, generator)
            )

    return tuple(occurrences)


def frequencies(occurrences: Sequence[Occurrence]) -> tuple[float, ...]:
    """Return the distinct expectation frequencies of the symbol.

    The result is the sorted tuple of differences of sums of one scaled
    eigenvalue per non-zero occurrence.  Only ``{0.0}`` means the
    expectation is symbol-independent.
    """
    sums: set[float] = {0.0}
    for occurrence in occurrences:
        if occurrence.zero:
            continue
        sums = {
            value + eigenvalue for value in sums for eigenvalue in occurrence.spectrum
        }
    return tuple(sorted({first - second for first in sums for second in sums}))


def _solve_weights(
    freqs: Sequence[float], shifts: Sequence[float]
) -> tuple[np.ndarray, float]:
    cosines = np.array([[math.cos(w * s) for s in shifts] for w in freqs])
    sines = np.array([[math.sin(w * s) for s in shifts] for w in freqs])
    real_system = np.vstack((cosines, sines))
    real_target = np.concatenate(
        (np.zeros(len(freqs), dtype=np.float64), np.asarray(freqs, dtype=np.float64))
    )
    weights = np.linalg.lstsq(real_system, real_target, rcond=None)[0]
    complex_system = cosines + 1j * sines
    complex_target = 1j * np.asarray(freqs, dtype=np.float64)
    residual = (
        float(np.max(np.abs(complex_system @ weights - complex_target)))
        if freqs
        else 0.0
    )
    return weights, residual


_SHIFT_CACHE: dict[
    tuple[float, ...], tuple[tuple[float, ...], tuple[complex, ...]]
] = {}


def shift_plan(
    occurrences: Sequence[Occurrence],
) -> tuple[tuple[float, ...], tuple[complex, ...]]:
    """Return exact shift points and weights for one symbol's gradient.

    The plan satisfies ``sum_j w_j * exp(i * nu * s_j) == i * nu`` for
    every expectation frequency of the symbol, so
    ``dE/dt = sum_j w_j * E(t + s_j)`` holds exactly for any estimator
    of the expectation.  Candidate shift sets are tried in order of
    increasing evaluation count, starting with the classical two-point
    rules; the first set with a numerically consistent solution is
    accepted and cached per frequency set.
    """
    freqs = frequencies(occurrences)
    if not any(freqs):
        return (), ()
    cached = _SHIFT_CACHE.get(freqs)
    if cached is not None:
        return cached

    candidates: list[tuple[float, ...]] = [
        (math.pi / 2, -math.pi / 2),
        (0.0, math.pi),
    ]
    for count in range(3, len(freqs) + 5):
        candidates.append(tuple(2.0 * math.pi * j / count for j in range(count)))

    for shifts in candidates:
        weights, residual = _solve_weights(freqs, shifts)
        if residual <= _WEIGHT_TOLERANCE:
            plan = (shifts, tuple(complex(weight) for weight in weights))
            _SHIFT_CACHE[freqs] = plan
            return plan
    raise ValueError(
        f"failed to derive an exact parameter-shift rule for frequency set {freqs!r}"
    )
