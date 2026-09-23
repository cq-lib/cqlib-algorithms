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
to a state amplitude array.  It also derives exact local parameter-shift
evaluation points and weights for expectation-value gradients.

A parametric gate ``G(angle)`` is written as a spectral sum
``G(angle) = sum(exp(-i * lambda * angle) * P_lambda)`` over real branch
eigenvalues ``lambda``.  For a symbol ``t`` entering a slot through an
affine expression ``angle = a * t + b`` the branch phase becomes
``exp(-i * (a * lambda) * t)``, so the occurrence contributes the scaled
spectrum ``{a * lambda}``.  Parameter-shift differentiates one gate slot at
a time and adds its contribution according to the multivariate chain rule;
this keeps the evaluation count linear when one symbol is shared by many
gates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TypeAlias

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
            if not math.isfinite(coefficient):
                raise ValueError(
                    "parameter expressions must have finite affine coefficients, "
                    f"found {coefficient!r} for symbol {symbol!r} at operation "
                    f"{index}, slot {slot}"
                )
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


def local_shift_plan(
    occurrence: Occurrence,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return an exact rule for one gate-parameter occurrence.

    Shifts are expressed in the gate slot's angle, not in the shared
    circuit symbol.  The caller therefore multiplies the returned weights
    by ``occurrence.coefficient`` to apply the affine chain rule.  Every
    currently supported two-eigenvalue slot uses the ordinary two-point
    rule.  Controlled rotations have the two positive local frequencies
    ``1/2`` and ``1`` and use the corresponding four-point rule.
    """
    if occurrence.zero:
        return (), ()
    if occurrence.coefficient == 0.0:
        return (), ()

    spectrum = tuple(value / occurrence.coefficient for value in occurrence.spectrum)
    frequencies = sorted(
        {
            abs(first - second)
            for first in spectrum
            for second in spectrum
            if first != second
        }
    )
    if len(frequencies) == 1 and math.isclose(
        frequencies[0], 1.0, rel_tol=1e-12, abs_tol=1e-12
    ):
        return (
            (math.pi / 2.0, -math.pi / 2.0),
            (0.5, -0.5),
        )
    if (
        len(frequencies) == 2
        and math.isclose(frequencies[0], 0.5, rel_tol=1e-12, abs_tol=1e-12)
        and math.isclose(frequencies[1], 1.0, rel_tol=1e-12, abs_tol=1e-12)
    ):
        # Solve 2 * sum_j a_j sin(nu * s_j) = nu for
        # s = (pi/2, pi).  The paired weights are (+a_j, -a_j).
        first = 0.5
        second = 0.25 - math.sqrt(2.0) / 4.0
        return (
            (math.pi / 2.0, -math.pi / 2.0, math.pi, -math.pi),
            (first, -first, second, -second),
        )
    raise ValueError(
        "gate parameter occurrence has an unsupported local frequency set "
        f"{tuple(frequencies)!r}"
    )
