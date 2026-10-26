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

r"""Circuit-rebuild primitives shared by error-mitigation techniques.

Several techniques transform a circuit rather than only its parameters:
folding, identity insertion, and layer splitting all rebuild the operation
list. This module provides the primitives they share: classifying and
stripping measurements, and rebuilding, expanding, or splicing circuits while
preserving the global phase that ``Circuit.from_operations`` does not carry
over.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

from cqlib import Circuit
from cqlib.circuit import Qubit, ValueOperation

_MEASUREMENT_NAMES = frozenset({"measure_bit", "measure_bits"})


def is_measurement(operation: ValueOperation, /) -> bool:
    r"""Return whether ``operation`` measures a qubit.

    Parameters
    ----------
    operation
        Top-level operation to classify.

    Returns
    -------
    bool
        ``True`` for either measurement shape, ``False`` otherwise.
    """
    # A measurement reaches the operation list in two shapes: the
    # ``Directive::Measure`` directive, and the ``MeasureBit`` / ``MeasureBits``
    # records that ``Circuit.measure`` emits. Missing either shape lets a
    # measurement slip into a fold, which ``Circuit.inverse`` then rejects.
    if operation.is_directive:
        directive = operation.instruction.directive
        return directive is not None and directive.is_measure()
    if operation.is_classical_data:
        return operation.name in _MEASUREMENT_NAMES
    return False


def strip_measurements(circuit: Circuit, /) -> tuple[Circuit, list[Qubit]]:
    r"""Remove every measurement, returning the gate body and measured qubits.

    Parameters
    ----------
    circuit
        Circuit to strip.

    Returns
    -------
    tuple[Circuit, list[Qubit]]
        The measurement-free body (sharing the source global phase) and the
        measured qubits in measurement order, ready to be re-measured at the
        tail of the folded circuit via :meth:`Circuit.measure`. The source
        circuit is returned unchanged, with an empty qubit list, when it
        carries no measurement.
    """
    operations = list(circuit.operations)
    measured: list[Qubit] = []
    body: list[ValueOperation] = []
    for operation in operations:
        if is_measurement(operation):
            measured.extend(operation.qubits)
        else:
            body.append(operation)
    if not measured:
        return circuit, []
    # A fold such as U-dagger U is unitary only on the gate body, and
    # ``Circuit.inverse`` raises on a measurement inside it. The classical
    # value table goes with the measurements, because the values they defined
    # no longer exist and ``Circuit.validate`` rejects entries of
    # ``classical_values`` that no operation defines.
    stripped = Circuit.from_operations(
        list(circuit.qubits),
        body,
        list(circuit.classical_vars),
        [],
    )
    stripped.set_global_phase(circuit.global_phase)
    return stripped, measured


def rebuild(circuit: Circuit, operations: Sequence[ValueOperation], /) -> Circuit:
    r"""Rebuild ``circuit`` around a replacement top-level operation list.

    All five state components of the source circuit are preserved: qubits,
    classical variables, classical values, the global phase, and the qubit
    identifiers referenced by operations.

    Parameters
    ----------
    circuit
        Circuit supplying qubits, classical state, and global phase.
    operations
        Replacement top-level operation list, in execution order.

    Returns
    -------
    Circuit
        A new circuit carrying ``operations`` plus the full state of
        ``circuit``, including ``global_phase``.
    """
    rebuilt = Circuit.from_operations(
        list(circuit.qubits),
        list[ValueOperation](operations),
        list(circuit.classical_vars),
        list(circuit.classical_values),
    )
    rebuilt.set_global_phase(circuit.global_phase)
    return rebuilt


def expand(
    circuit: Circuit,
    expander: Callable[[ValueOperation], Iterable[ValueOperation]],
    /,
) -> Circuit:
    r"""Replace every operation by a sequence and rebuild the circuit.

    The rewrite is sequential and order-preserving: the operations yielded by
    ``expander(operation)`` take the place of ``operation`` in place. This is
    the natural shape for "append a correction immediately after each gate"
    rewrites, where the position of the rewritten gate need not be known.

    Parameters
    ----------
    circuit
        Circuit whose operations are expanded.
    expander
        Pure callable mapping one operation to its replacement sequence.

    Returns
    -------
    Circuit
        A new circuit with the expanded operation list.
    """
    operations: list[ValueOperation] = []
    for operation in circuit.operations:
        operations.extend(expander(operation))
    return rebuild(circuit, operations)


def splice(
    circuit: Circuit,
    insertions: Mapping[int, Sequence[ValueOperation]],
    /,
) -> Circuit:
    r"""Insert operations after selected indices of the original circuit.

    Keys of ``insertions`` index ``circuit.operations`` as it stood before the
    rebuild; the mapped operations are appended immediately after the
    operation at that index. This is the shape used by position-addressed
    rewrites such as idle-window insertions and measurement-basis rotations.

    Parameters
    ----------
    circuit
        Circuit receiving the insertions.
    insertions
        Map from original operation index to the operations inserted after it.

    Returns
    -------
    Circuit
        A new circuit with the inserted operations in place.
    """
    operations: list[ValueOperation] = []
    for index, operation in enumerate(circuit.operations):
        operations.append(operation)
        operations.extend(insertions.get(index, ()))
    return rebuild(circuit, operations)
