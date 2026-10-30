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

"""Shared fixed-reference composition for ordered evolution ansatze."""

import copy

from cqlib.circuit import Circuit, CircuitError

from .base_ansatz import _InnerAnsatz


def validate_fixed_reference_evolution(
    initial_circuit: Circuit | None,
    evolution: _InnerAnsatz,
) -> None:
    """Validate an evolution component and its optional fixed reference."""
    evolution.validate()
    if initial_circuit is None:
        return
    initial_circuit.validate()
    if initial_circuit.num_qubits != evolution.num_qubits():
        raise ValueError("initial_state snapshot has an inconsistent width")
    if initial_circuit.used_symbols:
        raise ValueError("initial_state snapshot must not contain active symbols")
    try:
        initial_circuit.inverse()
    except CircuitError as error:
        raise ValueError(
            "initial_state snapshot must contain only statically invertible operations"
        ) from error


def build_fixed_reference_evolution(
    initial_circuit: Circuit | None,
    evolution: _InnerAnsatz,
    *,
    parameter_prefix: str,
) -> Circuit:
    """Return a reference snapshot followed by the configured evolution."""
    circuit = (
        Circuit(evolution.num_qubits())
        if initial_circuit is None
        else copy.copy(initial_circuit)
    )
    evolved = evolution.build_circuit(parameter_prefix)
    circuit.compose(evolved, circuit.qubits)
    return circuit
