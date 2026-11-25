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

"""Build preparation and ansatz components and snapshot fixed initial states."""

from __future__ import annotations

from cqlib.circuit import Circuit, CircuitError

from ..protocols.ansatz import Ansatz
from ..protocols.state_preparation import StatePreparation
from .numeric import validate_nonnegative_integer, validate_positive_integer


def build_state_preparation_circuit(
    state_preparation: StatePreparation, *, prefix: str
) -> Circuit:
    """Build a preparation and verify its declared width and parameter slots."""
    if not isinstance(state_preparation, StatePreparation):
        raise TypeError("state_preparation must be a StatePreparation")
    return build_component_circuit(state_preparation, prefix=prefix)


def build_component_circuit(
    component: StatePreparation | Ansatz, *, prefix: str
) -> Circuit:
    """Build a preparation or ansatz and verify its width and parameter slots."""
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a string")
    if not isinstance(component, (StatePreparation, Ansatz)):
        raise TypeError("component must be a StatePreparation or Ansatz")
    role = "state_preparation" if isinstance(component, StatePreparation) else "ansatz"
    component.validate()
    num_qubits = component.num_qubits()
    num_parameters = component.num_parameters()
    validate_positive_integer(num_qubits, f"{role} num_qubits")
    validate_nonnegative_integer(num_parameters, name=f"{role} num_parameters")
    circuit = component.build_circuit(prefix)
    if not isinstance(circuit, Circuit):
        raise TypeError(f"{role} build_circuit must return a Circuit")
    if circuit.num_qubits != num_qubits:
        raise ValueError(f"built circuit qubit count must equal {role} num_qubits")
    if len(circuit.used_symbols) > num_parameters:
        raise ValueError(
            "built circuit active symbol count must not exceed num_parameters"
        )
    return circuit


def snapshot_fixed_initial_state(
    initial_state: StatePreparation | None,
    *,
    num_qubits: int,
    prefix: str,
) -> Circuit | None:
    """Validate and snapshot an optional zero-parameter preparation."""
    if initial_state is None:
        return None
    if not isinstance(initial_state, StatePreparation):
        raise TypeError("initial_state must be a StatePreparation or None")
    initial_state.validate()
    if initial_state.num_parameters() != 0:
        raise ValueError("initial_state must have zero logical parameters")
    if initial_state.num_qubits() != num_qubits:
        raise ValueError("initial_state and ansatz must have the same width")

    built = initial_state.build_circuit(prefix)
    if not isinstance(built, Circuit):
        raise TypeError("initial_state must build a Circuit")
    built.validate()
    if built.num_qubits != num_qubits:
        raise ValueError("initial_state circuit has an inconsistent width")
    if built.used_symbols:
        raise ValueError("initial_state circuit must not contain active symbols")
    snapshot = Circuit(num_qubits)
    snapshot.compose(built, snapshot.qubits)
    snapshot.validate()
    try:
        snapshot.inverse()
    except CircuitError as error:
        raise ValueError(
            "initial_state circuit must contain only statically invertible operations"
        ) from error
    return snapshot


__all__: list[str] = []
