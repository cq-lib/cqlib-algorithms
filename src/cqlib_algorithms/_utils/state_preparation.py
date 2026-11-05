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

"""Shared snapshots of fixed state-preparation components."""

from __future__ import annotations

from cqlib.circuit import Circuit, CircuitError

from ..protocols.state_preparation import StatePreparation


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
