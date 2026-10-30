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

"""Shared validation helpers for ansatz components."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence

from cqlib.circuit import Circuit, CircuitError
from cqlib.circuit.ansatz import EntanglementTopology
from cqlib.circuit.gates import StandardGate
from cqlib.qis import Hamiltonian

from ...protocols.state_preparation import StatePreparation

_ROTATION_GATES = (StandardGate.RX, StandardGate.RY, StandardGate.RZ)
_ENTANGLEMENT_GATES = (StandardGate.CX, StandardGate.CY, StandardGate.CZ)


def _is_finite_complex(value: complex) -> bool:
    return math.isfinite(value.real) and math.isfinite(value.imag)


def copy_hermitian_hamiltonian(
    hamiltonian: Hamiltonian,
    *,
    name: str,
) -> Hamiltonian:
    """Return a simplified snapshot of a finite, non-empty Hamiltonian."""
    validate_hermitian_hamiltonian(hamiltonian, name=name)
    snapshot = hamiltonian.copy()
    snapshot.simplify()
    if snapshot.num_terms == 0:
        raise ValueError(f"{name} must not be empty after simplification")
    return snapshot


def validate_hermitian_hamiltonian(
    hamiltonian: Hamiltonian,
    *,
    name: str,
) -> None:
    """Validate a finite, non-empty Hermitian Hamiltonian without mutation."""
    if not isinstance(hamiltonian, Hamiltonian):
        raise TypeError(f"{name} must be a Hamiltonian")

    aggregated: dict[tuple[int, int, int], complex] = {}
    for pauli, coefficient in hamiltonian.terms:
        effective = complex(coefficient) * pauli.phase.to_complex()
        if not _is_finite_complex(effective):
            raise ValueError(f"{name} coefficients must be finite")

        key = (pauli.num_qubits, pauli.x_mask, pauli.z_mask)
        combined = aggregated.get(key, 0.0j) + effective
        if not _is_finite_complex(combined):
            raise ValueError(f"{name} coefficients must be finite")
        aggregated[key] = combined

    for coefficient in aggregated.values():
        if coefficient != 0.0j and coefficient.imag != 0.0:
            raise ValueError(f"{name} is not Hermitian")

    if not any(coefficient != 0.0j for coefficient in aggregated.values()):
        raise ValueError(f"{name} must not be empty after simplification")


def validate_positive_qubit_count(num_qubits: int) -> None:
    """Validate a positive qubit count."""
    if isinstance(num_qubits, bool) or not isinstance(num_qubits, int):
        raise TypeError("num_qubits must be an integer")
    if num_qubits <= 0:
        raise ValueError("num_qubits must be greater than zero")


def validate_repetitions(reps: int) -> None:
    """Validate a non-negative repetition count."""
    if isinstance(reps, bool) or not isinstance(reps, int):
        raise TypeError("reps must be an integer")
    if reps < 0:
        raise ValueError("reps must be non-negative")


def validate_prefix(prefix: str) -> None:
    """Validate a circuit-parameter prefix."""
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a string")


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


def validate_boolean(value: bool, name: str) -> None:
    """Validate a Boolean option."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


def snapshot_rotation_gates(
    gates: Sequence[StandardGate],
) -> tuple[StandardGate, ...]:
    """Validate RX/RY/RZ templates and snapshot their sequence as a tuple.

    The tuple isolates the container from later edits to the caller's list;
    this is not a deep copy or a claim that arbitrary gate objects are frozen.
    Bound rotations and unsupported gate templates are rejected.
    """
    if isinstance(gates, (str, bytes)) or not isinstance(gates, Sequence):
        raise TypeError("rotation_gates must be a sequence of StandardGate values")
    frozen = tuple(gates)
    for gate in frozen:
        if not isinstance(gate, StandardGate):
            raise TypeError("rotation_gates must contain only StandardGate values")
        if gate not in _ROTATION_GATES:
            raise ValueError("rotation_gates may contain only unbound RX, RY, or RZ")
    return frozen


def validate_entanglement_gate(gate: StandardGate) -> None:
    """Validate an unbound Core two-qubit entangling gate."""
    if not isinstance(gate, StandardGate):
        raise TypeError("entanglement_gate must be a StandardGate")
    if gate not in _ENTANGLEMENT_GATES:
        raise ValueError("entanglement_gate must be an unbound CX, CY, or CZ")


def resolve_entanglement(
    entanglement: EntanglementTopology | None,
) -> EntanglementTopology:
    """Validate and snapshot a topology, using linear connectivity by default."""
    if entanglement is None:
        return EntanglementTopology.linear()
    if not isinstance(entanglement, EntanglementTopology):
        raise TypeError("entanglement must be an EntanglementTopology or None")
    return copy.copy(entanglement)
