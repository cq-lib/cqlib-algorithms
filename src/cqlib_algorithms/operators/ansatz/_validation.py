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

from cqlib.circuit.ansatz import EntanglementTopology
from cqlib.circuit.gates import StandardGate
from cqlib.qis import Hamiltonian

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
    if not isinstance(hamiltonian, Hamiltonian):
        raise TypeError(f"{name} must be a Hamiltonian")

    snapshot = hamiltonian.copy()
    aggregated: dict[tuple[int, int, int], complex] = {}
    for pauli, coefficient in snapshot.terms:
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

    snapshot.simplify()
    if snapshot.num_terms == 0:
        raise ValueError(f"{name} must not be empty after simplification")
    return snapshot


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


def validate_boolean(value: bool, name: str) -> None:
    """Validate a Boolean option."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


def snapshot_rotation_gates(
    gates: Sequence[StandardGate],
) -> tuple[StandardGate, ...]:
    """Validate and freeze unbound one-parameter rotation gates."""
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
