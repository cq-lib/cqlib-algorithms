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

"""Excitation-number-preserving variational ansatz."""

from __future__ import annotations

import copy
from typing import Literal, final

from cqlib.circuit import Circuit, Parameter
from cqlib.circuit.ansatz import EntanglementTopology
from cqlib.circuit.gates import StandardGate

from ._validation import (
    validate_boolean,
    validate_positive_qubit_count,
    validate_repetitions,
)
from .base_ansatz import BaseAnsatz

_ExcitationMode = Literal["iswap", "fsim"]


def _snapshot_edges(
    num_qubits: int,
    entanglement: EntanglementTopology | None,
) -> tuple[tuple[int, int], ...]:
    """Return a validated, ordered snapshot of topology edges."""
    if entanglement is None:
        topology = EntanglementTopology.full()
    elif isinstance(entanglement, EntanglementTopology):
        topology = copy.copy(entanglement)
    else:
        raise TypeError("entanglement must be an EntanglementTopology or None")

    edges = tuple((pair[0], pair[1]) for pair in topology.generate_pairs(num_qubits))
    seen: set[tuple[int, int]] = set()
    for edge in edges:
        left, right = edge
        if left == right:
            raise ValueError("entanglement edges must connect distinct qubits")
        if not 0 <= left < num_qubits or not 0 <= right < num_qubits:
            raise ValueError("entanglement edge index is out of range")
        undirected = (min(left, right), max(left, right))
        if undirected in seen:
            raise ValueError("entanglement edges must not be duplicated")
        seen.add(undirected)
    return edges


@final
class ExcitationPreservingAnsatz(BaseAnsatz):
    """Alternate local Z rotations with number-preserving exchange blocks.

    Each exchange parameter is shared by ``RXX(theta / 2)`` and
    ``RYY(theta / 2)``. In ``fsim`` mode, every exchange block additionally
    applies a controlled ``Phase(-phi)`` with an independent parameter.
    The resulting circuit commutes with total computational-basis excitation
    number. A useful variational state therefore normally starts from a
    separately prepared state in the desired nonzero excitation sector.
    """

    __slots__ = (
        "_edges",
        "_eligible_qubits",
        "_mode",
        "_num_qubits",
        "_reps",
        "_skip_final_rotation_layer",
        "_skip_unentangled_qubits",
    )

    def __init__(
        self,
        num_qubits: int,
        *,
        reps: int = 3,
        mode: _ExcitationMode = "iswap",
        entanglement: EntanglementTopology | None = None,
        skip_unentangled_qubits: bool = False,
        skip_final_rotation_layer: bool = False,
    ) -> None:
        validate_positive_qubit_count(num_qubits)
        validate_repetitions(reps)
        if not isinstance(mode, str):
            raise TypeError("mode must be a string")
        if mode not in ("iswap", "fsim"):
            raise ValueError("mode must be 'iswap' or 'fsim'")
        validate_boolean(skip_unentangled_qubits, "skip_unentangled_qubits")
        validate_boolean(skip_final_rotation_layer, "skip_final_rotation_layer")

        self._num_qubits = num_qubits
        self._reps = reps
        self._mode: _ExcitationMode = mode
        self._edges = _snapshot_edges(num_qubits, entanglement)
        self._skip_unentangled_qubits = skip_unentangled_qubits
        self._skip_final_rotation_layer = skip_final_rotation_layer
        if skip_unentangled_qubits:
            active = {qubit for edge in self._edges for qubit in edge}
            self._eligible_qubits = tuple(
                qubit for qubit in range(num_qubits) if qubit in active
            )
        else:
            self._eligible_qubits = tuple(range(num_qubits))
        self.validate()

    def _validate_configuration(self) -> None:
        """Validate the frozen excitation-preserving architecture."""
        validate_positive_qubit_count(self._num_qubits)
        validate_repetitions(self._reps)
        if self._mode not in ("iswap", "fsim"):
            raise ValueError("mode must be 'iswap' or 'fsim'")
        validate_boolean(
            self._skip_unentangled_qubits,
            "skip_unentangled_qubits",
        )
        validate_boolean(
            self._skip_final_rotation_layer,
            "skip_final_rotation_layer",
        )

    def _append_rotation_layer(
        self,
        circuit: Circuit,
        prefix: str,
        parameter_index: int,
    ) -> int:
        for qubit in self._eligible_qubits:
            circuit.rz(qubit, Parameter(f"{prefix}_{parameter_index}"))
            parameter_index += 1
        return parameter_index

    def _append_exchange_block(
        self,
        circuit: Circuit,
        edge: tuple[int, int],
        prefix: str,
        parameter_index: int,
    ) -> int:
        theta = Parameter(f"{prefix}_{parameter_index}")
        circuit.append_gate(StandardGate.RXX(theta / 2), list(edge))
        circuit.append_gate(StandardGate.RYY(theta / 2), list(edge))
        parameter_index += 1
        if self._mode == "fsim":
            phi = Parameter(f"{prefix}_{parameter_index}")
            circuit.append_mc_gate(StandardGate.Phase(-phi).control(1), list(edge))
            parameter_index += 1
        return parameter_index

    def _build_circuit(self, prefix: str) -> Circuit:
        """Build layers with parameters in exact operation order."""
        circuit = Circuit(self._num_qubits)
        parameter_index = 0
        for _ in range(self._reps):
            parameter_index = self._append_rotation_layer(
                circuit,
                prefix,
                parameter_index,
            )
            for edge in self._edges:
                parameter_index = self._append_exchange_block(
                    circuit,
                    edge,
                    prefix,
                    parameter_index,
                )

        if not self._skip_final_rotation_layer:
            self._append_rotation_layer(circuit, prefix, parameter_index)
        return circuit

    def _parameter_count(self) -> int:
        """Return rotation and exchange parameter slots."""
        rotation_layers = self._reps + int(not self._skip_final_rotation_layer)
        exchange_parameters = 2 if self._mode == "fsim" else 1
        return (
            len(self._eligible_qubits) * rotation_layers
            + self._reps * len(self._edges) * exchange_parameters
        )

    def _qubit_count(self) -> int:
        """Return the configured register width."""
        return self._num_qubits
