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

"""Private topology helpers for tensor-network-inspired ansatze."""

from __future__ import annotations

from cqlib.circuit import Circuit, Parameter

_Edge = tuple[int, int]
_Level = tuple[_Edge, ...]
_MeraScale = tuple[_Level, _Level]


def append_real_two_qubit_block(
    circuit: Circuit,
    edge: _Edge,
    prefix: str,
    parameter_index: int,
) -> int:
    """Append ``RY(left) -> RY(right) -> CX(left, right)``."""
    left, right = edge
    circuit.ry(left, Parameter(f"{prefix}_{parameter_index}"))
    circuit.ry(right, Parameter(f"{prefix}_{parameter_index + 1}"))
    circuit.cx(left, right)
    return parameter_index + 2


def mps_edges(num_qubits: int) -> _Level:
    """Return the left-to-right nearest-neighbor MPS sweep."""
    return tuple((left, left + 1) for left in range(num_qubits - 1))


def ttn_levels(num_qubits: int) -> tuple[tuple[_Level, ...], int]:
    """Return reducing TTN levels and the retained root qubit."""
    active = list(range(num_qubits))
    levels: list[_Level] = []
    while len(active) > 1:
        edges = tuple(
            (active[index], active[index + 1]) for index in range(0, len(active) - 1, 2)
        )
        levels.append(edges)
        retained = [right for _, right in edges]
        if len(active) % 2:
            retained.append(active[-1])
        active = retained
    return tuple(levels), active[0]


def mera_scales(num_qubits: int) -> tuple[tuple[_MeraScale, ...], int]:
    """Return complete binary MERA scales and the retained root qubit."""
    active = list(range(num_qubits))
    scales: list[_MeraScale] = []
    while len(active) > 1:
        disentanglers = tuple(
            (active[index], active[index + 1]) for index in range(1, len(active) - 1, 2)
        )
        isometries = tuple(
            (active[index], active[index + 1]) for index in range(0, len(active), 2)
        )
        scales.append((disentanglers, isometries))
        active = [right for _, right in isometries]
    return tuple(scales), active[0]
