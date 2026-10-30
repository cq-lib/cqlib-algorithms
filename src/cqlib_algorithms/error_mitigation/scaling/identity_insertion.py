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

r"""Noise scaling by identity-layer insertion.

Identity insertion inflates the circuit depth without changing its logical
action: a layer of :math:`\mathrm{id}` gates is appended after existing
operations, so the idle time each qubit sees grows while the compiled
unitary stays exactly the same. On a device whose noise grows with circuit
duration, this is a hardware-cheap way to produce the higher-noise samples
that ZNE extrapolates from.

For a body of :math:`D` operations and scale factor :math:`\lambda`, each
operation receives

.. math::

    n = \lfloor \lambda - 1 \rfloor

uniform identity layers, and a residual fraction :math:`\lambda - 1 - n`
distributes :math:`\lfloor D (\lambda - 1 - n) \rfloor` further layers over
randomly chosen positions. The total inserted layer count is therefore
:math:`\lfloor D (\lambda - 1) \rfloor`, which reproduces the requested depth
inflation up to one layer per operation.

A layer here is one index into ``Circuit.operations``: each inserted layer
applies :math:`\mathrm{id}` to every qubit, and a partial layer is placed
after a randomly chosen subset of the operations.

References
----------
[1] A. He, B. Nachman, W. A. de Jong, and C. W. Bauer, "Zero-noise extrapolation for
    quantum-gate error mitigation with identity insertions," Physical Review A 102,
    012426, 2020. arXiv:2003.04941.
"""

from __future__ import annotations

import numpy as np

from cqlib import Circuit
from cqlib.circuit import Qubit, StandardGate, ValueOperation

from .._circuits import rebuild, strip_measurements
from .folding import _normalize_scale_factor, _restore_measurements


def insert_id_layers(
    circuit: Circuit,
    scale_factor: float,
    /,
    *,
    seed: int,
) -> Circuit:
    r"""Inflate ``circuit`` to ``scale_factor`` by inserting identity layers.

    Every operation is followed by :math:`n = \lfloor \lambda - 1 \rfloor`
    uniform identity layers, and the residual fraction is spent as one extra
    layer after each of :math:`\lfloor D (\lambda - 1 - n) \rfloor` randomly
    chosen operations.

    Args:
        circuit: Circuit to scale. Measurements are stripped and re-appended.
        scale_factor: Noise scale factor :math:`\lambda \ge 1`. ``1`` inserts nothing.
        seed: Seed for the random selection of partially scaled positions.

    Returns:
        Circuit with identity layers inserted, carrying the source global
            phase and its measurements restored at the tail.

    Raises:
        ValueError: If ``scale_factor < 1``.

    Note:
        ``seed`` is a required keyword-only argument so the partially scaled
        positions are reproducible instead of reading the global random stream.
    """
    scale = _normalize_scale_factor(scale_factor)
    body, measured = strip_measurements(circuit)
    operations = list(body.operations)
    depth = len(operations)
    num_uniform_layers, num_partial_layers = _calculate_id_layers(depth, scale)
    partial = _select_partial_layers(depth, num_partial_layers, seed=seed)
    layer = _identity_layer(list(body.qubits))
    scaled: list[ValueOperation] = []
    for index, operation in enumerate(operations):
        scaled.append(operation)
        for _ in range(num_uniform_layers):
            scaled.extend(layer)
        if index in partial:
            scaled.extend(layer)
    return _restore_measurements(rebuild(body, scaled), measured)


def _calculate_id_layers(
    depth: int,
    scale_factor: float,
    /,
) -> tuple[int, int]:
    r"""Split the depth inflation into a uniform and a partial layer count.

    A scale factor that lands on an integer needs no partial layers at all;
    otherwise the residual fraction is spread over ``depth`` positions.
    """
    num_uniform_layers = int(scale_factor - 1.0)
    integer_scale_factor = num_uniform_layers + 1.0
    if np.isclose(integer_scale_factor, scale_factor):
        return num_uniform_layers, 0
    num_partial_layers = int(depth * (scale_factor - 1.0 - num_uniform_layers))
    return num_uniform_layers, num_partial_layers


def _select_partial_layers(
    depth: int,
    num_partial_layers: int,
    /,
    *,
    seed: int,
) -> set[int]:
    r"""Pick the operation indices that receive one extra identity layer."""
    if num_partial_layers <= 0:
        return set()
    chosen = np.random.RandomState(seed).choice(
        depth,
        size=num_partial_layers,
        replace=False,
    )
    return {int(index) for index in chosen}


def _identity_layer(qubits: list[Qubit], /) -> list[ValueOperation]:
    r"""Return one identity layer acting on every qubit in ``qubits``."""
    return [
        ValueOperation.from_standard_gate(StandardGate.I, [qubit])
        for qubit in qubits
    ]
