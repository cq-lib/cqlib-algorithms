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

r"""Layer-selective noise scaling.

Global folding scales every gate by the same factor, which is wrong when the
device noise is layer dependent. ``fold_layers`` instead takes a mapping from
layer index to the scale factor that layer alone should see, so the
multivariate extrapolation in
:mod:`cqlib_algorithms.error_mitigation.inference.multivariate` can fit one
noise parameter per layer.

A layer here is one index into ``Circuit.operations``: layer ``i`` is folded
in place and every other operation is copied through untouched. Consecutive
layers that share a scale factor form one block, and each block is folded the
same way :func:`~cqlib_algorithms.error_mitigation.scaling.fold_global` folds a
whole circuit, so a block realises both the integer and the fractional part of
its factor.

References
----------
[1] A. Russo and A. Mari, "Quantum error mitigation by layerwise Richardson
    extrapolation," Physical Review A 110, 062420, 2024. arXiv:2402.04000.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from cqlib import Circuit
from cqlib.circuit import ValueOperation

from .._circuits import rebuild, strip_measurements
from .folding import (
    _fold_operations,
    _normalize_scale_factor,
    _restore_measurements,
)


def fold_layers(
    circuit: Circuit,
    /,
    *,
    scales: Mapping[int, float],
) -> Circuit:
    r"""Fold the addressed layers of ``circuit`` by individual scale factors.

    Consecutive indices that carry the same scale factor form one block, which
    is folded the way
    :func:`~cqlib_algorithms.error_mitigation.scaling.fold_global` folds a
    whole circuit: the block is folded
    :math:`n = \lfloor (\lambda - 1) / 2 \rfloor` times and its residual
    :math:`f = \lambda - (2n + 1)` is spent on the last
    :math:`k = \mathrm{round}(f\,G_b / 2)` gates of the block, so a block of
    :math:`G_b` gates grows to :math:`(2n + 1)G_b + 2k` and realises
    :math:`\lambda` to within :math:`2k / G_b`. Layers absent from ``scales``
    are copied through unfolded.

    Args:
        circuit: Circuit to scale. Measurements are stripped and re-appended.
        scales: Map from operation index to that layer's scale factor. Indices index
            the gate body: ``circuit.operations`` with measurements removed, as
            inspected before any folding.

    Returns:
        Folded circuit carrying the source global phase.

    Raises:
        ValueError: If any scale factor is below ``1``, any index is negative or beyond
            the gate-body operation list, or the body contains an operation that
            cannot be inverted.
    """
    # Measurements are stripped before the body is grouped, so the indices and
    # the body they address are the same length; indexing the raw operation
    # list would offset every index by the measurement count.
    body, measured = strip_measurements(circuit)
    operations = list(body.operations)
    _validate_layer_indices(scales, len(operations))
    folded: list[ValueOperation] = []
    for block, scale_factor in _layer_blocks(operations, scales):
        if scale_factor is None or scale_factor == 1.0:
            folded.extend(block)
            continue
        folded.extend(_fold_operations(body, block, scale_factor).operations)
    return _restore_measurements(rebuild(body, folded), measured)


def _validate_layer_indices(
    scales: Mapping[int, float],
    num_operations: int,
    /,
) -> None:
    r"""Validate circuit-layer indices."""
    for index in scales:
        if index < 0 or index >= num_operations:
            raise ValueError(
                f"Layer index {index} is outside the {num_operations} "
                "operations of the circuit."
            )


def _layer_blocks(
    operations: Sequence[ValueOperation],
    scales: Mapping[int, float],
    /,
) -> list[tuple[list[ValueOperation], float | None]]:
    r"""Split the body into runs of layers that share one scale factor.

    Each returned block carries its normalized scale factor, or ``None`` when
    none of its indices is addressed. An unaddressed index therefore breaks
    the run, so only explicitly addressed neighbours merge into one block.
    """
    blocks: list[tuple[list[ValueOperation], float | None]] = []
    current: list[ValueOperation] = []
    current_scale: float | None = None
    for index, operation in enumerate(operations):
        factor = scales.get(index)
        normalized = None if factor is None else _normalize_scale_factor(factor)
        if current and normalized != current_scale:
            blocks.append((current, current_scale))
            current = []
        current.append(operation)
        current_scale = normalized
    if current:
        blocks.append((current, current_scale))
    return blocks
