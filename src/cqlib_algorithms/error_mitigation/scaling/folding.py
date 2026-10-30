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

r"""Noise scaling by gate folding.

Folding replaces a unitary block :math:`U` by :math:`U U^\dagger U`, which
leaves the logical action of the circuit unchanged while tripling the noise
accumulated on that block. The scale factor :math:`\lambda` measures how many
times the original gate count is reproduced:

.. math::

    \lambda = (2 n + 1) + f, \qquad n = \left\lfloor \tfrac{\lambda - 1}{2}
    \right\rfloor, \quad f \in [0, 2)

The integer part :math:`2n + 1` comes from replacing every gate by
:math:`G (G^\dagger G)^{n}`, i.e. from folding the whole body :math:`n`
times. The residual :math:`f` is realised by repeating the last
:math:`k = \mathrm{round}(f G / 2)` gates of the (already folded) body as
:math:`L^\dagger L`, where :math:`G` is the gate count.

:func:`fold_global` folds the whole body uniformly, :func:`fold_local` folds
contiguous chunks independently, and :func:`fold_gates_at_random` folds a
random subset of the gates. Measurements are removed before folding and
re-appended afterwards, since the :math:`U^\dagger U` pattern is unitary only
on the gate body.

References
----------
[1] T. Giurgica-Tiron, Y. Hindy, R. LaRose, A. Mari, and W. J. Zeng, "Digital zero noise
    extrapolation for quantum error mitigation," IEEE International Conference on
    Quantum Computing and Engineering (QCE), pp. 306-316, 2020. arXiv:2005.10921.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cqlib import Circuit
from cqlib.circuit import (
    Instruction,
    Qubit,
    UnitaryGate,
    ValueOperation,
)
from cqlib.error_mitigation import ZNEMitigation

from .._circuits import rebuild, strip_measurements


def fold_global(circuit: Circuit, scale_factor: float, /) -> Circuit:
    r"""Fold every gate of ``circuit`` globally to ``scale_factor``.

    The whole body is folded uniformly, so the resulting gate count is
    exactly :math:`(2n + 1) G + 2 k` with :math:`k = \mathrm{round}(f G / 2)`.
    Folding every gate by the same amount keeps the noise model layer
    independent, which is what makes the extrapolation in
    :mod:`cqlib_algorithms.error_mitigation.inference` well defined.

    Args:
        circuit: Circuit to scale. Measurements are stripped and re-appended.
        scale_factor: Noise scale factor :math:`\lambda \ge 1`. ``1`` folds nothing.

    Returns:
        Folded circuit carrying the source global phase.

    Raises:
        ValueError: If ``scale_factor < 1``, or if the body contains an operation that
            cannot be inverted (reset, classical data, classical control).
    """
    scale = _normalize_scale_factor(scale_factor)
    body, measured = strip_measurements(circuit)
    if scale > 1.0:
        body = _fold_operations(body, list(body.operations), scale)
    return _restore_measurements(body, measured)


def fold_gates_at_random(
    circuit: Circuit,
    scale_factor: float,
    /,
    *,
    seed: int,
) -> Circuit:
    r"""Fold a random subset of the gates of ``circuit``.

    Each gate contributes the same weight :math:`0.99^n` for an ``n``-qubit
    gate, so the scale factor is reached by folding
    :math:`\mathrm{round}(f G / 2)` randomly chosen gates one extra time. The
    result is a deterministic function of ``seed``.

    Args:
        circuit: Circuit to scale. Measurements are stripped and re-appended.
        scale_factor: Noise scale factor :math:`\lambda \ge 1`.
        seed: Seed for the gate-selection shuffle.

    Returns:
        Folded circuit carrying the source global phase.

    Raises:
        ValueError: If ``scale_factor < 1``, or if the body contains an operation that
            cannot be inverted (reset, classical data, classical control).
    """
    scale = _normalize_scale_factor(scale_factor)
    body, measured = strip_measurements(circuit)
    if scale > 1.0:
        operations = list(body.operations)
        weights = [_default_weight(operation) for operation in operations]
        # One shuffle drives a greedy mask, so the residual fold count is
        # spent without a second draw.
        mask = _create_fold_mask(weights, scale, seed=seed)
        body = rebuild(body, _apply_fold_mask(operations, mask))
    return _restore_measurements(body, measured)


def fold_local(
    circuit: Circuit,
    scale_factor: float,
    /,
    *,
    num_chunks: int,
) -> Circuit:
    r"""Split ``circuit`` into contiguous chunks and fold each one globally.

    Chunk-wise folding emulates a layer-dependent noise model: only the
    :math:`\lceil G / \text{num\_chunks} \rceil` gates of each chunk are
    scaled, so gates in noise-free regions of the device stay unfolded. The
    split is deterministic, unlike :func:`fold_gates_at_random`.

    Args:
        circuit: Circuit to scale. Measurements are stripped and re-appended.
        scale_factor: Noise scale factor :math:`\lambda \ge 1`, applied to every chunk.
        num_chunks: Number of contiguous chunks, ``>= 1``. Chunks are as even as
            possible; a chunk may be empty when ``num_chunks`` exceeds the gate
            count.

    Returns:
        Folded circuit carrying the source global phase.

    Raises:
        ValueError: If ``num_chunks < 1``, if ``scale_factor < 1``, or if the body
            contains an operation that cannot be inverted.
    """
    if num_chunks < 1:
        raise ValueError(f"num_chunks must be >= 1 but was {num_chunks!r}.")
    scale = _normalize_scale_factor(scale_factor)
    body, measured = strip_measurements(circuit)
    if scale > 1.0:
        chunks = _split_into_chunks(list(body.operations), num_chunks)
        folded: list[ValueOperation] = []
        for chunk in chunks:
            folded.extend(_fold_operations(body, chunk, scale).operations)
        body = rebuild(body, folded)
    return _restore_measurements(body, measured)


def _normalize_scale_factor(scale_factor: float, /) -> float:
    r"""Validate and return a scale factor in normalized form."""
    value = float(scale_factor)
    if value < 1.0:
        raise ValueError(
            f"Scale factors must be >= 1 but the scale factor was {value!r}; "
            "folding cannot shrink a circuit."
        )
    return value


def _split_scale_factor(scale_factor: float, /) -> tuple[int, float]:
    r"""Split a scale factor into an integer fold count and a fraction.

    Returns ``(n, f)`` with ``scale_factor = (2 n + 1) + f`` and
    :math:`f \in [0, 2)`. ``divmod`` is used rather than a floor of the
    quotient so that both segments stay exact for float inputs.
    """
    num_folds, fraction = divmod(scale_factor - 1.0, 2.0)
    return int(num_folds), float(fraction)


def _default_weight(operation: ValueOperation, /) -> float:
    r"""Return the default folding weight :math:`0.99^{n}` of an operation."""
    return 0.99**operation.num_qubits


def _fold_operations(
    anchor: Circuit,
    operations: Sequence[ValueOperation],
    scale_factor: float,
    /,
) -> Circuit:
    r"""Fold ``operations`` globally and rebuild the result on ``anchor``.

    ``anchor`` supplies the qubits, classical state, and global phase for the
    rebuilt circuit; its own operation list is discarded.
    """
    num_folds, fraction = _split_scale_factor(scale_factor)
    # The residual is a fraction of the *original* gate count, so it is read
    # before the integer fold inflates the body; reading it afterwards would
    # scale the residual by (2n + 1) and overshoot the requested factor.
    num_gates = len(operations)
    current = list(operations)
    if num_folds:
        current = list(
            _fold_integer_blocks(rebuild(anchor, current), num_folds).operations
        )
    tail = _fraction_tail(current, num_gates, fraction)
    if tail:
        current = [*current, *tail]
    return rebuild(anchor, current)


def _fold_integer_blocks(body: Circuit, num_folds: int, /) -> Circuit:
    r"""Apply the :math:`U (U^\dagger U)^{n}` pattern to every gate.
    """
    folded = ZNEMitigation(body, [num_folds]).fold_circuits()[0]

    # The folded circuit is rebuilt from the bare qubit list, so the global
    # phase has to be carried over again on the returned circuit.
    folded.set_global_phase(body.global_phase)
    return folded


def _fraction_tail(
    operations: Sequence[ValueOperation],
    num_gates: int,
    fraction: float,
    /,
) -> list[ValueOperation]:
    r"""Return the residual :math:`L^\dagger L` append for ``fraction``.

    The last :math:`k = \mathrm{round}(f\,G/2)` gates form the block
    :math:`L`, where :math:`G` is ``num_gates``, the gate count *before* any
    integer fold; appending :math:`L^\dagger L` reproduces those :math:`k`
    gates twice and hence multiplies the gate count by :math:`1 + 2k/G`. The
    appended block is :math:`[G_k^\dagger \ldots G_1^\dagger, G_1 \ldots
    G_k]` on the tail ordered :math:`L = [G_1 \ldots G_k]`.
    """
    if fraction <= 0.0:
        return []
    num_to_fold = round(fraction * num_gates / 2.0)
    if num_to_fold == 0:
        return []
    tail = list(operations[-num_to_fold:])
    return [_inverse_operation(operation) for operation in reversed(tail)] + tail


def _create_fold_mask(
    weights: Sequence[float],
    scale_factor: float,
    /,
    *,
    seed: int,
) -> list[int]:
    r"""Return how many times each gate must be folded.

    The returned mask never overshoots the requested scale factor, so the
    approximation is never worsened by an extra fold.
    """
    num_uniform_folds = int((scale_factor - 1.0) / 2.0)
    odd_integer_scale_factor = 2 * num_uniform_folds + 1
    mask = [
        0 if np.isclose(weight, 0.0) else num_uniform_folds for weight in weights
    ]
    if np.isclose(odd_integer_scale_factor, scale_factor):
        return mask

    # Every gate starts at the largest odd scale factor below ``scale_factor``.
    # The residual is then spent on randomly ordered gates, one fold at a time,
    # until an extra fold would overshoot the request.
    folding_order = list(range(len(weights)))
    np.random.RandomState(seed).shuffle(folding_order)

    input_weight = float(sum(weights))
    output_weight = odd_integer_scale_factor * input_weight
    approx_error = abs(output_weight - scale_factor * input_weight)
    for index in folding_order:
        if np.isclose(weights[index], 0.0):
            continue
        candidate = output_weight + 2 * weights[index]
        candidate_error = abs(candidate - scale_factor * input_weight)
        if candidate_error >= approx_error:
            break
        approx_error = candidate_error
        output_weight = candidate
        mask[index] += 1
    return mask


def _apply_fold_mask(
    operations: Sequence[ValueOperation],
    mask: Sequence[int],
    /,
) -> list[ValueOperation]:
    r"""Replace each ``G`` by :math:`G (G^\dagger G)^{m}` for its mask ``m``."""
    if len(operations) != len(mask):
        raise ValueError(
            f"The circuit and the folding mask have incompatible sizes: "
            f"{len(operations)} operations and {len(mask)} mask entries."
        )
    folded: list[ValueOperation] = []
    for operation, num_folds in zip(operations, mask, strict=True):
        folded.append(operation)
        for _ in range(num_folds):
            folded.append(_inverse_operation(operation))
            folded.append(operation)
    return folded


def _inverse_operation(operation: ValueOperation, /) -> ValueOperation:
    r"""Return the inverse of one gate-body operation.

    Raises:
        ValueError: If the operation is not invertible, i.e. it measures, resets, or
            carries classical data or classical control.

    """
    # Standard gates invert through their own handle, and directives and delays
    # are self-inverse. The handle stored in the instruction carries no
    # parameters, so a parametric gate such as RX(theta) is rebound to the
    # operation parameters before inversion; otherwise the binding rejects the
    # call.
    if operation.is_standard:
        gate = operation.instruction.standard_gate
        if gate is not None:
            bound = gate(*operation.params) if operation.params else gate
            return ValueOperation.from_standard_gate(
                bound.inverse(),
                list(operation.qubits),
                label=operation.label,
            )
    if operation.is_directive:
        directive = operation.instruction.directive
        inverse = None if directive is None else directive.inverse()
        if inverse is not None:
            return ValueOperation.from_instruction(
                Instruction.from_directive(inverse),
                list(operation.qubits),
            )
    if operation.is_delay:
        return operation

    # Multi-controlled gates, custom unitaries, and circuit
    # gates expose no such handle on the Python binding, so they invert
    # through their matrix.
    if operation.is_mcgate or operation.is_unitary or operation.is_circuit_gate:
        matrix_gate = UnitaryGate(
            f"{operation.name}^dagger",
            operation.num_qubits,
        ).with_matrix(np.conjugate(operation.matrix().T))
        return ValueOperation.from_instruction(
            Instruction.from_unitary_gate(matrix_gate),
            list(operation.qubits),
        )
    raise ValueError(
        f"Operation {operation.name!r} has no unitary inverse, so the circuit "
        "cannot be folded."
    )


def _split_into_chunks(
    operations: Sequence[ValueOperation],
    num_chunks: int,
    /,
) -> list[list[ValueOperation]]:
    r"""Split ``operations`` into ``num_chunks`` contiguous, even chunks."""
    size, remainder = divmod(len(operations), num_chunks)
    chunks: list[list[ValueOperation]] = []
    start = 0
    for index in range(num_chunks):
        stop = start + size + (1 if index < remainder else 0)
        chunks.append(list(operations[start:stop]))
        start = stop
    return chunks


def _restore_measurements(
    circuit: Circuit,
    measured: Sequence[Qubit],
    /,
) -> Circuit:
    r"""Re-measure ``measured`` at the tail of ``circuit``, in order."""
    for qubit in measured:
        circuit.measure(qubit)
    return circuit
