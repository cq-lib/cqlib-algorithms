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

"""Tests for digital dynamical decoupling."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from cqlib import Circuit

from cqlib_algorithms.error_mitigation.techniques._ddd_rules import (
    DDDRule,
    XX,
    XYXY,
)
from cqlib_algorithms.error_mitigation.techniques._equivalence import (
    allclose_up_to_global_phase,
)
from cqlib_algorithms.error_mitigation.techniques.ddd import (
    DynamicalDecoupling,
    combine_results,
)
from cqlib_algorithms.protocols import EstimatorResult

_ATOL = 1e-9

# Symmetric XY4 timing: the free stretch at each end of the window covers half
# a slot while the stretches in between cover a full one. Only this layout
# leaves the static detuning without a first-order term.
_END_STRETCH = 0.5
_MIDDLE_STRETCH = 1.0


def _idle_window_circuit() -> Circuit:
    """Return a two-qubit body whose second wire idles across four layers."""
    circuit = Circuit(2)
    for _ in range(4):
        circuit.h(0)
    circuit.cx(0, 1)
    return circuit


def _free_stretch(detuning: float, scale: float, /) -> np.ndarray:
    """Return ``exp(-i scale detuning Z / 2)`` as a diagonal matrix."""
    phase = np.exp(-0.5j * scale * detuning * np.array([1.0, -1.0]))
    return np.diag(phase)


def _propagator(rule: DDDRule, detuning: float, /) -> np.ndarray:
    r"""Return the propagator of one rule laid out with symmetric timing.

    A static detuning :math:`H = \Delta Z / 2` drives the free stretches while
    the pulses act instantaneously, so the propagator is the ordered product
    of the two. The stretches wrap the pulses with half-length stretches at
    the two ends.

    Parameters
    ----------
    rule
        Pulse sequence to propagate.
    detuning
        Strength :math:`\Delta` of the static detuning.

    Returns
    -------
    numpy.ndarray
        The :math:`2 \times 2` propagator of one full pulse period.
    """
    last = len(rule.gates) - 1
    steps: list[np.ndarray] = [_free_stretch(detuning, _END_STRETCH)]
    for index, gate in enumerate(rule.gates):
        steps.append(gate.matrix())
        steps.append(
            _free_stretch(
                detuning,
                _END_STRETCH if index == last else _MIDDLE_STRETCH,
            )
        )

    # Walking the steps in circuit order leaves each step left-multiplied, so
    # the last step ends up leftmost, which is the action order of a circuit.
    total = np.eye(2, dtype=np.complex128)
    for step in steps:
        total = step @ total
    return total


def test_apply_inserts_only_into_idle_windows(
    names: Callable[[Circuit], list[str]],
) -> None:
    """Pulses land on the idle wire only, leaving the computed wire alone."""
    source = _idle_window_circuit()
    decoupled = DynamicalDecoupling(rule=XX).apply(source)

    computed = [
        operation.name
        for operation in decoupled.operations
        if operation.instruction.standard_gate not in XX.gates
    ]
    assert computed == names(source)

    idle_wire = source.qubits[1]
    pulses = [
        operation
        for operation in decoupled.operations
        if operation.instruction.standard_gate in XX.gates
    ]
    assert pulses
    for operation in pulses:
        assert list(operation.qubits) == [idle_wire]


def test_the_inserted_sequence_is_the_identity() -> None:
    """The decoupled circuit realises the same unitary as the source."""
    source = _idle_window_circuit()
    decoupled = DynamicalDecoupling(rule=XX).apply(source)

    assert len(decoupled.operations) > len(source.operations)
    assert np.allclose(
        source.to_matrix(),
        decoupled.to_matrix(),
        atol=_ATOL,
    )


def test_the_inserted_sequence_preserves_the_global_phase(
    phases: Callable[[Circuit], complex],
) -> None:
    """Decoupling keeps the global phase the source circuit carries."""
    source = _idle_window_circuit()
    source.set_global_phase(0.25)
    decoupled = DynamicalDecoupling(rule=XX).apply(source)

    assert phases(decoupled) == phases(source)


def test_xyxy_propagator_averages_a_static_detuning_to_first_order() -> None:
    r"""XY4 removes the first-order term a static detuning contributes."""
    detuning = 1e-5

    # The pulses multiply to the identity up to a global phase, so a window
    # with no free evolution collapses to that phase.
    assert allclose_up_to_global_phase(_propagator(XYXY, 0.0), np.eye(2))

    forward = _propagator(XYXY, detuning)
    backward = _propagator(XYXY, -detuning)
    first_order = (forward - backward) / (2.0 * detuning)
    assert np.allclose(first_order, 0.0, atol=1e-6)

    # A bare free stretch does carry a first-order term, so the cancellation
    # above is a property of the pulse layout rather than of the probe.
    bare = (1.0 / (2.0 * detuning)) * (
        _free_stretch(detuning, _MIDDLE_STRETCH)
        - _free_stretch(-detuning, _MIDDLE_STRETCH)
    )
    assert not np.allclose(bare, 0.0)


def test_apply_is_the_identity_when_there_is_no_idle_window() -> None:
    """A circuit that keeps every wire busy is returned untouched."""
    packed = Circuit(2)
    packed.cx(0, 1)
    packed.cx(0, 1)

    decoupled = DynamicalDecoupling(rule=XX).apply(packed)

    assert decoupled is packed
    assert len(decoupled.operations) == len(packed.operations)


def test_combine_results_requires_exactly_one_result() -> None:
    """The reduction is the identity on a one-element family."""
    assert combine_results([EstimatorResult(expectation=0.5)]) == 0.5

    with pytest.raises(ValueError, match="exactly one result"):
        combine_results([])
    with pytest.raises(ValueError, match="exactly one result"):
        combine_results(
            [
                EstimatorResult(expectation=0.1),
                EstimatorResult(expectation=0.2),
            ]
        )
