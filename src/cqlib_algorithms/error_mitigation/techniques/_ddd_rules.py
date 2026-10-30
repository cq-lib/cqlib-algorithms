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

r"""Dynamical-decoupling pulse rules.

A dynamical-decoupling rule is a short pulse sequence whose pulses multiply to
the identity while refocusing the environmentally induced phase that an idle
qubit accumulates. The rule fixes which pulses are used and how far apart they
sit, so one rule fills idle windows of different lengths.

A rule is laid into a window as a list of time slots: every slot holds either
one pulse or nothing, and the pulses are spread as evenly as the window
allows. :data:`XX`, :data:`XYXY` and :data:`YY` are the standard sequences, and
:func:`general_rule` builds further ones from an arbitrary pulse list.

References
----------
[1] L. Viola and S. Lloyd, "Dynamical suppression of decoherence in two-state quantum
    systems," Physical Review A 58, 2733, 1998. arXiv:quant-ph/9803057.
[2] L. Viola, E. Knill, and S. Lloyd, "Dynamical decoupling of open quantum systems,"
    Physical Review Letters 82, 2417, 1999. doi:10.1103/PhysRevLett.82.2417.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from cqlib.circuit import StandardGate
from numpy.typing import NDArray

from ._equivalence import allclose_up_to_global_phase

# A negative spacing asks the rule for the widest spacing the window can hold,
# which is the only spacing that fits a single pulse sequence.
_DEFAULT_SPACING = -1


@dataclass(frozen=True, slots=True)
class DDDRule:
    r"""Pulse sequence inserted into idle windows.

    Args:
        gates: Pulses in application order. At least two single-qubit standard gates
            are required, and their product must be the identity up to a global
            phase.
        spacing: Number of idle slots kept before the first pulse and between two
            consecutive pulses. A negative value selects the widest spacing that
            lets the pulses fit the window.

    Raises:
        TypeError: If ``gates`` is not a sequence of ``StandardGate`` instances, or if
            ``spacing`` is not an integer.
        ValueError: If fewer than two pulses are given, or if a pulse acts on more than
            one qubit.

    Examples:
        >>> from cqlib.circuit import StandardGate
        >>> DDDRule(gates=(StandardGate.X, StandardGate.X)).sequence(4)
        (None, StandardGate.X, StandardGate.X, None)

    """

    gates: tuple[StandardGate, ...]
    spacing: int = field(default=_DEFAULT_SPACING, kw_only=True)

    def __post_init__(self) -> None:
        _validate_gates(self.gates)
        _validate_spacing(self.spacing)
        # A list is accepted on input, and the stored value has to stay a
        # tuple so the rule remains hashable.
        object.__setattr__(self, "gates", tuple(self.gates))

    def sequence(self, slack_length: int, /) -> tuple[StandardGate | None, ...]:
        r"""Lay the pulses into an idle window of ``slack_length`` slots.

        A window of :math:`L` slots holds :math:`n` pulses separated by
        :math:`\lfloor (L - n) / (n + 1) \rfloor` idle slots, and the slots
        left over are spread over the two ends of the window. A window too
        short for the pulse list is left untouched.

        Args:
            slack_length: Number of time slots in the idle window.

        Returns:
            One entry per slot of the window, holding the pulse applied in
                that slot or ``None`` for an idle slot. Empty when the window
                cannot hold the pulses, or when ``spacing`` does not fit.

        Raises:
            TypeError: If ``slack_length`` is not an integer.
            ValueError: If ``slack_length`` is negative, or if the pulses do not multiply
                to the identity up to a global phase.

        """
        _validate_slack_length(slack_length)
        num_pulses = len(self.gates)
        if slack_length < 2 or slack_length < num_pulses:
            return ()

        span = slack_length - num_pulses
        spacing = self.spacing
        if spacing < 0:
            spacing = span // (num_pulses + 1)
        remainder = slack_length - (spacing * (num_pulses + 1) + num_pulses)
        if remainder < 0:
            return ()
        _validate_pulse_identity(self.gates)

        # The sequence opens with the spacing block and repeats it after every
        # pulse, so the pulses end up evenly spaced across the window.
        slots: list[StandardGate | None] = [None] * spacing
        for gate in self.gates:
            slots.append(gate)
            slots.extend([None] * spacing)
        # The slots that the evenly spaced layout does not spend go alternately
        # to the two ends, which keeps the pulses centred in the window.
        for index in range(remainder):
            if index % 2:
                slots.append(None)
            else:
                slots.insert(0, None)
        return tuple(slots)


def general_rule(
    gates: Sequence[StandardGate],
    /,
    *,
    spacing: int = _DEFAULT_SPACING,
) -> DDDRule:
    r"""Build a decoupling rule from an arbitrary pulse list.

    Args:
        gates: Single-qubit standard gates whose product is the identity up to a
            global phase, in application order.
        spacing: Number of idle slots kept before the first pulse and between two
            consecutive pulses. A negative value selects the widest spacing that
            lets the pulses fit the window.

    Returns:
        The rule described by ``gates`` and ``spacing``.

    """
    return DDDRule(gates=tuple(gates), spacing=spacing)


def _validate_gates(gates: Sequence[StandardGate], /) -> None:
    r"""Validate a dynamical-decoupling pulse sequence."""
    if isinstance(gates, (str, bytes)) or not isinstance(gates, Sequence):
        raise TypeError("gates must be a sequence of StandardGate instances")
    if len(gates) < 2:
        raise ValueError(
            "gates must hold at least two pulses to form a decoupling "
            f"sequence, but {len(gates)} were given."
        )
    for gate in gates:
        if not isinstance(gate, StandardGate):
            raise TypeError(
                "gates must hold StandardGate instances, but received "
                f"{type(gate).__name__}"
            )
        if gate.num_qubits != 1:
            raise ValueError(
                "every decoupling pulse must act on one qubit, but "
                f"{gate!r} acts on {gate.num_qubits}."
            )


def _validate_spacing(spacing: int, /) -> None:
    r"""Validate an integer pulse spacing."""
    if isinstance(spacing, bool) or not isinstance(spacing, int):
        raise TypeError("spacing must be an integer")


def _validate_slack_length(slack_length: int, /) -> None:
    r"""Validate a non-negative slack length."""
    if isinstance(slack_length, bool) or not isinstance(slack_length, int):
        raise TypeError("slack_length must be an integer")
    if slack_length < 0:
        raise ValueError(
            f"slack_length must not be negative but was {slack_length!r}."
        )


def _validate_pulse_identity(gates: Sequence[StandardGate], /) -> None:
    r"""Validate the ordered pulse product.

    The idling qubit must return to its original state once the sequence is
    over, so the pulses are applied in order and their product is compared to
    the identity up to a global phase.
    """
    product: NDArray[np.complex128] = np.eye(2, dtype=np.complex128)
    for gate in gates:
        product = gate.matrix() @ product
    if not allclose_up_to_global_phase(product, np.eye(2, dtype=np.complex128)):
        raise ValueError(
            "input rule violates the condition: the pulses must multiply to "
            "the identity up to a global phase, see Viola and Lloyd 1998 in "
            "detail."
        )


# Spin echo: two pulses around a single axis refocus static dephasing.
XX = general_rule((StandardGate.X, StandardGate.X))

# XY4: alternating axes cancel the first-order error term around every axis.
XYXY = general_rule(
    (StandardGate.X, StandardGate.Y, StandardGate.X, StandardGate.Y)
)

# Twin of XX around the other transverse axis.
YY = general_rule((StandardGate.Y, StandardGate.Y))
