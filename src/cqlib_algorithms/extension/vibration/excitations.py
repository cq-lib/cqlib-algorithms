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

"""Reference-relative excitations in finite vibrational modal spaces."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from types import MappingProxyType
from typing import cast

from ..many_body import VibrationalLadder, VibrationalOp
from .modal_integrals import ModalTransition
from .modal_space import ModalSpace


def _copy_reference_modals(
    modal_space: ModalSpace,
    reference_modals: Sequence[int] | None,
) -> tuple[int, ...]:
    if reference_modals is None:
        return (0,) * modal_space.num_modes
    if isinstance(reference_modals, (str, bytes)) or not isinstance(
        reference_modals, Sequence
    ):
        raise TypeError("reference_modals must be a sequence of integers or None")
    copied = tuple(reference_modals)
    modal_space.modal_basis_index(copied)
    return copied


def _copy_transitions(
    modal_space: ModalSpace,
    transitions: Iterable[ModalTransition],
) -> tuple[ModalTransition, ...]:
    if isinstance(transitions, (str, bytes)):
        raise TypeError("transitions must be an iterable of ModalTransition values")
    try:
        copied = tuple(transitions)
    except TypeError as exc:
        raise TypeError(
            "transitions must be an iterable of ModalTransition values"
        ) from exc
    if not copied:
        raise ValueError("an excitation must contain at least one transition")

    seen_modes: set[int] = set()
    for position, transition in enumerate(copied):
        if not isinstance(transition, ModalTransition):
            raise TypeError(f"transitions[{position}] must be a ModalTransition")
        modal_space.qubit_index(transition.mode, transition.bra_modal)
        modal_space.qubit_index(transition.mode, transition.ket_modal)
        if transition.bra_modal == transition.ket_modal:
            raise ValueError("excitation source and target modals must differ")
        if transition.mode in seen_modes:
            raise ValueError("an excitation may change each mode at most once")
        seen_modes.add(transition.mode)
    return tuple(sorted(copied, key=lambda transition: transition.mode))


@dataclass(frozen=True, slots=True)
class VibrationalExcitation:
    """A simultaneous modal substitution on distinct vibrational modes.

    Each :class:`ModalTransition` is interpreted as ``target <- source``:
    ``bra_modal`` is the target and ``ket_modal`` is the source. This keeps
    excitation data identical to the matrix-unit convention used by modal
    integrals.

    Args:
        modal_space: Finite modal basis.
        transitions: Non-empty transitions on distinct modes.

    Raises:
        TypeError: If the modal space or transitions have the wrong type.
        ValueError: If indices, source/target pairs, or mode uniqueness fail.
    """

    modal_space: ModalSpace
    transitions: tuple[ModalTransition, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.modal_space, ModalSpace):
            raise TypeError("modal_space must be a ModalSpace")
        object.__setattr__(
            self,
            "transitions",
            _copy_transitions(self.modal_space, self.transitions),
        )

    @classmethod
    def from_transitions(
        cls,
        modal_space: ModalSpace,
        transitions: Iterable[ModalTransition],
    ) -> VibrationalExcitation:
        """Construct and defensively copy an explicit custom excitation."""
        return cls(modal_space, cast(tuple[ModalTransition, ...], transitions))

    @property
    def rank(self) -> int:
        """Return the number of simultaneously changed modes."""
        return len(self.transitions)

    @property
    def sources(self) -> tuple[tuple[int, int], ...]:
        """Return ``(mode, source_modal)`` pairs in mode order."""
        return tuple(
            (transition.mode, transition.ket_modal) for transition in self.transitions
        )

    @property
    def targets(self) -> tuple[tuple[int, int], ...]:
        """Return ``(mode, target_modal)`` pairs in mode order."""
        return tuple(
            (transition.mode, transition.bra_modal) for transition in self.transitions
        )

    def apply(self, configuration: Sequence[int]) -> tuple[int, ...]:
        """Apply this substitution to one physical modal configuration."""
        if isinstance(configuration, (str, bytes)) or not isinstance(
            configuration, Sequence
        ):
            raise TypeError("configuration must be a sequence of integers")
        copied = tuple(configuration)
        self.modal_space.modal_basis_index(copied)
        output = list(copied)
        for transition in self.transitions:
            if copied[transition.mode] != transition.ket_modal:
                raise ValueError(
                    "configuration does not occupy every excitation source modal"
                )
            output[transition.mode] = transition.bra_modal
        return tuple(output)

    def adjoint(self) -> VibrationalExcitation:
        """Return the inverse modal substitution."""
        return VibrationalExcitation.from_transitions(
            self.modal_space,
            (transition.adjoint() for transition in self.transitions),
        )

    def to_vibrational_op(self) -> VibrationalOp:
        """Return the ordered modal-transfer monomial for this excitation."""
        ladders: list[VibrationalLadder] = []
        for transition in self.transitions:
            ladders.extend(
                (
                    VibrationalLadder("+", transition.mode, transition.bra_modal),
                    VibrationalLadder("-", transition.mode, transition.ket_modal),
                )
            )
        return VibrationalOp.from_terms(self.modal_space, ((tuple(ladders), 1.0),))


def generate_vibrational_single_excitations(
    modal_space: ModalSpace,
    reference_modals: Sequence[int] | None = None,
) -> tuple[VibrationalExcitation, ...]:
    """Generate every one-mode excitation away from a reference configuration.

    Args:
        modal_space: Finite modal basis.
        reference_modals: Occupied modal per mode; modal zero when omitted.

    Returns:
        Deterministically ordered single-mode excitations.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the reference configuration is invalid.
    """
    if not isinstance(modal_space, ModalSpace):
        raise TypeError("modal_space must be a ModalSpace")
    reference = _copy_reference_modals(modal_space, reference_modals)
    return tuple(
        VibrationalExcitation.from_transitions(
            modal_space,
            (ModalTransition(mode, target, reference[mode]),),
        )
        for mode, count in enumerate(modal_space.num_modals)
        for target in range(count)
        if target != reference[mode]
    )


def generate_vibrational_double_excitations(
    modal_space: ModalSpace,
    reference_modals: Sequence[int] | None = None,
) -> tuple[VibrationalExcitation, ...]:
    """Generate every two-distinct-mode excitation from a reference.

    Args:
        modal_space: Finite modal basis.
        reference_modals: Occupied modal per mode; modal zero when omitted.

    Returns:
        Deterministically ordered double-mode excitations.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the reference configuration is invalid.
    """
    singles = generate_vibrational_single_excitations(
        modal_space,
        reference_modals,
    )
    return tuple(
        VibrationalExcitation.from_transitions(
            modal_space,
            (*left.transitions, *right.transitions),
        )
        for left, right in combinations(singles, 2)
        if left.transitions[0].mode != right.transitions[0].mode
    )


def vscf_initial_point(
    excitations: Sequence[VibrationalExcitation],
    *,
    prefix: str = "uvcc",
    reps: int = 1,
) -> Mapping[str, float]:
    """Return zero amplitudes that leave a repeated UVCC circuit unchanged.

    Args:
        excitations: Compatible vibrational excitation sequence.
        prefix: Prefix used by the corresponding UVCC circuit.
        reps: Non-negative number of repeated excitation layers.

    Returns:
        Immutable mapping from every UVCC parameter to zero.

    Raises:
        TypeError: If an argument or excitation has the wrong type.
        ValueError: If repetitions are negative or modal spaces disagree.
    """
    if isinstance(excitations, (str, bytes)) or not isinstance(excitations, Sequence):
        raise TypeError(
            "excitations must be a sequence of VibrationalExcitation values"
        )
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a str")
    if type(reps) is not int:
        raise TypeError("reps must be an int")
    if reps < 0:
        raise ValueError("reps must be non-negative")
    copied = tuple(excitations)
    if any(not isinstance(excitation, VibrationalExcitation) for excitation in copied):
        raise TypeError("excitations must contain only VibrationalExcitation values")
    if copied:
        modal_space = copied[0].modal_space
        if any(excitation.modal_space != modal_space for excitation in copied[1:]):
            raise ValueError("all excitations must use the same modal space")
    return MappingProxyType(
        {f"{prefix}_theta_{index}": 0.0 for index in range(reps * len(copied))}
    )


__all__ = [
    "VibrationalExcitation",
    "generate_vibrational_double_excitations",
    "generate_vibrational_single_excitations",
    "vscf_initial_point",
]
