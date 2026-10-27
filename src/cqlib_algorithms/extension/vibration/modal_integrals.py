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

"""Sparse modal-basis integrals for finite vibrational systems."""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np

from ..many_body import VibrationalCoefficient, VibrationalLadder, VibrationalOp
from .modal_space import ModalSpace

_HERMITIAN_ATOL = 1e-12


@dataclass(frozen=True, slots=True, order=True)
class ModalTransition:
    r"""One matrix unit :math:`|bra\rangle\langle ket|` in a mode.

    Args:
        mode: Non-negative vibrational-mode index.
        bra_modal: Non-negative target modal index.
        ket_modal: Non-negative source modal index.

    Raises:
        TypeError: If an index is not an integer.
        ValueError: If an index is negative.
    """

    mode: int
    bra_modal: int
    ket_modal: int

    def __post_init__(self) -> None:
        for name, value in (
            ("mode", self.mode),
            ("bra_modal", self.bra_modal),
            ("ket_modal", self.ket_modal),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an int")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

    def adjoint(self) -> ModalTransition:
        """Return the conjugate-transposed modal matrix unit."""
        return ModalTransition(self.mode, self.ket_modal, self.bra_modal)


ModalIntegralTerm = tuple[ModalTransition, ...]
ModalIntegralEntry = tuple[ModalIntegralTerm, complex]


def _normalize_coefficient(
    coefficient: VibrationalCoefficient,
    *,
    name: str,
) -> complex:
    """Validate and return a finite complex coefficient."""
    if isinstance(coefficient, (bool, np.bool_)) or not isinstance(
        coefficient, numbers.Complex
    ):
        raise TypeError(f"{name} must be a complex number")
    converted = complex(coefficient)
    if not math.isfinite(converted.real) or not math.isfinite(converted.imag):
        raise ValueError(f"{name} must be finite")
    return converted


def _normalize_constant(constant: float) -> float:
    """Validate and return a finite real constant."""
    if isinstance(constant, (bool, np.bool_)) or not isinstance(constant, numbers.Real):
        raise TypeError("constant must be a real number")
    converted = float(constant)
    if not math.isfinite(converted):
        raise ValueError("constant must be finite")
    return converted


def _canonicalize_term(
    modal_space: ModalSpace,
    term: Iterable[ModalTransition],
    *,
    position: int,
) -> ModalIntegralTerm:
    if isinstance(term, (str, bytes)):
        raise TypeError(f"terms[{position}][0] must be an iterable of transitions")
    try:
        copied = tuple(term)
    except TypeError as exc:
        raise TypeError(
            f"terms[{position}][0] must be an iterable of transitions"
        ) from exc
    if not copied:
        raise ValueError("modal integral terms must contain at least one transition")

    seen_modes: set[int] = set()
    for transition_index, transition in enumerate(copied):
        if not isinstance(transition, ModalTransition):
            raise TypeError(
                f"terms[{position}][0][{transition_index}] must be a ModalTransition"
            )
        modal_space.qubit_index(transition.mode, transition.bra_modal)
        modal_space.qubit_index(transition.mode, transition.ket_modal)
        if transition.mode in seen_modes:
            raise ValueError("a modal integral term may contain each mode at most once")
        seen_modes.add(transition.mode)
    return tuple(sorted(copied, key=lambda transition: transition.mode))


def _copy_terms(
    modal_space: ModalSpace,
    entries: Iterable[tuple[Iterable[ModalTransition], VibrationalCoefficient]],
) -> tuple[ModalIntegralEntry, ...]:
    if isinstance(entries, (str, bytes)):
        raise TypeError("terms must be iterable")
    try:
        copied = tuple(entries)
    except TypeError as exc:
        raise TypeError("terms must be iterable") from exc

    merged: dict[ModalIntegralTerm, complex] = {}
    for position, entry in enumerate(copied):
        if isinstance(entry, (str, bytes)) or not isinstance(entry, Sequence):
            raise TypeError(f"terms[{position}] must be a term-coefficient pair")
        if len(entry) != 2:
            raise ValueError(f"terms[{position}] must contain exactly two items")
        term = _canonicalize_term(modal_space, entry[0], position=position)
        coefficient = _normalize_coefficient(
            entry[1],
            name=f"terms[{position}][1]",
        )
        combined = merged.get(term, 0.0j) + coefficient
        if not math.isfinite(combined.real) or not math.isfinite(combined.imag):
            raise ValueError("combined modal integral coefficients must be finite")
        merged[term] = combined

    return tuple(
        sorted(
            (
                (term, coefficient)
                for term, coefficient in merged.items()
                if coefficient != 0.0
            ),
            key=lambda item: item[0],
        )
    )


def _adjoint_term(term: ModalIntegralTerm) -> ModalIntegralTerm:
    return tuple(sorted(transition.adjoint() for transition in term))


def _is_hermitian(
    terms: tuple[ModalIntegralEntry, ...],
    *,
    atol: float,
) -> bool:
    coefficients = dict(terms)
    return all(
        abs(coefficient - coefficients.get(_adjoint_term(term), 0.0j).conjugate())
        <= atol
        for term, coefficient in terms
    )


def _project_hermitian_terms(
    terms: tuple[ModalIntegralEntry, ...],
    *,
    atol: float,
) -> tuple[ModalIntegralEntry, ...]:
    """Validate and project sparse terms onto an exactly Hermitian snapshot."""
    coefficients = dict(terms)
    projected: dict[ModalIntegralTerm, complex] = {}
    visited: set[ModalIntegralTerm] = set()
    for term, coefficient in terms:
        if term in visited:
            continue
        adjoint = _adjoint_term(term)
        adjoint_coefficient = coefficients.get(adjoint, 0.0j)
        residual = coefficient - adjoint_coefficient.conjugate()
        if (
            not math.isfinite(residual.real)
            or not math.isfinite(residual.imag)
            or math.hypot(residual.real, residual.imag) > atol
        ):
            raise ValueError(
                "modal integrals must satisfy Hermitian conjugate symmetry"
            )

        if term == adjoint:
            value = complex(coefficient.real, 0.0)
            if value != 0.0:
                projected[term] = value
        else:
            conjugate_adjoint = adjoint_coefficient.conjugate()
            value = 0.5 * coefficient + 0.5 * conjugate_adjoint
            if not math.isfinite(value.real) or not math.isfinite(value.imag):
                raise ValueError("projected modal integral coefficients must be finite")
            if value != 0.0:
                projected[term] = value
                projected[adjoint] = value.conjugate()
        visited.add(term)
        visited.add(adjoint)
    return tuple(sorted(projected.items(), key=lambda item: item[0]))


@dataclass(frozen=True, slots=True)
class ModalIntegrals:
    r"""Hermitian modal integrals without a general-purpose tensor layer.

    A term containing transitions for modes :math:`m_1,\ldots,m_r` stores the
    coefficient of

    .. math::

       \prod_j a^\dagger_{m_j,bra_j} a_{m_j,ket_j}.

    Hence term length is the coupling order. Each mode may occur at most once
    in a term. One-mode, two-mode, and higher-order couplings share the same
    sparse representation.

    Args:
        modal_space: Finite modal basis for all transition indices.
        terms: Sparse modal transition products and coefficients.
        constant: Finite real scalar energy offset.

    Raises:
        TypeError: If an input has the wrong type.
        ValueError: If indices, coefficients, or Hermitian symmetry are invalid.
    """

    modal_space: ModalSpace
    terms: tuple[ModalIntegralEntry, ...]
    constant: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.modal_space, ModalSpace):
            raise TypeError("modal_space must be a ModalSpace")
        copied = _copy_terms(self.modal_space, self.terms)
        object.__setattr__(
            self,
            "terms",
            _project_hermitian_terms(copied, atol=_HERMITIAN_ATOL),
        )
        object.__setattr__(self, "constant", _normalize_constant(self.constant))

    @classmethod
    def from_terms(
        cls,
        modal_space: ModalSpace,
        terms: Iterable[tuple[Iterable[ModalTransition], VibrationalCoefficient]],
        *,
        constant: float = 0.0,
    ) -> ModalIntegrals:
        """Construct validated sparse modal integrals."""
        return cls(
            modal_space,
            cast(tuple[ModalIntegralEntry, ...], terms),
            constant,
        )

    @property
    def max_order(self) -> int:
        """Return the largest stored mode-coupling order, or zero."""
        return max((len(term) for term, _ in self.terms), default=0)

    def terms_of_order(self, order: int) -> tuple[ModalIntegralEntry, ...]:
        """Return the canonical terms of one coupling order."""
        if type(order) is not int:
            raise TypeError("order must be an int")
        if order < 1:
            raise ValueError("order must be at least 1")
        return tuple(entry for entry in self.terms if len(entry[0]) == order)

    def truncated(self, max_order: int) -> ModalIntegrals:
        """Return integrals containing at most ``max_order`` coupled modes."""
        if type(max_order) is not int:
            raise TypeError("max_order must be an int")
        if max_order < 1 or max_order > self.modal_space.num_modes:
            raise ValueError("max_order must be between 1 and the number of modes")
        return ModalIntegrals.from_terms(
            self.modal_space,
            (entry for entry in self.terms if len(entry[0]) <= max_order),
            constant=self.constant,
        )

    def is_hermitian(self, *, atol: float = _HERMITIAN_ATOL) -> bool:
        """Return whether all integral coefficients obey conjugate symmetry."""
        if isinstance(atol, (bool, np.bool_)) or not isinstance(atol, numbers.Real):
            raise TypeError("atol must be a real number")
        threshold = float(atol)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("atol must be finite and non-negative")
        return _is_hermitian(self.terms, atol=threshold)

    def to_vibrational_op(self) -> VibrationalOp:
        """Return the exact sparse second-quantized operator."""
        entries: list[tuple[tuple[VibrationalLadder, ...], VibrationalCoefficient]] = []
        if self.constant != 0.0:
            entries.append(((), self.constant))
        for term, coefficient in self.terms:
            ladders: list[VibrationalLadder] = []
            for transition in term:
                ladders.extend(
                    (
                        VibrationalLadder("+", transition.mode, transition.bra_modal),
                        VibrationalLadder("-", transition.mode, transition.ket_modal),
                    )
                )
            entries.append((tuple(ladders), coefficient))
        return VibrationalOp.from_terms(self.modal_space, entries)


__all__ = [
    "ModalIntegralEntry",
    "ModalIntegralTerm",
    "ModalIntegrals",
    "ModalTransition",
]
