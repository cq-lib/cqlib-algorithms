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

"""Sparse operators on a finite vibrational modal space."""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..vibration.modal_space import ModalSpace

VibrationalLadderKind = Literal["+", "-"]
VibrationalCoefficient = complex | float | int


def _validate_non_negative_index(value: int, *, name: str) -> None:
    """Validate a non-negative integer index."""
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


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


def _normalize_atol(atol: float) -> float:
    """Validate and return a finite non-negative tolerance."""
    if isinstance(atol, (bool, np.bool_)) or not isinstance(atol, numbers.Real):
        raise TypeError("atol must be a real number")
    converted = float(atol)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError("atol must be finite and non-negative")
    return converted


@dataclass(frozen=True, slots=True, order=True)
class VibrationalLadder:
    """One modal creation or annihilation operator.

    Args:
        kind: ``"+"`` for creation or ``"-"`` for annihilation.
        mode: Non-negative vibrational-mode index.
        modal: Non-negative modal index within the mode.

    Raises:
        TypeError: If an index is not an integer.
        ValueError: If ``kind`` or an index is invalid.
    """

    kind: VibrationalLadderKind
    mode: int
    modal: int

    def __post_init__(self) -> None:
        if self.kind not in ("+", "-"):
            raise ValueError("kind must be '+' or '-'")
        _validate_non_negative_index(self.mode, name="mode")
        _validate_non_negative_index(self.modal, name="modal")

    def adjoint(self) -> VibrationalLadder:
        """Return the opposite ladder operation at the same mode/modal."""
        return VibrationalLadder(
            "-" if self.kind == "+" else "+", self.mode, self.modal
        )


VibrationalTerm = tuple[VibrationalLadder, ...]
VibrationalTermEntry = tuple[VibrationalTerm, complex]


def _canonicalize_term(
    term: Iterable[VibrationalLadder],
    *,
    modal_space: ModalSpace,
    position: int,
) -> VibrationalTerm:
    if isinstance(term, (str, bytes)):
        raise TypeError(f"terms[{position}][0] must be an iterable of ladders")
    try:
        copied = tuple(term)
    except TypeError as exc:
        raise TypeError(f"terms[{position}][0] must be an iterable of ladders") from exc
    for factor_index, ladder in enumerate(copied):
        if not isinstance(ladder, VibrationalLadder):
            raise TypeError(
                f"terms[{position}][0][{factor_index}] must be a VibrationalLadder"
            )
        modal_space.qubit_index(ladder.mode, ladder.modal)

    # Operators on different modal qubits commute. Python's stable sort keeps
    # the physically significant order of repeated factors on the same modal.
    return tuple(sorted(copied, key=lambda ladder: (ladder.mode, ladder.modal)))


def _copy_terms(
    modal_space: ModalSpace,
    entries: Iterable[tuple[Iterable[VibrationalLadder], VibrationalCoefficient]],
) -> tuple[VibrationalTermEntry, ...]:
    if isinstance(entries, (str, bytes)):
        raise TypeError("terms must be iterable")
    try:
        copied = tuple(entries)
    except TypeError as exc:
        raise TypeError("terms must be iterable") from exc

    merged: dict[VibrationalTerm, complex] = {}
    for position, entry in enumerate(copied):
        if isinstance(entry, (str, bytes)) or not isinstance(entry, Sequence):
            raise TypeError(f"terms[{position}] must be a term-coefficient pair")
        if len(entry) != 2:
            raise ValueError(f"terms[{position}] must contain exactly two items")
        term = _canonicalize_term(
            entry[0],
            modal_space=modal_space,
            position=position,
        )
        coefficient = _normalize_coefficient(
            entry[1],
            name=f"terms[{position}][1]",
        )
        merged[term] = merged.get(term, 0.0j) + coefficient

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


@dataclass(frozen=True, slots=True)
class VibrationalOp:
    """Immutable sparse sum of modal ladder products.

    Factors are written in ordinary operator-product order, so the rightmost
    factor acts first. Different mode/modal coordinates commute; factor order
    on the same coordinate is retained.

    Args:
        modal_space: Finite modal basis defining every ladder coordinate.
        terms: Sparse ladder products and finite complex coefficients.

    Raises:
        TypeError: If the modal space, terms, ladders, or coefficients are invalid.
        ValueError: If ladder indices or coefficient values are invalid.
    """

    modal_space: ModalSpace
    terms: tuple[VibrationalTermEntry, ...]

    def __post_init__(self) -> None:
        from ..vibration.modal_space import ModalSpace

        if not isinstance(self.modal_space, ModalSpace):
            raise TypeError("modal_space must be a ModalSpace")
        object.__setattr__(self, "terms", _copy_terms(self.modal_space, self.terms))

    @classmethod
    def from_terms(
        cls,
        modal_space: ModalSpace,
        terms: Iterable[tuple[Iterable[VibrationalLadder], VibrationalCoefficient]],
    ) -> VibrationalOp:
        """Construct a validated sparse vibrational operator."""
        return cls(
            modal_space,
            cast(tuple[VibrationalTermEntry, ...], terms),
        )

    @classmethod
    def zero(cls, modal_space: ModalSpace) -> VibrationalOp:
        """Return the zero operator on a fixed modal space."""
        return cls.from_terms(modal_space, ())

    @classmethod
    def identity(cls, modal_space: ModalSpace) -> VibrationalOp:
        """Return the identity of the full one-hot qubit register."""
        return cls.from_terms(modal_space, (((), 1.0),))

    @property
    def num_terms(self) -> int:
        """Return the number of nonzero canonical terms."""
        return len(self.terms)

    def is_zero(self) -> bool:
        """Return whether no nonzero terms are stored."""
        return not self.terms

    def normalized(self, *, atol: float = 0.0) -> VibrationalOp:
        """Return canonical sparse form with small coefficients removed.

        This operation does not rescale coefficients or change the represented
        operator. Construction already merges exact duplicate terms; ``atol``
        optionally removes numerical residuals.
        """
        threshold = _normalize_atol(atol)
        return VibrationalOp.from_terms(
            self.modal_space,
            (
                (term, coefficient)
                for term, coefficient in self.terms
                if abs(coefficient) > threshold
            ),
        )

    def scaled(self, factor: VibrationalCoefficient) -> VibrationalOp:
        """Return this operator multiplied by a finite scalar."""
        converted = _normalize_coefficient(factor, name="factor")
        return VibrationalOp.from_terms(
            self.modal_space,
            ((term, converted * coefficient) for term, coefficient in self.terms),
        )

    def add(self, other: VibrationalOp) -> VibrationalOp:
        """Return the sum of operators on the same modal space."""
        self._validate_compatible(other)
        return VibrationalOp.from_terms(
            self.modal_space,
            (*self.terms, *other.terms),
        )

    def compose(self, other: VibrationalOp) -> VibrationalOp:
        """Return the operator product ``self * other``."""
        self._validate_compatible(other)
        return VibrationalOp.from_terms(
            self.modal_space,
            (
                (left_term + right_term, left_coefficient * right_coefficient)
                for left_term, left_coefficient in self.terms
                for right_term, right_coefficient in other.terms
            ),
        )

    def adjoint(self) -> VibrationalOp:
        """Return the conjugate transpose."""
        return VibrationalOp.from_terms(
            self.modal_space,
            (
                (
                    tuple(ladder.adjoint() for ladder in reversed(term)),
                    coefficient.conjugate(),
                )
                for term, coefficient in self.terms
            ),
        )

    def equiv(self, other: VibrationalOp, *, atol: float = 0.0) -> bool:
        """Compare canonical sparse coefficients within absolute tolerance."""
        if not isinstance(other, VibrationalOp):
            return False
        if self.modal_space != other.modal_space:
            return False
        threshold = _normalize_atol(atol)
        left = dict(self.terms)
        right = dict(other.terms)
        return all(
            abs(left.get(term, 0.0j) - right.get(term, 0.0j)) <= threshold
            for term in left.keys() | right.keys()
        )

    def is_hermitian(self, *, atol: float = 0.0) -> bool:
        """Return whether the exact one-hot qubit representation is Hermitian."""
        threshold = _normalize_atol(atol)
        # The import is local to keep the sparse representation independent of
        # the concrete mapper during module initialization.
        from .one_hot_vibrational_mapping import OneHotVibrationalMapper

        return OneHotVibrationalMapper().map_operator(self).is_hermitian(atol=threshold)

    def preserves_one_hot_subspace(self) -> bool:
        """Return whether every monomial conserves occupancy of every mode."""
        for term, _ in self.terms:
            balance = [0] * self.modal_space.num_modes
            for ladder in term:
                balance[ladder.mode] += 1 if ladder.kind == "+" else -1
            if any(value != 0 for value in balance):
                return False
        return True

    def to_modal_matrix(self) -> NDArray[np.complex128]:
        """Return the physical modal-space matrix for small-system validation."""
        dimension = self.modal_space.dimension
        result = np.zeros((dimension, dimension), dtype=np.complex128)
        encoded_to_modal = {
            encoded: modal
            for modal, encoded in enumerate(self.modal_space.physical_basis_indices)
        }

        for term, coefficient in self.terms:
            for column, encoded_input in enumerate(
                self.modal_space.physical_basis_indices
            ):
                encoded_output = encoded_input
                nonzero = True
                for ladder in reversed(term):
                    bit = 1 << self.modal_space.qubit_index(
                        ladder.mode,
                        ladder.modal,
                    )
                    occupied = bool(encoded_output & bit)
                    if ladder.kind == "+":
                        if occupied:
                            nonzero = False
                            break
                        encoded_output |= bit
                    else:
                        if not occupied:
                            nonzero = False
                            break
                        encoded_output &= ~bit
                if nonzero and encoded_output in encoded_to_modal:
                    result[encoded_to_modal[encoded_output], column] += coefficient
        return result

    def _validate_compatible(self, other: VibrationalOp) -> None:
        if not isinstance(other, VibrationalOp):
            raise TypeError("other must be a VibrationalOp")
        if self.modal_space != other.modal_space:
            raise ValueError("vibrational operators must use the same modal space")

    def __add__(self, other: VibrationalOp, /) -> VibrationalOp:
        return self.add(other)

    def __sub__(self, other: VibrationalOp, /) -> VibrationalOp:
        self._validate_compatible(other)
        return self.add(other.scaled(-1.0))

    def __mul__(self, factor: VibrationalCoefficient, /) -> VibrationalOp:
        return self.scaled(factor)

    def __rmul__(self, factor: VibrationalCoefficient, /) -> VibrationalOp:
        return self.scaled(factor)

    def __matmul__(self, other: VibrationalOp, /) -> VibrationalOp:
        return self.compose(other)

    def __neg__(self) -> VibrationalOp:
        return self.scaled(-1.0)


__all__ = [
    "VibrationalCoefficient",
    "VibrationalLadder",
    "VibrationalLadderKind",
    "VibrationalOp",
    "VibrationalTerm",
]
