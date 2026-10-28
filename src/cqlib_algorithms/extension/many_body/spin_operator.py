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

"""Sparse physical-spin operators for many-body model construction."""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from typing import Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

SpinAxis: TypeAlias = Literal["X", "Y", "Z"]
SpinFactor: TypeAlias = tuple[SpinAxis, int]
SpinTerm: TypeAlias = tuple[SpinFactor, ...]
SpinCoefficient: TypeAlias = complex | float | int
SpinTermEntry: TypeAlias = tuple[SpinTerm, complex]


def _normalize_num_spins(num_spins: int) -> int:
    """Validate and return a positive spin-register width."""
    if type(num_spins) is not int:
        raise TypeError("num_spins must be an int")
    if num_spins < 1:
        raise ValueError("num_spins must be at least 1")
    return num_spins


def _normalize_spin(spin: Fraction | float) -> Fraction:
    """Validate and return a positive integer or half-integer spin."""
    if isinstance(spin, (bool, np.bool_)) or not isinstance(spin, numbers.Real):
        raise TypeError("spin must be a real integer or half-integer")
    if not math.isfinite(float(spin)):
        raise ValueError("spin must be finite")
    value = Fraction(spin)
    if value <= 0 or value.denominator not in (1, 2):
        raise ValueError("spin must be a positive integer or half-integer")
    return value


def _normalize_coefficient(value: SpinCoefficient, *, name: str) -> complex:
    """Validate and return a finite complex coefficient."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Complex):
        raise TypeError(f"{name} must be a real or complex number")
    result = complex(value)
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise ValueError(f"{name} must be finite")
    return result


def _canonicalize_term(
    term: Iterable[SpinFactor],
    *,
    num_spins: int,
    position: int,
) -> SpinTerm:
    try:
        factors = tuple(term)
    except TypeError as exc:
        raise TypeError(f"terms[{position}][0] must be iterable") from exc

    validated: list[SpinFactor] = []
    for factor_position, factor in enumerate(factors):
        if isinstance(factor, (str, bytes)) or not isinstance(factor, Sequence):
            raise TypeError(
                f"terms[{position}][0][{factor_position}] must be an axis-index pair"
            )
        if len(factor) != 2:
            raise ValueError(
                f"terms[{position}][0][{factor_position}] must contain two items"
            )
        axis = factor[0]
        index = factor[1]
        if axis not in ("X", "Y", "Z"):
            raise ValueError(
                f"terms[{position}][0][{factor_position}][0] must be X, Y, or Z"
            )
        if type(index) is not int:
            raise TypeError(
                f"terms[{position}][0][{factor_position}][1] must be an int"
            )
        if index < 0 or index >= num_spins:
            raise ValueError(
                f"terms[{position}][0][{factor_position}][1] is out of range"
            )
        validated.append((axis, index))

    # Operators on different sites commute. Stable sorting groups each site
    # while preserving the physically relevant order within the same site.
    validated.sort(key=lambda item: item[1])
    return tuple(validated)


def _copy_terms(
    num_spins: int,
    entries: Iterable[tuple[Iterable[SpinFactor], SpinCoefficient]],
) -> tuple[SpinTermEntry, ...]:
    try:
        copied = tuple(entries)
    except TypeError as exc:
        raise TypeError("terms must be iterable") from exc

    merged: dict[SpinTerm, complex] = {}
    for position, entry in enumerate(copied):
        if isinstance(entry, (str, bytes)) or not isinstance(entry, Sequence):
            raise TypeError(f"terms[{position}] must be a term-coefficient pair")
        if len(entry) != 2:
            raise ValueError(f"terms[{position}] must contain exactly two items")
        term = _canonicalize_term(
            entry[0],
            num_spins=num_spins,
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
            key=lambda item: tuple((index, axis) for axis, index in item[0]),
        )
    )


def _single_spin_matrices(
    spin: Fraction,
) -> dict[SpinAxis, NDArray[np.complex128]]:
    dimension = 2 * spin.numerator // spin.denominator + 1
    raising = np.zeros((dimension, dimension), dtype=np.complex128)
    spin_float = float(spin)
    for column in range(1, dimension):
        magnetic_number = spin_float - column
        raising[column - 1, column] = math.sqrt(
            spin_float * (spin_float + 1.0) - magnetic_number * (magnetic_number + 1.0)
        )
    lowering = raising.conjugate().T
    return {
        "X": 0.5 * (raising + lowering),
        "Y": (raising - lowering) / (2.0j),
        "Z": np.diag([spin_float - position for position in range(dimension)]).astype(
            np.complex128
        ),
    }


@dataclass(frozen=True, slots=True)
class SpinOp:
    """Immutable sparse sum of products of physical spin operators.

    ``X``, ``Y`` and ``Z`` denote :math:`S^x`, :math:`S^y` and :math:`S^z`.
    For spin one half these are one half of the corresponding Pauli matrices.
    Factors on different sites are canonically reordered, while factor order on
    the same site is retained because those factors need not commute.

    Args:
        num_spins: Positive number of physical spin sites.
        terms: Sparse products and finite complex coefficients.
        spin: Positive integer or half-integer spin quantum number.

    Raises:
        TypeError: If a count, factor, or coefficient has the wrong type.
        ValueError: If indices, coefficients, or ``spin`` are invalid.
    """

    num_spins: int
    terms: tuple[SpinTermEntry, ...]
    spin: Fraction = Fraction(1, 2)

    def __post_init__(self) -> None:
        num_spins = _normalize_num_spins(self.num_spins)
        spin = _normalize_spin(self.spin)
        terms = _copy_terms(num_spins, self.terms)
        object.__setattr__(self, "spin", spin)
        object.__setattr__(self, "terms", terms)

    @classmethod
    def from_terms(
        cls,
        num_spins: int,
        terms: Iterable[tuple[Iterable[SpinFactor], SpinCoefficient]],
        *,
        spin: Fraction | float = Fraction(1, 2),
    ) -> SpinOp:
        """Construct a validated sparse spin operator."""
        return cls(
            num_spins=num_spins,
            terms=cast(tuple[SpinTermEntry, ...], terms),
            spin=cast(Fraction, spin),
        )

    @classmethod
    def zero(
        cls,
        num_spins: int,
        *,
        spin: Fraction | float = Fraction(1, 2),
    ) -> SpinOp:
        """Return the zero operator at a fixed register width."""
        return cls.from_terms(num_spins, (), spin=spin)

    @classmethod
    def identity(
        cls,
        num_spins: int,
        *,
        spin: Fraction | float = Fraction(1, 2),
    ) -> SpinOp:
        """Return the identity operator at a fixed register width."""
        return cls.from_terms(num_spins, (((), 1.0),), spin=spin)

    @property
    def num_terms(self) -> int:
        """Return the number of nonzero canonical terms."""
        return len(self.terms)

    def is_zero(self) -> bool:
        """Return whether no nonzero terms are stored."""
        return not self.terms

    def scaled(self, factor: SpinCoefficient) -> SpinOp:
        """Return this operator multiplied by a finite scalar."""
        factor = _normalize_coefficient(factor, name="factor")
        return SpinOp.from_terms(
            self.num_spins,
            ((term, factor * coefficient) for term, coefficient in self.terms),
            spin=self.spin,
        )

    def add(self, other: SpinOp) -> SpinOp:
        """Return the sum of compatible spin operators."""
        self._validate_compatible(other)
        return SpinOp.from_terms(
            self.num_spins,
            (*self.terms, *other.terms),
            spin=self.spin,
        )

    def compose(self, other: SpinOp) -> SpinOp:
        """Return the operator product ``self * other``."""
        self._validate_compatible(other)
        return SpinOp.from_terms(
            self.num_spins,
            (
                (left_term + right_term, left_coefficient * right_coefficient)
                for left_term, left_coefficient in self.terms
                for right_term, right_coefficient in other.terms
            ),
            spin=self.spin,
        )

    def adjoint(self) -> SpinOp:
        """Return the conjugate transpose."""
        return SpinOp.from_terms(
            self.num_spins,
            (
                (tuple(reversed(term)), coefficient.conjugate())
                for term, coefficient in self.terms
            ),
            spin=self.spin,
        )

    def chopped(self, *, atol: float = 0.0) -> SpinOp:
        """Return a copy without coefficients of magnitude at most ``atol``."""
        if isinstance(atol, (bool, np.bool_)) or not isinstance(atol, numbers.Real):
            raise TypeError("atol must be a real number")
        if not math.isfinite(float(atol)) or atol < 0:
            raise ValueError("atol must be finite and non-negative")
        return SpinOp.from_terms(
            self.num_spins,
            (
                (term, coefficient)
                for term, coefficient in self.terms
                if abs(coefficient) > float(atol)
            ),
            spin=self.spin,
        )

    def equiv(self, other: SpinOp, *, atol: float = 0.0) -> bool:
        """Compare represented operators within an absolute tolerance.

        The fallback matrix comparison is necessary because distinct ordered
        products on one site may represent the same spin operator. This helper
        is therefore intended for small-system validation, like ``to_matrix``.
        """
        if not isinstance(other, SpinOp):
            return False
        if self.num_spins != other.num_spins or self.spin != other.spin:
            return False
        if isinstance(atol, (bool, np.bool_)) or not isinstance(atol, numbers.Real):
            raise TypeError("atol must be a real number")
        if not math.isfinite(float(atol)) or atol < 0:
            raise ValueError("atol must be finite and non-negative")
        left = dict(self.terms)
        right = dict(other.terms)
        keys = left.keys() | right.keys()
        if all(abs(left.get(key, 0.0j) - right.get(key, 0.0j)) <= atol for key in keys):
            return True
        return bool(
            np.allclose(
                self.to_matrix(),
                other.to_matrix(),
                rtol=0.0,
                atol=float(atol),
            )
        )

    def is_hermitian(self, *, atol: float = 0.0) -> bool:
        """Return whether this operator equals its adjoint.

        Distinct ordered products on one site can represent the same operator,
        so an inconclusive sparse comparison uses the same dense small-system
        fallback as :meth:`equiv`. This method is not a scalable many-spin
        certification routine.
        """
        return self.equiv(self.adjoint(), atol=atol)

    def to_matrix(self) -> NDArray[np.complex128]:
        """Return the dense matrix, intended for small-system verification."""
        matrices = _single_spin_matrices(self.spin)
        local_dimension = next(iter(matrices.values())).shape[0]
        identity = np.eye(local_dimension, dtype=np.complex128)
        full_dimension = local_dimension**self.num_spins
        result = np.zeros((full_dimension, full_dimension), dtype=np.complex128)

        for term, coefficient in self.terms:
            local_operators = [identity.copy() for _ in range(self.num_spins)]
            for axis, index in term:
                local_operators[index] = local_operators[index] @ matrices[axis]
            tensor = reduce(np.kron, reversed(local_operators))
            result += coefficient * tensor
        return result

    def _validate_compatible(self, other: SpinOp) -> None:
        if not isinstance(other, SpinOp):
            raise TypeError("other must be a SpinOp")
        if self.num_spins != other.num_spins:
            raise ValueError("spin operators must have the same num_spins")
        if self.spin != other.spin:
            raise ValueError("spin operators must have the same spin value")

    def __add__(self, other: SpinOp, /) -> SpinOp:
        return self.add(other)

    def __sub__(self, other: SpinOp, /) -> SpinOp:
        self._validate_compatible(other)
        return self.add(other.scaled(-1.0))

    def __mul__(self, factor: SpinCoefficient, /) -> SpinOp:
        return self.scaled(factor)

    def __rmul__(self, factor: SpinCoefficient, /) -> SpinOp:
        return self.scaled(factor)

    def __matmul__(self, other: SpinOp, /) -> SpinOp:
        return self.compose(other)

    def __neg__(self) -> SpinOp:
        return self.scaled(-1.0)
