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

r"""Abstract base class for standard block-encoding operators."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from numbers import Real

from cqlib import Circuit


class BlockEncoding(ABC):
    r"""Represent a standard ``(alpha, a, epsilon)`` block-encoding.

    A standard block-encoding uses a unitary ``U`` such that

    .. math::

        \left\|A - \alpha (\langle 0^a | \otimes I) U (|0^a\rangle \otimes I)
        \right\| \le \epsilon.

    This abstract base class stores the public specification shared by concrete
    block-encoding operators while leaving the actual circuit synthesis to
    subclasses.

    Parameters
    ----------
    alpha
        Positive subnormalization factor.
    ancilla_qubits
        Number of ancilla qubits ``a`` used by the block-encoding.
    epsilon
        Non-negative approximation error bound.
    matrix_shape
        Shape ``(rows, columns)`` of the encoded matrix ``A``.
    system_qubits
        Width of the system register. When omitted, the minimal width able to
        contain both matrix dimensions is inferred.

    Raises
    ------
    TypeError
        If any argument has an invalid type.
    ValueError
        If any argument is outside its valid range.
    """

    __slots__ = (
        "_alpha",
        "_ancilla_qubits",
        "_epsilon",
        "_matrix_shape",
        "_system_qubits",
    )

    _alpha: float
    _ancilla_qubits: int
    _epsilon: float
    _matrix_shape: tuple[int, int]
    _system_qubits: int

    def __init__(
        self,
        *,
        alpha: float,
        ancilla_qubits: int,
        epsilon: float,
        matrix_shape: tuple[int, int],
        system_qubits: int | None = None,
    ) -> None:
        self._alpha = _validate_positive_real(alpha, name="alpha")
        self._ancilla_qubits = _validate_non_negative_integer(
            ancilla_qubits,
            name="ancilla_qubits",
        )
        self._epsilon = _validate_non_negative_real(epsilon, name="epsilon")
        self._matrix_shape = _validate_matrix_shape(matrix_shape)

        minimum_system_qubits = _minimum_system_qubits(
            max(self._matrix_shape[0], self._matrix_shape[1])
        )
        if system_qubits is None:
            self._system_qubits = minimum_system_qubits
        else:
            system_qubits_int = _validate_non_negative_integer(
                system_qubits,
                name="system_qubits",
            )
            if system_qubits_int < minimum_system_qubits:
                raise ValueError(
                    "system_qubits must be large enough to contain matrix_shape"
                )
            self._system_qubits = system_qubits_int

    @property
    def alpha(self) -> float:
        r"""Return the subnormalization factor ``alpha``."""
        return self._alpha

    @property
    def ancilla_qubits(self) -> int:
        r"""Return the ancilla width ``a``."""
        return self._ancilla_qubits

    @property
    def epsilon(self) -> float:
        r"""Return the block-encoding error bound ``epsilon``."""
        return self._epsilon

    @property
    def matrix_shape(self) -> tuple[int, int]:
        r"""Return the encoded matrix shape ``(rows, columns)``."""
        return self._matrix_shape

    @property
    def system_qubits(self) -> int:
        r"""Return the width of the system register."""
        return self._system_qubits

    @property
    def total_qubits(self) -> int:
        r"""Return the total circuit width."""
        return self.ancilla_qubits + self.system_qubits

    @property
    def spec(self) -> tuple[float, int, float]:
        r"""Return the standard ``(alpha, a, epsilon)`` specification tuple."""
        return (self.alpha, self.ancilla_qubits, self.epsilon)

    @property
    def is_standard(self) -> bool:
        r"""Return whether the block-encoding uses the standard zero projector."""
        return True

    @property
    def left_projector_dimension(self) -> int:
        r"""Return the left projected matrix dimension."""
        return self.matrix_shape[0]

    @property
    def right_projector_dimension(self) -> int:
        r"""Return the right projected matrix dimension."""
        return self.matrix_shape[1]

    @abstractmethod
    def build_circuit(self) -> Circuit:
        r"""Build and return a fresh block-encoding circuit."""

    def build_adjoint_circuit(self) -> Circuit:
        r"""Build and return a fresh adjoint circuit."""
        return self.build_circuit().inverse()


def _validate_positive_real(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real finite scalar")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise TypeError(f"{name} must be a real finite scalar")
    if scalar <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return scalar


def _validate_non_negative_real(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real finite scalar")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise TypeError(f"{name} must be a real finite scalar")
    if scalar < 0.0:
        raise ValueError(f"{name} must be greater than or equal to zero")
    return scalar


def _validate_non_negative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be greater than or equal to zero")
    return value


def _validate_matrix_shape(matrix_shape: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(matrix_shape, tuple) or len(matrix_shape) != 2:
        raise TypeError("matrix_shape must be a tuple of two integers")

    rows = _validate_positive_shape_entry(matrix_shape[0], name="matrix_shape[0]")
    columns = _validate_positive_shape_entry(
        matrix_shape[1],
        name="matrix_shape[1]",
    )
    return (rows, columns)


def _validate_positive_shape_entry(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _minimum_system_qubits(max_dimension: int) -> int:
    r"""Return the minimum qubit count able to index ``max_dimension`` rows/columns."""
    return (max_dimension - 1).bit_length()
