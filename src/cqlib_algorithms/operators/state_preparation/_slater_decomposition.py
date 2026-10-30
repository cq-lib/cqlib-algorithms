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

"""Private validation and adjacent-mode decomposition for Slater states."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from cqlib.circuit import Circuit
from cqlib.circuit.gates import StandardGate
from numpy.typing import ArrayLike, NDArray

_ORTHONORMALITY_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class _AdjacentGivens:
    """An adjacent-mode SU(2) block [[alpha,beta],[-beta*,alpha*]]."""

    left: int
    right: int
    alpha: complex
    beta: complex


def snapshot_orbital_coefficients(
    value: ArrayLike,
) -> NDArray[np.complex128]:
    """Return a read-only complex isometry snapshot."""
    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "orbital_coefficients must be a rectangular complex matrix"
        ) from error
    if source.ndim != 2:
        raise ValueError("orbital_coefficients must be two-dimensional")

    try:
        object_values = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "orbital_coefficients must be a rectangular complex matrix"
        ) from error
    if any(isinstance(item, (bool, np.bool_)) for item in object_values.flat):
        raise TypeError("orbital_coefficients must not contain bool values")

    try:
        copied = np.array(value, dtype=np.complex128, copy=True)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "orbital_coefficients must be a rectangular complex matrix"
        ) from error
    _validate_orbital_coefficients(copied)
    return np.frombuffer(copied.tobytes(), dtype=np.complex128).reshape(copied.shape)


def validate_orbital_coefficients(value: NDArray[np.complex128]) -> None:
    """Validate an existing orbital-coefficient snapshot without changing it."""
    _validate_orbital_coefficients(value)


def _validate_orbital_coefficients(value: Any) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError("orbital_coefficients must be a numpy array")
    if value.dtype != np.dtype(np.complex128):
        raise TypeError("orbital_coefficients must use complex128 values")
    if value.ndim != 2:
        raise ValueError("orbital_coefficients must be two-dimensional")
    num_modes, num_particles = value.shape
    if num_modes < 1:
        raise ValueError("orbital_coefficients must contain at least one mode")
    if num_particles > num_modes:
        raise ValueError("num_particles must not exceed num_modes")
    if not np.all(np.isfinite(value)):
        raise ValueError("orbital_coefficients must contain only finite values")
    gram = value.conj().T @ value
    identity = np.eye(num_particles, dtype=np.complex128)
    if not np.allclose(
        gram,
        identity,
        rtol=_ORTHONORMALITY_TOLERANCE,
        atol=_ORTHONORMALITY_TOLERANCE,
    ):
        raise ValueError(
            "orbital_coefficients columns must be orthonormal within 1e-12"
        )


def decompose_slater_isometry(
    coefficients: NDArray[np.complex128],
) -> tuple[_AdjacentGivens, ...]:
    """Return adjacent rotations that prepare the isometry up to column phases.

    Adjacent row Givens rotations reduce the input isometry to an occupied
    diagonal block. Replaying their adjoints in reverse elimination order maps
    the first occupied modes to the requested orbitals. The ignored diagonal
    column phases multiply the determinant by one global phase only.
    """
    num_modes, num_particles = coefficients.shape
    if num_particles == 0 or num_particles == num_modes:
        return ()

    reduced = np.array(coefficients, dtype=np.complex128, copy=True)
    eliminations: list[tuple[int, int, NDArray[np.complex128]]] = []
    for column in range(num_particles):
        for lower in range(num_modes - 1, column, -1):
            upper = lower - 1
            a = reduced[upper, column]
            b = reduced[lower, column]
            if b == 0.0:
                continue
            norm = math.hypot(abs(a), abs(b))
            givens = np.array(
                (
                    (a.conjugate() / norm, b.conjugate() / norm),
                    (-b / norm, a / norm),
                ),
                dtype=np.complex128,
            )
            reduced[[upper, lower], :] = givens @ reduced[[upper, lower], :]
            reduced[lower, column] = 0.0
            eliminations.append((upper, lower, givens))

    rotations: list[_AdjacentGivens] = []
    for left, right, givens in reversed(eliminations):
        inverse = givens.conj().T
        rotations.append(
            _AdjacentGivens(
                left,
                right,
                complex(inverse[0, 0]),
                complex(inverse[0, 1]),
            )
        )
    return tuple(rotations)


def append_adjacent_givens(circuit: Circuit, rotation: _AdjacentGivens) -> None:
    """Append the exact number-preserving Fock lift of one adjacent SU(2) block."""
    alpha = rotation.alpha
    beta = rotation.beta
    mixing_angle = math.atan2(abs(beta), abs(alpha))
    alpha_phase = math.atan2(alpha.imag, alpha.real)
    if beta == 0.0:
        before_phase = 0.5 * alpha_phase
        after_phase = before_phase
    else:
        beta_phase = math.atan2(beta.imag, beta.real)
        before_phase = 0.5 * (alpha_phase - beta_phase - math.pi / 2.0)
        after_phase = 0.5 * (alpha_phase + beta_phase + math.pi / 2.0)

    _append_balanced_mode_phase(
        circuit,
        rotation.left,
        rotation.right,
        before_phase,
    )
    if mixing_angle != 0.0:
        circuit.append_gate(
            StandardGate.RXX(mixing_angle),
            [rotation.left, rotation.right],
        )
        circuit.append_gate(
            StandardGate.RYY(mixing_angle),
            [rotation.left, rotation.right],
        )
    _append_balanced_mode_phase(
        circuit,
        rotation.left,
        rotation.right,
        after_phase,
    )


def _append_balanced_mode_phase(
    circuit: Circuit,
    left: int,
    right: int,
    phase: float,
) -> None:
    """Apply diag(exp(i*phase), exp(-i*phase)) on the one-particle sector."""
    if phase == 0.0:
        return
    circuit.rz(left, phase)
    circuit.rz(right, -phase)


__all__: list[str] = []
