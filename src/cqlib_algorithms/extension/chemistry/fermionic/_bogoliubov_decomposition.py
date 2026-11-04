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

"""Private canonical validation and Majorana-plane decomposition."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from cqlib.circuit import Circuit
from cqlib.circuit.gates import StandardGate
from numpy.typing import ArrayLike, NDArray

from .._numeric import immutable_complex_array

_CANONICAL_ATOL = 1e-12
_DECOMPOSITION_ATOL = 5e-11


@dataclass(frozen=True, slots=True)
class _MajoranaPlaneRotation:
    """A nearest-neighbour Majorana rotation implemented by one Core gate."""

    index: int
    angle: float


@dataclass(frozen=True, slots=True)
class _MajoranaDecomposition:
    """Ordered plane rotations followed by an optional last-mode reflection."""

    rotations: tuple[_MajoranaPlaneRotation, ...]
    particle_hole_reflection: bool


def snapshot_bogoliubov_transformation(
    value: ArrayLike,
) -> NDArray[np.complex128]:
    """Return a canonical read-only ``[C D]`` Bogoliubov snapshot.

    An ``N x N`` input is the number-conserving shorthand ``[C 0]``.
    """
    source = immutable_complex_array(value, name="transformation")
    if source.ndim != 2:
        raise ValueError("transformation must be two-dimensional")
    num_modes, width = source.shape
    if num_modes < 1:
        raise ValueError("transformation must contain at least one mode")
    if width == num_modes:
        expanded = np.concatenate(
            (source, np.zeros_like(source, dtype=np.complex128)),
            axis=1,
        )
        snapshot = np.frombuffer(expanded.tobytes(), dtype=np.complex128).reshape(
            expanded.shape
        )
    elif width == 2 * num_modes:
        snapshot = source
    else:
        raise ValueError("transformation shape must be (N, N) or (N, 2*N)")
    validate_bogoliubov_transformation(snapshot)
    return snapshot


def validate_bogoliubov_transformation(
    value: NDArray[np.complex128],
) -> None:
    """Validate a frozen ``[C D]`` snapshot without modifying it."""
    if not isinstance(value, np.ndarray):
        raise TypeError("transformation must be a numpy array")
    if value.dtype != np.dtype(np.complex128):
        raise TypeError("transformation must use complex128 values")
    if value.ndim != 2:
        raise ValueError("transformation must be two-dimensional")
    num_modes, width = value.shape
    if num_modes < 1 or width != 2 * num_modes:
        raise ValueError("transformation snapshot shape must be (N, 2*N)")
    if not np.all(np.isfinite(value)):
        raise ValueError("transformation must contain only finite values")
    creation = value[:, :num_modes]
    annihilation = value[:, num_modes:]
    first_car = creation @ creation.conj().T + annihilation @ annihilation.conj().T
    second_car = creation @ annihilation.T + annihilation @ creation.T
    if not np.allclose(
        first_car,
        np.eye(num_modes, dtype=np.complex128),
        rtol=0.0,
        atol=_CANONICAL_ATOL,
    ):
        raise ValueError("transformation must satisfy C C† + D D† = I")
    if not np.allclose(
        second_car,
        np.zeros((num_modes, num_modes), dtype=np.complex128),
        rtol=0.0,
        atol=_CANONICAL_ATOL,
    ):
        raise ValueError("transformation must satisfy C D^T + D C^T = 0")


def bogoliubov_to_majorana(
    value: NDArray[np.complex128],
) -> NDArray[np.float64]:
    """Convert canonical ``[C D]`` rows to the associated real orthogonal map."""
    validate_bogoliubov_transformation(value)
    num_modes = value.shape[0]
    creation = value[:, :num_modes]
    annihilation = value[:, num_modes:]
    orthogonal = np.empty((2 * num_modes, 2 * num_modes), dtype=np.float64)
    orthogonal[0::2, 0::2] = np.real(creation + annihilation)
    orthogonal[0::2, 1::2] = np.imag(creation - annihilation)
    orthogonal[1::2, 0::2] = -np.imag(creation + annihilation)
    orthogonal[1::2, 1::2] = np.real(creation - annihilation)
    identity = np.eye(2 * num_modes, dtype=np.float64)
    if not np.allclose(
        orthogonal @ orthogonal.T,
        identity,
        rtol=0.0,
        atol=_CANONICAL_ATOL,
    ):
        raise ValueError("transformation does not define an orthogonal Majorana map")
    return orthogonal


def majorana_to_bogoliubov(
    value: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Convert a real orthogonal Majorana map to canonical ``[C D]`` rows."""
    matrix = np.asarray(value)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Majorana transformation must be square")
    if matrix.shape[0] < 2 or matrix.shape[0] % 2 != 0:
        raise ValueError("Majorana transformation must have positive even width")
    if np.iscomplexobj(matrix) or not np.all(np.isfinite(matrix)):
        raise ValueError("Majorana transformation must be finite and real")
    orthogonal = np.array(matrix, dtype=np.float64, copy=True)
    if not np.allclose(
        orthogonal @ orthogonal.T,
        np.eye(orthogonal.shape[0]),
        rtol=0.0,
        atol=_CANONICAL_ATOL,
    ):
        raise ValueError("Majorana transformation must be orthogonal")

    even_even = orthogonal[0::2, 0::2]
    even_odd = orthogonal[0::2, 1::2]
    odd_even = orthogonal[1::2, 0::2]
    odd_odd = orthogonal[1::2, 1::2]
    creation = 0.5 * (even_even + odd_odd + 1j * (even_odd - odd_even))
    annihilation = 0.5 * (even_even - odd_odd - 1j * (even_odd + odd_even))
    combined = np.concatenate((creation, annihilation), axis=1)
    snapshot = np.frombuffer(
        np.asarray(combined, dtype=np.complex128).tobytes(),
        dtype=np.complex128,
    ).reshape(combined.shape)
    validate_bogoliubov_transformation(snapshot)
    return snapshot


def decompose_majorana_orthogonal(
    orthogonal: NDArray[np.float64],
) -> _MajoranaDecomposition:
    """Factor an orthogonal map into nearest-neighbour Majorana rotations."""
    candidate = np.array(orthogonal, dtype=np.float64, copy=True)
    if candidate.ndim != 2 or candidate.shape[0] != candidate.shape[1]:
        raise ValueError("Majorana transformation must be square")
    if candidate.shape[0] < 2 or candidate.shape[0] % 2 != 0:
        raise ValueError("Majorana transformation must have positive even width")
    identity = np.eye(candidate.shape[0], dtype=np.float64)
    if not np.all(np.isfinite(candidate)) or not np.allclose(
        candidate @ candidate.T,
        identity,
        rtol=0.0,
        atol=_CANONICAL_ATOL,
    ):
        raise ValueError("Majorana transformation must be finite and orthogonal")

    determinant = float(np.linalg.det(candidate))
    if not math.isclose(abs(determinant), 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("Majorana transformation determinant must be +1 or -1")
    reflection = determinant < 0.0
    if reflection:
        candidate[:, -1] *= -1.0

    eliminations: list[_MajoranaPlaneRotation] = []
    width = candidate.shape[0]
    for column in range(width - 1):
        for lower in range(width - 1, column, -1):
            upper = lower - 1
            a = float(candidate[upper, column])
            b = float(candidate[lower, column])
            if b == 0.0:
                continue
            norm = math.hypot(a, b)
            cosine = a / norm
            sine = b / norm
            upper_row = candidate[upper, :].copy()
            lower_row = candidate[lower, :].copy()
            candidate[upper, :] = cosine * upper_row + sine * lower_row
            candidate[lower, :] = -sine * upper_row + cosine * lower_row
            candidate[lower, column] = 0.0
            eliminations.append(_MajoranaPlaneRotation(upper, -math.atan2(b, a)))
        if candidate[column, column] < 0.0:
            candidate[[column, column + 1], :] *= -1.0
            eliminations.append(_MajoranaPlaneRotation(column, -math.pi))

    if not np.allclose(
        candidate,
        identity,
        rtol=0.0,
        atol=_DECOMPOSITION_ATOL,
    ):
        residual = float(np.max(np.abs(candidate - identity)))
        raise ValueError(
            "Majorana transformation decomposition failed validation "
            f"(maximum residual {residual:.3e})"
        )
    return _MajoranaDecomposition(tuple(eliminations), reflection)


def append_majorana_decomposition(
    circuit: Circuit,
    decomposition: _MajoranaDecomposition,
) -> None:
    """Append a decomposition with ``U a† U†`` transformation semantics."""
    for rotation in decomposition.rotations:
        if rotation.index % 2 == 0:
            circuit.rz(rotation.index // 2, rotation.angle)
        else:
            circuit.append_gate(
                StandardGate.RXX(rotation.angle),
                [rotation.index // 2, rotation.index // 2 + 1],
            )
    if decomposition.particle_hole_reflection:
        circuit.x(circuit.num_qubits - 1)


__all__: list[str] = []
