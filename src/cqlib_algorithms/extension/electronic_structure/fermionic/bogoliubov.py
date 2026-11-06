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

"""Structured circuits for canonical fermionic Bogoliubov transformations."""

from __future__ import annotations

from typing import final

import numpy as np
from cqlib.circuit import Circuit
from numpy.typing import ArrayLike, NDArray

from ._bogoliubov_decomposition import (
    _MajoranaDecomposition,
    append_majorana_decomposition,
    bogoliubov_to_majorana,
    decompose_majorana_orthogonal,
    snapshot_bogoliubov_transformation,
    validate_bogoliubov_transformation,
)


@final
class BogoliubovTransform:
    """Build a fixed structured fermionic Gaussian transformation.

    An ``N x 2N`` matrix is interpreted as ``W=[C D]`` with rows
    ``b†_i = sum_p (C[i,p] a†_p + D[i,p] a_p)``. The circuit follows the
    active convention ``U a†_i U† = b†_i``. An ``N x N`` input is shorthand
    for the number-conserving case ``[C 0]``.

    The decomposition uses adjacent Majorana-plane rotations and, when the
    orthogonal transformation has negative determinant, one last-mode
    particle-hole reflection. Such a reflection changes occupation parity;
    rotations alone preserve it.

    Args:
        transformation: Canonical ``N x 2N`` matrix ``[C D]`` or an ``N x N``
            number-conserving shorthand.

    Raises:
        TypeError: If the transformation cannot be interpreted numerically.
        ValueError: If dimensions or canonical anticommutation constraints fail.
    """

    __slots__ = ("_decomposition", "_majorana", "_transformation")

    def __init__(self, transformation: ArrayLike) -> None:
        snapshot = snapshot_bogoliubov_transformation(transformation)
        majorana = bogoliubov_to_majorana(snapshot)
        self._transformation = snapshot
        self._majorana = np.frombuffer(
            np.asarray(majorana, dtype=np.float64).tobytes(), dtype=np.float64
        ).reshape(majorana.shape)
        self._decomposition: _MajoranaDecomposition = decompose_majorana_orthogonal(
            self._majorana
        )
        self.validate()

    @property
    def transformation(self) -> NDArray[np.complex128]:
        """Return an immutable independent snapshot of ``[C D]``."""
        copied = np.array(self._transformation, dtype=np.complex128, copy=True)
        return np.frombuffer(copied.tobytes(), dtype=np.complex128).reshape(
            copied.shape
        )

    @property
    def num_modes(self) -> int:
        """Return the number of fermionic modes and occupation qubits."""
        return int(self._transformation.shape[0])

    @property
    def changes_parity(self) -> bool:
        """Return whether the selected canonical branch changes parity."""
        return self._decomposition.particle_hole_reflection

    def validate(self) -> None:
        """Validate the canonical snapshot and its structured decomposition."""
        validate_bogoliubov_transformation(self._transformation)
        rebuilt = bogoliubov_to_majorana(self._transformation)
        if not np.allclose(rebuilt, self._majorana, rtol=0.0, atol=0.0):
            raise ValueError("stored Majorana transformation is inconsistent")

    def build_circuit(self) -> Circuit:
        """Return an independent fixed structured transformation circuit.

        Returns:
            Circuit implementing the stored active Bogoliubov convention.

        Raises:
            ValueError: If the stored canonical decomposition is inconsistent.
        """
        self.validate()
        circuit = Circuit(self.num_modes)
        append_majorana_decomposition(circuit, self._decomposition)
        circuit.validate()
        return circuit


__all__ = ["BogoliubovTransform"]
