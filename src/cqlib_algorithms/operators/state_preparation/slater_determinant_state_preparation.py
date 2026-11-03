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

"""Slater-determinant state preparation in occupation encoding."""

from __future__ import annotations

from typing import final

from cqlib.circuit import Circuit
from numpy.typing import ArrayLike

from ..._utils.parameters import validate_parameter_prefix
from ...protocols.state_preparation import StatePreparation
from ._slater_decomposition import (
    _AdjacentGivens,
    append_adjacent_givens,
    decompose_slater_isometry,
    snapshot_orbital_coefficients,
    validate_orbital_coefficients,
)
from .basis_state_initialization import BasisStateInitialization


@final
class SlaterDeterminantStatePreparation(StatePreparation):
    """Prepare a fixed Slater determinant up to one global phase.

    For an ``(M, N)`` column-isometry ``C``, occupied orbital ``i`` is
    ``d_i† = sum_p C[p, i] a_p†``. Mode ``p`` maps directly to qubit ``p``
    in little-endian occupation encoding. The prepared computational-basis
    amplitude at an ascending occupied-mode set ``S`` is proportional to
    ``det(C[S, :])``; column phases and column permutations therefore change
    only the physically irrelevant overall phase.

    Args:
        orbital_coefficients: Complex ``(num_modes, num_particles)`` column
            isometry defining occupied one-particle orbitals.

    Raises:
        TypeError: If coefficients cannot be interpreted numerically.
        ValueError: If dimensions, finiteness, or column orthonormality fail.
    """

    __slots__ = (
        "_basis_preparation",
        "_num_modes",
        "_num_particles",
        "_orbital_coefficients",
        "_rotations",
    )

    def __init__(self, orbital_coefficients: ArrayLike) -> None:
        coefficients = snapshot_orbital_coefficients(orbital_coefficients)
        self._orbital_coefficients = coefficients
        self._num_modes = int(coefficients.shape[0])
        self._num_particles = int(coefficients.shape[1])
        self._rotations: tuple[_AdjacentGivens, ...] = decompose_slater_isometry(
            coefficients
        )
        reference_index = (1 << self._num_particles) - 1
        self._basis_preparation = BasisStateInitialization(
            self._num_modes,
            reference_index,
        )
        self.validate()

    def validate(self) -> None:
        """Validate the immutable orbital and reference-state snapshots."""
        validate_orbital_coefficients(self._orbital_coefficients)
        if self._num_modes != self._orbital_coefficients.shape[0]:
            raise ValueError("num_modes is inconsistent with orbital_coefficients")
        if self._num_particles != self._orbital_coefficients.shape[1]:
            raise ValueError("num_particles is inconsistent with orbital_coefficients")
        self._basis_preparation.validate()

    def build_circuit(self, prefix: str) -> Circuit:
        """Return an independent adjacent-mode Slater preparation circuit."""
        validate_parameter_prefix(prefix)
        self.validate()
        circuit = self._basis_preparation.build_circuit(prefix)
        for rotation in self._rotations:
            append_adjacent_givens(circuit, rotation)
        circuit.validate()
        return circuit

    def num_parameters(self) -> int:
        """Return zero because the orbital coefficients are fixed."""
        return 0

    def num_qubits(self) -> int:
        """Return the number of fermionic modes and occupation qubits."""
        return self._num_modes


__all__ = ["SlaterDeterminantStatePreparation"]
