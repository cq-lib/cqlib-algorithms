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

"""Fixed pure fermionic Gaussian eigenstate preparation."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import final

import numpy as np
from cqlib.circuit import Circuit
from numpy.typing import ArrayLike

from ..._utils.parameters import validate_parameter_prefix
from ...extension.chemistry import BogoliubovTransform
from ...protocols.state_preparation import StatePreparation
from .basis_state_initialization import BasisStateInitialization
from .slater_determinant_state_preparation import (
    SlaterDeterminantStatePreparation,
)


def _snapshot_occupied_quasiparticles(
    value: Sequence[int],
    *,
    num_modes: int,
) -> tuple[int, ...]:
    """Return a strictly ascending immutable quasiparticle occupation."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("occupied_quasiparticles must be a sequence of ints")
    occupied = tuple(value)
    _validate_occupied_quasiparticles(occupied, num_modes=num_modes)
    return occupied


def _validate_occupied_quasiparticles(
    occupied: tuple[int, ...],
    *,
    num_modes: int,
) -> None:
    """Validate a frozen ascending quasiparticle occupation."""
    if any(type(index) is not int for index in occupied):
        raise TypeError("occupied_quasiparticles must contain only ints")
    if any(index < 0 or index >= num_modes for index in occupied):
        raise ValueError("occupied_quasiparticles contains an out-of-range index")
    if any(left >= right for left, right in itertools.pairwise(occupied)):
        raise ValueError("occupied_quasiparticles must be strictly increasing")


@final
class FermionicGaussianStatePreparation(StatePreparation):
    """Prepare a quasiparticle occupation of a canonical transformation.

    For ``W=[C D]`` and its structured circuit ``U`` satisfying
    ``U a†_i U† = b†_i``, the default prepares the quasiparticle vacuum
    ``U|vac>``. Selected indices prepare
    ``product_i b†_i |vac_b>`` in ascending order. Mode ``p`` maps directly
    to qubit ``p`` in little-endian Jordan--Wigner occupation encoding.

    Number-conserving inputs delegate to
    :class:`SlaterDeterminantStatePreparation`. General inputs delegate the
    canonical transformation to :class:`BogoliubovTransform` and may change
    occupation parity when its physical orthogonal branch requires one
    particle-hole reflection.

    Args:
        transformation: Canonical Bogoliubov matrix or number-conserving block.
        occupied_quasiparticles: Strictly increasing occupied quasiparticle modes.

    Raises:
        TypeError: If an argument cannot be interpreted as the required type.
        ValueError: If the transformation or quasiparticle indices are invalid.
    """

    __slots__ = (
        "_basis_preparation",
        "_occupied_quasiparticles",
        "_slater_preparation",
        "_transform",
    )

    def __init__(
        self,
        transformation: ArrayLike,
        *,
        occupied_quasiparticles: Sequence[int] = (),
    ) -> None:
        transform = BogoliubovTransform(transformation)
        occupied = _snapshot_occupied_quasiparticles(
            occupied_quasiparticles,
            num_modes=transform.num_modes,
        )
        snapshot = transform.transformation
        annihilation = snapshot[:, transform.num_modes :]
        self._transform = transform
        self._occupied_quasiparticles = occupied
        if np.count_nonzero(annihilation) == 0:
            orbitals = snapshot[
                np.asarray(occupied, dtype=np.intp), : transform.num_modes
            ]
            coefficients = np.asarray(orbitals.T, dtype=np.complex128)
            self._slater_preparation: SlaterDeterminantStatePreparation | None = (
                SlaterDeterminantStatePreparation(coefficients)
            )
            self._basis_preparation: BasisStateInitialization | None = None
        else:
            basis_index = sum(1 << index for index in occupied)
            self._basis_preparation = BasisStateInitialization(
                transform.num_modes,
                basis_index,
            )
            self._slater_preparation = None
        self.validate()

    @property
    def occupied_quasiparticles(self) -> tuple[int, ...]:
        """Return the frozen ascending quasiparticle occupation."""
        return self._occupied_quasiparticles

    @property
    def particle_parity(self) -> int:
        """Return the fixed physical occupation parity of the prepared state."""
        return (
            len(self._occupied_quasiparticles) + int(self._transform.changes_parity)
        ) % 2

    def validate(self) -> None:
        """Validate the canonical transform and delegated preparation."""
        self._transform.validate()
        _validate_occupied_quasiparticles(
            self._occupied_quasiparticles,
            num_modes=self._transform.num_modes,
        )
        if (self._slater_preparation is None) == (self._basis_preparation is None):
            raise ValueError("exactly one Gaussian preparation branch must be active")
        if self._slater_preparation is not None:
            self._slater_preparation.validate()
        if self._basis_preparation is not None:
            self._basis_preparation.validate()

    def build_circuit(self, prefix: str) -> Circuit:
        """Return an independent fixed fermionic Gaussian preparation circuit."""
        validate_parameter_prefix(prefix)
        self.validate()
        if self._slater_preparation is not None:
            return self._slater_preparation.build_circuit(prefix)
        if self._basis_preparation is None:
            raise ValueError("Gaussian preparation branch is inconsistent")
        circuit = self._basis_preparation.build_circuit(prefix)
        circuit.compose(self._transform.build_circuit(), circuit.qubits)
        circuit.validate()
        return circuit

    def num_parameters(self) -> int:
        """Return zero because the canonical transformation is fixed."""
        return 0

    def num_qubits(self) -> int:
        """Return the number of fermionic modes and occupation qubits."""
        return self._transform.num_modes


__all__ = ["FermionicGaussianStatePreparation"]
