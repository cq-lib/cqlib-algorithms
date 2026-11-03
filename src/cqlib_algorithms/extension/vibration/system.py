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

"""Validated physical input for vibrational algorithms."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from cqlib.qis import Hamiltonian
from numpy.typing import NDArray

from .._numeric import normalize_vibrational_atol
from ..many_body import ModalSpace, OneHotVibrationalMapper, VibrationalOp
from .energy_model import VibrationalEnergy
from .observables import OccupiedModals, OneHotLeakage


@dataclass(frozen=True, slots=True)
class VibrationalSystem:
    """A finite-modal vibrational Hamiltonian and its physical one-hot sector.

    Args:
        energy: Vibrational energy model that preserves one modal per mode.

    Raises:
        TypeError: If ``energy`` has the wrong type.
        ValueError: If the Hamiltonian does not preserve the one-hot sector.
    """

    energy: VibrationalEnergy

    def __post_init__(self) -> None:
        if not isinstance(self.energy, VibrationalEnergy):
            raise TypeError("energy must be a VibrationalEnergy")
        operator = self.energy.to_vibrational_op()
        if not operator.preserves_one_hot_subspace():
            raise ValueError("vibrational energy must preserve one modal per mode")

    @property
    def modal_space(self) -> ModalSpace:
        """Return the system's finite modal space."""
        return self.energy.modal_space

    @property
    def num_modes(self) -> int:
        """Return the number of vibrational modes."""
        return self.modal_space.num_modes

    @property
    def num_qubits(self) -> int:
        """Return the one-hot qubit-register width."""
        return self.modal_space.num_qubits

    @property
    def occupied_modals(self) -> OccupiedModals:
        """Return the per-mode physical-sector observables."""
        return OccupiedModals(self.modal_space)

    @property
    def one_hot_leakage(self) -> OneHotLeakage:
        """Return strict per-mode one-hot leakage observables."""
        return OneHotLeakage(self.modal_space)

    def to_vibrational_op(self) -> VibrationalOp:
        """Return the truncated physical vibrational Hamiltonian."""
        return self.energy.to_vibrational_op()

    def to_qubit_hamiltonian(
        self,
        mapper: OneHotVibrationalMapper | None = None,
    ) -> Hamiltonian:
        """Return the complete one-hot qubit Hamiltonian."""
        return self.energy.to_qubit_hamiltonian(mapper)

    def is_physical_configuration(self, configuration: Sequence[int]) -> bool:
        """Return whether one modal index is validly selected for every mode."""
        try:
            self.modal_space.modal_basis_index(configuration)
        except (TypeError, ValueError):
            return False
        return True

    def is_physical_statevector(
        self,
        statevector: NDArray[np.complex128],
        *,
        atol: float = 1e-12,
    ) -> bool:
        """Return whether a normalized qubit state lies in the one-hot sector."""
        threshold = normalize_vibrational_atol(atol)
        array = np.asarray(statevector, dtype=np.complex128)
        if array.ndim != 1 or array.shape != (1 << self.num_qubits,):
            raise ValueError("statevector shape must be (2**num_qubits,)")
        if not np.all(np.isfinite(array)):
            raise ValueError("statevector must contain only finite values")
        norm = float(np.vdot(array, array).real)
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=threshold):
            return False
        physical_probability = float(
            np.sum(np.abs(array[list(self.modal_space.physical_basis_indices)]) ** 2)
        )
        return math.isclose(
            physical_probability,
            1.0,
            rel_tol=0.0,
            abs_tol=threshold,
        )


__all__ = ["VibrationalSystem"]
