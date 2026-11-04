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

"""Exact physical-spin-one-half to qubit mapping."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from cqlib.qis import Hamiltonian

from ._qubit_mapping import map_spin_half_terms
from .spin_operator import SpinOp


@dataclass(frozen=True, slots=True)
class SpinHalfMapper:
    """Map ``S^alpha`` to ``sigma_alpha / 2`` on the same qubit."""

    def map_operator(self, operator: SpinOp) -> Hamiltonian:
        """Return the exact qubit Hamiltonian for a spin-one-half operator."""
        if not isinstance(operator, SpinOp):
            raise TypeError("operator must be a SpinOp")
        if operator.spin != Fraction(1, 2):
            raise ValueError("SpinHalfMapper supports only spin one half")

        return map_spin_half_terms(operator.num_spins, operator.terms)

    @staticmethod
    def num_qubits(num_spins: int) -> int:
        """Return one qubit per spin-one-half site."""
        if type(num_spins) is not int:
            raise TypeError("num_spins must be an int")
        if num_spins < 1:
            raise ValueError("num_spins must be at least 1")
        return num_spins
