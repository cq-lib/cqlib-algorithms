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

"""Fermionic excitations, generators, and canonical transformations."""

from .bogoliubov import BogoliubovTransform
from .excitation import FermionicExcitation
from .excitation_generation import (
    generate_double_excitations,
    generate_paired_double_excitations,
    generate_puccsd_generators,
    generate_single_excitations,
    generate_uccsd_excitations,
)
from .generators import GroupedFermionicGenerator
from .quadratic_model import QuadraticFermionicHamiltonian

__all__ = [
    "BogoliubovTransform",
    "FermionicExcitation",
    "GroupedFermionicGenerator",
    "QuadraticFermionicHamiltonian",
    "generate_double_excitations",
    "generate_paired_double_excitations",
    "generate_puccsd_generators",
    "generate_single_excitations",
    "generate_uccsd_excitations",
]
