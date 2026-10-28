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

"""Vibrational-domain data models, Hamiltonians, and observables."""

from ..many_body import ModalSpace, OneHotVibrationalMapper, VibrationalOp
from .energy_model import VibrationalEnergy
from .excitations import (
    VibrationalExcitation,
    generate_vibrational_double_excitations,
    generate_vibrational_single_excitations,
    vscf_initial_point,
)
from .modal_integrals import (
    ModalIntegralEntry,
    ModalIntegrals,
    ModalIntegralTerm,
    ModalTransition,
)
from .observables import ModalOccupation, OccupiedModals, OneHotLeakage
from .system import VibrationalSystem

__all__ = [
    "ModalIntegralEntry",
    "ModalIntegralTerm",
    "ModalIntegrals",
    "ModalOccupation",
    "ModalSpace",
    "ModalTransition",
    "OccupiedModals",
    "OneHotLeakage",
    "OneHotVibrationalMapper",
    "VibrationalEnergy",
    "VibrationalExcitation",
    "VibrationalOp",
    "VibrationalSystem",
    "generate_vibrational_double_excitations",
    "generate_vibrational_single_excitations",
    "vscf_initial_point",
]
