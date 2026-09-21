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
"""Quantum arithmetic circuit library.

Construction helper subpackage for quantum arithmetic building blocks.
Exposes the submodule categories documented in the architecture design.
Concrete implementations (and hence public exports) are added here as
each component is landed in its respective module / subpackage.
"""

from __future__ import annotations

from . import (
    adders,
    amplitude_encoded_rotation,
    comparators,
    multipliers,
    weighted_sums,
)
from .adders import DraperConstantAdder, FullAdder, HalfAdder, ModularAdder
from .amplitude_encoded_rotation import (
    UCRX,
    UCRY,
    UCRZ,
    ExactReciprocal,
)
from .comparators import IntegerComparator
from .multipliers import Multiplier
from .weighted_sums import WeightedSum

__all__ = [
    "UCRX",
    "UCRY",
    "UCRZ",
    "DraperConstantAdder",
    "ExactReciprocal",
    "FullAdder",
    "HalfAdder",
    "IntegerComparator",
    "ModularAdder",
    "Multiplier",
    "WeightedSum",
    "adders",
    "amplitude_encoded_rotation",
    "comparators",
    "multipliers",
    "weighted_sums",
]
