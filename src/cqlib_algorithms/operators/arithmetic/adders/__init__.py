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
"""Quantum adder circuit collection.

This package exports the public adder classes used by the arithmetic module.
Concrete implementations and wrapper interfaces are documented in their
individual modules.
"""

from __future__ import annotations

from .ripple_carry_adder import RippleCarryAdder, AdderKind
from .draper_qft_adder import DraperQFTAdder
from .constant_adder import DraperConstantAdder
from .half_adder import HalfAdder
from .full_adder import FullAdder
from .modular_adder import ModularAdder

__all__ = [
    "RippleCarryAdder",
    "AdderKind",
    "DraperQFTAdder",
    "DraperConstantAdder",
    "HalfAdder",
    "FullAdder",
    "ModularAdder",
]

