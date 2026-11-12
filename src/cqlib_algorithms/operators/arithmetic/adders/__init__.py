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

This package exposes the four public adder interfaces used by the arithmetic
module. Concrete synthesis implementations live in internal modules prefixed
with an underscore.
"""

from __future__ import annotations

from .constant_adder import DraperConstantAdder
from .full_adder import FullAdder
from .half_adder import HalfAdder
from .modular_adder import ModularAdder

__all__ = [
    "DraperConstantAdder",
    "FullAdder",
    "HalfAdder",
    "ModularAdder",
]
