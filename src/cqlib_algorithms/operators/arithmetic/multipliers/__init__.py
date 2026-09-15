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
"""Quantum multiplier circuits.

This package exports the public multiplier wrapper used by the arithmetic
module. Concrete synthesis implementations live in internal modules.
"""

from __future__ import annotations

from .multiplier import Multiplier

__all__ = [
    "Multiplier",
]
