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

"""Amplitude-encoded rotation operators.

Subpackage exposing three uniformly controlled Pauli rotation gates
(UCRX / UCRY / UCRZ) and function-table controlled rotation families
built on top of them:

* :class:`ExactReciprocal` — ``1 / x`` amplitude encoding (HHL-style
  eigenvalue inversion).

The synthesis helpers live inside the private ``_synthesis`` module and
are not exported.  Each axis gate and each functional rotation exposes
its own alpha-table construction as a module-private helper rather than
promoting those routines to the generic synthesis layer.
"""

from __future__ import annotations

from .exact_reciprocal import ExactReciprocal
from .ucrx import UCRX
from .ucry import UCRY
from .ucrz import UCRZ

__all__ = [
    "UCRX",
    "UCRY",
    "UCRZ",
    "ExactReciprocal",
]
