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

"""Multiplexed (UCR / Gray-code table) amplitude-encoded rotations.

This subpackage collects the uniformly-controlled-rotation building
blocks and the function-table families built on top of them.  All
circuits here use the Möttönen et al. (2005) Gray-code / CX
interleaving synthesis, which decomposes a block-diagonal target

.. math::

    \\operatorname{diag}(R(\\alpha_0), R(\\alpha_1), \\dots, R(\\alpha_{2^n-1}))

into ``2**n`` single-qubit rotations and ``2**n`` CX gates.  Concrete
families include the base axis gates (:class:`UCRX`, :class:`UCRY`,
:class:`UCRZ`) and two higher-level function tables:

* :class:`ExactReciprocal` — HHL reciprocal-eigenvalue inversion encoded as
  ``2 * arcsin(s * 2**n / k)``.
* :class:`ArbitraryFunction` — a user-supplied real-valued per-control
  table ``f(k)`` that is encoded either as amplitudes (``arcsin``) or
  as raw angles depending on the selected ``mode``.
"""

from __future__ import annotations

from ._synthesis import (
    gray_code,
    gray_code_sequence,
    gray_control_positions,
    gray_matrix,
    solve_theta_from_alpha,
)
from .arbitrary_function import ArbitraryFunction
from .exact_reciprocal import ExactReciprocal
from .ucrx import UCRX
from .ucry import UCRY
from .ucrz import UCRZ

__all__ = [
    "UCRX",
    "UCRY",
    "UCRZ",
    "ArbitraryFunction",
    "ExactReciprocal",
    "gray_code",
    "gray_code_sequence",
    "gray_control_positions",
    "gray_matrix",
    "solve_theta_from_alpha",
]
