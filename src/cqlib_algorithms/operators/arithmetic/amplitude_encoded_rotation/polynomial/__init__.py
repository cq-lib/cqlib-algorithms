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

"""Bit-wise polynomial Pauli rotations on an auxiliary qubit.

Unlike the :mod:`multiplexed` subpackage, which synthesises arbitrary
per-control tables through the Möttönen Gray-code decomposition
(cost ``O(2**n)`` gates), the families collected here exploit the
additive structure of polynomial functions and rely on the
angle-addition property of co-axial Pauli rotations,

.. math::

    R_{\\text{axis}}(\\theta_1) \\, R_{\\text{axis}}(\\theta_2)
        = R_{\\text{axis}}(\\theta_1 + \\theta_2),

to reduce the gate count to a polynomial in the control-register bit
width.  Currently implemented families:

* :class:`LinearPauliRotation` — ``theta(k) = slope * k + offset``
  using one unconditional rotation plus ``n`` singly controlled
  rotations (Stamatopoulos et al., 2020).
"""

from __future__ import annotations

from .linear_pauli_rotation import LinearPauliRotation

__all__ = [
    "LinearPauliRotation",
]
