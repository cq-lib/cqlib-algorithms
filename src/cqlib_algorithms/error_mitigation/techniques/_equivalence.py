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

r"""Operator comparison up to a global phase.

Some error-mitigation invariants are statements about operators rather than
about circuits: a dynamical-decoupling rule is usable only when the product of
its pulses is the identity, and a Pauli-twirling table entry is usable only
when the dressed gate reproduces the original gate. In both cases the two
sides are allowed to differ by a global phase, which carries no observable
consequence, so the comparison is made modulo that phase.

Both checks share the single helper in this module.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def allclose_up_to_global_phase(
    first: NDArray[np.complex128],
    second: NDArray[np.complex128],
    /,
    *,
    atol: float = 1.0e-9,
) -> bool:
    r"""Return whether two operators agree up to a global phase.

    Two operators :math:`A` and :math:`B` agree up to a global phase when
    :math:`A = e^{\mathrm{i}\varphi} B` for some real :math:`\varphi`. The
    phase is recovered from the entry of ``second`` of largest magnitude, and
    the two operators are then compared entrywise against
    :math:`\varphi`-rotated ``second``.

    Parameters
    ----------
    first
        Operator to compare, as a complex array.
    second
        Operator to compare against, as a complex array of the same shape.
    atol
        Absolute tolerance of the entrywise comparison.

    Returns
    -------
    bool
        ``True`` when a global phase exists that makes the two operators
        equal to within ``atol``, ``False`` otherwise, including when the
        shapes differ.
    """
    left = np.asarray(first, dtype=np.complex128)
    right = np.asarray(second, dtype=np.complex128)
    if left.shape != right.shape:
        return False

    flat_right = right.reshape(-1)
    pivot = int(np.argmax(np.abs(flat_right)))
    magnitude = float(np.abs(flat_right[pivot]))
    if magnitude <= atol:
        # Neither operator carries an entry a phase could be read off, so the
        # two are compared directly as they stand.
        return bool(np.allclose(left, right, atol=atol))

    phase = complex(left.reshape(-1)[pivot] / flat_right[pivot])
    return bool(np.allclose(left, phase * right, atol=atol))
