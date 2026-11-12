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

"""Uniformly Controlled Rotation around the Pauli Y axis."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from cqlib import Circuit

from ._synthesis import _UCRBase


class UCRY(_UCRBase):
    """Uniformly Controlled Rotation around the Y axis.

    Implements the family

    .. math::

        F_k(\\hat y, \\boldsymbol{\\alpha})
            = \\bigoplus_{c=0}^{2^k - 1}
                |c\\rangle\\langle c| \\otimes R_y(\\alpha_c),

    which is the building block used by the function-table rotation
    (``ExactReciprocal``) for amplitude encoding.

    The local qubit layout is ``[control_0, ..., control_{k-1}, target]``
    with wires ``0..k-1`` for controls and wire ``k`` for the target.

    Parameters
    ----------
    num_controls : int
        Non-negative integer number of control qubits ``k``.  Must not
        be a ``bool``.
    alpha : Sequence[float] | np.ndarray[Any, Any]
        Target rotation angles, one per control basis state.  The
        length must equal ``2**num_controls`` (or ``1`` when
        ``num_controls == 0``).
    name : str | None, keyword-only
        Optional human readable label.  Defaults to ``"UCRY"``.

    Attributes
    ----------
    num_controls : int
    alpha : np.ndarray[Any, Any]
    theta : np.ndarray[Any, Any]
    axis : Literal["y"]
        Always ``"y"`` for this subclass.
    gray_codes : list[int]
    controls : list[int]
    gray_matrix : np.ndarray[Any, Any]
    num_state_qubits : int
    num_qubits : int
    control_register_indices : list[int]
    result_qubit_index : int
    name : str

    Examples
    --------
    >>> import numpy as np
    >>> from cqlib_algorithms.operators.arithmetic import UCRY
    >>> ucr = UCRY(2, np.array([0.1, 0.2, 0.3, 0.4]))
    >>> circ = ucr.build_circuit()
    >>> circ.num_qubits
    3
    """

    _ROTATION_AXIS = "y"
    _DEFAULT_NAME = "UCRY"

    def __init__(
        self,
        num_controls: int,
        alpha: Sequence[float] | np.ndarray[Any, Any],
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(num_controls, alpha, name=name)

    def build_circuit(self) -> Circuit:
        """Return a fresh local-numbering UCRY circuit."""

        return super().build_circuit()
