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

"""Uniformly Controlled Rotation around the Pauli X axis."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from cqlib import Circuit

from ._synthesis import _UCRBase


class UCRX(_UCRBase):
    """Uniformly Controlled Rotation around the X axis.

    Implements the family

    .. math::

        F_k(\\hat x, \\boldsymbol{\\alpha})
            = \\bigoplus_{c=0}^{2^k - 1}
                |c\\rangle\\langle c| \\otimes R_x(\\alpha_c),

    where ``k = num_controls``.  Circuit synthesis follows the Gray-code
    / CX alternating decomposition of Möttönen et al. (2005).

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
        Optional human readable label.  Defaults to ``"UCRX"``.

    Attributes
    ----------
    num_controls : int
        Number of control qubits.
    alpha : np.ndarray[Any, Any]
        Defensive copy of the target angles.
    theta : np.ndarray[Any, Any]
        Gray-expanded circuit-level angles solved from ``alpha``.
    axis : Literal["x"]
        Always ``"x"`` for this subclass.
    gray_codes : list[int]
        Binary-reflected Gray code sequence.
    controls : list[int]
        Local CX control positions for the alternating sequence.
    gray_matrix : np.ndarray[Any, Any]
        The ``M^k`` Gray matrix used to solve ``M^k theta = alpha``.
    num_state_qubits : int
        Alias for ``num_controls``.
    num_qubits : int
        Total circuit width, ``num_controls + 1``.
    control_register_indices : list[int]
        Local wire indices ``[0, ..., num_controls - 1]``.
    result_qubit_index : int
        Local target wire index (``num_controls``).
    name : str
        Configured human-readable name.

    Examples
    --------
    >>> import numpy as np
    >>> from cqlib_algorithms.operators.arithmetic import UCRX
    >>> ucr = UCRX(2, np.array([0.1, 0.2, 0.3, 0.4]))
    >>> circ = ucr.build_circuit()
    >>> circ.num_qubits
    3
    """

    _ROTATION_AXIS = "x"
    _DEFAULT_NAME = "UCRX"

    def __init__(
        self,
        num_controls: int,
        alpha: Sequence[float] | np.ndarray[Any, Any],
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(num_controls, alpha, name=name)

    def build_circuit(self) -> Circuit:
        """Return a fresh local-numbering UCRX circuit."""

        return super().build_circuit()
