# This code is part of Cqlib.
#
# (C) Copyright China Telecom Quantum Group 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# use this file except in compliance with the License, obtained from the
# LICENSE.txt file in the root directory of this source tree or
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Quantum Fisher information from the statevector QGT metric.

References
----------
[1] J. Stokes, J. Izaac, N. Killoran, and G. Carleo, "Quantum natural
    gradient," Quantum, 4, 269, 2020.
    DOI: 10.22331/q-2020-05-25-269.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import final

import numpy as np

from ...protocols.ansatz import Ansatz
from ...protocols.gradient import GradientRequest
from .qgt import QGTResult, StatevectorQGT

_HERMITIAN_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class QFIResult:
    """The quantum Fisher information matrix of one request.

    ``qfi`` is the real symmetric matrix ``F_ij = 4 * g_ij`` where ``g``
    is the real metric of the quantum geometric tensor (Stokes et al.,
    arXiv:1909.02108 convention).  Rows and columns follow
    ``parameter_names``.  Array-like inputs are copied into a real
    NumPy array.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If names are duplicated, the shape disagrees with
            the names, a value is not finite, or the matrix is not
            symmetric within 1e-9.
    """

    parameter_names: tuple[str, ...]
    qfi: np.ndarray

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.parameter_names, tuple):
            raise TypeError("parameter_names must be a tuple")
        if any(not isinstance(name, str) for name in self.parameter_names):
            raise TypeError("parameter names must be strings")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("parameter names must be unique")

        count = len(self.parameter_names)
        matrix_input = np.asarray(self.qfi)
        if np.iscomplexobj(matrix_input):
            if np.any(matrix_input.imag != 0.0):
                raise ValueError("qfi must be real-valued")
            matrix_input = matrix_input.real
        qfi = np.array(matrix_input, dtype=np.float64, copy=True)
        if qfi.shape != (count, count):
            raise ValueError("qfi must be a square matrix matching the names")
        if not np.all(np.isfinite(qfi)):
            raise ValueError("qfi must be finite")
        if not np.allclose(qfi, qfi.T, atol=_HERMITIAN_TOLERANCE, rtol=0.0):
            raise ValueError("qfi must be symmetric")
        object.__setattr__(self, "qfi", qfi)


@final
class StatevectorQFI:
    """Exact quantum Fisher information on statevectors.

    ``compute`` evaluates ``QFI = 4 * g`` where ``g`` is the real part
    of the quantum geometric tensor computed by
    :class:`StatevectorQGT`.  For a single ``RY`` rotation the value is
    ``1``.  Positive semidefiniteness is a mathematical property of the
    construction and is not re-checked here.

    Using the QFI instead of ``g`` as the natural-gradient metric only
    rescales the step size by ``1/4`` and leaves the direction
    unchanged, so :class:`NaturalGradient` keeps using ``g``.

    Raises:
        TypeError: If the request has the wrong type.
        ValueError: If the request is invalid or the circuit contains
            unsupported instructions.
    """

    __slots__ = ()

    def compute(
        self,
        request: GradientRequest | Ansatz,
        parameter_bindings: Mapping[str, float] | None = None,
        *,
        parameter_names: tuple[str, ...] | None = None,
        prefix: str = "theta",
    ) -> QFIResult:
        """Return pure-state QFI from an ansatz or existing gradient request.

        With a GradientRequest, use its stored bindings and parameter selection;
        additional bindings or parameter names and a nondefault prefix are rejected.

        Args:
            request: Trainable component or existing gradient request.
            parameter_bindings: Exact bindings for a component input.
            parameter_names: Ordered subset; None uses active-symbol order.
            prefix: Naming prefix used to build a component.

        Returns:
            Four times the real QGT metric in the selected parameter order.

        Raises:
            TypeError: If the component or request has the wrong type.
            ValueError: If parameters, options, or circuit instructions are invalid.
        """
        result: QGTResult = StatevectorQGT().compute(
            request, parameter_bindings, parameter_names=parameter_names, prefix=prefix
        )
        qfi = 4.0 * result.metric
        return QFIResult(result.parameter_names, qfi)
