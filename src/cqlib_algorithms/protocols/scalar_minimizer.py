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

"""Callable minimization of scalar objectives and immutable results."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ScalarMinimizerResult:
    """A selected point, objective value and termination metadata.

    ``evaluations`` counts actual objective calls. The point is copied to a
    finite real tuple. Invalid types raise TypeError; invalid values raise
    ValueError.
    """

    point: tuple[float, ...]
    value: float
    success: bool
    status: int
    message: str
    evaluations: int

    def __post_init__(self) -> None:
        if not isinstance(self.point, tuple):
            raise TypeError("point must be a tuple")
        point: list[float] = []
        for value in self.point:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("point must contain real numbers")
            if not math.isfinite(float(value)):
                raise ValueError("point must contain finite values")
            point.append(float(value))
        object.__setattr__(self, "point", tuple(point))
        if isinstance(self.value, bool) or not isinstance(self.value, Real):
            raise TypeError("value must be a real number")
        if not math.isfinite(float(self.value)):
            raise ValueError("value must be finite")
        object.__setattr__(self, "value", float(self.value))
        if not isinstance(self.success, bool):
            raise TypeError("success must be bool")
        if isinstance(self.status, bool) or not isinstance(self.status, int):
            raise TypeError("status must be an integer")
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")
        if isinstance(self.evaluations, bool) or not isinstance(self.evaluations, int):
            raise TypeError("evaluations must be an integer")
        if self.evaluations <= 0:
            raise ValueError("evaluations must be positive")


class ScalarMinimizer(Protocol):
    """Search an objective using an independent initial-point array.

    Evaluate the supplied objective and return a selected point with its
    observed value and exact evaluation count.
    """

    def __call__(
        self,
        objective: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        /,
    ) -> ScalarMinimizerResult: ...
