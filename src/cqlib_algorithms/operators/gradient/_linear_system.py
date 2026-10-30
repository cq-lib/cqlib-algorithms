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

"""Private metric-system solve shared by geometry-aware algorithms."""

from __future__ import annotations

import numpy as np


def solve_metric_system(
    metric: np.ndarray,
    right_hand_side: np.ndarray,
    regularization: float,
) -> np.ndarray:
    """Return the minimum-norm or ridge-regularized metric solution."""
    if regularization == 0.0:
        solution: np.ndarray = np.linalg.lstsq(
            metric,
            right_hand_side,
            rcond=None,
        )[0]
        return solution
    system = metric + regularization * np.eye(len(right_hand_side))
    solution = np.linalg.solve(system, right_hand_side)
    return solution
