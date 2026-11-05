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

"""Balanced uncentered kernel-target alignment.

Reference: Hubregtsen et al., arXiv:2105.02276v1, Section IV, Eq. (27).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..._utils.numeric import copy_finite_real_array
from ...protocols.quantum_kernel import KernelLoss


def _alignment(matrix: NDArray[np.float64], target: NDArray[np.float64]) -> float:
    # Normalize the arrays before the inner product to avoid denominator overflow.
    kernel_norm = float(np.linalg.norm(matrix, ord="fro"))
    target_norm = float(np.linalg.norm(target, ord="fro"))
    if (
        not np.isfinite(kernel_norm)
        or not np.isfinite(target_norm)
        or kernel_norm <= 0.0
        or target_norm <= 0.0
    ):
        raise ValueError("alignment norms must be finite and positive")
    result = float(np.sum((matrix / kernel_norm) * (target / target_norm)))
    tolerance = 16.0 * np.finfo(np.float64).eps * matrix.size
    if not np.isfinite(result) or not -1.0 - tolerance <= result <= 1.0 + tolerance:
        raise ValueError("alignment must lie within [-1, 1]")
    return min(1.0, max(-1.0, result))


class BalancedKernelAlignmentLoss(KernelLoss):
    """Return negative alignment with the balanced binary target matrix.

    The smaller label maps to -1 and the larger to +1. For class size n_c,
    t_i=z_i/n_c and T=t*t.T. The loss is -<K,T>/(||K||*||T||).
    Invalid data raises TypeError or ValueError.
    """

    def classes(self, targets: NDArray[np.float64]) -> tuple[float, float]:
        """Return the two finite numeric labels in sorted order."""
        values = copy_finite_real_array(targets, "y", 1)
        unique = np.unique(values)
        if len(unique) != 2:
            raise ValueError("y must contain exactly two distinct classes")
        return float(unique[0]), float(unique[1])

    def __call__(
        self, matrix: NDArray[np.float64], targets: NDArray[np.float64], /
    ) -> float:
        values = copy_finite_real_array(targets, "y", 1)
        classes = self.classes(values)
        gram = copy_finite_real_array(matrix, "kernel matrix", 2)
        if gram.shape != (len(values), len(values)) or not np.array_equal(gram, gram.T):
            raise ValueError("kernel must return a symmetric square training matrix")
        balanced = np.where(
            values == classes[0],
            -1.0 / np.count_nonzero(values == classes[0]),
            1.0 / np.count_nonzero(values == classes[1]),
        )
        return -_alignment(gram, np.outer(balanced, balanced))
