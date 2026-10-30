# This code is part of Cqlib.
# (C) Copyright China Telecom Quantum Group 2026
# Licensed under the Apache License, Version 2.0; see LICENSE.txt.
"""Independent matrix references for balanced kernel alignment."""

import numpy as np
import pytest

from cqlib_algorithms.operators import BalancedKernelAlignmentLoss


@pytest.mark.reference
def test_alignment_imbalanced_three_to_one_labels():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]])
    gram = vectors @ vectors.T
    weights = np.array([-1 / 3, -1 / 3, -1 / 3, 1.0])
    target = weights[:, None] * weights[None, :]
    numerator = sum(gram[i, j] * target[i, j] for i in range(4) for j in range(4))
    expected = -numerator / (
        np.sqrt(sum(float(x * x) for x in gram.flat))
        * np.sqrt(sum(float(x * x) for x in target.flat))
    )
    assert BalancedKernelAlignmentLoss()(
        gram, np.array([2.0, 2.0, 2.0, 7.0])
    ) == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("labels", [[1.0, 1.0], [1.0, 2.0, 3.0], [np.nan, 1.0]])
def test_target_semantics(labels):
    with pytest.raises(ValueError):
        BalancedKernelAlignmentLoss()(np.eye(len(labels)), np.array(labels))
