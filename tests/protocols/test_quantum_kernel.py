# This code is part of Cqlib.
# (C) Copyright China Telecom Quantum Group 2026
# Licensed under the Apache License, Version 2.0; see LICENSE.txt.
"""Quantum-kernel abstraction and scalar minimizer result contracts."""

from dataclasses import FrozenInstanceError, replace

import pytest

from cqlib_algorithms.protocols import (
    QuantumKernel,
    ScalarMinimizerResult,
    TrainableQuantumKernel,
)


@pytest.mark.parametrize("abstract", [QuantumKernel, TrainableQuantumKernel])
def test_abstract_kernels_require_implementations(abstract):
    with pytest.raises(TypeError):
        abstract()


def test_scalar_result_is_immutable_and_replaceable():
    result = ScalarMinimizerResult((0.2,), 0.3, True, 0, "Done", 1)
    assert replace(result, value=0.4).value == 0.4
    with pytest.raises(FrozenInstanceError):
        result.value = 0.4
