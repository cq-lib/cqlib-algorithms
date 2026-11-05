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

"""Tests for the public state-fidelity contract."""

from __future__ import annotations

import inspect
from collections.abc import Sequence

from cqlib_algorithms.operators.state_fidelity import ComputeUncomputeFidelity
from cqlib_algorithms.protocols import StateFidelity


class _FakeFidelity(StateFidelity[int, float]):
    def run(self, requests: Sequence[int], /) -> list[float]:
        return [float(value) for value in requests]


def test_state_fidelity_is_a_nominal_abstract_contract() -> None:
    assert inspect.isabstract(StateFidelity)
    assert isinstance(_FakeFidelity(), StateFidelity)
    assert issubclass(ComputeUncomputeFidelity, StateFidelity)


def test_state_fidelity_preserves_batch_order_and_empty_batches() -> None:
    fidelity = _FakeFidelity()

    assert fidelity.run([2, 1]) == [2.0, 1.0]
    assert fidelity.run([]) == []
