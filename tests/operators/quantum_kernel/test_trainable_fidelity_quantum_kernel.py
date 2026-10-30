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

"""Trainable fidelity binding and configuration snapshots."""

import numpy as np
import pytest
from cqlib import Circuit, Parameter

from cqlib_algorithms.operators import TrainableFidelityQuantumKernel


def test_fidelity_family_snapshot_and_fixed_result_after_new_bindings():
    circuit = Circuit(1)
    circuit.ry(0, Parameter("w") * Parameter("x"))
    inputs = ["x"]
    names = ["w"]
    family = TrainableFidelityQuantumKernel(circuit, inputs, names)
    first = family.bind_parameters({"w": 0.4})
    circuit.x(0)
    inputs[:] = ["other"]
    names[:] = ["other"]
    second = family.bind_parameters({"w": 1.3})
    X = [[0.0], [1.0]]
    assert first.evaluate(X)[0, 1] == pytest.approx(np.cos(0.2) ** 2)
    assert second.evaluate(X)[0, 1] == pytest.approx(np.cos(0.65) ** 2)
    for bad in [{}, {"w": np.inf}, {"w": True}, {"w": 1j}, {"z": 1.0}]:
        with pytest.raises((TypeError, ValueError)):
            family.bind_parameters(bad)
