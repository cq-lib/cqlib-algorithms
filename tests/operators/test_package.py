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

"""Smoke checks for the public operators packages."""

from importlib import import_module

from cqlib_algorithms.protocols import StatePreparation


def test_state_preparation_operators_are_exported_from_functional_packages():
    operators = import_module("cqlib_algorithms.operators")
    state_preparation = import_module("cqlib_algorithms.operators.state_preparation")
    expected = {
        "AmplitudeEncoding",
        "AngleEncoding",
        "ArbitraryStatePreparation",
        "BasisEncoding",
        "BasisStateInitialization",
        "FeatureMapPreparation",
        "GHZStatePreparation",
        "ProductStatePreparation",
        "SparseAmplitudeEncoding",
        "UniformSuperposition",
        "UserCircuitPreparation",
    }
    assert expected <= set(operators.__all__)
    assert set(state_preparation.__all__) == expected
    for name in expected:
        assert getattr(operators, name) is getattr(state_preparation, name)
        assert issubclass(getattr(state_preparation, name), StatePreparation)
