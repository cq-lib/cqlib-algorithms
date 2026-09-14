"""Smoke checks for the public operators packages."""

from importlib import import_module


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
    assert set(operators.__all__) == expected
    assert set(state_preparation.__all__) == expected
    for name in expected:
        assert getattr(operators, name) is getattr(state_preparation, name)
