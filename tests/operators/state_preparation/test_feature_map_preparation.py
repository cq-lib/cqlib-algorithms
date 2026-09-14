"""Tests for the stable Cqlib feature-map adapter."""

import pytest

from cqlib.circuit.ansatz import (
    IQPFeatureMap,
    PauliFeatureMap,
    ZFeatureMap,
    ZZFeatureMap,
)
from cqlib_algorithms.operators.state_preparation import FeatureMapPreparation


@pytest.mark.parametrize(
    "feature_map",
    [ZFeatureMap(3), IQPFeatureMap(3), ZZFeatureMap(3), PauliFeatureMap(3)],
)
def test_snapshots_and_delegates_supported_feature_maps(feature_map: object) -> None:
    preparation = FeatureMapPreparation(feature_map)  # type: ignore[arg-type]
    direct = feature_map.build_circuit("data")  # type: ignore[union-attr]
    first = preparation.build_circuit("data")
    second = preparation.build_circuit("data")

    preparation.validate()
    assert preparation.num_qubits() == 3
    assert preparation.num_parameters() == 3
    assert first is not second
    assert first.used_symbols == direct.used_symbols
    assert [repr(operation) for operation in first.operations] == [
        repr(operation) for operation in direct.operations
    ]


def test_snapshot_is_unaffected_by_functional_reconfiguration() -> None:
    original = ZFeatureMap(2)
    preparation = FeatureMapPreparation(original)
    reconfigured = original.reps(5)
    assert len(preparation.build_circuit("x").operations) != len(
        reconfigured.build_circuit("x").operations
    )


def test_rejects_unsupported_type_and_propagates_core_validation_error() -> None:
    with pytest.raises(TypeError, match="feature_map must be"):
        FeatureMapPreparation(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        FeatureMapPreparation(PauliFeatureMap(1))


def test_validates_prefix_type() -> None:
    with pytest.raises(TypeError, match="prefix must be a str"):
        FeatureMapPreparation(ZFeatureMap(2)).build_circuit(1)  # type: ignore[arg-type]
