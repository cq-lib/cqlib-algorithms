"""Stable adapter for Cqlib feature-map preparations."""

from __future__ import annotations

import copy
from typing import TypeAlias, final

from cqlib.circuit import Circuit
from cqlib.circuit.ansatz import (
    IQPFeatureMap,
    PauliFeatureMap,
    ZFeatureMap,
    ZZFeatureMap,
)

from ._state_synthesis import validate_prefix

_CoreFeatureMap: TypeAlias = (
    ZFeatureMap | IQPFeatureMap | ZZFeatureMap | PauliFeatureMap
)
_SUPPORTED_FEATURE_MAPS = (
    ZFeatureMap,
    IQPFeatureMap,
    ZZFeatureMap,
    PauliFeatureMap,
)


@final
class FeatureMapPreparation:
    """Snapshot and delegate one supported Cqlib feature-map component.

    Supported inputs are :class:`ZFeatureMap`, :class:`IQPFeatureMap`,
    :class:`ZZFeatureMap`, and :class:`PauliFeatureMap`. Their construction,
    parameters, and feature semantics remain owned by Cqlib Core.
    """

    __slots__ = ("_feature_map",)

    def __init__(self, feature_map: _CoreFeatureMap) -> None:
        if not isinstance(feature_map, _SUPPORTED_FEATURE_MAPS):
            raise TypeError(
                "feature_map must be a ZFeatureMap, IQPFeatureMap, "
                "ZZFeatureMap, or PauliFeatureMap"
            )
        self._feature_map = copy.copy(feature_map)
        self.validate()

    def validate(self) -> None:
        """Validate the snapshotted Core feature-map configuration."""
        self._feature_map.validate()

    def build_circuit(self, prefix: str) -> Circuit:
        """Return a new circuit delegated to the Core feature map."""
        validate_prefix(prefix)
        return self._feature_map.build_circuit(prefix)

    def num_parameters(self) -> int:
        """Return the Core feature map's logical feature-slot count."""
        return self._feature_map.num_parameters()

    def num_qubits(self) -> int:
        """Return the Core feature map's register width."""
        return self._feature_map.num_qubits()


__all__ = ["FeatureMapPreparation"]
