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

from ...protocols.state_preparation import StatePreparation
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
class FeatureMapPreparation(StatePreparation):
    """Snapshot and delegate one supported Cqlib feature-map component.

    Supported inputs are :class:`ZFeatureMap`, :class:`IQPFeatureMap`,
    :class:`ZZFeatureMap`, and :class:`PauliFeatureMap`. Their construction,
    parameters, and feature semantics remain owned by Cqlib Core.

    Args:
        feature_map: Supported Cqlib feature-map instance to snapshot.

    Raises:
        TypeError: If the component type is unsupported.
        ValueError: If the feature map fails validation.
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
