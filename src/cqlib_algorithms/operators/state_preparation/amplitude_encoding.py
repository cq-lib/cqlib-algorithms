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

"""Dense classical-data amplitude encoding."""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from cqlib.circuit import Circuit

from ...protocols.state_preparation import StatePreparation
from ._state_synthesis import (
    build_mottonen_circuit,
    copy_complex_vector,
    normalize_nonzero,
    validate_prefix,
)


@final
class AmplitudeEncoding(StatePreparation):
    r"""Normalize and encode dense data as :math:`\sum_i x_i|i\rangle`."""

    __slots__ = ("_features",)

    def __init__(self, features: Sequence[complex]) -> None:
        copied = copy_complex_vector(features, name="features")
        self._features = normalize_nonzero(copied, name="features")

    def validate(self) -> None:
        """Validate the immutable normalized feature vector."""
        normalize_nonzero(self._features, name="features")

    def build_circuit(self, prefix: str) -> Circuit:
        """Return a new exact dense amplitude-encoding circuit."""
        validate_prefix(prefix)
        return build_mottonen_circuit(self._features)

    def num_parameters(self) -> int:
        """Return zero because all feature values are fixed numeric data."""
        return 0

    def num_qubits(self) -> int:
        """Return the base-two logarithm of the feature count."""
        return len(self._features).bit_length() - 1
