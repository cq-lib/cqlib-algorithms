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

"""Arbitrary pure-state preparation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from cqlib.circuit import Circuit

from ...protocols.state_preparation import StatePreparation
from ._state_synthesis import (
    build_mottonen_circuit,
    copy_complex_vector,
    require_normalized,
    validate_prefix,
)


@final
class ArbitraryStatePreparation(StatePreparation):
    """Prepare a normalized complex state using Möttönen multiplexors.

    Amplitude index bits follow Cqlib's little-endian convention, with qubit
    zero as the least-significant bit. The copied input must have unit norm
    within relative and absolute tolerance 1e-12. Accepted roundoff is
    corrected before synthesis.
    """

    __slots__ = ("_amplitudes",)

    def __init__(self, amplitudes: Sequence[complex]) -> None:
        copied = copy_complex_vector(amplitudes, name="amplitudes")
        self._amplitudes = require_normalized(copied, name="amplitudes")

    def validate(self) -> None:
        """Validate the immutable normalized state data."""
        require_normalized(self._amplitudes, name="amplitudes")

    def build_circuit(self, prefix: str) -> Circuit:
        """Return a new exact Möttönen state-preparation circuit."""
        validate_prefix(prefix)
        return build_mottonen_circuit(self._amplitudes)

    def num_parameters(self) -> int:
        """Return zero because all amplitudes are fixed numeric data."""
        return 0

    def num_qubits(self) -> int:
        """Return the base-two logarithm of the amplitude count."""
        return len(self._amplitudes).bit_length() - 1
