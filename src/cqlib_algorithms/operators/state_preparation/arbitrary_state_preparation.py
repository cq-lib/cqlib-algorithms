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

"""Arbitrary pure-state preparation.

References
----------
[1] M. Möttönen, J. J. Vartiainen, V. Bergholm, and M. M. Salomaa,
    "Transformation of quantum states using uniformly controlled rotations,"
    Quantum Information and Computation, 5(6), 467-473, 2005.
    DOI: 10.26421/QIC5.6-5.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from cqlib.circuit import Circuit

from ..._utils.parameters import validate_parameter_prefix
from ...protocols.state_preparation import StatePreparation
from ._state_synthesis import (
    build_mottonen_circuit,
    copy_complex_vector,
    normalize_unit_vector,
    validate_normalized_vector,
)


@final
class ArbitraryStatePreparation(StatePreparation):
    """Prepare a normalized complex state using Möttönen multiplexors.

    Amplitude index bits follow Cqlib's little-endian convention, with qubit
    zero as the least-significant bit. The copied input must have unit norm
    within relative and absolute tolerance 1e-12. Accepted roundoff is
    corrected before synthesis.

    Args:
        amplitudes: Finite, normalized complex vector of power-of-two length.

    Raises:
        TypeError: If ``amplitudes`` is not a numeric sequence.
        ValueError: If its length, entries, or norm are invalid.
    """

    __slots__ = ("_amplitudes",)

    def __init__(self, amplitudes: Sequence[complex]) -> None:
        copied = copy_complex_vector(amplitudes, name="amplitudes")
        self._amplitudes = normalize_unit_vector(copied, name="amplitudes")

    def validate(self) -> None:
        """Validate the immutable normalized state data."""
        validate_normalized_vector(self._amplitudes, name="amplitudes")

    def build_circuit(self, prefix: str) -> Circuit:
        """Return a new exact Möttönen state-preparation circuit."""
        validate_parameter_prefix(prefix)
        return build_mottonen_circuit(self._amplitudes)

    def num_parameters(self) -> int:
        """Return zero because all amplitudes are fixed numeric data."""
        return 0

    def num_qubits(self) -> int:
        """Return the base-two logarithm of the amplitude count."""
        return len(self._amplitudes).bit_length() - 1
