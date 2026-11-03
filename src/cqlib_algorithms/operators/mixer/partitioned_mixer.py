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

"""Ordered composition of one-parameter partial mixers.

References
----------
[1] S. Hadfield, Z. Wang, B. O'Gorman, E. G. Rieffel, D. Venturelli, and
    R. Biswas, "From the Quantum Approximate Optimization Algorithm to a
    Quantum Alternating Operator Ansatz," Algorithms, 12(2), 34, 2019.
    DOI: 10.3390/a12020034.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import final

from cqlib.circuit import Circuit

from ..._utils.numeric import validate_positive_integer
from ...protocols.ansatz import Ansatz
from ._base import _BaseMixer


@dataclass(frozen=True, slots=True)
class MixerApplication:
    """Place one local mixer on an ordered tuple of global qubits.

    Attributes:
        mixer: Local one-parameter Ansatz.
        qubits: Unique global qubit indices in local-input order.
    """

    mixer: Ansatz
    qubits: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.mixer, Ansatz):
            raise TypeError("mixer must be an Ansatz")
        if isinstance(self.qubits, (str, bytes)) or not isinstance(
            self.qubits, Sequence
        ):
            raise TypeError("qubits must be a sequence of integers")
        frozen = tuple(self.qubits)
        for qubit in frozen:
            if isinstance(qubit, bool) or not isinstance(qubit, int):
                raise TypeError("qubits must contain only integers")
            if qubit < 0:
                raise ValueError("qubits must be non-negative")
        if len(set(frozen)) != len(frozen):
            raise ValueError("qubits must not contain duplicates")
        if len(frozen) != self.mixer.num_qubits():
            raise ValueError("qubits must match the local mixer width")
        object.__setattr__(self, "qubits", frozen)


@final
class PartitionedMixer(_BaseMixer):
    """Compose an ordered sequence of partial mixers sharing one beta.

    Applications are executed in input order. Each local mixer must expose
    exactly one parameter and must follow the ``{prefix}_beta`` naming
    convention. Applications may overlap because order is part of the mixer.

    Args:
        num_qubits: Positive global circuit width.
        parts: Ordered local mixer applications.

    Raises:
        TypeError: If the width or applications have invalid types.
        ValueError: If indices, local widths, or parameter contracts are invalid.
    """

    __slots__ = ("_parts",)

    def __init__(
        self,
        num_qubits: int,
        parts: Sequence[MixerApplication],
    ) -> None:
        validate_positive_integer(num_qubits, name="num_qubits")
        if isinstance(parts, (str, bytes)) or not isinstance(parts, Sequence):
            raise TypeError("parts must be a sequence of MixerApplication values")
        frozen = tuple(parts)
        if not frozen:
            raise ValueError("parts must not be empty")
        for part in frozen:
            if not isinstance(part, MixerApplication):
                raise TypeError("parts must contain only MixerApplication values")
            if any(qubit >= num_qubits for qubit in part.qubits):
                raise ValueError("a mixer application qubit is out of range")
            if part.mixer.num_parameters() != 1:
                raise ValueError("each local mixer must have one parameter")
        self._parts = frozen
        super().__init__(num_qubits)
        self.validate()

    def _validate_configuration(self) -> None:
        for part in self._parts:
            part.mixer.validate()
            if part.mixer.num_parameters() != 1:
                raise ValueError("each local mixer must have one parameter")
            if part.mixer.num_qubits() != len(part.qubits):
                raise ValueError("a local mixer width changed after construction")

    def _build_mixer(self, prefix: str) -> Circuit:
        circuit = Circuit(self._num_qubits)
        parameter_name = f"{prefix}_beta"
        for part in self._parts:
            local = part.mixer.build_circuit(prefix)
            if set(local.used_symbols) != {parameter_name}:
                raise ValueError("each local mixer must use the shared beta parameter")
            circuit.compose(local, list(part.qubits))
        return circuit
