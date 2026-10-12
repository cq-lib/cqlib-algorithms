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

"""Single-qudit ring mixer."""

from __future__ import annotations

from typing import final

from cqlib.circuit import Circuit, Parameter
from cqlib.circuit.ansatz import EvolutionStrategy

from ._base import _BaseMixer
from .nearby_values_mixer import NearbyValuesMixer


@final
class RingMixer(_BaseMixer):
    """Specialize :class:`NearbyValuesMixer` to cyclic radius one."""

    __slots__ = ()

    def __init__(
        self,
        num_values: int,
        *,
        strategy: EvolutionStrategy | None = None,
    ) -> None:
        self._inner = NearbyValuesMixer(num_values, 1, strategy=strategy)
        super().__init__(self._inner.num_qubits())

    def _validate_configuration(self) -> None:
        self._inner.validate()

    def _build_mixer(self, prefix: str, beta: Parameter) -> Circuit:
        del beta
        return self._inner.build_circuit(prefix)
