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

"""Computational-state-controlled bit-flip mixer.

References
----------
[1] S. Hadfield, Z. Wang, B. O'Gorman, E. G. Rieffel, D. Venturelli, and
    R. Biswas, "From the Quantum Approximate Optimization Algorithm to a
    Quantum Alternating Operator Ansatz," Algorithms, 12(2), 34, 2019.
    DOI: 10.3390/a12020034.
"""

from __future__ import annotations

from typing import final

from cqlib.circuit import Circuit
from cqlib.circuit.gates import StandardGate

from ._base import _BaseMixer
from ._validation import resolve_control_state


@final
class ControlledBitFlipMixer(_BaseMixer):
    """Apply ``exp(-i beta X)`` when controls equal ``control_state``.

    Controls occupy local qubits ``0..num_controls-1`` in little-endian order;
    the target is local qubit ``num_controls``. ``None`` selects the all-zero
    state used by the NOR predicate; an explicit integer selects an extension
    to any computational control state.
    """

    __slots__ = ("_control_state", "_num_controls")

    def __init__(self, num_controls: int, *, control_state: int | None = None) -> None:
        self._num_controls = num_controls
        self._control_state = resolve_control_state(num_controls, control_state)
        super().__init__(num_controls + 1)
        self.validate()

    def _validate_configuration(self) -> None:
        return None

    def _build_mixer(self, prefix: str) -> Circuit:
        beta = self._beta_parameter(prefix)
        circuit = Circuit(self._num_controls + 1)
        controls = list(range(self._num_controls))
        target = self._num_controls
        for position, qubit in enumerate(controls):
            if not (self._control_state >> position) & 1:
                circuit.x(qubit)
        if controls:
            circuit.append_mc_gate(
                StandardGate.RX(2.0 * beta).control(self._num_controls),
                [*controls, target],
            )
        else:
            circuit.rx(target, 2.0 * beta)
        for position, qubit in reversed(tuple(enumerate(controls))):
            if not (self._control_state >> position) & 1:
                circuit.x(qubit)
        return circuit
