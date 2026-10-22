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

"""Computational-state-controlled null exchange partial mixer.

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

from ._base import _BaseMixer
from ._exchange import append_controlled_exchange
from ._validation import resolve_control_state


@final
class ControlledNullSwapMixer(_BaseMixer):
    """Exchange one-hot null/value targets under a control-state predicate.

    Controls are local qubits ``0..num_controls-1``; the final two qubits are
    the null and selected-value one-hot targets. ``None`` selects the all-zero
    state used by the NONE predicate; an explicit integer selects an extension
    to any computational control state. This is a partial mixer intended for
    ordered composition.

    Args:
        num_controls: Positive number of control qubits.
        control_state: Selected little-endian control value, or all-zero by default.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the width or control state is invalid.
    """

    __slots__ = ("_control_state", "_num_controls")

    def __init__(self, num_controls: int, *, control_state: int | None = None) -> None:
        self._num_controls = num_controls
        self._control_state = resolve_control_state(num_controls, control_state)
        super().__init__(num_controls + 2)
        self.validate()

    def _validate_configuration(self) -> None:
        return None

    def _build_mixer(self, prefix: str) -> Circuit:
        beta = self._beta_parameter(prefix)
        circuit = Circuit(self._num_controls + 2)
        append_controlled_exchange(
            circuit,
            range(self._num_controls),
            self._num_controls,
            self._num_controls + 1,
            beta,
            self._control_state,
        )
        return circuit
