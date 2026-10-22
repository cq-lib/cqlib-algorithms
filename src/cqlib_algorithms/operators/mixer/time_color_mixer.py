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

"""Exact time-color partitioned scheduling mixer.

References
----------
[1] S. Hadfield, Z. Wang, B. O'Gorman, E. G. Rieffel, D. Venturelli, and
    R. Biswas, "From the Quantum Approximate Optimization Algorithm to a
    Quantum Alternating Operator Ansatz," Algorithms, 12(2), 34, 2019.
    DOI: 10.3390/a12020034.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from cqlib.circuit import Circuit

from ._base import _BaseMixer
from ._exchange import (
    absolute_register_offsets,
    absolute_width,
    complete_graph_edge_colors,
    transition_circuit,
)
from ._validation import (
    resolve_horizon,
    snapshot_processing_times,
    validate_positive_integer,
)


@final
class TimeColorMixer(_BaseMixer):
    """Compose time-swap partial mixers in time then edge-color order.

    Args:
        processing_times: Positive processing duration for each job.
        horizon: Optional positive schedule horizon.
        repetitions: Positive schedule repetition count.

    Raises:
        TypeError: If an option has the wrong type.
        ValueError: If durations, horizon, or repetitions are invalid.
    """

    __slots__ = ("_horizon", "_offsets", "_processing_times", "_repetitions")

    def __init__(
        self,
        processing_times: Sequence[int],
        *,
        horizon: int | None = None,
        repetitions: int = 1,
    ) -> None:
        frozen = snapshot_processing_times(processing_times)
        resolved_horizon = resolve_horizon(frozen, horizon)
        validate_positive_integer(repetitions, name="repetitions")
        self._processing_times = frozen
        self._horizon = resolved_horizon
        self._repetitions = repetitions
        self._offsets = absolute_register_offsets(frozen, resolved_horizon)
        super().__init__(absolute_width(frozen, resolved_horizon))
        self.validate()

    def _validate_configuration(self) -> None:
        return None

    def _build_mixer(self, prefix: str) -> Circuit:
        circuit = Circuit(self._num_qubits)
        local = transition_circuit(prefix)
        colors = complete_graph_edge_colors(len(self._processing_times))
        for _ in range(self._repetitions):
            for start_time in range(self._horizon + 1):
                for color in colors:
                    for first, second in color:
                        first_duration = self._processing_times[first]
                        second_duration = self._processing_times[second]
                        if (
                            start_time
                            > self._horizon - first_duration - second_duration
                        ):
                            continue
                        targets = [
                            self._offsets[first] + start_time + second_duration,
                            self._offsets[second] + start_time,
                            self._offsets[first] + start_time,
                            self._offsets[second] + start_time + first_duration,
                        ]
                        circuit.compose(local, targets)
        return circuit
