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

"""Exact absolute-time job swap partial mixer.

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
from ._exchange import absolute_register_offsets, absolute_width, transition_circuit
from ._validation import (
    resolve_horizon,
    snapshot_processing_times,
    validate_job,
    validate_nonnegative_integer,
)


@final
class TimeSwapMixer(_BaseMixer):
    """Swap two adjacent jobs at one absolute start time.

    Job ``i`` owns one-hot qubits for starts ``0..horizon-p_i``. The four-body
    transition maps starts ``(t, t+p_i)`` to ``(t+p_j, t)`` and back.
    """

    __slots__ = (
        "_first_job",
        "_horizon",
        "_offsets",
        "_processing_times",
        "_second_job",
        "_start_time",
    )

    def __init__(
        self,
        processing_times: Sequence[int],
        first_job: int,
        second_job: int,
        start_time: int,
        *,
        horizon: int | None = None,
    ) -> None:
        frozen = snapshot_processing_times(processing_times)
        validate_job(first_job, num_jobs=len(frozen), name="first_job")
        validate_job(second_job, num_jobs=len(frozen), name="second_job")
        if first_job == second_job:
            raise ValueError("first_job and second_job must be distinct")
        validate_nonnegative_integer(start_time, name="start_time")
        resolved_horizon = resolve_horizon(frozen, horizon)
        maximum = resolved_horizon - frozen[first_job] - frozen[second_job]
        if start_time > maximum:
            raise ValueError("start_time leaves insufficient horizon for both jobs")
        self._processing_times = frozen
        self._first_job = first_job
        self._second_job = second_job
        self._start_time = start_time
        self._horizon = resolved_horizon
        self._offsets = absolute_register_offsets(frozen, resolved_horizon)
        super().__init__(absolute_width(frozen, resolved_horizon))
        self.validate()

    def _validate_configuration(self) -> None:
        return None

    def _build_mixer(self, prefix: str) -> Circuit:
        first_duration = self._processing_times[self._first_job]
        second_duration = self._processing_times[self._second_job]
        first_offset = self._offsets[self._first_job]
        second_offset = self._offsets[self._second_job]
        targets = [
            first_offset + self._start_time + second_duration,
            second_offset + self._start_time,
            first_offset + self._start_time,
            second_offset + self._start_time + first_duration,
        ]
        circuit = Circuit(self._num_qubits)
        circuit.compose(transition_circuit(prefix), targets)
        return circuit
