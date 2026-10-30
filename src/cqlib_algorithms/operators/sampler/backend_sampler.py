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

"""Synchronous batch sampling through an externally supplied backend adapter."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import copy
from typing import final

from cqlib import Circuit
from cqlib.circuit import Measurement
from cqlib.device import ExecutionResult

from ...protocols.sampler import Sampler, SamplerRequest

_BackendExecutor = Callable[
    [Circuit, Measurement, int],
    ExecutionResult,
]


def _validate_result(result: ExecutionResult, request: SamplerRequest) -> None:
    """Validate the part of the Sampler contract observable after execution."""
    if not result.status.is_terminal():
        raise ValueError("backend executor must return terminal results")
    if not result.status.is_success():
        return
    if result.shots != request.shots:
        raise ValueError("successful backend result shots must match the request")
    if result.num_qubits != len(request.qubits):
        raise ValueError(
            "successful backend result width must match the requested qubits"
        )
    if result.qubits != list(request.qubits):
        raise ValueError("successful backend result qubits must preserve request order")


@final
class BackendSampler(Sampler):
    """Delegate measured private circuits to a synchronous backend adapter.

    ``executor`` receives a private circuit copy, the exact Core measurement
    receipt appended in the requested qubit order, and the shot count. Passing
    the receipt avoids guessing which classical value represents the target
    readout when the source circuit already contains dynamic operations. The
    executor must synchronously return one terminal Core ``ExecutionResult``.

    Authentication, asynchronous job submission, polling, cancellation and
    transport are deliberately outside this component. A platform integration
    owns those concerns and exposes the small synchronous callable consumed
    here. Failed or cancelled terminal results are preserved; successful
    results are checked against the frozen Sampler contract.

    The adapter callable receives no seed, so ``request.seed`` cannot reach a
    remote device and the counts of this sampler are not reproducible. A
    simulator-backed adapter that can honour a seed should be exposed instead
    through :class:`~cqlib_algorithms.operators.sampler.StatevectorSampler`.

    Args:
        executor: Synchronous single-execution callback supplied by an adapter.

    Raises:
        TypeError: If ``executor`` is not callable, a request has the wrong
            type, or the executor returns an invalid result.
        ValueError: If a returned terminal state or successful result metadata
            violates the Sampler contract.
    """

    __slots__ = ("_executor",)

    def __init__(self, executor: _BackendExecutor) -> None:
        if not callable(executor):
            raise TypeError("executor must be callable")
        self._executor = executor

    def run(
        self,
        requests: Sequence[SamplerRequest],
        /,
    ) -> list[ExecutionResult]:
        """Execute a validated request batch through the configured adapter."""
        results: list[ExecutionResult] = []
        for request in requests:
            if not isinstance(request, SamplerRequest):
                raise TypeError("requests must contain only SamplerRequest values")
            request.validate()

            execution_circuit = copy(request.circuit)
            measurement = execution_circuit.measure_bits(list(request.qubits))
            result = self._executor(execution_circuit, measurement, request.shots)
            if not isinstance(result, ExecutionResult):
                raise TypeError(
                    "backend executor must return cqlib.device.ExecutionResult"
                )
            _validate_result(result, request)
            results.append(result)
        return results
