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

"""Finite-shot statevector sampling on a private compact circuit."""

from __future__ import annotations

from collections.abc import Sequence
from copy import copy
from typing import final

from cqlib.device import ExecutionResult
from cqlib.qis.state import Statevector

from cqlib import Circuit

from ...protocols.sampler import Sampler, SamplerRequest


def _validate_static_unitary(circuit: Circuit) -> None:
    """Reject classical and non-unitary operations before state simulation."""
    # Core cannot decompose dynamic circuits, so reject those at the outer
    # boundary before recursively inspecting any circuit-defined gates.
    for operation in circuit.operations:
        if operation.instruction_type == "circuit":
            continue
        if operation.instruction_type in {"standard", "mcgate", "unitary", "delay"}:
            continue
        if operation.instruction_type == "directive" and operation.name == "barrier":
            continue
        raise ValueError(
            "circuit must contain only static unitary operations "
            "(barriers and delays are permitted)"
        )

    for operation in circuit.decompose().operations:
        if operation.instruction_type in {"standard", "mcgate", "unitary", "delay"}:
            continue
        if operation.instruction_type == "directive" and operation.name == "barrier":
            continue
        raise ValueError(
            "circuit must contain only static unitary operations "
            "(barriers and delays are permitted)"
        )


@final
class StatevectorSampler(Sampler):
    """Sample fully bound, static circuits with Core's statevector simulator.

    Each request is validated immediately before use. The source circuit is
    copied and its qubits are mapped by insertion position onto ``0..n-1``.
    State evolution happens before a measurement receipt is created on this
    private compact circuit. The resulting Core counts are finite-shot
    frequencies, not exact probabilities or quasi-probabilities. The result
    retains the original request qubit order and shot count.

    No random seed is offered because the pinned Core ``Statevector.sample``
    interface does not expose one. Stochastic counts are not reproducible.

    Raises:
        TypeError: If a batch member is not a ``SamplerRequest``.
        ValueError: If the circuit contains classical or non-unitary operations.
        cqlib.circuit.CircuitError: If a Core circuit operation fails.
    """

    __slots__ = ()

    def run(
        self,
        requests: Sequence[SamplerRequest],
        /,
    ) -> list[ExecutionResult]:
        """Return one successful terminal Core result for each request."""
        results: list[ExecutionResult] = []
        for request in requests:
            if not isinstance(request, SamplerRequest):
                raise TypeError("requests must contain only SamplerRequest values")
            request.validate()
            source = copy(request.circuit)
            source.validate()
            _validate_static_unitary(source)

            source_qubits = source.qubits
            positions = {
                qubit: position for position, qubit in enumerate(source_qubits)
            }
            execution_circuit = Circuit(len(source_qubits))
            execution_circuit.compose(source, list(range(len(source_qubits))))
            state = Statevector.from_circuit(execution_circuit)
            measured_positions = [positions[qubit] for qubit in request.qubits]
            receipt = execution_circuit.measure_bits(measured_positions)
            sampled = state.sample(receipt, request.shots)
            results.append(
                ExecutionResult.from_counts(
                    sampled.task_id,
                    list(request.qubits),
                    request.shots,
                    len(request.qubits),
                    sampled.counts,
                    sampled.backend,
                )
            )
        return results
