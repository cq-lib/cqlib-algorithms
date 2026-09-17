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
"""Structural sampling protocol and request for Cqlib circuits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from cqlib.device import ExecutionResult

from cqlib import Circuit, Qubit


@dataclass(frozen=True, slots=True)
class SamplerRequest:
    """A synchronous sampling request for one fully bound circuit.

    ``qubits`` is the explicit measurement order and is retained exactly. A
    sampler must not mutate the request circuit. When execution requires a
    Core measurement, the sampler creates it on a private execution circuit
    and reports result qubits and bits in the requested order. Keeping the
    measurement circuit-local avoids accepting stale or forged Core
    measurement handles in the public request.

    The dataclass is only shallowly frozen: ``Circuit`` remains mutable.
    Callers must not modify it during the request's lifetime. A sampler must
    nevertheless call :meth:`validate` immediately before execution. Such
    revalidation detects invalid mutations, but cannot detect every
    semantically different mutation that leaves the circuit valid.

    Args:
        circuit: A valid circuit with no used symbolic parameters.
        qubits: A non-empty, ordered tuple of distinct circuit qubits.
        shots: A positive number of samples. Boolean values are not accepted.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument violates a request invariant.
        cqlib.circuit.CircuitError: If ``Circuit.validate`` fails.
    """

    circuit: Circuit
    qubits: tuple[Qubit, ...]
    shots: int

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Revalidate the request against the circuit's current state.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If an argument violates a request invariant.
            cqlib.circuit.CircuitError: If ``Circuit.validate`` fails.
        """
        if not isinstance(self.circuit, Circuit):
            raise TypeError("circuit must be a cqlib.Circuit")
        if not isinstance(self.qubits, tuple):
            raise TypeError("qubits must be a tuple")
        if any(not isinstance(qubit, Qubit) for qubit in self.qubits):
            raise TypeError("qubits must contain only cqlib.Qubit values")
        if isinstance(self.shots, bool) or not isinstance(self.shots, int):
            raise TypeError("shots must be an integer")
        if self.shots <= 0:
            raise ValueError("shots must be greater than zero")

        self.circuit.validate()
        if self.circuit.used_symbols:
            raise ValueError("circuit must not contain used symbolic parameters")
        if not self.qubits:
            raise ValueError("qubits must not be empty")
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError("qubits must be unique")
        if any(qubit not in self.circuit.qubits for qubit in self.qubits):
            raise ValueError("qubits must belong to circuit")


class Sampler(Protocol):
    """Structural protocol for synchronous batch circuit sampling.

    Implementations must call :meth:`SamplerRequest.validate` and must not
    mutate request circuits. A Core measurement, when needed, is created on a
    private execution circuit for ``request.qubits``. Implementations return
    exactly one terminal
    :class:`cqlib.device.ExecutionResult` per request, in input order. An empty
    request sequence produces an empty list. Successful results use the
    request's shot count, and their qubit and result-bit order equals
    ``request.qubits``.
    """

    def run(
        self,
        requests: Sequence[SamplerRequest],
        /,
    ) -> list[ExecutionResult]:
        """Execute a batch and return terminal results in input order."""
        ...



