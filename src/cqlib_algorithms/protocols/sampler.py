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

"""Abstract sampling interface and request for Cqlib circuits."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from cqlib.device import ExecutionResult

from cqlib import Circuit, Qubit


def _validate_seed(seed: int | None) -> None:
    if seed is None:
        return
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer or None")
    if seed < 0:
        raise ValueError("seed must be non-negative")


def spawn_seeds(seed: int | None, count: int, /) -> list[int]:
    """Derive ``count`` independent child seeds from one parent seed.

    A caller that spends one seed across several requests must give each
    request its own seed. Repeating one seed would make the requests replay
    the same draw, so their noise would not average out. The children are a
    pure function of ``seed``, which keeps the parent seed sufficient to
    reproduce the whole run.

    Args:
        seed: Parent seed, or ``None`` to derive fresh child seeds.
        count: Number of child seeds to derive.

    Returns:
        ``count`` non-negative integer seeds.

    Raises:
        TypeError: If ``seed`` or ``count`` has the wrong type.
        ValueError: If ``seed`` or ``count`` is negative.
    """
    _validate_seed(seed)
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer")
    if count < 0:
        raise ValueError("count must be non-negative")
    return [int(value) for value in np.random.SeedSequence(seed).generate_state(count)]


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
        seed: Non-negative seed of the count draw, or ``None`` to draw fresh
            randomness. A sampler that draws counts from a random source must
            derive that source from this value, so that one request is
            reproducible.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument violates a request invariant.
        cqlib.circuit.CircuitError: If ``Circuit.validate`` fails.
    """

    circuit: Circuit
    qubits: tuple[Qubit, ...]
    shots: int
    seed: int | None = None

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
        _validate_seed(self.seed)

        self.circuit.validate()
        if self.circuit.used_symbols:
            raise ValueError("circuit must not contain used symbolic parameters")
        if not self.qubits:
            raise ValueError("qubits must not be empty")
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError("qubits must be unique")
        if any(qubit not in self.circuit.qubits for qubit in self.qubits):
            raise ValueError("qubits must belong to circuit")


class Sampler(ABC):
    """Abstract base class for synchronous batch circuit sampling.

    Implementations must call :meth:`SamplerRequest.validate` and must not
    mutate request circuits. A Core measurement, when needed, is created on a
    private execution circuit for ``request.qubits``. Implementations return
    exactly one terminal
    :class:`cqlib.device.ExecutionResult` per request, in input order. An empty
    request sequence produces an empty list. Successful results use the
    request's shot count, and their qubit and result-bit order equals
    ``request.qubits``.

    An implementation whose counts come from a random source must derive that
    source from :attr:`SamplerRequest.seed`, so that a request with an explicit
    seed reproduces its counts. An implementation backed by a device that does
    not accept a seed may ignore it, and must then document that its counts are
    not reproducible.
    """

    __slots__ = ()

    @abstractmethod
    def run(
        self,
        requests: Sequence[SamplerRequest],
        /,
    ) -> list[ExecutionResult]:
        """Execute a batch and return terminal results in input order.

        Args:
            requests: Valid sampling requests in output order.

        Returns:
            One successful terminal result per request.
        """
        raise NotImplementedError


def validate_sampler_result(
    result: ExecutionResult,
    request: SamplerRequest,
) -> None:
    """Validate one result against its sampling request.

    This helper defines the result-side half of the public synchronous
    :class:`Sampler` contract. It checks execution status and metadata as well
    as the complete, unfiltered count distribution used by algorithms.

    Args:
        result: Terminal Core execution result returned by a Sampler.
        request: Sampling request that produced ``result``.

    Raises:
        TypeError: If ``request`` has the wrong type.
        ValueError: If result status, metadata, or counts violate the request.
    """
    if not isinstance(request, SamplerRequest):
        raise TypeError("request must be a SamplerRequest")
    if not result.status.is_terminal() or not result.status.is_success():
        raise ValueError("sampler must return successful terminal results")
    if result.shots != request.shots:
        raise ValueError("sampler result shots must match the request")
    if result.num_qubits != len(request.qubits):
        raise ValueError("sampler result width must match the requested qubits")
    if result.qubits != list(request.qubits):
        raise ValueError("sampler result qubits must preserve request order")

    width = len(request.qubits)
    total = 0
    for bitstring, count in result.counts.items():
        if not isinstance(bitstring, str) or len(bitstring) != width:
            raise ValueError("sampler count keys must match the requested width")
        if any(bit not in "01" for bit in bitstring):
            raise ValueError("sampler count keys must be binary strings")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("sampler counts must be non-negative integers")
        total += count
    if total != request.shots:
        raise ValueError("sampler counts must sum to the requested shots")


def validate_sampler_results(
    results: object,
    requests: Sequence[SamplerRequest],
) -> list[ExecutionResult]:
    """Validate a complete ordered result batch from a Sampler.

    Args:
        results: Object returned by :meth:`Sampler.run`.
        requests: Requests passed to the Sampler in the same order.

    Returns:
        The validated result list, unchanged.

    Raises:
        TypeError: If the result container, one of its entries, or a request
            has the wrong type.
        ValueError: If the batch size or an individual result violates the
            public Sampler contract.
    """
    if not isinstance(results, list):
        raise TypeError("sampler must return a list of ExecutionResult values")
    if len(results) != len(requests):
        raise ValueError("sampler must return one result per request")

    for result, request in zip(results, requests):
        if not isinstance(result, ExecutionResult):
            raise TypeError("sampler results must be cqlib.device.ExecutionResult")
        validate_sampler_result(result, request)
    return results
