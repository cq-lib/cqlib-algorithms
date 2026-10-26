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

r"""Unified execution backend for error-mitigation techniques.

Techniques drive one of two surfaces: an expectation-value surface and a
sampling surface. :class:`QemExecutor` carries both behind a single backend
object, with optional request caching, so a technique can drive either
surface and compose with another technique without introducing a second
execution contract.
"""

from __future__ import annotations

from collections.abc import Sequence

from cqlib import Circuit
from cqlib.device import ExecutionResult
from cqlib.qis import Hamiltonian, PauliString

from ...protocols import (
    Estimator,
    EstimatorRequest,
    EstimatorResult,
    Sampler,
    SamplerRequest,
)


def _pauli_key(pauli: PauliString) -> tuple[int, int, int, complex]:
    return (
        pauli.num_qubits,
        pauli.x_mask,
        pauli.z_mask,
        pauli.phase.to_complex(),
    )


def _observable_key(observable: Hamiltonian | PauliString) -> tuple[object, ...]:
    if isinstance(observable, PauliString):
        return ("pauli", _pauli_key(observable))
    terms = tuple(
        sorted(
            (
                (_pauli_key(pauli), complex(coefficient))
                for pauli, coefficient in observable.terms
            ),
            key=lambda entry: (entry[0], entry[1].real, entry[1].imag),
        )
    )
    return ("hamiltonian", observable.num_qubits, terms)


def _circuit_key(circuit: Circuit) -> tuple[object, ...]:
    return (
        tuple(circuit.qubits),
        tuple(repr(operation) for operation in circuit.operations),
        circuit.global_phase,
        tuple(repr(item) for item in circuit.classical_vars),
        tuple(repr(item) for item in circuit.classical_values),
    )


def _estimator_cache_key(request: EstimatorRequest) -> tuple[object, ...]:
    return (
        _circuit_key(request.circuit),
        _observable_key(request.observable),
        request.shots,
        request.seed,
        tuple(sorted(request.parameter_bindings.items())),
    )


def _sampler_cache_key(request: SamplerRequest) -> tuple[object, ...]:
    return (
        _circuit_key(request.circuit),
        tuple(request.qubits),
        request.shots,
        request.seed,
    )


class QemExecutor:
    r"""Bundled execution backend for error-mitigation techniques.

    Holds one expectation-value backend and one count-based backend. The
    request and result types are the frozen library contracts: expectation
    requests and results are
    :class:`~cqlib_algorithms.protocols.EstimatorRequest` and
    :class:`~cqlib_algorithms.protocols.EstimatorResult`, sampling requests and
    results are :class:`~cqlib_algorithms.protocols.SamplerRequest` and
    :class:`cqlib.device.ExecutionResult`.

    Args:
        estimator: Expectation-value backend, or ``None`` when only sampling is used.
        sampler: Count-based backend, or ``None`` when only expectation values are used.
        use_cache: When ``True``, repeated requests are served from a per-instance cache
            keyed on circuit structure, observable, shots, seed, and parameter
            bindings. Disabled by default so execution stays explicit; caching is
            the prerequisite for chaining one technique on top of another.

    """

    __slots__ = (
        "_estimator",
        "_estimator_cache",
        "_sampler",
        "_sampler_cache",
        "_use_cache",
    )

    _estimator: Estimator | None
    _sampler: Sampler | None
    _use_cache: bool
    _estimator_cache: dict[tuple[object, ...], EstimatorResult]
    _sampler_cache: dict[tuple[object, ...], ExecutionResult]

    def __init__(
        self,
        *,
        estimator: Estimator | None = None,
        sampler: Sampler | None = None,
        use_cache: bool = False,
    ) -> None:
        self._estimator = estimator
        self._sampler = sampler
        self._use_cache = use_cache
        self._estimator_cache = {}
        self._sampler_cache = {}

    @property
    def estimator(self) -> Estimator | None:
        r"""Return the injected expectation-value backend, or ``None``.

        A technique executor that consumes only one execution surface needs to
        rebuild a wrapper around the other one so the result still satisfies
        the full executor protocol.
        """
        return self._estimator

    @property
    def sampler(self) -> Sampler | None:
        r"""Return the injected count-based backend, or ``None``.

        Mirror of :attr:`estimator` for the sampling surface.
        """
        return self._sampler

    @property
    def use_cache(self) -> bool:
        r"""Return whether repeated requests are served from the cache.

        Wrappers reproduce this flag on the executor they return, so chaining
        one technique on top of another keeps the caching semantics.
        """
        return self._use_cache

    def estimate(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        r"""Estimate a batch of observable expectations in input order.

        Args:
            requests: Expectation requests, delegated unchanged to the estimator.

        Returns:
            One result per request, in input order.

        Raises:
            ValueError: If this executor holds no estimator backend.

        """
        if self._estimator is None:
            raise ValueError("QemExecutor has no estimator backend")
        if not self._use_cache:
            return self._estimator.run(requests)

        misses: list[EstimatorRequest] = []
        miss_keys: list[tuple[object, ...]] = []
        seen: set[tuple[object, ...]] = set()
        for request in requests:
            key = _estimator_cache_key(request)
            if key in seen or key in self._estimator_cache:
                continue
            seen.add(key)
            misses.append(request)
            miss_keys.append(key)

        for key, result in zip(miss_keys, self._estimator.run(misses)):
            self._estimator_cache[key] = result

        return [
            self._estimator_cache[_estimator_cache_key(request)] for request in requests
        ]

    def sample(
        self,
        requests: Sequence[SamplerRequest],
        /,
    ) -> list[ExecutionResult]:
        r"""Sample a batch of ``counts`` distributions in input order.

        Args:
            requests: Sampling requests, delegated unchanged to the sampler.

        Returns:
            One terminal result per request, in input order.

        Raises:
            ValueError: If this executor holds no sampler backend.

        """
        if self._sampler is None:
            raise ValueError("QemExecutor has no sampler backend")
        if not self._use_cache:
            return self._sampler.run(requests)

        misses: list[SamplerRequest] = []
        miss_keys: list[tuple[object, ...]] = []
        seen: set[tuple[object, ...]] = set()
        for request in requests:
            key = _sampler_cache_key(request)
            if key in seen or key in self._sampler_cache:
                continue
            seen.add(key)
            misses.append(request)
            miss_keys.append(key)

        for key, result in zip(miss_keys, self._sampler.run(misses)):
            self._sampler_cache[key] = result

        return [self._sampler_cache[_sampler_cache_key(request)] for request in requests]
