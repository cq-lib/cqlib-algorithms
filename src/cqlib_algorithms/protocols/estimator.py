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

"""Abstract expectation-estimation interface and value types."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType

from cqlib import Circuit
from cqlib.qis import Hamiltonian, PauliString


def _validate_positive_shots(shots: int | None) -> None:
    if shots is None:
        return
    if isinstance(shots, bool) or not isinstance(shots, int):
        raise TypeError("shots must be an integer or None")
    if shots <= 0:
        raise ValueError("shots must be greater than zero")


def _validate_seed(seed: int | None) -> None:
    if seed is None:
        return
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer or None")
    if seed < 0:
        raise ValueError("seed must be non-negative")


def _immutable_bindings(bindings: Mapping[str, float]) -> Mapping[str, float]:
    if not isinstance(bindings, Mapping):
        raise TypeError("parameter_bindings must be a mapping")

    normalized: dict[str, float] = {}
    for name, value in bindings.items():
        if not isinstance(name, str):
            raise TypeError("parameter binding names must be strings")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("parameter binding values must be real numbers")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError("parameter binding values must be finite")
        normalized[name] = numeric_value
    return MappingProxyType(normalized)


def _is_finite(value: complex) -> bool:
    return math.isfinite(value.real) and math.isfinite(value.imag)


def _validate_hermitian(observable: Hamiltonian | PauliString) -> None:
    if isinstance(observable, PauliString):
        phase = observable.phase.to_complex()
        if phase not in (1.0 + 0.0j, -1.0 + 0.0j):
            raise ValueError("observable PauliString phase must be +1 or -1")
        return

    aggregated: dict[tuple[int, int, int], complex] = {}
    for pauli, coefficient in observable.terms:
        effective = complex(coefficient) * pauli.phase.to_complex()
        if not _is_finite(effective):
            raise ValueError("observable coefficients must be finite")

        key = (pauli.num_qubits, pauli.x_mask, pauli.z_mask)
        combined = aggregated.get(key, 0.0j) + effective
        if not _is_finite(combined):
            raise ValueError("observable coefficients must be finite")
        aggregated[key] = combined

    for coefficient in aggregated.values():
        if coefficient != 0.0j and coefficient.imag != 0.0:
            raise ValueError("observable must be Hermitian")


@dataclass(frozen=True, slots=True)
class EstimatorRequest:
    """A request for one observable expectation value.

    Observable qubit index ``i`` maps to ``circuit.qubits[i]``. This explicit
    insertion-order rule also applies to circuits with non-contiguous or
    reordered qubit identifiers. Bindings cover exactly the circuit's current
    used symbols. ``shots=None`` requests exact, non-sampling evaluation; a
    positive value requests shot-based estimation. An estimator that does not
    support the requested mode may raise ``ValueError``.

    The dataclass is only shallowly frozen: the Core circuit and observable
    remain mutable. Callers must not modify them during the request's lifetime.
    An estimator must nevertheless call :meth:`validate` immediately before
    execution. Revalidation detects invalid mutations but cannot detect every
    semantically different mutation that remains valid.

    Args:
        circuit: The circuit that prepares the state to evaluate.
        observable: A finite Hermitian Core Hamiltonian or Pauli string with
            matching width.
        parameter_bindings: Exact finite bindings for all used symbols. Values
            are defensively copied into an immutable mapping.
        shots: ``None`` for exact evaluation or a positive shot count.
        seed: Non-negative seed of the estimation, or ``None`` to draw fresh
            randomness. An estimator that draws shot-based results from a
            random source must derive that source from this value, so that one
            request is reproducible.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument violates a request invariant.
        cqlib.circuit.CircuitError: If ``Circuit.validate`` fails.
    """

    circuit: Circuit
    observable: Hamiltonian | PauliString
    parameter_bindings: Mapping[str, float] = field(default_factory=dict)
    shots: int | None = None
    seed: int | None = None

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        bindings = _immutable_bindings(self.parameter_bindings)
        object.__setattr__(self, "parameter_bindings", bindings)
        self.validate()

    def validate(self) -> None:
        """Revalidate the request against its current Core objects.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If an argument violates a request invariant.
            cqlib.circuit.CircuitError: If ``Circuit.validate`` fails.
        """
        if not isinstance(self.circuit, Circuit):
            raise TypeError("circuit must be a cqlib.Circuit")
        if not isinstance(self.observable, (Hamiltonian, PauliString)):
            raise TypeError("observable must be a cqlib.qis.Hamiltonian or PauliString")

        self.circuit.validate()
        if self.observable.num_qubits != self.circuit.num_qubits:
            raise ValueError("observable and circuit must have the same qubit count")

        required = set(self.circuit.used_symbols)
        provided = set(self.parameter_bindings)
        if provided != required:
            missing = sorted(required - provided)
            unexpected = sorted(provided - required)
            details: list[str] = []
            if missing:
                details.append(f"missing={missing!r}")
            if unexpected:
                details.append(f"unexpected={unexpected!r}")
            raise ValueError(
                "parameter bindings must match used circuit symbols exactly: "
                + ", ".join(details)
            )

        _validate_positive_shots(self.shots)
        _validate_seed(self.seed)
        _validate_hermitian(self.observable)


@dataclass(frozen=True, slots=True)
class EstimatorResult:
    """An expectation estimate and the statistical variance of that estimate.

    ``variance`` is ``Var(expectation estimator)``, not the physical variance
    of a single observable measurement. Deterministic exact estimators should
    return ``0.0``; implementations that cannot provide it may return ``None``.

    Raises:
        TypeError: If a value has the wrong type.
        ValueError: If a value is non-finite or variance is negative.
    """

    expectation: float
    variance: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.expectation, bool) or not isinstance(self.expectation, Real):
            raise TypeError("expectation must be a real number")
        expectation = float(self.expectation)
        if not math.isfinite(expectation):
            raise ValueError("expectation must be finite")
        object.__setattr__(self, "expectation", expectation)

        if self.variance is None:
            return
        if isinstance(self.variance, bool) or not isinstance(self.variance, Real):
            raise TypeError("variance must be a real number or None")
        variance = float(self.variance)
        if not math.isfinite(variance):
            raise ValueError("variance must be finite")
        if variance < 0.0:
            raise ValueError("variance must be non-negative")
        object.__setattr__(self, "variance", variance)


class Estimator(ABC):
    """Abstract base class for synchronous batch expectation estimation.

    Implementations call :meth:`EstimatorRequest.validate` immediately before
    execution and return one :class:`EstimatorResult` per request, in input
    order. Empty input produces an empty list. Implementations may raise
    ``ValueError`` when they do not support exact requests, shot-based
    requests, or another requested execution mode.

    An implementation whose estimates come from a random source must derive
    that source from :attr:`EstimatorRequest.seed`, so that a request with an
    explicit seed reproduces its estimate. An implementation backed by a
    device or simulator that does not accept a seed may ignore it, and must
    then document that its estimates are not reproducible.

    This type is distinct from ``cqlib.error_mitigation.Estimator``, which is a
    single-call callback alias in a separate namespace.
    """

    __slots__ = ()

    @abstractmethod
    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        """Estimate a batch and return results in input order.

        Args:
            requests: Valid expectation requests in output order.

        Returns:
            One estimate per request.
        """
        raise NotImplementedError
