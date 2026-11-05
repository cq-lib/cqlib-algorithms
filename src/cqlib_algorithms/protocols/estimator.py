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

from cqlib import Circuit
from cqlib.qis import Hamiltonian, PauliString

from .._utils.hamiltonian import validate_hermitian_observable
from .._utils.numeric import validate_optional_positive_integer, validate_optional_seed
from .._utils.parameters import snapshot_parameter_bindings


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
        bindings = snapshot_parameter_bindings(
            self.parameter_bindings,
            name="parameter_bindings",
        )
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

        validate_optional_positive_integer(self.shots, "shots")
        validate_optional_seed(self.seed)
        validate_hermitian_observable(self.observable, name="observable")


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


def require_estimator_results(
    results: object,
    requests: Sequence[EstimatorRequest],
    *,
    count_error: str = "estimator must return one result per request",
) -> list[EstimatorResult]:
    """Validate and return an ordered Estimator result batch unchanged.

    Args:
        results: Object returned by :meth:`Estimator.run`.
        requests: Requests passed to the Estimator in the same order.
        count_error: Error text used when the result count is invalid.

    Returns:
        The validated result list, unchanged.

    Raises:
        TypeError: If the result container or one of its entries has the
            wrong type.
        ValueError: If the result count does not match the request count.
    """
    if not isinstance(results, list):
        raise TypeError("estimator must return a list of EstimatorResult values")
    if len(results) != len(requests):
        raise ValueError(count_error)
    if any(not isinstance(result, EstimatorResult) for result in results):
        raise TypeError("estimator must return EstimatorResult values")
    return results


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
