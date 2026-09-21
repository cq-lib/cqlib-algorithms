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

"""Pure-state circuit fidelity through the compute-uncompute construction."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import final

from cqlib import Circuit
from cqlib.circuit import CircuitError
from cqlib.device import ExecutionResult

from ...protocols.sampler import Sampler, SamplerRequest


def _immutable_bindings(bindings: Mapping[str, float]) -> Mapping[str, float]:
    """Return a finite, immutable copy of one parameter-binding mapping."""
    if not isinstance(bindings, Mapping):
        raise TypeError("parameter bindings must be a mapping")

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


def _validate_exact_bindings(
    circuit: Circuit,
    bindings: Mapping[str, float],
    *,
    label: str,
) -> None:
    """Require bindings to cover exactly the circuit's current live symbols."""
    required = set(circuit.used_symbols)
    provided = set(bindings)
    if provided == required:
        return

    missing = sorted(required - provided)
    unexpected = sorted(provided - required)
    details: list[str] = []
    if missing:
        details.append(f"missing={missing!r}")
    if unexpected:
        details.append(f"unexpected={unexpected!r}")
    raise ValueError(
        f"{label} parameter bindings must match used circuit symbols exactly: "
        + ", ".join(details)
    )


def _validate_static_unitary(circuit: Circuit, *, label: str) -> None:
    """Reject dynamic, non-unitary, delayed, and non-decomposable circuits."""
    if circuit.classical_vars or circuit.classical_values:
        raise ValueError(f"{label} must be a static unitary circuit")

    try:
        expanded = circuit.decompose()
    except CircuitError as exc:
        raise ValueError(
            f"{label} must be a decomposable static unitary circuit"
        ) from exc

    for operation in expanded.operations:
        if (
            operation.is_classical_data
            or operation.is_classical_control
            or operation.is_delay
            or (operation.is_directive and operation.name != "barrier")
        ):
            raise ValueError(
                f"{label} must contain only static unitary operations "
                "without measurement, reset, delay, or classical control"
            )


def _bound_circuit_and_inverse(
    circuit: Circuit,
    bindings: Mapping[str, float],
    *,
    label: str,
) -> tuple[Circuit, Circuit]:
    """Bind, validate, and invert a circuit without mutating the source."""
    _validate_static_unitary(circuit, label=label)
    bound = circuit.assign_parameters(dict(bindings))
    bound.validate()
    _validate_static_unitary(bound, label=label)
    try:
        inverse = bound.inverse()
    except CircuitError as exc:
        raise ValueError(
            f"{label} must contain only statically invertible operations"
        ) from exc
    return bound, inverse


@dataclass(frozen=True, slots=True)
class CircuitFidelityRequest:
    """Request the squared overlap of two pure states prepared by circuits.

    The two circuits are associated by qubit insertion position, even when
    their concrete qubit identifiers differ. Both circuits must contain the
    same positive number of qubits and only statically invertible unitary
    operations. Each binding mapping must cover exactly the corresponding
    circuit's currently used symbols.

    The dataclass is only shallowly frozen: Core circuits remain mutable.
    Callers must not modify them during the request lifetime, and execution
    revalidates all detectable invariants immediately before use. Parameter
    bindings are defensively copied into immutable mappings.

    Args:
        circuit_a: Circuit preparing the first pure state.
        circuit_b: Circuit preparing the second pure state.
        shots: Positive number of compute-uncompute samples.
        parameter_bindings_a: Exact finite bindings for ``circuit_a``.
        parameter_bindings_b: Exact finite bindings for ``circuit_b``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument violates a request invariant.
        cqlib.circuit.CircuitError: If direct Core circuit validation fails.
    """

    circuit_a: Circuit
    circuit_b: Circuit
    shots: int
    parameter_bindings_a: Mapping[str, float] = field(default_factory=dict)
    parameter_bindings_b: Mapping[str, float] = field(default_factory=dict)

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameter_bindings_a",
            _immutable_bindings(self.parameter_bindings_a),
        )
        object.__setattr__(
            self,
            "parameter_bindings_b",
            _immutable_bindings(self.parameter_bindings_b),
        )
        self.validate()

    def validate(self) -> None:
        """Revalidate the request against the circuits' current states."""
        self._prepare()

    def _prepare(self) -> tuple[Circuit, Circuit]:
        """Validate once and return the bound execution components."""
        if not isinstance(self.circuit_a, Circuit):
            raise TypeError("circuit_a must be a cqlib.Circuit")
        if not isinstance(self.circuit_b, Circuit):
            raise TypeError("circuit_b must be a cqlib.Circuit")
        if isinstance(self.shots, bool) or not isinstance(self.shots, int):
            raise TypeError("shots must be an integer")
        if self.shots <= 0:
            raise ValueError("shots must be greater than zero")

        self.circuit_a.validate()
        self.circuit_b.validate()
        if self.circuit_a.num_qubits < 1:
            raise ValueError("circuits must contain at least one qubit")
        if self.circuit_a.num_qubits != self.circuit_b.num_qubits:
            raise ValueError("circuits must have the same qubit count")

        _validate_exact_bindings(
            self.circuit_a,
            self.parameter_bindings_a,
            label="circuit_a",
        )
        _validate_exact_bindings(
            self.circuit_b,
            self.parameter_bindings_b,
            label="circuit_b",
        )
        _, inverse_a = _bound_circuit_and_inverse(
            self.circuit_a,
            self.parameter_bindings_a,
            label="circuit_a",
        )
        bound_b, _ = _bound_circuit_and_inverse(
            self.circuit_b,
            self.parameter_bindings_b,
            label="circuit_b",
        )
        return bound_b, inverse_a


@dataclass(frozen=True, slots=True)
class CircuitFidelityResult:
    """A pure-state fidelity estimate and variance of that estimator.

    ``fidelity`` is the estimated all-zero probability and lies in ``[0, 1]``.
    ``variance`` is the statistical variance of that probability estimator,
    not a quantum-state variance. It is ``None`` for a one-shot estimate.
    """

    fidelity: float
    variance: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.fidelity, bool) or not isinstance(self.fidelity, Real):
            raise TypeError("fidelity must be a real number")
        fidelity = float(self.fidelity)
        if not math.isfinite(fidelity):
            raise ValueError("fidelity must be finite")
        if not 0.0 <= fidelity <= 1.0:
            raise ValueError("fidelity must be between zero and one")
        object.__setattr__(self, "fidelity", fidelity)

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


def _validate_sample_result(
    result: ExecutionResult,
    request: SamplerRequest,
) -> None:
    """Validate all sampler metadata and counts used by the estimator."""
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


@final
class ComputeUncomputeFidelity:
    """Estimate pure-state fidelity with an injected batch Sampler.

    For circuits ``U_A`` and ``U_B``, the submitted circuit applies ``U_B``
    followed by ``U_A†`` to ``|0...0>``. The probability of observing the
    all-zero bitstring is therefore ``|<0|U_A† U_B|0>|**2``, the squared
    overlap of the two prepared pure states. Qubits are associated by circuit
    insertion position and the source circuits are never mutated.

    A complete input batch is validated and transformed before one flattened
    Sampler call. For more than one shot, ``p * (1 - p) / (shots - 1)`` is the
    unbiased sample estimate of the variance of the sample-mean estimator.

    Args:
        sampler: Concrete Sampler implementation used for all measurements.

    Raises:
        TypeError: If ``sampler``, a request, or a result has the wrong type.
        ValueError: If request invariants, invertibility, or sampler-result
            contracts are violated.
    """

    __slots__ = ("_sampler",)

    def __init__(self, sampler: Sampler) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError("sampler must be a Sampler instance")
        self._sampler = sampler

    def run(
        self,
        requests: Sequence[CircuitFidelityRequest],
        /,
    ) -> list[CircuitFidelityResult]:
        """Estimate a complete batch in input order using one Sampler call."""
        sampler_requests: list[SamplerRequest] = []
        for request in requests:
            if not isinstance(request, CircuitFidelityRequest):
                raise TypeError(
                    "requests must contain CircuitFidelityRequest instances"
                )
            bound_b, inverse_a = request._prepare()

            compact = Circuit(request.circuit_a.num_qubits)
            positions = list(range(request.circuit_a.num_qubits))
            compact.compose(bound_b, positions)
            compact.compose(inverse_a, positions)
            sampler_requests.append(
                SamplerRequest(compact, tuple(compact.qubits), request.shots)
            )

        if not sampler_requests:
            return []

        sample_results = self._sampler.run(sampler_requests)
        if len(sample_results) != len(sampler_requests):
            raise ValueError("sampler must return one result per request")

        results: list[CircuitFidelityResult] = []
        for result, sampler_request in zip(sample_results, sampler_requests):
            if not isinstance(result, ExecutionResult):
                raise TypeError("sampler results must be cqlib.device.ExecutionResult")
            _validate_sample_result(result, sampler_request)
            all_zero = "0" * len(sampler_request.qubits)
            fidelity = result.counts.get(all_zero, 0) / result.shots
            variance = (
                None
                if result.shots == 1
                else fidelity * (1.0 - fidelity) / (result.shots - 1)
            )
            results.append(CircuitFidelityResult(fidelity, variance))
        return results
