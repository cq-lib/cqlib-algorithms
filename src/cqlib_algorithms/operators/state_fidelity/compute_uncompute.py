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

"""Pure-state circuit fidelity through the compute-uncompute construction.

References
----------
[1] V. Havlíček et al., "Supervised learning with quantum-enhanced feature
    spaces," Nature, 567, 209-212, 2019.
    DOI: 10.1038/s41586-019-0980-2.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import final

from cqlib import Circuit
from cqlib.circuit import CircuitError

from ..._utils.circuit import (
    validate_static_unitary_structure,
)
from ..._utils.parameters import snapshot_parameter_bindings
from ...protocols.sampler import (
    Sampler,
    SamplerRequest,
    require_sampler_results,
)
from ...protocols.state_fidelity import StateFidelity


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


def _bound_circuit_and_inverse(
    circuit: Circuit,
    bindings: Mapping[str, float],
    *,
    label: str,
) -> tuple[Circuit, Circuit]:
    """Bind, validate, and invert a circuit without mutating the source."""
    validate_static_unitary_structure(circuit, label=label)
    bound = circuit.assign_parameters(dict(bindings))
    bound.validate()
    validate_static_unitary_structure(bound, label=label)
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
        seed: Non-negative sampler seed, or ``None`` for fresh randomness.

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
    seed: int | None = None

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameter_bindings_a",
            snapshot_parameter_bindings(
                self.parameter_bindings_a,
                name="parameter bindings",
            ),
        )
        object.__setattr__(
            self,
            "parameter_bindings_b",
            snapshot_parameter_bindings(
                self.parameter_bindings_b,
                name="parameter bindings",
            ),
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
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise TypeError("seed must be an integer or None")
            if self.seed < 0:
                raise ValueError("seed must be non-negative")

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

    Attributes:
        fidelity: Estimated squared state overlap in ``[0, 1]``.
        variance: Statistical variance of the estimator, when available.
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


@final
class ComputeUncomputeFidelity(
    StateFidelity[CircuitFidelityRequest, CircuitFidelityResult]
):
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
                SamplerRequest(
                    compact,
                    tuple(compact.qubits),
                    request.shots,
                    request.seed,
                )
            )

        if not sampler_requests:
            return []

        sample_results = require_sampler_results(
            self._sampler.run(sampler_requests),
            sampler_requests,
        )

        results: list[CircuitFidelityResult] = []
        for result, sampler_request in zip(
            sample_results,
            sampler_requests,
            strict=True,
        ):
            all_zero = "0" * len(sampler_request.qubits)
            fidelity = result.counts.get(all_zero, 0) / result.shots
            variance = (
                None
                if result.shots == 1
                else fidelity * (1.0 - fidelity) / (result.shots - 1)
            )
            results.append(CircuitFidelityResult(fidelity, variance))
        return results
