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

"""Deutsch-Jozsa constant-vs-balanced discrimination.

References
----------
[1] D. Deutsch and R. Jozsa, "Rapid solution of problems by quantum
    computation," Proceedings of the Royal Society A, 439, 553-558, 1992.
    DOI: 10.1098/rspa.1992.0167.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import final

from cqlib import Circuit

from ..operators.composite import HadamardPhaseQuery
from ..operators.oracle import (
    BooleanFunction,
    BooleanVectorFunction,
)
from ..protocols.sampler import Sampler, SamplerRequest, require_sampler_results


@dataclass(frozen=True, slots=True)
class DeutschJozsaRequest:
    """One constant-vs-balanced discrimination query.

    The oracle is the immutable Boolean marking function; multi-output
    functions are reduced to a scalar predicate by output-mask parity, so
    the constant-or-balanced promise applies to that parity function.

    Args:
        function: Immutable scalar or vector Boolean function, promised to
            be constant or balanced.
        shots: Positive number of discrimination shots.
        output_mask: Selected output bits for vector functions, as in
            :class:`PhaseOracle`.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If ``shots`` is not positive or the mask is invalid.
    """

    function: BooleanFunction | BooleanVectorFunction
    shots: int
    output_mask: int | None = None

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Revalidate the request fields before execution."""
        if isinstance(self.shots, bool) or not isinstance(self.shots, int):
            raise TypeError("shots must be an integer")
        if self.shots <= 0:
            raise ValueError("shots must be greater than zero")
        HadamardPhaseQuery(self.function, output_mask=self.output_mask)


@dataclass(frozen=True, slots=True)
class DeutschJozsaResult:
    """Constant/balanced verdict and all-zero measurement statistics.

    ``zero_count`` is the number of shots whose outcome was the all-zero
    bitstring. Under an ideal sampler a constant function yields
    ``zero_count == shots`` and a balanced function ``zero_count == 0``; the
    verdict is derived from the observed counts only.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If the counts are inconsistent.
    """

    is_constant: bool
    zero_count: int
    shots: int

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.is_constant, bool):
            raise TypeError("is_constant must be a bool")
        if isinstance(self.zero_count, bool) or not isinstance(self.zero_count, int):
            raise TypeError("zero_count must be an integer")
        if self.zero_count < 0:
            raise ValueError("zero_count must be non-negative")
        if isinstance(self.shots, bool) or not isinstance(self.shots, int):
            raise TypeError("shots must be an integer")
        if self.shots <= 0:
            raise ValueError("shots must be greater than zero")
        if self.zero_count > self.shots:
            raise ValueError("zero_count must not exceed shots")
        if self.is_constant != (self.zero_count == self.shots):
            raise ValueError("is_constant must match whether zero_count equals shots")

    @property
    def is_balanced(self) -> bool:
        """Return the complementary balanced verdict."""
        return not self.is_constant


@final
class DeutschJozsa:
    """Discriminate constant from balanced Boolean functions.

    One phase-oracle query ``S_f |x> = (-1)**f(x) |x>`` between two Hadamard
    layers leaves the all-zero computational-basis amplitude at
    ``(1/2**n) * sum_x (-1)**f(x)``: exactly 1 for a constant function and
    exactly 0 for a balanced one. The submitted circuit therefore ends with
    an all-qubit measurement (added by the Sampler on a private execution
    circuit). Under the promise and ideal noiseless execution, one shot is
    sufficient: the all-zero outcome means "constant" and any non-zero
    outcome means "balanced". For a general injected Sampler, the component
    applies the same rule to observed counts; noise can therefore change the
    verdict.

    The oracle circuit is built from the immutable Boolean function by
    :class:`PhaseOracle`; multi-output functions define the derived scalar
    function ``g(x)`` as the parity selected by ``output_mask``, and the
    promise applies to ``g``. The algorithm layer does not classically inspect
    the truth table to decide the promise, although explicit-table oracle
    synthesis necessarily reads it. An input violating the promise produces
    a formally computed but semantically unspecified verdict.

    The complete input batch is validated and transformed before one
    flattened Sampler call. The component performs no classical pre-check,
    iteration management, or backend administration. Each constructed circuit
    contains one phase-oracle application; requesting multiple shots repeats
    the circuit-execution experiment and must not be described as one total
    physical oracle evaluation. References: Deutsch and Jozsa,
    *Rapid solution of problems by quantum computation*, Proc. R. Soc. Lond.
    A 439, 553 (1992); Cleve et al., *Quantum Algorithms Revisited*, Proc. R.
    Soc. Lond. A 454, 339 (1998).

    Args:
        sampler: Concrete Sampler implementation used for all measurements.

    Raises:
        TypeError: If ``sampler``, a request, or a result has the wrong type.
        ValueError: If request invariants or sampler-result contracts are
            violated.
    """

    __slots__ = ("_sampler",)

    def __init__(self, sampler: Sampler) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError("sampler must be a Sampler instance")
        self._sampler = sampler

    @property
    def sampler(self) -> Sampler:
        """Return the injected Sampler implementation."""
        return self._sampler

    @staticmethod
    def build_circuit(
        function: BooleanFunction | BooleanVectorFunction,
        *,
        output_mask: int | None = None,
    ) -> Circuit:
        """Build the unmeasured ``H**n -> S_f -> H**n`` query circuit.

        The returned circuit contains no terminal measurement; the Sampler
        appends it on a private execution circuit.
        """
        return HadamardPhaseQuery(
            function,
            output_mask=output_mask,
        ).build_circuit()

    def run(
        self,
        requests: Sequence[DeutschJozsaRequest],
        /,
    ) -> list[DeutschJozsaResult]:
        """Evaluate a complete batch in input order with one Sampler call."""
        sampler_requests: list[SamplerRequest] = []
        for request in requests:
            if not isinstance(request, DeutschJozsaRequest):
                raise TypeError("requests must contain DeutschJozsaRequest instances")
            request.validate()
            circuit = self.build_circuit(
                request.function, output_mask=request.output_mask
            )
            sampler_requests.append(
                SamplerRequest(circuit, tuple(circuit.qubits), request.shots)
            )

        if not sampler_requests:
            return []

        sample_results = require_sampler_results(
            self._sampler.run(sampler_requests), sampler_requests
        )

        results: list[DeutschJozsaResult] = []
        for result, sampler_request in zip(sample_results, sampler_requests):
            zero_count = result.counts.get("0" * len(sampler_request.qubits), 0)
            results.append(
                DeutschJozsaResult(
                    is_constant=zero_count == sampler_request.shots,
                    zero_count=zero_count,
                    shots=sampler_request.shots,
                )
            )
        return results
