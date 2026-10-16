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

"""Simon's hidden-period algorithm.

References
----------
[1] D. R. Simon, "On the power of quantum computation," SIAM Journal on
    Computing, 26(5), 1474-1483, 1997. DOI: 10.1137/S0097539796298637.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import final

from cqlib import Circuit

from ..operators.oracle import (
    BitFlipOracle,
    BooleanFunction,
    BooleanVectorFunction,
)
from ..protocols.sampler import Sampler, SamplerRequest, validate_sampler_results


def _bits_to_integer(bits: tuple[bool, ...]) -> int:
    return sum(int(bit) << qubit for qubit, bit in enumerate(bits))


def _integer_to_bits(value: int, width: int) -> tuple[bool, ...]:
    return tuple(bool((value >> qubit) & 1) for qubit in range(width))


def _reduce_equations(
    equations: Sequence[tuple[bool, ...]],
    num_qubits: int,
) -> tuple[tuple[tuple[bool, ...], ...], tuple[tuple[bool, ...], ...]]:
    """Return canonical RREF rows and a canonical GF(2) nullspace basis."""
    rows = sorted(
        {_bits_to_integer(row) for row in equations if any(row)}, reverse=True
    )
    pivot_columns: list[int] = []
    next_row = 0

    for column in range(num_qubits - 1, -1, -1):
        pivot_index = next(
            (
                index
                for index in range(next_row, len(rows))
                if (rows[index] >> column) & 1
            ),
            None,
        )
        if pivot_index is None:
            continue
        rows[next_row], rows[pivot_index] = rows[pivot_index], rows[next_row]
        pivot = rows[next_row]
        for index, row in enumerate(rows):
            if index != next_row and (row >> column) & 1:
                rows[index] = row ^ pivot
        pivot_columns.append(column)
        next_row += 1

    pivot_rows = rows[: len(pivot_columns)]
    free_columns = [
        column for column in range(num_qubits) if column not in pivot_columns
    ]
    nullspace: list[int] = []
    for free_column in free_columns:
        vector = 1 << free_column
        for row, pivot_column in zip(pivot_rows, pivot_columns):
            if (row >> free_column) & 1:
                vector |= 1 << pivot_column
        nullspace.append(vector)

    return (
        tuple(_integer_to_bits(row, num_qubits) for row in pivot_rows),
        tuple(_integer_to_bits(vector, num_qubits) for vector in nullspace),
    )


@dataclass(frozen=True, slots=True)
class SimonRequest:
    """One Simon hidden-period query.

    Args:
        function: Immutable scalar or vector Boolean function promised to obey
            ``F(x) = F(y)`` exactly when ``x = y`` or ``x xor y = s`` for one
            hidden period ``s``. The zero period denotes a one-to-one function.
        shots: Positive number of samples used to collect GF(2) equations.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If ``shots`` is not positive.
    """

    function: BooleanFunction | BooleanVectorFunction
    shots: int

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Revalidate the request fields before execution."""
        if isinstance(self.shots, bool) or not isinstance(self.shots, int):
            raise TypeError("shots must be an integer")
        if self.shots <= 0:
            raise ValueError("shots must be greater than zero")
        BitFlipOracle(self.function)


@dataclass(frozen=True, slots=True)
class SimonResult:
    """Canonical independent equations and their GF(2) nullspace.

    Tuple coordinates use little-endian qubit order: entry ``q`` corresponds
    to input qubit ``q``. ``period`` is available only when the equations have
    rank ``num_qubits - 1`` and identify one nonzero candidate, or when they
    have full rank and identify the zero period of a one-to-one function.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If the equations, basis, or shot count are inconsistent.
    """

    num_qubits: int
    independent_equations: tuple[tuple[bool, ...], ...]
    nullspace_basis: tuple[tuple[bool, ...], ...]
    shots: int

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if isinstance(self.num_qubits, bool) or not isinstance(self.num_qubits, int):
            raise TypeError("num_qubits must be an integer")
        if self.num_qubits <= 0:
            raise ValueError("num_qubits must be greater than zero")
        if not isinstance(self.independent_equations, tuple):
            raise TypeError("independent_equations must be a tuple")
        if not isinstance(self.nullspace_basis, tuple):
            raise TypeError("nullspace_basis must be a tuple")
        for name, vectors in (
            ("independent_equations", self.independent_equations),
            ("nullspace_basis", self.nullspace_basis),
        ):
            if any(not isinstance(vector, tuple) for vector in vectors):
                raise TypeError(f"{name} entries must be tuples")
            if any(len(vector) != self.num_qubits for vector in vectors):
                raise ValueError(f"{name} entries must match num_qubits")
            if any(not isinstance(bit, bool) for vector in vectors for bit in vector):
                raise TypeError(f"{name} entries must contain only bool values")
        if isinstance(self.shots, bool) or not isinstance(self.shots, int):
            raise TypeError("shots must be an integer")
        if self.shots <= 0:
            raise ValueError("shots must be greater than zero")
        if len(self.independent_equations) > self.shots:
            raise ValueError("equation rank must not exceed shots")

        expected_rows, expected_nullspace = _reduce_equations(
            self.independent_equations, self.num_qubits
        )
        if self.independent_equations != expected_rows:
            raise ValueError("independent_equations must be in canonical RREF form")
        if self.nullspace_basis != expected_nullspace:
            raise ValueError(
                "nullspace_basis must be the canonical basis of the equation nullspace"
            )

    @property
    def rank(self) -> int:
        """Return the rank of the sampled GF(2) equation system."""
        return len(self.independent_equations)

    @property
    def is_complete(self) -> bool:
        """Return whether the system uniquely identifies a promised period."""
        return self.rank in {self.num_qubits - 1, self.num_qubits}

    @property
    def period(self) -> tuple[bool, ...] | None:
        """Return the unique period, or ``None`` if it is not identified."""
        if self.rank == self.num_qubits:
            return (False,) * self.num_qubits
        if self.rank != self.num_qubits - 1 or len(self.nullspace_basis) != 1:
            return None
        candidate = self.nullspace_basis[0]
        return candidate if any(candidate) else None

    @property
    def period_integer(self) -> int | None:
        """Return the identified period as an integer, if available."""
        return None if self.period is None else _bits_to_integer(self.period)


@final
class Simon:
    """Recover a promised XOR period from sampled equations.

    For a function ``F: {0,1}**n -> {0,1}**m`` obeying
    ``F(x) = F(y)`` exactly when ``x = y`` or ``x xor y = s``, the circuit
    applies Hadamards to the input register, one :class:`BitFlipOracle`, and
    another input-register Hadamard layer. Measuring only the input register
    yields vectors ``z`` satisfying ``z . s = 0`` over GF(2). Row reduction
    identifies a nonzero ``s`` when the collected equations have rank
    ``n - 1``. For the allowed one-to-one case ``s = 0``, the samples span
    the complete input space and rank ``n`` identifies the all-zero period.

    The algorithm does not inspect the truth table to verify the promise and
    does not retry until sufficient rank is reached. Insufficient or noisy
    equations are returned faithfully. Rank below ``n - 1`` leaves ``period``
    unresolved; rank ``n - 1`` returns the unique nonzero nullspace vector;
    rank ``n`` returns zero. Noise can falsely change the rank, and this
    implementation does not perform robust decoding. Explicit truth-table oracle synthesis has
    exponential classical input and synthesis cost and is outside the query
    complexity model. Reference: D. R. Simon, *On the Power of Quantum
    Computation*, FOCS 1994, 116--123.

    Args:
        sampler: Concrete Sampler implementation used for input measurements.
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
    ) -> Circuit:
        """Build the unmeasured Simon query circuit for ``function``."""
        oracle = BitFlipOracle(function)
        oracle_circuit = oracle.build_circuit()
        num_inputs = function.num_variables
        circuit = Circuit(len(oracle_circuit.qubits))
        for qubit in range(num_inputs):
            circuit.h(qubit)
        circuit.compose(oracle_circuit, circuit.qubits)
        for qubit in range(num_inputs):
            circuit.h(qubit)
        return circuit

    def run(self, requests: Sequence[SimonRequest], /) -> list[SimonResult]:
        """Evaluate a complete batch in input order with one Sampler call."""
        sampler_requests: list[SamplerRequest] = []
        widths: list[int] = []
        for request in requests:
            if not isinstance(request, SimonRequest):
                raise TypeError("requests must contain SimonRequest instances")
            request.validate()
            circuit = self.build_circuit(request.function)
            num_inputs = request.function.num_variables
            sampler_requests.append(
                SamplerRequest(
                    circuit,
                    tuple(circuit.qubits[:num_inputs]),
                    request.shots,
                )
            )
            widths.append(num_inputs)

        if not sampler_requests:
            return []

        sample_results = validate_sampler_results(
            self._sampler.run(sampler_requests), sampler_requests
        )

        results: list[SimonResult] = []
        for result, sampler_request, num_qubits in zip(
            sample_results, sampler_requests, widths
        ):
            equations = tuple(
                tuple(bit == "1" for bit in reversed(outcome))
                for outcome, count in result.counts.items()
                if count > 0
            )
            independent, nullspace = _reduce_equations(equations, num_qubits)
            results.append(
                SimonResult(
                    num_qubits=num_qubits,
                    independent_equations=independent,
                    nullspace_basis=nullspace,
                    shots=sampler_request.shots,
                )
            )
        return results
