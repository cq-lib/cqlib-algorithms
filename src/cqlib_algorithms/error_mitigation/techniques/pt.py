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

r"""Pauli twirling.

A two-qubit gate on real hardware carries coherent errors, and coherent errors
accumulate linearly in the number of gates while stochastic ones accumulate
only in their standard deviation. Pauli twirling reshapes the former into the
latter by dressing each two-qubit gate with Pauli operators: the gate is
surrounded by a Pauli pair drawn from a table, so that the dressed gate still
realises the original gate while the error the gate carries is conjugated by
the pair.

Only CX and CZ are twirled, because it is for those two gates that
single-qubit Pauli pairs reproducing the gate are known. A twirling average
needs several dressed variants of the same circuit, which
:class:`PauliTwirling` produces.

The two-piece API is :func:`construct_circuits` plus :func:`combine_results`,
orchestrated by :func:`execute_with_pt`; :func:`pt_executor` returns the same
computation as a reusable executor for chaining.

References
----------
[1] J. J. Wallman and J. Emerson, "Noise tailoring for scalable quantum computation via
    randomized compiling," Physical Review A 94, 052325, 2016. arXiv:1512.01098.
[2] A. Saki, A. Katabarwa, S. Resch, and others, "Hypothesis testing for error
    mitigation: How to evaluate error mitigation," 2023. arXiv:2301.02690.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from cqlib import Circuit
from cqlib.circuit import StandardGate, ValueOperation
from cqlib.qis import Hamiltonian, PauliString

from ...protocols import Estimator, EstimatorRequest, EstimatorResult, spawn_seeds
from .._circuits import expand
from ..core import QemExecutor
from ._equivalence import allclose_up_to_global_phase

# One row of a twirling table. The four Pauli gates are the pair applied before
# the dressed gate and the pair applied after it, each acting on the first and
# on the second qubit of the gate in the order the gate lists them.
_TwirlRow = tuple[StandardGate, StandardGate, StandardGate, StandardGate]

# Rows for CX, listed as (P, Q, R, S).
_CNOT_TWIRLINGS: tuple[_TwirlRow, ...] = (
    (StandardGate.I, StandardGate.I, StandardGate.I, StandardGate.I),
    (StandardGate.I, StandardGate.X, StandardGate.I, StandardGate.X),
    (StandardGate.I, StandardGate.Y, StandardGate.Z, StandardGate.Y),
    (StandardGate.I, StandardGate.Z, StandardGate.Z, StandardGate.Z),
    (StandardGate.Y, StandardGate.I, StandardGate.Y, StandardGate.X),
    (StandardGate.Y, StandardGate.X, StandardGate.Y, StandardGate.I),
    (StandardGate.Y, StandardGate.Y, StandardGate.X, StandardGate.Z),
    (StandardGate.Y, StandardGate.Z, StandardGate.X, StandardGate.Y),
    (StandardGate.X, StandardGate.I, StandardGate.X, StandardGate.X),
    (StandardGate.X, StandardGate.X, StandardGate.X, StandardGate.I),
    (StandardGate.X, StandardGate.Y, StandardGate.Y, StandardGate.Z),
    (StandardGate.X, StandardGate.Z, StandardGate.Y, StandardGate.Y),
    (StandardGate.Z, StandardGate.I, StandardGate.Z, StandardGate.I),
    (StandardGate.Z, StandardGate.X, StandardGate.Z, StandardGate.X),
    (StandardGate.Z, StandardGate.Y, StandardGate.I, StandardGate.Y),
    (StandardGate.Z, StandardGate.Z, StandardGate.I, StandardGate.Z),
)

# Rows for CZ, listed as (P, Q, R, S).
_CZ_TWIRLINGS: tuple[_TwirlRow, ...] = (
    (StandardGate.I, StandardGate.I, StandardGate.I, StandardGate.I),
    (StandardGate.I, StandardGate.X, StandardGate.Z, StandardGate.X),
    (StandardGate.I, StandardGate.Y, StandardGate.Z, StandardGate.Y),
    (StandardGate.I, StandardGate.Z, StandardGate.I, StandardGate.Z),
    (StandardGate.X, StandardGate.I, StandardGate.X, StandardGate.Z),
    (StandardGate.X, StandardGate.X, StandardGate.Y, StandardGate.Y),
    (StandardGate.X, StandardGate.Y, StandardGate.Y, StandardGate.X),
    (StandardGate.X, StandardGate.Z, StandardGate.X, StandardGate.I),
    (StandardGate.Y, StandardGate.I, StandardGate.Y, StandardGate.Z),
    (StandardGate.Y, StandardGate.X, StandardGate.X, StandardGate.Y),
    (StandardGate.Y, StandardGate.Y, StandardGate.X, StandardGate.X),
    (StandardGate.Y, StandardGate.Z, StandardGate.Y, StandardGate.I),
    (StandardGate.Z, StandardGate.I, StandardGate.Z, StandardGate.I),
    (StandardGate.Z, StandardGate.X, StandardGate.I, StandardGate.X),
    (StandardGate.Z, StandardGate.Y, StandardGate.I, StandardGate.Y),
    (StandardGate.Z, StandardGate.Z, StandardGate.Z, StandardGate.Z),
)

_TWIRLING_TABLES: dict[StandardGate, tuple[_TwirlRow, ...]] = {
    StandardGate.CX: _CNOT_TWIRLINGS,
    StandardGate.CZ: _CZ_TWIRLINGS,
}


def _validate_twirl_tables() -> None:
    r"""Validate the gate-twirling tables.
    """
    for gate, table in _TWIRLING_TABLES.items():
        matrix = gate.matrix()
        for p, q, r, s in table:
            dressed = (
                np.kron(r.matrix(), s.matrix())
                @ matrix
                @ np.kron(p.matrix(), q.matrix())
            )
            if not allclose_up_to_global_phase(dressed, matrix):
                raise RuntimeError(
                    "input twirling table violates the condition: every row "
                    "must satisfy kron(R, S) @ G @ kron(P, Q) = G for the "
                    "dressed gate G, see Saki et al. 2023 in detail; the row "
                    f"(P, Q, R, S) = ({p!r}, {q!r}, {r!r}, {s!r}) dressed "
                    f"{gate!r} and did not."
                )


# These tables are the only correctness-critical data in the module, and a
# mistyped row would silently corrupt every circuit the module produces, so
# they are checked once at import time instead of once per twirled circuit.
_validate_twirl_tables()


@dataclass(frozen=True, slots=True)
class PauliTwirling:
    r"""Line transform dressing two-qubit gates with Pauli pairs.

    Args:
        num_randomizations: Number of dressed variants that :meth:`variants` produces.
        seed: Seed of the Pauli draw, or ``None`` to draw fresh randomness on every
            call.

    Raises:
        TypeError: If ``num_randomizations`` is not an integer, or if ``seed`` is neither
            an integer nor ``None``.
        ValueError: If ``num_randomizations`` is below one.
    """

    num_randomizations: int = 1
    seed: int | None = None

    def __post_init__(self) -> None:
        _validate_num_randomizations(self.num_randomizations)
        _validate_seed(self.seed)

    def apply(self, circuit: Circuit, /) -> Circuit:
        r"""Return one Pauli-dressed copy of ``circuit``.

        Every CX and CZ gate is dressed with a Pauli pair drawn independently
        for that gate, so the copy still realises the unitary of ``circuit``
        and differs from it only in the placement of those pairs.

        Args:
            circuit: Circuit whose two-qubit gates receive the Pauli pairs.

        Returns:
            One dressed copy of ``circuit``.
        """
        return _twirl(circuit, np.random.default_rng(self.seed))

    def variants(self, circuit: Circuit, /) -> list[Circuit]:
        r"""Return ``num_randomizations`` independently dressed copies.

        All copies are drawn from one stream seeded by ``seed``, so a given
        rule reproduces the same family and the copies differ from each other.

        Args:
            circuit: Circuit whose two-qubit gates receive the Pauli pairs.

        Returns:
            The dressed copies, in draw order.
        """
        rng = np.random.default_rng(self.seed)
        return [_twirl(circuit, rng) for _ in range(self.num_randomizations)]


def construct_circuits(
    circuit: Circuit,
    /,
    *,
    num_randomizations: int = 1,
    seed: int | None = None,
) -> list[Circuit]:
    r"""Dress ``circuit`` once per randomization.

    Args:
        circuit: Circuit whose two-qubit gates receive the Pauli pairs.
        num_randomizations: Number of dressed variants to build.
        seed: Seed of the Pauli draw, or ``None`` for a fresh draw.

    Returns:
        One dressed circuit per randomization, in draw order.

    Raises:
        TypeError: If ``num_randomizations`` is not an integer, or if ``seed`` is neither
            an integer nor ``None``.
        ValueError: If ``num_randomizations`` is below one.

    """
    return PauliTwirling(
        num_randomizations=num_randomizations,
        seed=seed,
    ).variants(circuit)


def combine_results(results: Sequence[EstimatorResult], /) -> float:
    r"""Average the expectation values of the dressed circuits.

    Args:
        results: Expectation values measured on the circuits returned by
            :func:`construct_circuits`, in the same order.

    Returns:
        The averaged expectation value.

    Raises:
        ValueError: If ``results`` is empty.

    """
    if not results:
        raise ValueError(
            "pauli twirling needs at least one result to average."
        )
    return float(np.mean([result.expectation for result in results]))


def execute_with_pt(
    circuit: Circuit,
    executor: QemExecutor,
    observable: Hamiltonian | PauliString,
    /,
    *,
    num_randomizations: int = 1,
    shots: int | None = None,
    seed: int | None = None,
) -> float:
    r"""Estimate one observable on ``circuit`` with Pauli twirling.

    This is the immediate form of :func:`pt_executor`::

        execute_with_pt(c, ex, obs, **params)
        == pt_executor(ex, **params).estimate(
               [EstimatorRequest(c, obs)]
           )[0].expectation

    Args:
        circuit: Circuit whose expectation value is to be mitigated.
        executor: Execution backend supplying the expectation surface.
        observable: Observable to estimate. A bare Pauli string is interpreted as a
            Hamiltonian with unit coefficient.
        num_randomizations: Number of dressed variants to average over.
        shots: Shot count for every dressed variant, or ``None`` for exact
            evaluation.
        seed: Seed of the Pauli draw and the shot sampling, or ``None`` for fresh
            randomness. Pinning it reproduces the whole estimate.

    Returns:
        Mitigated expectation value.

    Raises:
        TypeError: If ``num_randomizations`` is not an integer, or if ``seed`` is neither
            an integer nor ``None``.
        ValueError: If ``num_randomizations`` is below one.
    """
    wrapped = pt_executor(
        executor,
        num_randomizations=num_randomizations,
        shots=shots,
        seed=seed,
    )
    request = EstimatorRequest(circuit=circuit, observable=observable, shots=shots)
    return wrapped.estimate([request])[0].expectation


def pt_executor(
    executor: QemExecutor,
    /,
    *,
    num_randomizations: int = 1,
    shots: int | None = None,
    seed: int | None = None,
) -> QemExecutor:
    r"""Return an executor with Pauli twirling applied.

    The returned executor keeps the wrapped executor's sampling surface and
    caching flag, so Pauli twirling can be chained on top of another technique
    without changing either semantics.

    Args:
        executor: Execution backend to wrap.
        num_randomizations: Number of dressed variants averaged for every request.
        shots: Shot count bound to every request, or ``None`` to defer to the
            request's own shot count.
        seed: Seed of the Pauli draw and the shot sampling, or ``None`` to defer to
            the request's own seed.

    Returns:
        Executor whose expectation surface applies Pauli twirling.

    Raises:
        TypeError: If ``num_randomizations`` is not an integer, or if ``seed`` is neither
            an integer nor ``None``.
        ValueError: If ``num_randomizations`` is below one.

    """
    estimator = _PtEstimator(
        executor,
        num_randomizations=num_randomizations,
        shots=shots,
        seed=seed,
    )
    return QemExecutor(
        estimator=estimator,
        sampler=executor.sampler,
        use_cache=executor.use_cache,
    )


class _PtEstimator(Estimator):
    r"""Expectation surface that averages over Pauli-dressed variants.

    Args:
        executor: Execution backend supplying the expectation surface for each dressed
            circuit.
        num_randomizations: Number of dressed variants averaged for every request.
        shots: Shot count bound to every request, or ``None`` to defer to the
            request.
        seed: Seed of the Pauli draw, or ``None`` for a fresh draw.

    """

    __slots__ = ("_executor", "_num_randomizations", "_seed", "_shots")

    _executor: QemExecutor
    _num_randomizations: int
    _seed: int | None
    _shots: int | None

    def __init__(
        self,
        executor: QemExecutor,
        /,
        *,
        num_randomizations: int,
        shots: int | None,
        seed: int | None,
    ) -> None:
        _validate_num_randomizations(num_randomizations)
        _validate_seed(seed)
        self._executor = executor
        self._num_randomizations = num_randomizations
        self._shots = shots
        self._seed = seed

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        r"""Mitigate every request and return results in input order."""
        return [self._mitigate(request) for request in requests]

    def _mitigate(self, request: EstimatorRequest) -> EstimatorResult:
        r"""Estimate one request as the average over its dressed variants."""
        shots = self._shots if self._shots is not None else request.shots
        seed = self._seed if self._seed is not None else request.seed
        circuits = construct_circuits(
            request.circuit,
            num_randomizations=self._num_randomizations,
            seed=seed,
        )
        # Each dressed variant is measured independently, so it takes its own
        # child seed while one run seed still reproduces the whole average.
        seeds = spawn_seeds(seed, len(circuits))
        results = self._executor.estimate(
            [
                EstimatorRequest(
                    circuit=circuit,
                    observable=request.observable,
                    shots=shots,
                    seed=seeds[index],
                )
                for index, circuit in enumerate(circuits)
            ]
        )
        return EstimatorResult(expectation=combine_results(results))


def _twirl(
    circuit: Circuit,
    rng: np.random.Generator,
    /,
) -> Circuit:
    r"""Dress every two-qubit gate of ``circuit`` with one Pauli draw.

    Args:
        circuit: Circuit whose two-qubit gates receive the Pauli pairs.
        rng: Random stream the Pauli rows are drawn from.

    Returns:
        A dressed copy of ``circuit``.

    """
    return expand(circuit, lambda operation: _twirl_operation(operation, rng))


def _twirl_operation(
    operation: ValueOperation,
    rng: np.random.Generator,
    /,
) -> tuple[ValueOperation, ...]:
    r"""Replace one operation by its Pauli-dressed form.

    Args:
        operation: Operation to dress. Gates outside the twirling tables are returned
            unchanged.
        rng: Random stream the Pauli row is drawn from.

    Returns:
        The operations taking the place of ``operation``.

    """
    if not operation.is_standard:
        return (operation,)
    standard_gate = operation.instruction.standard_gate
    if standard_gate is None:
        return (operation,)
    table = _TWIRLING_TABLES.get(standard_gate)
    if table is None:
        return (operation,)

    # The row dresses the gate with the pair that precedes it and the pair that
    # follows it. ``operation.qubits`` lists the qubits in the order the gate
    # declares them, which is the order the rows use.
    first, second = operation.qubits
    p, q, r, s = table[int(rng.integers(len(table)))]
    return (
        ValueOperation.from_standard_gate(p, [first]),
        ValueOperation.from_standard_gate(q, [second]),
        operation,
        ValueOperation.from_standard_gate(r, [first]),
        ValueOperation.from_standard_gate(s, [second]),
    )


def _validate_num_randomizations(num_randomizations: int, /) -> None:
    r"""Validate a positive randomization count."""
    if isinstance(num_randomizations, bool) or not isinstance(
        num_randomizations, int
    ):
        raise TypeError("num_randomizations must be an integer")
    if num_randomizations < 1:
        raise ValueError(
            "num_randomizations must be at least one, but was "
            f"{num_randomizations!r}."
        )


def _validate_seed(seed: int | None, /) -> None:
    r"""Validate an optional random seed."""
    if seed is None:
        return
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer or None")
    if seed < 0:
        raise ValueError("seed must be non-negative")
