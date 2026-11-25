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

"""Bounded quantum rejection sampling from a supplied joint preparation.

The conditional-state construction follows Low, Yoder and Chuang,
arXiv:1402.7359v1. The explicit finite power schedule is a bounded variant,
not the paper's unbounded expected-query-complexity algorithm.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from types import MappingProxyType

from cqlib import Circuit

from ..._utils.circuit import snapshot_bound_static_unitary_circuit
from ..._utils.state_preparation import build_state_preparation_circuit
from ...operators.composite import GroverOperator
from ...operators.oracle.phase_oracle import BitPatternPhaseOracle
from ...operators.state_preparation.user_circuit_preparation import (
    UserCircuitPreparation,
)
from ...protocols.sampler import Sampler, SamplerRequest, require_sampler_results
from ...protocols.state_preparation import StatePreparation


def _integer(value: int, name: str, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _names(names: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(names, (str, bytes)) or not isinstance(names, Sequence):
        raise TypeError(f"{name} must be a finite sequence of names")
    if len(names) == 0:
        raise ValueError(f"{name} must not be empty")
    values = tuple(names)
    if any(not isinstance(item, str) for item in values):
        raise TypeError(f"{name} must contain strings")
    if any(not item for item in values):
        raise ValueError(f"{name} names must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} names must be unique")
    return values


def _evidence(values: Mapping[str, int | bool] | None) -> dict[str, int]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError("evidence must be a mapping or None")
    result: dict[str, int] = {}
    for name, bit in values.items():
        if not isinstance(name, str):
            raise TypeError("evidence names must be strings")
        if not name:
            raise ValueError("evidence names must not be empty")
        if not isinstance(bit, int):
            raise TypeError("evidence values must be binary integers or bool")
        if bit not in (0, 1):
            raise ValueError("evidence values must be zero or one")
        result[name] = int(bit)
    return result


class QBayesianStatus(Enum):
    """Termination reason for a finite inference schedule."""

    TARGET_REACHED = "target_reached"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True, slots=True)
class QBayesianRound:
    """Effective Grover power and complete observed batch totals."""

    grover_power: int
    shots: int
    accepted: int

    def __post_init__(self) -> None:
        _integer(self.grover_power, "grover_power", minimum=0)
        _integer(self.shots, "shots")
        _integer(self.accepted, "accepted", minimum=0)
        if self.accepted > self.shots:
            raise ValueError("accepted must not exceed shots")


@dataclass(frozen=True, slots=True)
class QBayesianResult:
    """Sparse empirical posterior in the caller's query order.

    No accepted samples means ``probabilities=None``, not zero evidence
    probability. Query counters include shot multiplicity. Amplified
    acceptance is never an estimate of the original evidence probability.
    """

    query: tuple[str, ...]
    evidence: Mapping[str, int]
    counts: Mapping[tuple[int, ...], int]
    probabilities: Mapping[tuple[int, ...], float] | None
    accepted: int
    attempted_shots: int
    target_accepted: int
    rounds: tuple[QBayesianRound, ...]
    status: QBayesianStatus

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        query = _names(self.query, "query")
        evidence = _evidence(self.evidence)
        _integer(self.accepted, "accepted", minimum=0)
        _integer(self.attempted_shots, "attempted_shots")
        _integer(self.target_accepted, "target_accepted")
        if not isinstance(self.rounds, tuple) or not self.rounds:
            raise ValueError("rounds must be a nonempty tuple")
        if any(not isinstance(item, QBayesianRound) for item in self.rounds):
            raise TypeError("rounds must contain QBayesianRound instances")
        if sum(item.shots for item in self.rounds) != self.attempted_shots:
            raise ValueError("round shots must sum to attempted_shots")
        if sum(item.accepted for item in self.rounds) != self.accepted:
            raise ValueError("round accepted counts must sum to accepted")
        if sum(item.accepted for item in self.rounds[:-1]) >= self.target_accepted:
            raise ValueError("rounds must stop when the target is first reached")
        if not evidence and any(
            item.grover_power != 0 or item.accepted != item.shots
            for item in self.rounds
        ):
            raise ValueError(
                "empty evidence requires zero powers and accepts all shots"
            )
        if not isinstance(self.status, QBayesianStatus):
            raise TypeError("status must be a QBayesianStatus")
        reached = self.accepted >= self.target_accepted
        if (self.status is QBayesianStatus.TARGET_REACHED) != reached:
            raise ValueError("status must agree with accepted and target_accepted")
        if not isinstance(self.counts, Mapping):
            raise TypeError("counts must be a mapping")
        counts = dict(self.counts)
        for bits, count in counts.items():
            if not isinstance(bits, tuple) or len(bits) != len(query):
                raise ValueError("count keys must be tuples matching query width")
            if any(type(bit) is not int or bit not in (0, 1) for bit in bits):
                raise ValueError("count keys must contain integer bits")
            _integer(count, "sparse counts")
            if any(
                name in evidence and bits[index] != evidence[name]
                for index, name in enumerate(query)
            ):
                raise ValueError("query counts must agree with overlapping evidence")
        if sum(counts.values()) != self.accepted:
            raise ValueError("counts must sum to accepted")
        probabilities: dict[tuple[int, ...], float] | None = None
        if not self.accepted:
            if self.probabilities is not None:
                raise ValueError("probabilities must be None without accepted samples")
        else:
            if not isinstance(self.probabilities, Mapping):
                raise TypeError("probabilities must be a mapping with accepted samples")
            if set(self.probabilities) != set(counts):
                raise ValueError(
                    "probabilities must have the same sparse keys as counts"
                )
            probabilities = {}
            for bits, value in self.probabilities.items():
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise TypeError("probabilities must contain real values")
                probability = float(value)
                if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    raise ValueError(
                        "probabilities must be finite and between zero and one"
                    )
                if not math.isclose(
                    probability,
                    counts[bits] / self.accepted,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError("probabilities must equal normalized counts")
                probabilities[bits] = probability
            if not math.isclose(
                math.fsum(probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError("probabilities must sum to one")
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "evidence", MappingProxyType(evidence))
        object.__setattr__(self, "counts", MappingProxyType(counts))
        object.__setattr__(
            self,
            "probabilities",
            None if probabilities is None else MappingProxyType(probabilities),
        )

    @property
    def num_grover_queries(self) -> int:
        """Return Grover iterations weighted by executed shots."""
        return sum(item.grover_power * item.shots for item in self.rounds)

    @property
    def num_state_preparation_queries(self) -> int:
        """Return uses of A or its inverse, weighted by executed shots."""
        return sum((2 * item.grover_power + 1) * item.shots for item in self.rounds)


class QBayesian:
    """Infer computational-basis marginals by bounded amplitude amplification.

    ``state_preparation`` supplies the joint distribution as a preparation
    component. Its circuit must be fully bound, static, and
    invertible. ``variable_qubits`` maps nonempty names
    to distinct insertion positions, including when Core qubit IDs are not
    contiguous. Unnamed qubits are marginalized. Inputs are copied.

    ``run`` follows exactly its finite schedule until the accepted target
    is reached or the schedule ends. The whole schedule must pass resource
    limits before execution. Ideal conditional probabilities are unchanged
    by amplification; noisy execution only yields empirical postselection.
    Backend errors propagate.

    Args:
        state_preparation: Fully bound component encoding the joint
            distribution. Components are built once with prefix ``"bayes"``.
        variable_qubits: Variable names mapped to distinct qubit insertion positions.
        sampler: Executor used by the finite amplification-and-sampling schedule.

    Raises:
        TypeError: If input types or component interfaces are invalid.
        ValueError: If the preparation or variable positions are invalid.
    """

    __slots__ = ("_sampler", "_state_preparation", "_variable_qubits")

    def __init__(
        self,
        state_preparation: StatePreparation,
        variable_qubits: Mapping[str, int],
        sampler: Sampler,
    ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError("sampler must be a Sampler instance")
        source = build_state_preparation_circuit(state_preparation, prefix="bayes")
        preparation = snapshot_bound_static_unitary_circuit(source, "state_preparation")
        if not isinstance(variable_qubits, Mapping):
            raise TypeError("variable_qubits must be a mapping")
        if not variable_qubits:
            raise ValueError("variable_qubits must not be empty")
        names = _names(tuple(variable_qubits), "variable_qubits")
        positions = dict(variable_qubits)
        for name in names:
            _integer(positions[name], "variable qubit position", minimum=0)
            if positions[name] >= preparation.num_qubits:
                raise ValueError("variable qubit positions must be in circuit range")
        if len(set(positions.values())) != len(positions):
            raise ValueError("variable qubit positions must be distinct")
        self._state_preparation = UserCircuitPreparation(preparation)
        self._variable_qubits = MappingProxyType(positions)
        self._sampler = sampler

    @property
    def state_preparation(self) -> Circuit:
        """Return an independent normalized preparation circuit."""
        return self._state_preparation.build_circuit("")

    @property
    def variable_qubits(self) -> Mapping[str, int]:
        """Return immutable variable insertion positions."""
        return self._variable_qubits

    def _normalize_evidence(
        self, evidence: Mapping[str, int | bool] | None
    ) -> dict[str, int]:
        result = _evidence(evidence)
        if any(name not in self._variable_qubits for name in result):
            raise ValueError("evidence contains an unknown variable")
        return result

    def build_circuit(
        self,
        evidence: Mapping[str, int | bool] | None = None,
        *,
        grover_power: int = 0,
        max_grover_power: int = 1024,
    ) -> Circuit:
        """Return an independent unmeasured A followed by k Grover iterations.

        Empty evidence uses effective power zero. The requested power must
        still respect the explicit nonnegative ``max_grover_power`` limit.
        """
        _integer(grover_power, "grover_power", minimum=0)
        _integer(max_grover_power, "max_grover_power", minimum=0)
        if grover_power > max_grover_power:
            raise ValueError("grover_power exceeds max_grover_power")
        assignments = self._normalize_evidence(evidence)
        circuit = self.state_preparation
        if not assignments or grover_power == 0:
            return circuit
        oracle = BitPatternPhaseOracle(
            circuit.num_qubits,
            {self._variable_qubits[name]: bit for name, bit in assignments.items()},
        )
        iteration = GroverOperator(
            oracle, state_preparation=self._state_preparation
        ).build_circuit()
        for _ in range(grover_power):
            circuit.compose(iteration, circuit.qubits)
        return circuit

    def run(
        self,
        query: Sequence[str],
        evidence: Mapping[str, int | bool] | None = None,
        *,
        target_accepted: int = 256,
        shots_per_round: int = 256,
        amplification_powers: Sequence[int] = (0, 1, 2, 4, 8, 16, 32),
        max_rounds: int = 128,
        max_grover_power: int = 1024,
        max_total_shots: int = 1_000_000,
        max_grover_queries: int = 10_000_000,
    ) -> QBayesianResult:
        """Execute a preflighted finite schedule and return sparse postselection.

        Each schedule entry requests ``shots_per_round`` shots. All accepted
        counts from the terminating round are retained, so the target can be
        exceeded. Limits reject the entire schedule rather than clipping it.
        A zero-acceptance result does not certify impossible evidence.

        Args:
            query: Variable names in the desired output-tuple order.
            evidence: Known binary variable values, or ``None`` for a marginal.
            target_accepted: Positive accepted-sample target.
            shots_per_round: Positive samples requested in each round.
            amplification_powers: Nonempty sequence of nonnegative Grover powers.
            max_rounds: Upper bound on the supplied schedule length.
            max_grover_power: Largest allowed individual amplification power.
            max_total_shots: Upper bound on the schedule's total requested samples.
            max_grover_queries: Upper bound on shot-weighted Grover iterations.

        Returns:
            Empirical conditional probabilities, accepted counts, per-round
            observations, query counts, and the termination status.

        Raises:
            TypeError: If input types or returned sampler results are invalid.
            ValueError: If names, budgets, or sampler results violate the contract.
        """
        names = _names(query, "query")
        if any(name not in self._variable_qubits for name in names):
            raise ValueError("query contains an unknown variable")
        assignments = self._normalize_evidence(evidence)
        for value, label in (
            (target_accepted, "target_accepted"),
            (shots_per_round, "shots_per_round"),
            (max_rounds, "max_rounds"),
            (max_total_shots, "max_total_shots"),
            (max_grover_queries, "max_grover_queries"),
        ):
            _integer(value, label)
        _integer(max_grover_power, "max_grover_power", minimum=0)
        if isinstance(amplification_powers, (str, bytes)) or not isinstance(
            amplification_powers, Sequence
        ):
            raise TypeError("amplification_powers must be a finite sequence")
        size = len(amplification_powers)
        if size == 0:
            raise ValueError("amplification_powers must not be empty")
        if size > max_rounds:
            raise ValueError("schedule exceeds max_rounds")
        if size * shots_per_round > max_total_shots:
            raise ValueError("schedule exceeds max_total_shots")
        powers = tuple(amplification_powers)
        for power in powers:
            _integer(power, "amplification power", minimum=0)
            if power > max_grover_power:
                raise ValueError("schedule exceeds max_grover_power")
        effective_powers = powers if assignments else (0,) * size
        if sum(effective_powers) * shots_per_round > max_grover_queries:
            raise ValueError("schedule exceeds max_grover_queries")
        positions = sorted(
            {self._variable_qubits[name] for name in (*names, *assignments)}
        )
        measured_index = {position: index for index, position in enumerate(positions)}
        counts: dict[tuple[int, ...], int] = {}
        rounds: list[QBayesianRound] = []
        accepted = 0
        for power in effective_powers:
            circuit = self.build_circuit(
                assignments, grover_power=power, max_grover_power=max_grover_power
            )
            request = SamplerRequest(
                circuit,
                tuple(circuit.qubits[position] for position in positions),
                shots_per_round,
            )
            result = require_sampler_results(self._sampler.run([request]), [request])[0]
            round_accepted = 0
            for bits, count in result.counts.items():
                decoded = {
                    name: int(bits[-1 - measured_index[position]])
                    for name, position in self._variable_qubits.items()
                    if position in measured_index
                }
                if any(decoded[name] != value for name, value in assignments.items()):
                    continue
                outcome = tuple(decoded[name] for name in names)
                if count:
                    counts[outcome] = counts.get(outcome, 0) + count
                round_accepted += count
            rounds.append(QBayesianRound(power, shots_per_round, round_accepted))
            accepted += round_accepted
            if accepted >= target_accepted:
                break
        probabilities = (
            None
            if not accepted
            else {key: count / accepted for key, count in counts.items()}
        )
        return QBayesianResult(
            names,
            assignments,
            counts,
            probabilities,
            accepted,
            sum(item.shots for item in rounds),
            target_accepted,
            tuple(rounds),
            QBayesianStatus.TARGET_REACHED
            if accepted >= target_accepted
            else QBayesianStatus.BUDGET_EXHAUSTED,
        )

    def infer(
        self,
        query: Sequence[str],
        evidence: Mapping[str, int | bool] | None = None,
        *,
        target_accepted: int = 256,
        shots_per_round: int = 256,
        amplification_powers: Sequence[int] = (0, 1, 2, 4, 8, 16, 32),
        max_rounds: int = 128,
        max_grover_power: int = 1024,
        max_total_shots: int = 1_000_000,
        max_grover_queries: int = 10_000_000,
    ) -> QBayesianResult:
        """Infer the posterior with the same schedule and result as :meth:`run`."""
        return self.run(
            query,
            evidence,
            target_accepted=target_accepted,
            shots_per_round=shots_per_round,
            amplification_powers=amplification_powers,
            max_rounds=max_rounds,
            max_grover_power=max_grover_power,
            max_total_shots=max_total_shots,
            max_grover_queries=max_grover_queries,
        )


__all__ = ["QBayesian", "QBayesianResult", "QBayesianRound", "QBayesianStatus"]
