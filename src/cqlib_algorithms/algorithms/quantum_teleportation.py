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

"""Ideal local simulation of the standard quantum teleportation protocol.

References
----------
[1] C. H. Bennett et al., "Teleporting an unknown quantum state via dual
    classical and Einstein-Podolsky-Rosen channels," Physical Review
    Letters, 70, 1895-1899, 1993. DOI: 10.1103/PhysRevLett.70.1895.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

from cqlib import Circuit
from cqlib.qis import metrics
from cqlib.qis.state import DensityMatrix, Statevector

from ..operators.estimator._circuit_validation import (
    validate_unitary_state_preparation,
)
from ..operators.state_preparation import (
    GHZStatePreparation,
    UserCircuitPreparation,
)

_NUMERICAL_TOLERANCE = 1e-12
_BELL_OUTCOMES = ((0, 0), (0, 1), (1, 0), (1, 1))


def _validate_shots(shots: int) -> None:
    if type(shots) is not int:
        raise TypeError("shots must be an integer")
    if shots < 1:
        raise ValueError("shots must be greater than zero")


def _validate_fidelity_tolerance(tolerance: float) -> None:
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
        raise TypeError("fidelity_tolerance must be a real number")
    numeric = float(tolerance)
    if not math.isfinite(numeric):
        raise ValueError("fidelity_tolerance must be finite")
    if not 0.0 <= numeric <= 1.0:
        raise ValueError("fidelity_tolerance must be between zero and one")


def _snapshot_state_preparation(state_preparation: Circuit) -> Circuit:
    if not isinstance(state_preparation, Circuit):
        raise TypeError("state_preparation must be a Circuit")
    state_preparation.validate()
    if state_preparation.num_qubits != 1:
        raise ValueError("state_preparation must contain exactly one qubit")
    if state_preparation.used_symbols:
        raise ValueError("state_preparation must not contain active symbols")
    validate_unitary_state_preparation(state_preparation)

    preparation = UserCircuitPreparation(state_preparation)
    preparation.validate()
    normalized = preparation.build_circuit("teleportation_input")
    validate_unitary_state_preparation(normalized)
    return normalized


def _normalize_fidelity(value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("teleportation fidelity must be finite")
    if numeric < -_NUMERICAL_TOLERANCE or numeric > 1.0 + _NUMERICAL_TOLERANCE:
        raise ValueError("teleportation fidelity must be between zero and one")
    return min(1.0, max(0.0, numeric))


def _copy_measurements(
    measurements: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    if not isinstance(measurements, tuple):
        raise TypeError("measurements must be a tuple")
    copied: list[tuple[int, int]] = []
    for outcome in measurements:
        if not isinstance(outcome, tuple) or len(outcome) != 2:
            raise TypeError("measurements must contain two-bit tuples")
        phase_bit, bit_flip_bit = outcome
        if type(phase_bit) is not int or type(bit_flip_bit) is not int:
            raise TypeError("measurement bits must be integers")
        if phase_bit not in (0, 1) or bit_flip_bit not in (0, 1):
            raise ValueError("measurement bits must be zero or one")
        copied.append((phase_bit, bit_flip_bit))
    return tuple(copied)


def _copy_fidelities(values: tuple[float, ...]) -> tuple[float, ...]:
    if not isinstance(values, tuple):
        raise TypeError("fidelities must be a tuple")
    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("fidelities must contain real numbers")
        normalized.append(_normalize_fidelity(float(value)))
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class QuantumTeleportationRequest:
    """One ideal quantum teleportation experiment.

    ``state_preparation`` must be a fixed one-qubit unitary circuit. The
    request is shallowly frozen, so its mutable Core circuit must not be
    changed during execution; the algorithm revalidates it before use.
    """

    state_preparation: Circuit
    shots: int
    fidelity_tolerance: float = 1e-12

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        _snapshot_state_preparation(self.state_preparation)
        _validate_shots(self.shots)
        _validate_fidelity_tolerance(self.fidelity_tolerance)
        object.__setattr__(self, "fidelity_tolerance", float(self.fidelity_tolerance))

    def validate(self) -> None:
        """Revalidate this request, including its mutable Core circuit."""
        _snapshot_state_preparation(self.state_preparation)
        _validate_shots(self.shots)
        _validate_fidelity_tolerance(self.fidelity_tolerance)


@dataclass(frozen=True, slots=True)
class QuantumTeleportationResult:
    """Bell outcomes and per-shot restored-state fidelities.

    Every measurement is ``(phase_bit, bit_flip_bit)``. Counts and
    probabilities use the fixed order ``00, 01, 10, 11`` in that bit order.
    """

    measurements: tuple[tuple[int, int], ...]
    fidelities: tuple[float, ...]
    fidelity_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        measurements = _copy_measurements(self.measurements)
        fidelities = _copy_fidelities(self.fidelities)
        _validate_fidelity_tolerance(self.fidelity_tolerance)
        if not measurements:
            raise ValueError("measurements must not be empty")
        if len(fidelities) != len(measurements):
            raise ValueError("fidelities must match the measurement count")
        object.__setattr__(self, "measurements", measurements)
        object.__setattr__(self, "fidelities", fidelities)
        object.__setattr__(self, "fidelity_tolerance", float(self.fidelity_tolerance))

    @property
    def shots(self) -> int:
        """Return the number of simulated protocol executions."""
        return len(self.measurements)

    @property
    def bell_counts(self) -> tuple[int, int, int, int]:
        """Return counts in fixed ``00, 01, 10, 11`` order."""
        return (
            self.measurements.count(_BELL_OUTCOMES[0]),
            self.measurements.count(_BELL_OUTCOMES[1]),
            self.measurements.count(_BELL_OUTCOMES[2]),
            self.measurements.count(_BELL_OUTCOMES[3]),
        )

    @property
    def bell_probabilities(self) -> tuple[float, float, float, float]:
        """Return empirical probabilities in fixed ``00, 01, 10, 11`` order."""
        counts = self.bell_counts
        return (
            counts[0] / self.shots,
            counts[1] / self.shots,
            counts[2] / self.shots,
            counts[3] / self.shots,
        )

    @property
    def minimum_fidelity(self) -> float:
        """Return the smallest restored-state fidelity."""
        return min(self.fidelities)

    @property
    def mean_fidelity(self) -> float:
        """Return the arithmetic mean restored-state fidelity."""
        return sum(self.fidelities) / self.shots

    @property
    def successful_shots(self) -> int:
        """Return shots with fidelity at least ``1 - fidelity_tolerance``."""
        threshold = 1.0 - self.fidelity_tolerance
        return sum(fidelity >= threshold for fidelity in self.fidelities)

    @property
    def success_rate(self) -> float:
        """Return the fraction of shots meeting the fidelity threshold."""
        return self.successful_shots / self.shots

    @property
    def all_succeeded(self) -> bool:
        """Return whether every shot meets the fidelity threshold."""
        return self.successful_shots == self.shots


class QuantumTeleportation:
    """Simulate Bennett et al.'s standard teleportation protocol locally.

    Circuit positions are input ``Q=0``, Alice's entangled qubit ``A=1``,
    and Bob's qubit ``B=2``. Each shot performs real Core measurements of Q
    and A, then applies X and Z corrections to B from those classical bits.
    """

    def build_premeasurement_circuit(self, state_preparation: Circuit) -> Circuit:
        """Build the three-qubit circuit immediately before Bell measurement."""
        input_circuit = _snapshot_state_preparation(state_preparation)
        return self._build_from_snapshot(input_circuit)

    @staticmethod
    def _build_from_snapshot(input_circuit: Circuit) -> Circuit:
        circuit = Circuit(3)
        circuit.compose(input_circuit, [0])
        bell_pair = GHZStatePreparation(2).build_circuit("teleportation_bell")
        circuit.compose(bell_pair, [1, 2])
        circuit.cx(0, 1)
        circuit.h(0)
        circuit.validate()
        return circuit

    def run(
        self,
        requests: Sequence[QuantumTeleportationRequest],
        /,
    ) -> list[QuantumTeleportationResult]:
        """Execute a validated batch of ideal local teleportation experiments."""
        prepared: list[
            tuple[QuantumTeleportationRequest, Statevector, Statevector]
        ] = []
        for request in requests:
            if not isinstance(request, QuantumTeleportationRequest):
                raise TypeError(
                    "requests must contain QuantumTeleportationRequest instances"
                )
            input_circuit = _snapshot_state_preparation(request.state_preparation)
            _validate_shots(request.shots)
            _validate_fidelity_tolerance(request.fidelity_tolerance)
            target_state = Statevector.from_circuit(input_circuit)
            premeasurement = Statevector.from_circuit(
                self._build_from_snapshot(input_circuit)
            )
            prepared.append((request, target_state, premeasurement))

        results: list[QuantumTeleportationResult] = []
        for request, target_state, premeasurement in prepared:
            measurements: list[tuple[int, int]] = []
            fidelities: list[float] = []
            for _ in range(request.shots):
                state = premeasurement.copy()
                phase_bit = int(state.measure(0))
                bit_flip_bit = int(state.measure(1))
                if bit_flip_bit:
                    state.apply_x(2)
                if phase_bit:
                    state.apply_z(2)

                bob_state = DensityMatrix.from_state(3, state.data).partial_trace([2])
                fidelity = metrics.state_fidelity_pure_mixed(target_state, bob_state)
                measurements.append((phase_bit, bit_flip_bit))
                fidelities.append(_normalize_fidelity(fidelity))

            results.append(
                QuantumTeleportationResult(
                    tuple(measurements),
                    tuple(fidelities),
                    request.fidelity_tolerance,
                )
            )
        return results
