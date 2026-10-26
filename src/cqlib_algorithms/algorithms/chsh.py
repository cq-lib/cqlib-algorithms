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

"""CHSH inequality evaluation through the public Estimator contract.

References
----------
[1] J. F. Clauser, M. A. Horne, A. Shimony, and R. A. Holt, "Proposed
    experiment to test local hidden-variable theories," Physical Review
    Letters, 23, 880-884, 1969.
    DOI: 10.1103/PhysRevLett.23.880.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import ClassVar

from cqlib.qis import Hamiltonian, Pauli, PauliString

from cqlib import Circuit

from ..operators.estimator._circuit_validation import (
    validate_unitary_state_preparation,
)
from ..operators.state_preparation import GHZStatePreparation
from ..protocols.estimator import Estimator, EstimatorRequest, EstimatorResult

_COEFFICIENT_TOLERANCE = 1e-15


def _validate_angles(angles: tuple[float, float], name: str) -> None:
    if not isinstance(angles, tuple):
        raise TypeError(f"{name} must be a tuple")
    if len(angles) != 2:
        raise ValueError(f"{name} must contain exactly two angles")
    for angle in angles:
        if isinstance(angle, bool) or not isinstance(angle, Real):
            raise TypeError(f"{name} must contain real numbers")
        if not math.isfinite(float(angle)):
            raise ValueError(f"{name} must contain finite angles")


def _validate_shots(shots: int | None) -> None:
    if shots is None:
        return
    if isinstance(shots, bool) or not isinstance(shots, int):
        raise TypeError("shots must be an integer or None")
    if shots <= 0:
        raise ValueError("shots must be greater than zero")


def _validate_tolerance(tolerance: float) -> None:
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
        raise TypeError("tolerance must be a real number")
    if not math.isfinite(float(tolerance)):
        raise ValueError("tolerance must be finite")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")


def _validate_state_preparation(state_preparation: Circuit | None) -> None:
    if state_preparation is None:
        return
    if not isinstance(state_preparation, Circuit):
        raise TypeError("state_preparation must be a Circuit or None")
    state_preparation.validate()
    if state_preparation.num_qubits != 2:
        raise ValueError("state_preparation must contain exactly two qubits")
    if state_preparation.used_symbols:
        raise ValueError("state_preparation must not contain active symbols")
    validate_unitary_state_preparation(state_preparation)


def _normalize_pair(values: tuple[float, float], name: str) -> tuple[float, float]:
    _validate_angles(values, name)
    return (float(values[0]), float(values[1]))


@dataclass(frozen=True, slots=True)
class CHSHRequest:
    """A CHSH evaluation request.

    Alice is circuit-position 0 and Bob is circuit-position 1. Their two
    measurement settings are ``M(theta) = cos(theta) Z + sin(theta) X``.
    ``shots=None`` requests exact estimation; a positive value delegates
    finite-shot evaluation to the injected estimator.

    The request is shallowly frozen. A caller-owned preparation circuit must
    not be modified during the request lifetime, and :class:`CHSH` revalidates
    it immediately before execution.

    Args:
        state_preparation: Fixed two-qubit preparation, or ``None`` for Bell state.
        alice_angles: Alice's two XZ-plane measurement angles.
        bob_angles: Bob's two XZ-plane measurement angles.
        shots: Samples per correlator, or ``None`` for exact estimation.
        tolerance: Numerical margin used for the classical-bound comparison.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If circuit, angles, shots, or tolerance are invalid.
    """

    state_preparation: Circuit | None = None
    alice_angles: tuple[float, float] = (0.0, math.pi / 2.0)
    bob_angles: tuple[float, float] = (math.pi / 4.0, -math.pi / 4.0)
    shots: int | None = None
    tolerance: float = 1e-12

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "alice_angles",
            _normalize_pair(self.alice_angles, "alice_angles"),
        )
        object.__setattr__(
            self,
            "bob_angles",
            _normalize_pair(self.bob_angles, "bob_angles"),
        )
        _validate_shots(self.shots)
        _validate_tolerance(self.tolerance)
        object.__setattr__(self, "tolerance", float(self.tolerance))
        _validate_state_preparation(self.state_preparation)

    def validate(self) -> None:
        """Revalidate this request and its mutable Core circuit."""
        _validate_state_preparation(self.state_preparation)
        _validate_angles(self.alice_angles, "alice_angles")
        _validate_angles(self.bob_angles, "bob_angles")
        _validate_shots(self.shots)
        _validate_tolerance(self.tolerance)


def _normalize_correlators(
    values: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if not isinstance(values, tuple):
        raise TypeError("correlators must be a tuple")
    if len(values) != 4:
        raise ValueError("correlators must contain E00, E01, E10, and E11")
    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("correlators must contain real numbers")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("correlators must be finite")
        normalized.append(numeric)
    return (normalized[0], normalized[1], normalized[2], normalized[3])


def _normalize_variances(
    values: tuple[float | None, float | None, float | None, float | None],
) -> tuple[float | None, float | None, float | None, float | None]:
    if not isinstance(values, tuple):
        raise TypeError("variances must be a tuple")
    if len(values) != 4:
        raise ValueError("variances must contain four entries")
    normalized: list[float | None] = []
    for value in values:
        if value is None:
            normalized.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("variances must contain real numbers or None")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("variances must be finite")
        if numeric < 0.0:
            raise ValueError("variances must be non-negative")
        normalized.append(numeric)
    return (normalized[0], normalized[1], normalized[2], normalized[3])


@dataclass(frozen=True, slots=True)
class CHSHResult:
    """Four correlators and their CHSH witness.

    Correlators are ordered ``(E00, E01, E10, E11)`` and define
    ``S = E00 + E01 + E10 - E11``. Values outside physical ideal bounds are
    retained because finite-shot, mitigated, or user-supplied estimators can
    produce them. ``tolerance`` is only a numerical comparison margin; it is
    not a statistical confidence interval.

    Attributes:
        correlators: ``(E00, E01, E10, E11)``.
        variances: Estimator variances in the same order.
        tolerance: Numerical bound-comparison margin.
    """

    CLASSICAL_BOUND: ClassVar[float] = 2.0
    TSIRELSON_BOUND: ClassVar[float] = 2.0 * math.sqrt(2.0)

    correlators: tuple[float, float, float, float]
    variances: tuple[float | None, float | None, float | None, float | None]
    tolerance: float = 1e-12

    def __post_init__(self) -> None:
        correlators = _normalize_correlators(self.correlators)
        variances = _normalize_variances(self.variances)
        _validate_tolerance(self.tolerance)
        object.__setattr__(self, "correlators", correlators)
        object.__setattr__(self, "variances", variances)
        object.__setattr__(self, "tolerance", float(self.tolerance))

    @property
    def e00(self) -> float:
        """Return the expectation for Alice setting 0 and Bob setting 0."""
        return self.correlators[0]

    @property
    def e01(self) -> float:
        """Return the expectation for Alice setting 0 and Bob setting 1."""
        return self.correlators[1]

    @property
    def e10(self) -> float:
        """Return the expectation for Alice setting 1 and Bob setting 0."""
        return self.correlators[2]

    @property
    def e11(self) -> float:
        """Return the expectation for Alice setting 1 and Bob setting 1."""
        return self.correlators[3]

    @property
    def chsh_value(self) -> float:
        """Return the signed CHSH witness ``E00 + E01 + E10 - E11``."""
        return self.e00 + self.e01 + self.e10 - self.e11

    @property
    def absolute_value(self) -> float:
        """Return the absolute CHSH witness."""
        return abs(self.chsh_value)

    @property
    def violation_margin(self) -> float:
        """Return ``abs(S) - 2`` without clipping negative values."""
        return self.absolute_value - self.CLASSICAL_BOUND

    @property
    def violates_classical_bound(self) -> bool:
        """Return whether ``abs(S)`` exceeds ``2 + tolerance``."""
        return self.absolute_value > self.CLASSICAL_BOUND + self.tolerance

    @property
    def classical_bound(self) -> float:
        """Return the local-hidden-variable CHSH bound."""
        return self.CLASSICAL_BOUND

    @property
    def tsirelson_bound(self) -> float:
        """Return the ideal quantum CHSH bound."""
        return self.TSIRELSON_BOUND

    @property
    def standard_deviation_upper_bound(self) -> float | None:
        """Return ``sum(sqrt(variance_i))`` when all variances are known.

        This is a conservative bound derived without covariance information,
        not a confidence interval or a significance test.
        """
        if any(variance is None for variance in self.variances):
            return None
        return sum(
            math.sqrt(variance) for variance in self.variances if variance is not None
        )


def _product_observable(alice_angle: float, bob_angle: float) -> Hamiltonian:
    alice_components = (
        (Pauli.z(), math.cos(alice_angle)),
        (Pauli.x(), math.sin(alice_angle)),
    )
    bob_components = (
        (Pauli.z(), math.cos(bob_angle)),
        (Pauli.x(), math.sin(bob_angle)),
    )
    observable = Hamiltonian(2)
    for alice_pauli, alice_coefficient in alice_components:
        for bob_pauli, bob_coefficient in bob_components:
            coefficient = alice_coefficient * bob_coefficient
            if abs(coefficient) <= _COEFFICIENT_TOLERANCE:
                continue
            term = PauliString(2)
            term.set_pauli(0, alice_pauli)
            term.set_pauli(1, bob_pauli)
            observable.add_term(term, coefficient)
    return observable


class CHSH:
    """Evaluate the CHSH witness with an injected expectation estimator.

    The default circuit prepares ``(|00> + |11>)/sqrt(2)``. Alice is circuit
    position 0 and Bob is position 1, including for custom circuits with
    reordered or non-contiguous qubit identifiers. Four product observables
    are submitted in ``E00, E01, E10, E11`` order in one flattened estimator
    batch. This class evaluates a witness; it does not model detector or
    locality loopholes, perform mitigation, or claim statistical significance.

    Args:
        estimator: Expectation estimator used for the four correlators.

    Raises:
        TypeError: If the estimator or a request has the wrong type.
        ValueError: If circuit or estimator-result validation fails.
    """

    __slots__ = ("_estimator",)

    def __init__(self, estimator: Estimator) -> None:
        if not isinstance(estimator, Estimator):
            raise TypeError("estimator must be an Estimator")
        self._estimator = estimator

    @property
    def estimator(self) -> Estimator:
        """Return the injected estimator."""
        return self._estimator

    def build_circuit(self, state_preparation: Circuit | None = None) -> Circuit:
        """Return an independent, fixed two-qubit preparation circuit."""
        _validate_state_preparation(state_preparation)
        if state_preparation is None:
            return GHZStatePreparation(2).build_circuit("chsh")

        snapshot = copy.deepcopy(state_preparation)
        circuit = Circuit(2)
        circuit.compose(snapshot, circuit.qubits)
        circuit.validate()
        return circuit

    def run(self, requests: Sequence[CHSHRequest], /) -> list[CHSHResult]:
        """Evaluate all requests in one flattened estimator call."""
        estimator_requests: list[EstimatorRequest] = []
        validated_requests: list[CHSHRequest] = []
        for request in requests:
            if not isinstance(request, CHSHRequest):
                raise TypeError("requests must contain CHSHRequest instances")
            request.validate()
            circuit = self.build_circuit(request.state_preparation)
            observables = (
                _product_observable(request.alice_angles[0], request.bob_angles[0]),
                _product_observable(request.alice_angles[0], request.bob_angles[1]),
                _product_observable(request.alice_angles[1], request.bob_angles[0]),
                _product_observable(request.alice_angles[1], request.bob_angles[1]),
            )
            for observable in observables:
                estimator_requests.append(
                    EstimatorRequest(
                        copy.deepcopy(circuit),
                        observable,
                        shots=request.shots,
                    )
                )
            validated_requests.append(request)

        if not estimator_requests:
            return []
        estimator_results = self._estimator.run(estimator_requests)
        if not isinstance(estimator_results, list):
            raise TypeError("estimator must return a list")
        if len(estimator_results) != len(estimator_requests):
            raise ValueError("estimator returned an unexpected result count")
        if any(not isinstance(result, EstimatorResult) for result in estimator_results):
            raise TypeError("estimator results must contain EstimatorResult instances")

        results: list[CHSHResult] = []
        for index, request in enumerate(validated_requests):
            group = estimator_results[4 * index : 4 * index + 4]
            results.append(
                CHSHResult(
                    (
                        group[0].expectation,
                        group[1].expectation,
                        group[2].expectation,
                        group[3].expectation,
                    ),
                    (
                        group[0].variance,
                        group[1].variance,
                        group[2].variance,
                        group[3].variance,
                    ),
                    request.tolerance,
                )
            )
        return results
