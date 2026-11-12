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

"""Tests for Clifford data regression."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import pytest
from cqlib import Circuit
from cqlib.circuit import StandardGate, ValueOperation
from cqlib.qis import PauliString

from cqlib_algorithms.error_mitigation.core import QemExecutor
from cqlib_algorithms.error_mitigation.techniques.cdr import (
    _near_clifford_operation,
    execute_with_cdr,
    fit_linear_regression,
    near_clifford_circuit,
)
from cqlib_algorithms.operators.estimator import StatevectorEstimator
from cqlib_algorithms.protocols import (
    Estimator,
    EstimatorRequest,
    EstimatorResult,
)

_ATOL = 1e-9

# The grid a snapped rotation lands on is a multiple of this half-turn.
_HALF_PI = math.pi / 2.0

# Affine response of the toy noisy backend reused by the training cloud and
# the linear-regime regression check: ``noisy = scale * ideal + offset``.
_CLOUD_SCALE = 0.8
_CLOUD_OFFSET = 0.05


class _AffineNoiseEstimator(Estimator):
    """Toy backend applying ``scale * exact + offset`` to every value."""

    __slots__ = ("_ideal", "_offset", "_scale")

    def __init__(self, scale: float, offset: float) -> None:
        self._ideal = StatevectorEstimator()
        self._scale = scale
        self._offset = offset

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        """Return the affine response of the exact values, in input order."""
        exact = self._ideal.run(requests)
        return [
            EstimatorResult(expectation=self._scale * result.expectation + self._offset)
            for result in exact
        ]


def _clifford_training_cloud(
    circuit: Circuit,
    observable: PauliString,
    executor: QemExecutor,
    ideal_estimator: Estimator,
    /,
    *,
    num_points: int = 6,
    seed: int = 0,
) -> tuple[list[float], list[float]]:
    """Return the ``(noisy, ideal)`` values of a near-Clifford cloud.

    The cloud is built by snapping half the rotations of ``circuit`` on every
    draw, so the variants differ in which rotations became Clifford gates and
    their exact values spread over more than one point.

    Parameters
    ----------
    circuit
        Target body the variants are drawn from.
    observable
        Observable evaluated on every variant.
    executor
        Noisy backend supplying the ``x`` values.
    ideal_estimator
        Exact backend supplying the ``y`` values.
    num_points
        Number of variants to draw.
    seed
        Seed of the snap decisions.

    Returns
    -------
    tuple[list[float], list[float]]
        The noisy and exact values, aligned by variant index.
    """
    rng = np.random.default_rng(seed)
    cloud = [near_clifford_circuit(circuit, 0.5, rng) for _ in range(num_points)]
    requests = [
        EstimatorRequest(circuit=variant, observable=observable) for variant in cloud
    ]
    noisy = executor.estimate(requests)
    ideal = ideal_estimator.run(requests)
    return (
        [result.expectation for result in noisy],
        [result.expectation for result in ideal],
    )


def _clifford_body() -> Circuit:
    """Return a two-qubit body mixing Clifford gates and generic rotations."""
    circuit = Circuit(2)
    circuit.ry(0, 0.3)
    circuit.ry(0, 1.1)
    circuit.ry(1, 2.2)
    circuit.cx(0, 1)
    return circuit


def test_near_clifford_circuit_maps_a_pure_clifford_circuit_to_itself(
    names: Callable[[Circuit], list[str]],
    phases: Callable[[Circuit], complex],
) -> None:
    body = Circuit(2)
    body.h(0)
    body.rz(0, math.pi / 2)
    body.cx(0, 1)
    body.s(1)
    body.set_global_phase(0.25)

    mapped = near_clifford_circuit(body, 0.5, np.random.default_rng(0))

    # Every rotation already sits on the Clifford grid, so nothing is rebuilt:
    # the same operations, unitary, and global phase must come back out.
    assert names(mapped) == names(body)
    assert np.allclose(mapped.to_matrix(), body.to_matrix(), atol=_ATOL)
    assert phases(mapped) == pytest.approx(phases(body), abs=_ATOL)


def test_near_clifford_keeps_the_original_rotation_gate_type() -> None:
    circuit = Circuit(1)
    circuit.rx(0, 1.0)

    mapped = near_clifford_circuit(circuit, 1.0, np.random.default_rng(0))
    operation = next(iter(mapped.operations))

    # Snapping changes the angle only; an RX stays an RX.
    assert operation.instruction.standard_gate == StandardGate.RX
    assert float(operation.params[0]) == pytest.approx(_HALF_PI, abs=_ATOL)


def test_near_clifford_snaps_to_the_nearest_multiple_of_half_pi() -> None:
    for angle in (0.3, 1.0, 2.4, 5.0, -0.7):
        circuit = Circuit(1)
        circuit.rx(0, angle)

        mapped = near_clifford_circuit(circuit, 1.0, np.random.default_rng(0))
        snapped = float(next(iter(mapped.operations)).params[0])

        # Land on the grid and stay within half a step of the original angle.
        assert snapped == pytest.approx(_HALF_PI * round(snapped / _HALF_PI), abs=_ATOL)
        assert abs(snapped - angle) <= _HALF_PI / 2.0 + _ATOL


def test_near_clifford_returns_the_same_operation_when_already_on_grid() -> None:
    qubits = list(Circuit(1).qubits)
    operation = ValueOperation.from_standard_gate(StandardGate.RZ(_HALF_PI), qubits)

    # A rotation on the grid is returned bit-identical even when selected,
    # which keeps a Clifford circuit from being rebuilt into a near copy.
    snapped = _near_clifford_operation(operation, 1.0, np.random.default_rng(0))
    assert snapped is operation


def test_fit_linear_regression_recovers_a_known_line(
    exact_estimator: StatevectorEstimator,
    zz_observable: PauliString,
) -> None:
    executor = QemExecutor(estimator=_AffineNoiseEstimator(_CLOUD_SCALE, _CLOUD_OFFSET))
    noisy, ideal = _clifford_training_cloud(
        _clifford_body(), zz_observable, executor, exact_estimator
    )

    slope, intercept = fit_linear_regression(noisy, ideal)

    # The cloud lies on ``ideal = noisy / scale - offset / scale`` exactly, so
    # the closed-form fit returns that line.
    assert slope == pytest.approx(1.0 / _CLOUD_SCALE, abs=_ATOL)
    assert intercept == pytest.approx(-_CLOUD_OFFSET / _CLOUD_SCALE, abs=_ATOL)


def test_fit_linear_regression_needs_at_least_two_points() -> None:
    with pytest.raises(ValueError, match="at least 2 training point"):
        fit_linear_regression([0.5], [0.4])


def test_execute_with_cdr_recovers_the_ideal_value_in_a_linear_regime(
    exact_estimator: StatevectorEstimator,
    z_observable: PauliString,
) -> None:
    target = Circuit(1)
    target.rx(0, 0.7)
    executor = QemExecutor(estimator=_AffineNoiseEstimator(_CLOUD_SCALE, _CLOUD_OFFSET))

    exact = exact_estimator.run(
        [EstimatorRequest(circuit=target, observable=z_observable)]
    )[0].expectation
    mitigated = execute_with_cdr(
        target,
        executor,
        z_observable,
        ideal_estimator=exact_estimator,
        num_training_circuits=20,
        fraction_non_clifford=0.5,
        seed=1,
    )

    # The noise is affine in the exact value, so every training point lands on
    # the fitted line and the correction removes it entirely.
    assert mitigated == pytest.approx(exact, abs=1e-12)


def test_cdr_rejects_a_non_z_diagonal_observable(
    exact_estimator: StatevectorEstimator,
) -> None:
    target = Circuit(1)
    target.rx(0, 0.7)
    executor = QemExecutor(estimator=_AffineNoiseEstimator(_CLOUD_SCALE, _CLOUD_OFFSET))

    with pytest.raises(ValueError, match="diagonal in the Z basis"):
        execute_with_cdr(
            target,
            executor,
            PauliString.from_str("X"),
            ideal_estimator=exact_estimator,
            seed=0,
        )
