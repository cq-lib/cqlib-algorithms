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

r"""Clifford data regression.

Clifford data regression estimates the ideal expectation value of a circuit
from a set of *training circuits* whose exact values are cheap to compute.
Each training circuit is a near-Clifford variant of the target: a fraction of
its single-qubit rotations are snapped to a multiple of :math:`\pi/2`, which
turns them into Clifford gates, so the variant can still be evaluated either
exactly or on the noisy device. Every variant is measured twice, giving a
noisy value :math:`x_i` and an exact value :math:`y_i`, and a straight line
:math:`y = m x + b` is fitted to the training cloud. Applying that line to the
noisy value :math:`z` of the target circuit yields the mitigated estimate
:math:`m z + b`.

A purely Clifford input is mapped to itself, so the method leaves such a
circuit untouched. The observable must be diagonal in the :math:`Z` basis,
the same restriction the sampling techniques impose.

Three pieces make up the module: :func:`near_clifford_circuit` builds one
training circuit, :func:`fit_linear_regression` fits the correction from a
cloud of training points, and :func:`execute_with_cdr` /
:func:`cdr_executor` orchestrate the full method. The first two are
independent of any executor and can be exercised on their own.

References
----------
[1] P. Czarnik, A. Arrasmith, P. J. Coles, and L. Cincio, "Error mitigation with
    Clifford quantum-circuit data," Quantum 5, 592, 2021. arXiv:2005.10189.
[2] A. Lowe, M. Medvidovic, R. Hayes, J. E. Rice, M. T. Rosenkrantz, T. J. Yoder, and H.
    Eguchi, "Unified approach to data-driven quantum error mitigation," Physical Review
    Research 3, 033098, 2021. arXiv:2011.01157.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real

import numpy as np
from cqlib import Circuit
from cqlib.circuit import StandardGate, ValueOperation
from cqlib.qis import Hamiltonian, PauliString

from ...protocols import Estimator, EstimatorRequest, EstimatorResult, spawn_seeds
from ...protocols.state_preparation import StatePreparation
from .._circuits import bind_request_circuit, build_estimator_request, expand
from ..core import QemExecutor
from .rem import _validate_z_diagonal

# The single-qubit rotations whose angle is snapped; other gates have no angle
# to round and are copied through.
_ROTATION_GATES: tuple[StandardGate, ...] = (
    StandardGate.RX,
    StandardGate.RY,
    StandardGate.RZ,
)

# A half-turn of the Bloch sphere: rotating by a multiple of this angle is a
# Clifford operation, so the snapped grid is ``k * _HALF_PI``.
_HALF_PI = math.pi / 2.0

# Relative slack within which a rotation is treated as already sitting on the
# Clifford grid. It absorbs the last-bit difference between an angle a caller
# wrote as ``k * math.pi / 2`` and the same product recomputed here, so a
# Clifford rotation is never rebuilt into a nearly identical gate.
_CLIFFORD_TOLERANCE = 1.0e-12


def near_clifford_circuit(
    circuit: Circuit,
    fraction_non_clifford: float,
    rng: np.random.Generator,
    /,
) -> Circuit:
    r"""Snap a random fraction of ``circuit``'s rotations onto the Clifford grid.

    Every single-qubit rotation :math:`R_\theta` of ``circuit`` is, with
    probability ``fraction_non_clifford``, replaced by the rotation whose angle
    is the nearest multiple of :math:`\pi/2`,

    .. math::

        \theta \mapsto \frac{\pi}{2}\,
        \operatorname{round}\!\left(\frac{2\theta}{\pi}\right),

    which is a Clifford operation; otherwise the rotation is kept. A rotation
    already on that grid is left bit-identical even when selected, and every
    non-rotation gate is copied through, so a purely Clifford input is
    returned as itself, with the same operation order and global phase.

    Args:
        circuit: Circuit whose rotations are snapped.
        fraction_non_clifford: Probability, in ``[0, 1]``, that any one rotation is snapped.
        rng: Random stream the per-rotation decisions are drawn from.

    Returns:
        A near-Clifford copy of ``circuit`` carrying its qubits, classical
            state, and global phase.

    Raises:
        TypeError: If ``circuit`` is not a :class:`cqlib.Circuit`, if
            ``fraction_non_clifford`` is not a real number, or if ``rng`` is not a
            :class:`numpy.random.Generator`.
        ValueError: If ``fraction_non_clifford`` lies outside ``[0, 1]``.

    """
    if not isinstance(circuit, Circuit):
        raise TypeError("circuit must be a cqlib.Circuit")
    _validate_fraction_non_clifford(fraction_non_clifford)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    fraction = float(fraction_non_clifford)
    # expand rebuilds through the shared primitive, which restores the global
    # phase that Circuit.from_operations would otherwise drop.
    return expand(
        circuit,
        lambda operation: (_near_clifford_operation(operation, fraction, rng),),
    )


def fit_linear_regression(
    noisy: Sequence[float],
    ideal: Sequence[float],
    /,
    *,
    intercept: bool = True,
) -> tuple[float, float]:
    r"""Fit the correction line mapping noisy values onto exact ones.

    The model is the straight line

    .. math::

        y = m x + b \quad (\text{intercept}) \qquad \text{or} \qquad y = m x
        \quad (\text{no intercept}),

    fitted by ordinary least squares, whose closed form is

    .. math::

        m = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}
                 {\sum_i (x_i - \bar{x})^2}, \qquad
        b = \bar{y} - m \bar{x},

    with the two means dropped when ``intercept`` is ``False``.

    Args:
        noisy: The independent values :math:`x_i`, nominally the expectation values
            measured on the noisy backend.
        ideal: The dependent values :math:`y_i`, nominally the exact expectation
            values. It must have the same length as ``noisy``.
        intercept: ``True`` to fit the offset :math:`b`, ``False`` to force
            :math:`b = 0`.

    Returns:
        The fitted slope :math:`m` and intercept :math:`b`; ``b`` is ``0.0``
            when ``intercept`` is ``False``.

    Raises:
        TypeError: If ``intercept`` is not a bool, or if ``noisy`` or ``ideal`` is not a
            sequence of real numbers.
        ValueError: If the two inputs have different lengths, if they are non-finite, if
            there are fewer points than the fit has parameters (two for the
            intercept fit, one otherwise), or if the noisy values leave the slope
            undetermined.
    """
    if not isinstance(intercept, bool):
        raise TypeError("intercept must be a bool")
    x = _as_float_array(noisy, "noisy")
    y = _as_float_array(ideal, "ideal")
    if x.size != y.size:
        raise ValueError(
            f"noisy and ideal must have the same length, but got {x.size} and {y.size}"
        )

    # A fit consumes one training point per parameter it estimates, so the
    # intercept fit needs at least two points and the no-intercept fit one.
    required = 2 if intercept else 1
    if x.size < required:
        raise ValueError(
            f"intercept={intercept} requires at least {required} training "
            f"point(s), but {x.size} were supplied"
        )

    if intercept:
        x_mean = float(x.mean())
        y_mean = float(y.mean())
        centered = x - x_mean
        sxx = float(centered @ centered)
        if sxx == 0.0:
            raise ValueError(
                "the noisy values are all equal, so the slope of the fit is "
                "undetermined"
            )
        slope = float(centered @ (y - y_mean) / sxx)
        return slope, y_mean - slope * x_mean

    sxx = float(x @ x)
    if sxx == 0.0:
        raise ValueError(
            "the noisy values are all zero, so the slope of the "
            "no-intercept fit is undetermined"
        )
    return float(x @ y / sxx), 0.0


def execute_with_cdr(
    circuit: StatePreparation | Circuit,
    executor: QemExecutor,
    observable: Hamiltonian | PauliString,
    /,
    *,
    ideal_estimator: Estimator,
    num_training_circuits: int = 10,
    fraction_non_clifford: float = 0.1,
    shots: int | None = None,
    seed: int | None = None,
    parameter_bindings: Mapping[str, float] | None = None,
    prefix: str = "qem",
) -> float:
    r"""Estimate one observable on ``circuit`` with Clifford data regression.

    This is the immediate form of :func:`cdr_executor`. The same task
    can be evaluated by calling ``run()`` on that executor.

    The numeric work is not split into a circuit-construction and a
    result-combination piece, because the training set depends on the exact
    values a noiseless backend supplies for it; both halves are therefore run
    inside the returned executor.

    Args:
        circuit: State-preparation component or circuit to evaluate.
        executor: Execution backend supplying the noisy expectation surface.
        observable: Observable to estimate. It must be diagonal in the :math:`Z` basis,
            that is, a sum of :math:`Z` and identity products.
        ideal_estimator: Noiseless backend used to evaluate the exact training values
            :math:`y_i`. It is driven with shot-free requests.
        num_training_circuits: Number of near-Clifford training circuits to build and measure.
        fraction_non_clifford: Probability, in ``[0, 1]``, that any one rotation of a training
            circuit is snapped onto the Clifford grid.
        shots: Shot count for every measurement on ``executor``, or ``None`` for
            exact evaluation.
        seed: Seed of the near-Clifford draws, or ``None`` to draw fresh randomness.
            It is threaded into a :class:`numpy.random.Generator`.
        parameter_bindings: Finite values covering exactly all active symbols.
            Values are bound before circuit transformations; None requires no symbols.
        prefix: Build prefix for a preparation component, defaulting to ``qem``;
            existing Circuit symbols keep their names.

    Returns:
        Mitigated expectation value.

    Raises:
        TypeError: If an input or configuration argument has the wrong type.
        ValueError: If bindings are non-finite or do not cover active symbols exactly,
            or if ``num_training_circuits`` is below two, if
            ``fraction_non_clifford`` lies outside ``[0, 1]``, or if
            ``observable`` is not diagonal in the :math:`Z` basis.
    """
    wrapped = cdr_executor(
        executor,
        ideal_estimator=ideal_estimator,
        num_training_circuits=num_training_circuits,
        fraction_non_clifford=fraction_non_clifford,
        shots=shots,
        seed=seed,
    )
    request = build_estimator_request(
        circuit,
        observable,
        shots=shots,
        parameter_bindings=parameter_bindings,
        prefix=prefix,
    )
    return wrapped.estimate([request])[0].expectation


def cdr_executor(
    executor: QemExecutor,
    /,
    *,
    ideal_estimator: Estimator,
    num_training_circuits: int = 10,
    fraction_non_clifford: float = 0.1,
    shots: int | None = None,
    seed: int | None = None,
) -> QemExecutor:
    r"""Return an executor with Clifford data regression applied.

    The returned executor keeps the wrapped executor's sampling surface and
    caching flag, so the technique can be chained on top of another one without
    changing either semantics.

    Args:
        executor: Execution backend to wrap.
        ideal_estimator: Noiseless backend used to evaluate the exact training values.
        num_training_circuits: Number of near-Clifford training circuits to build and measure.
        fraction_non_clifford: Probability, in ``[0, 1]``, that any one rotation of a training
            circuit is snapped onto the Clifford grid.
        shots: Shot count bound to every request, or ``None`` to defer to the
            request's own shot count.
        seed: Seed of the near-Clifford draws, or ``None`` for a fresh draw.

    Returns:
        Executor whose expectation surface applies Clifford data regression.

    Raises:
        TypeError: If ``ideal_estimator`` is not an
            :class:`~cqlib_algorithms.protocols.Estimator`, if
            ``num_training_circuits`` is not an integer, if
            ``fraction_non_clifford`` is not a real number, or if ``seed`` is
            neither an integer nor ``None``.
        ValueError: If ``num_training_circuits`` is below two or if
            ``fraction_non_clifford`` lies outside ``[0, 1]``.

    """
    estimator = _CdrEstimator(
        executor,
        ideal_estimator=ideal_estimator,
        num_training_circuits=num_training_circuits,
        fraction_non_clifford=fraction_non_clifford,
        shots=shots,
        seed=seed,
    )
    return QemExecutor(
        estimator=estimator,
        sampler=executor.sampler,
        use_cache=executor.use_cache,
    )


class _CdrEstimator(Estimator):
    r"""Expectation surface that trains on near-Clifford data and regresses.

    Args:
        executor: Execution backend supplying the noisy expectation surface.
        ideal_estimator: Noiseless backend used to evaluate the exact training values.
        num_training_circuits: Number of near-Clifford training circuits to build and measure.
        fraction_non_clifford: Probability, in ``[0, 1]``, that any one rotation of a training
            circuit is snapped onto the Clifford grid.
        shots: Shot count bound to every request, or ``None`` to defer to the
            request.
        seed: Seed of the near-Clifford draws, or ``None`` for a fresh draw.

    """

    __slots__ = (
        "_executor",
        "_fraction_non_clifford",
        "_ideal_estimator",
        "_num_training_circuits",
        "_seed",
        "_shots",
    )

    _executor: QemExecutor
    _fraction_non_clifford: float
    _ideal_estimator: Estimator
    _num_training_circuits: int
    _seed: int | None
    _shots: int | None

    def __init__(
        self,
        executor: QemExecutor,
        /,
        *,
        ideal_estimator: Estimator,
        num_training_circuits: int,
        fraction_non_clifford: float,
        shots: int | None,
        seed: int | None,
    ) -> None:
        if not isinstance(ideal_estimator, Estimator):
            raise TypeError("ideal_estimator must be an Estimator")
        _validate_num_training_circuits(num_training_circuits)
        _validate_fraction_non_clifford(fraction_non_clifford)
        _validate_seed(seed)
        self._executor = executor
        self._ideal_estimator = ideal_estimator
        self._num_training_circuits = num_training_circuits
        self._fraction_non_clifford = float(fraction_non_clifford)
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
        r"""Train on near-Clifford data, fit the line, and correct the target."""
        bound_circuit = bind_request_circuit(request)
        observable = request.observable
        _validate_z_diagonal(observable)
        shots = self._shots if self._shots is not None else request.shots

        # A fresh generator per request keeps every request reproducible from
        # the single seed and stops one request's draws from shifting the next.
        seed = self._seed if self._seed is not None else request.seed
        rng = np.random.default_rng(seed)
        training = [
            near_clifford_circuit(
                bound_circuit,
                self._fraction_non_clifford,
                rng,
            )
            for _ in range(self._num_training_circuits)
        ]

        # Every noisy measurement is independent, so it takes its own child
        # seed while one run seed still reproduces the whole fit.
        seeds = spawn_seeds(seed, len(training) + 1)
        noisy = self._executor.estimate(
            [
                EstimatorRequest(
                    circuit=circuit,
                    observable=observable,
                    shots=shots,
                    seed=seeds[index],
                )
                for index, circuit in enumerate(training)
            ]
        )
        # The training values are exact, so the ideal backend is driven without
        # a shot count regardless of how the noisy surface was configured.
        ideal = self._ideal_estimator.run(
            [
                EstimatorRequest(circuit=circuit, observable=observable)
                for circuit in training
            ]
        )
        slope, intercept = fit_linear_regression(
            [result.expectation for result in noisy],
            [result.expectation for result in ideal],
        )

        target = self._executor.estimate(
            [
                EstimatorRequest(
                    circuit=bound_circuit,
                    observable=observable,
                    shots=shots,
                    seed=seeds[-1],
                )
            ]
        )[0]
        return EstimatorResult(expectation=slope * target.expectation + intercept)


def _near_clifford_operation(
    operation: ValueOperation,
    fraction_non_clifford: float,
    rng: np.random.Generator,
    /,
) -> ValueOperation:
    r"""Return the near-Clifford image of one operation.

    Args:
        operation: Operation to snap. Gates outside the single-qubit rotations are
            returned unchanged.
        fraction_non_clifford: Probability that a rotation is snapped.
        rng: Random stream the decision is drawn from.

    Returns:
        The original operation, or the rotation carrying the snapped angle.

    """
    if not operation.is_standard:
        return operation
    gate = operation.instruction.standard_gate
    if gate is None:
        return operation
    base = _rotation_base(gate)
    # Only a single-qubit rotation has one angle to round; everything else,
    # including multi-qubit rotations and non-parametric gates, is copied.
    if base is None or operation.num_params != 1:
        return operation

    value = operation.params[0]
    # A symbolic angle has no numeric multiple of pi/2 to snap onto.
    if isinstance(value, bool) or not isinstance(value, Real):
        return operation
    if rng.random() >= fraction_non_clifford:
        return operation

    angle = float(value)
    snapped = _HALF_PI * round(angle / _HALF_PI)
    if abs(angle - snapped) <= _CLIFFORD_TOLERANCE * max(1.0, abs(angle)):
        # The rotation already sits on the Clifford grid, so snapping is a
        # no-op; returning the original keeps a Clifford circuit bit-identical.
        return operation
    return ValueOperation.from_standard_gate(base(snapped), list(operation.qubits))


def _rotation_base(gate: StandardGate, /) -> StandardGate | None:
    r"""Return the rotation definition ``gate`` matches, or ``None``.

    Args:
        gate: Standard-gate definition read off an operation.

    Returns:
        :data:`StandardGate.RX`, :data:`StandardGate.RY`, or
            :data:`StandardGate.RZ` when ``gate`` is that rotation, else ``None``.

    """
    for base in _ROTATION_GATES:
        if gate == base:
            return base
    return None


def _as_float_array(values: Sequence[float], name: str, /) -> np.ndarray:
    r"""Return ``values`` as a one-dimensional float array, checking its type.

    Args:
        values: Candidate sequence of real numbers.
        name: Parameter name used in the error messages.

    Returns:
        The values as a float array, in input order.

    Raises:
        TypeError: If ``values`` is not a sequence of real numbers.
        ValueError: If any value is non-finite.

    """
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of real numbers")
    numbers: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must contain only real numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain only finite values")
        numbers.append(number)
    return np.asarray(numbers, dtype=float)


def _validate_num_training_circuits(num_training_circuits: int, /) -> None:
    r"""Validate the training-circuit count."""
    if isinstance(num_training_circuits, bool) or not isinstance(
        num_training_circuits, int
    ):
        raise TypeError("num_training_circuits must be an integer")
    if num_training_circuits < 2:
        raise ValueError(
            "num_training_circuits must be at least two, because the "
            "intercept fit needs at least two training points, but was "
            f"{num_training_circuits!r}."
        )


def _validate_fraction_non_clifford(fraction_non_clifford: float, /) -> None:
    r"""Validate a near-Clifford snap probability."""
    if isinstance(fraction_non_clifford, bool) or not isinstance(
        fraction_non_clifford, Real
    ):
        raise TypeError("fraction_non_clifford must be a real number")
    if not 0.0 <= float(fraction_non_clifford) <= 1.0:
        raise ValueError(
            "fraction_non_clifford must lie in [0, 1], but was "
            f"{float(fraction_non_clifford)!r}."
        )


def _validate_seed(seed: int | None, /) -> None:
    r"""Validate an optional random seed."""
    if seed is None:
        return
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer or None")
    if seed < 0:
        raise ValueError("seed must be non-negative")
