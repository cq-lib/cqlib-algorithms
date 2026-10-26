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

r"""Classical shadows with robust readout-error calibration.

Classical shadows estimate many observables from one fixed family of
measurements. Every shot draws, independently on each qubit, one of the three
single-qubit Clifford axes :math:`\{X, Y, Z\}`; the circuit is rotated onto
that axis and read in the computational basis. The collected snapshots
reconstruct the state as

.. math::

    \hat{\rho} = \frac{1}{N} \sum_{s=1}^{N}
        \bigotimes_{i=1}^{n} \bigl(3\,|b_i^s\rangle\langle b_i^s| - I\bigr),

so that for a Pauli string :math:`O` supported on
:math:`\operatorname{supp}(O)` the snapshot estimator averages

.. math::

    \hat{\langle O \rangle} = \frac{1}{N} \sum_{s=1}^{N}
        \prod_{i \in \operatorname{supp}(O)}
        3\,\langle b_i^s | O_i | b_i^s \rangle .

A single-qubit factor is :math:`3 (+1)` or :math:`3 (-1)` when the drawn axis
matches the Pauli factor axis and the read-out bit is :math:`0` or :math:`1`,
and :math:`0` when the two axes differ. The identity factor is :math:`1`, so
the reconstruction is trace-preserving, :math:`\operatorname{Tr}\hat{\rho} = 1`,
for every shot set.

When per-qubit read-out fidelities :math:`\lambda_i \in (0, 1]` are known, the
robust-shadow-estimation calibration divides each matching factor by
:math:`\lambda_i`, giving :math:`3 / \lambda_i`. The value :math:`\lambda_i = 1`
reproduces the standard shadow estimator exactly.

:func:`snapshot_expectation` applies the estimator to recorded outcomes and
axes, :func:`execute_with_shadows` samples the rotated circuits and returns the
estimate, and :func:`shadows_executor` exposes the same computation as a
reusable executor for chaining.

References
----------
[1] H.-Y. Huang, R. Kueng, and J. Preskill, "Predicting many properties of a quantum
    system from very few measurements," Nature Physics 16, 1050-1057, 2020.
    arXiv:2002.08953.
[2] S. Chen, W. Yu, P. Zeng, and S. T. Flammia, "Robust shadow estimation," Quantum 5,
    509, 2021. arXiv:2011.09636.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from cqlib import Circuit
from cqlib.circuit import StandardGate, ValueOperation
from cqlib.qis import Hamiltonian, PauliString

from ...protocols import (
    Estimator,
    EstimatorRequest,
    EstimatorResult,
    SamplerRequest,
    spawn_seeds,
)
from .._circuits import rebuild
from ..core import QemExecutor

# Measurement axes in the order the uniform draw indexes them.
_AXES = ("X", "Y", "Z")

# Symplectic bits (x_bit, z_bit) of the non-identity single-qubit Paulis.
_AXIS_BY_BITS = {(1, 0): "X", (1, 1): "Y", (0, 1): "Z"}

# Single-qubit factor applied to a matching factor. It inverts the 1/3 chance
# that a uniformly drawn axis matches the Pauli factor, making the snapshot an
# unbiased estimator of the term.
_SHADOW_SCALE = 3.0


def snapshot_expectation(
    outcomes: Sequence[Sequence[int]],
    bases: Sequence[Sequence[str]],
    observable: PauliString,
    /,
    *,
    readout_fidelities: Sequence[float] | None = None,
) -> float:
    r"""Estimate one Pauli string from recorded shadow outcomes.

    The estimate averages the per-shot snapshot product

    .. math::

        \hat{\langle O \rangle} = \frac{1}{N} \sum_{s=1}^{N}
            \prod_{i \in \operatorname{supp}(O)}
            \frac{3}{\lambda_i}\,\langle b_i^s | O_i | b_i^s \rangle,

    where each single-qubit factor is
    :math:`\frac{3}{\lambda_i}(+1)` or :math:`\frac{3}{\lambda_i}(-1)` when the
    shot measured qubit :math:`i` on the matching axis and read :math:`0` or
    :math:`1`, and :math:`0` when the axes differ. The identity factor is
    :math:`1`, so an unsupported observable (the identity) yields exactly
    :math:`1`, the trace of the reconstructed state.

    Args:
        outcomes: Per-shot read-out bits; entry ``s`` is one bit per qubit with index
            ``i`` holding qubit ``i``. Bits must be ``0`` or ``1``.
        bases: Per-shot measurement axes aligned with ``outcomes``; entry ``s`` holds
            one label per qubit drawn from ``"X"``, ``"Y"``, or ``"Z"``.
        observable: Pauli string whose expectation is estimated. Its global phase is
            included in the result.
        readout_fidelities: Per-qubit read-out fidelities :math:`\lambda_i \in (0, 1]`, or ``None``
            for the standard estimator. Fidelities are plain floats; no noise
            model is involved. With every :math:`\lambda_i = 1` the result equals
            the standard shadow estimator exactly.

    Returns:
        The snapshot estimate of the observable's expectation value.

    Raises:
        TypeError: If ``observable`` is not a :class:`cqlib.qis.PauliString`, if
            ``outcomes`` or ``bases`` is not a sequence of per-shot sequences, if a
            bit is not an integer, or if a fidelity is not a real number.
        ValueError: If ``outcomes`` and ``bases`` differ in shot count or qubit width, if
            no shot was recorded, if a bit is not ``0`` or ``1``, if an axis label
            is not ``"X"``, ``"Y"``, or ``"Z"``, or if a fidelity is not finite
            or non-positive.

    """
    if not isinstance(observable, PauliString):
        raise TypeError("observable must be a cqlib.qis.PauliString")
    _validate_shadow_records(outcomes, bases, observable.num_qubits)
    _validate_readout_fidelities(readout_fidelities, observable.num_qubits)

    value = _snapshot_average(outcomes, bases, observable, readout_fidelities)
    phase = complex(observable.phase.to_complex())
    return float((phase * value).real)


def execute_with_shadows(
    circuit: Circuit,
    executor: QemExecutor,
    observable: Hamiltonian | PauliString,
    /,
    *,
    num_total_measurements_shadow: int,
    seed: int | None = None,
) -> float:
    r"""Estimate one observable on ``circuit`` with classical shadows.

    This is the immediate form of :func:`shadows_executor`::

        execute_with_shadows(c, ex, obs, num_total_measurements_shadow=n)
        == shadows_executor(ex, num_total_measurements_shadow=n).estimate(
               [EstimatorRequest(c, obs)]
           )[0].expectation

    A single-qubit :math:`Z` observable measured through the random-basis
    protocol converges to the same value as the direct
    :math:`Z`-basis mean of the shots.

    Args:
        circuit: Circuit whose expectation value is estimated.
        executor: Execution backend supplying the sampling surface.
        observable: Observable to estimate, any Hamiltonian or Pauli string. A bare Pauli
            string is wrapped to share the Hamiltonian code path, and every term is
            estimated from the same shadow set.
        num_total_measurements_shadow: Number of independent shadows :math:`N`, at least one.
        seed: Seed of the random basis draw and the shadow measurements, or ``None``
            for fresh randomness.

    Returns:
        The shadow estimate of the observable's expectation value.

    Raises:
        TypeError: If ``num_total_measurements_shadow`` is not an integer or ``seed`` is
            neither an integer nor ``None``.
        ValueError: If ``num_total_measurements_shadow`` is below one.

    """
    wrapped = shadows_executor(
        executor,
        num_total_measurements_shadow=num_total_measurements_shadow,
        seed=seed,
    )
    request = EstimatorRequest(
        circuit=circuit,
        observable=observable,
        shots=num_total_measurements_shadow,
    )
    return wrapped.estimate([request])[0].expectation


def shadows_executor(
    executor: QemExecutor,
    /,
    *,
    num_total_measurements_shadow: int,
    seed: int | None = None,
) -> QemExecutor:
    r"""Return an executor with classical shadows applied.

    The returned executor keeps the wrapped executor's sampling surface and
    caching flag, so shadows can be chained on top of another technique. The
    random-basis draw and the snapshot average replace the expectation surface
    while leaving sampling untouched.

    Args:
        executor: Execution backend to wrap.
        num_total_measurements_shadow: Number of independent shadows
            :math:`N` bound to every request, at least one.
        seed: Seed of the random basis draw and the shadow measurements, or ``None``
            to defer to the request's own seed.

    Returns:
        Executor whose expectation surface applies classical shadows.

    Raises:
        TypeError: If ``num_total_measurements_shadow`` is not an integer or ``seed`` is
            neither an integer nor ``None``.
        ValueError: If ``num_total_measurements_shadow`` is below one.

    """
    _validate_shadow_count(num_total_measurements_shadow)
    _validate_seed(seed)
    estimator = _ShadowsEstimator(
        executor,
        num_shots=num_total_measurements_shadow,
        seed=seed,
    )
    return QemExecutor(
        estimator=estimator,
        sampler=executor.sampler,
        use_cache=executor.use_cache,
    )


class _ShadowsEstimator(Estimator):
    r"""Expectation surface that draws random bases and averages snapshots.

    Args:
        executor: Execution backend supplying the sampling surface.
        num_shots: Number of independent shadows drawn for every request.
        seed: Seed of the random basis draw and the shadow measurements, or ``None``
            to defer to the request's own seed.

    """

    __slots__ = ("_executor", "_num_shots", "_seed")

    _executor: QemExecutor
    _num_shots: int
    _seed: int | None

    def __init__(
        self,
        executor: QemExecutor,
        /,
        *,
        num_shots: int,
        seed: int | None,
    ) -> None:
        self._executor = executor
        self._num_shots = num_shots
        self._seed = seed

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        r"""Estimate every request and return results in input order."""
        return [self._mitigate(request) for request in requests]

    def _mitigate(self, request: EstimatorRequest) -> EstimatorResult:
        r"""Draw the shadow set and average every term's snapshot estimate."""
        observable = _as_hamiltonian(request.observable)
        num_qubits = observable.num_qubits

        seed = self._seed if self._seed is not None else request.seed
        rng = np.random.default_rng(seed)
        assignments = rng.integers(0, len(_AXES), size=(self._num_shots, num_qubits))
        outcomes, bases = self._collect_shadows(request.circuit, assignments, seed)

        # Every term reads the same shadow set; only the coefficient and the
        # term's global phase differ, and their product is real for the
        # Hermitian observable the request validated.
        total = 0.0
        for pauli, coefficient in observable.terms:
            value = _snapshot_average(outcomes, bases, pauli, None)
            effective = complex(coefficient) * complex(pauli.phase.to_complex())
            total += (effective * value).real
        return EstimatorResult(expectation=float(total))

    def _collect_shadows(
        self,
        circuit: Circuit,
        assignments: np.ndarray,
        seed: int | None,
    ) -> tuple[list[list[int]], list[list[str]]]:
        r"""Sample the rotated circuits and expand counts into per-shot records.

        Args:
            circuit: State-preparation circuit shared by every shadow.
            assignments: Integer axis codes of shape ``(num_shots, num_qubits)``.
            seed: Seed of the sampled shadows, or ``None`` for fresh randomness.

        Returns:
            Per-shot read-out bits and matching per-shot basis labels, aligned
                in shot order.

        """
        num_qubits = assignments.shape[1]

        # Shots that share an axis assignment are one sampling request, so the
        # number of circuit executions is bounded by the distinct assignments.
        groups: dict[tuple[int, ...], int] = {}
        for row in assignments:
            key = tuple(int(axis) for axis in row)
            groups[key] = groups.get(key, 0) + 1

        keys = list(groups)
        # Each distinct axis assignment is sampled independently, so it takes
        # its own child seed while one run seed still reproduces the shadow set.
        seeds = spawn_seeds(seed, len(keys))
        requests = []
        for index, key in enumerate(keys):
            rotated = _rotate_into_basis(circuit, key)
            requests.append(
                SamplerRequest(
                    circuit=rotated,
                    qubits=tuple(rotated.qubits),
                    shots=groups[key],
                    seed=seeds[index],
                )
            )
        results = self._executor.sample(requests)

        outcomes: list[list[int]] = []
        bases: list[list[str]] = []
        for key, result in zip(keys, results):
            axes = [_AXES[axis] for axis in key]
            for bitstring, count in result.counts.items():
                # Count keys are little-endian, so qubit i sits at position
                # num_qubits - 1 - i.
                bits = [
                    int(bitstring[num_qubits - 1 - index])
                    for index in range(num_qubits)
                ]
                for _ in range(count):
                    outcomes.append(bits)
                    bases.append(axes)
        return outcomes, bases


def _as_hamiltonian(observable: Hamiltonian | PauliString, /) -> Hamiltonian:
    r"""Wrap a bare Pauli string so every observable shares one code path."""
    if isinstance(observable, PauliString):
        return Hamiltonian.from_pauli(observable)
    return observable


def _rotate_into_basis(
    circuit: Circuit,
    axes: Sequence[int],
    /,
) -> Circuit:
    r"""Return ``circuit`` extended by the rotations of one measurement basis.

    Args:
        circuit: State-preparation circuit.
        axes: One integer axis code per qubit: ``0`` for :math:`X`, ``1`` for
            :math:`Y`, and ``2`` for :math:`Z`.

    Returns:
        A rebuilt circuit that keeps the source global phase with the axis
            rotations appended.

    """
    rotations: list[ValueOperation] = []
    for qubit, axis in zip(circuit.qubits, axes):
        label = _AXES[axis]
        if label == "X":
            rotations.append(ValueOperation.from_standard_gate(StandardGate.H, [qubit]))
        elif label == "Y":
            # Sdg followed by H maps the Y eigenbasis onto the Z basis.
            rotations.append(
                ValueOperation.from_standard_gate(StandardGate.SDG, [qubit])
            )
            rotations.append(ValueOperation.from_standard_gate(StandardGate.H, [qubit]))
    return rebuild(circuit, [*circuit.operations, *rotations])


def _snapshot_average(
    outcomes: Sequence[Sequence[int]],
    bases: Sequence[Sequence[str]],
    observable: PauliString,
    readout_fidelities: Sequence[float] | None,
) -> float:
    r"""Return the phase-free shot-averaged snapshot estimator.
    """
    targets: list[tuple[int, str]] = []
    scale = 1.0
    for index in observable.support():
        x_bit = (observable.x_mask >> index) & 1
        z_bit = (observable.z_mask >> index) & 1
        targets.append((index, _AXIS_BY_BITS[(x_bit, z_bit)]))
        if readout_fidelities is None:
            scale *= _SHADOW_SCALE
        else:
            scale *= _SHADOW_SCALE / float(readout_fidelities[index])

    total = 0.0
    for shot_bits, shot_bases in zip(outcomes, bases):
        product = 1.0
        for index, axis in targets:
            # A shot contributes only when every support axis was measured on
            # the matching axis; otherwise its snapshot product is zero.
            if shot_bases[index] != axis:
                product = 0.0
                break
            if shot_bits[index] == 1:
                product = -product
        total += product
    return scale * total / len(outcomes)


def _validate_shadow_records(
    outcomes: Sequence[Sequence[int]],
    bases: Sequence[Sequence[str]],
    num_qubits: int,
) -> None:
    r"""Validate classical-shadow shot records."""
    if isinstance(outcomes, (str, bytes)) or not isinstance(outcomes, Sequence):
        raise TypeError("outcomes must be a sequence of per-shot bit sequences")
    if isinstance(bases, (str, bytes)) or not isinstance(bases, Sequence):
        raise TypeError("bases must be a sequence of per-shot basis sequences")
    if len(outcomes) != len(bases):
        raise ValueError("outcomes and bases must hold the same number of shots")
    if len(outcomes) == 0:
        raise ValueError("at least one shot is required")

    for bits in outcomes:
        if isinstance(bits, (str, bytes)) or not isinstance(bits, Sequence):
            raise TypeError("every outcome shot must be a sequence of bits")
        if len(bits) != num_qubits:
            raise ValueError("every outcome shot must carry one bit per qubit")
        for bit in bits:
            if isinstance(bit, bool) or not isinstance(bit, int):
                raise TypeError("outcome bits must be integers")
            if bit not in (0, 1):
                raise ValueError("outcome bits must be 0 or 1")

    for shot_bases in bases:
        if isinstance(shot_bases, (str, bytes)) or not isinstance(shot_bases, Sequence):
            raise TypeError("every basis shot must be a sequence of axis labels")
        if len(shot_bases) != num_qubits:
            raise ValueError("every basis shot must carry one axis per qubit")
        if any(axis not in _AXES for axis in shot_bases):
            raise ValueError('basis axes must be "X", "Y", or "Z"')


def _validate_readout_fidelities(
    readout_fidelities: Sequence[float] | None,
    num_qubits: int,
) -> None:
    r"""Validate readout-fidelity values."""
    if readout_fidelities is None:
        return
    if isinstance(readout_fidelities, (str, bytes)) or not isinstance(
        readout_fidelities, (Sequence, np.ndarray)
    ):
        raise TypeError("readout_fidelities must be a sequence of floats or None")
    if len(readout_fidelities) != num_qubits:
        raise ValueError("readout_fidelities must hold one value per qubit")
    for value in readout_fidelities:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
            raise TypeError("readout_fidelities must contain real numbers")
        if not math.isfinite(value):
            raise ValueError("readout_fidelities must be finite")
        if value <= 0.0:
            raise ValueError("readout_fidelities must be positive")


def _validate_shadow_count(num_total_measurements_shadow: int) -> None:
    r"""Validate a positive classical-shadow count."""
    if isinstance(num_total_measurements_shadow, bool) or not isinstance(
        num_total_measurements_shadow, int
    ):
        raise TypeError("num_total_measurements_shadow must be an integer")
    if num_total_measurements_shadow < 1:
        raise ValueError("num_total_measurements_shadow must be at least 1")


def _validate_seed(seed: int | None) -> None:
    r"""Validate an optional random seed."""
    if seed is None:
        return
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer or None")
    if seed < 0:
        raise ValueError("seed must be non-negative")
