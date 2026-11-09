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

r"""Zero-noise extrapolation.

Zero-noise extrapolation runs one circuit at several artificially inflated
noise levels and extrapolates the measured expectation values back to the
noise-free limit. A circuit is made noisier without changing its logical
action by *folding*: replacing a gate block :math:`U` by
:math:`U (U^\dagger U)^n` multiplies the error accumulated on that block by
:math:`2n + 1` while leaving :math:`U` itself untouched.

Two routes are available:

* The local route builds the folded family from
  :mod:`cqlib_algorithms.error_mitigation.scaling` and fits it with
  :mod:`cqlib_algorithms.error_mitigation.inference`. It covers every
  scaling strategy and every factory defined in this library.
* The cqlib route delegates folding and extrapolation to
  ``cqlib.error_mitigation.ZNEMitigation`` and accepts a narrower
  configuration: scale factors that are odd integers at or above one, global
  folding, and a polynomial or zero-asymptote exponential model.

The two-piece API is :func:`construct_circuits` (fold one circuit per scale
factor) plus :func:`combine_results` (reduce the measured values to the
limit), orchestrated by :func:`execute_with_zne`; :func:`zne_executor`
returns the same computation as a reusable executor for chaining.

References
----------
[1] K. Temme, S. Bravyi, and J. M. Gambetta, "Error mitigation for short-depth quantum
    circuits," Physical Review Letters 119, 180509, 2017. arXiv:1612.02058.
[2] T. Giurgica-Tiron, Y. Hindy, R. LaRose, A. Mari, and W. J. Zeng, "Digital zero noise
    extrapolation for quantum error mitigation," IEEE International Conference on
    Quantum Computing and Engineering (QCE), pp. 306-316, 2020. arXiv:2005.10921.
[3] L. F. Richardson and J. A. Gaunt, "The deferred approach to the limit. Part I.
    Single lattice. Part II. Interpenetrating lattices," Philosophical Transactions of
    the Royal Society A 226, 299-361, 1927. doi:10.1098/rsta.1927.0008.
[4] A. Kandala, K. Temme, A. D. Corcoles, A. Mezzacapo, J. M. Chow, and J. M. Gambetta,
    "Error mitigation extends the computational reach of a noisy quantum processor,"
    Nature 567, 491-495, 2019. arXiv:1805.04492.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Sequence
from enum import Enum

import numpy as np

from cqlib import Circuit
from cqlib.circuit import Instruction
from cqlib.error_mitigation import ExtrapolateMethod, ZNEMitigation
from cqlib.qis import Hamiltonian, PauliString

from ...protocols import (
    Estimator,
    EstimatorRequest,
    EstimatorResult,
    spawn_seeds,
)
from .._circuits import strip_measurements
from ..core import QemExecutor, make_cqlib_estimator
from ..inference import (
    ConvergenceWarning,
    ExpFactory,
    Factory,
    PolyFactory,
    RichardsonFactory,
)
from ..scaling import fold_gates_at_random, fold_global, fold_local

# Random gate folding draws its selection from a shuffle. Callers who do not
# pin a seed still need a reproducible circuit family, so a fixed fallback is
# used instead of an entropy-derived default.
_DEFAULT_FOLD_SEED = 0

# Condition number of the extrapolation Vandermonde matrix above which the
# fitted coefficients amplify their inputs more than an order of magnitude per
# decade of point spacing. Reaching it usually means the scale factors are too
# close together or not evenly spread.
_ILL_CONDITIONED_LIMIT = 1.0e8


class ScalingMethod(str, Enum):
    r"""Noise-scaling strategy used to build the folded circuits.

    A plain :class:`str` enumeration keeps the public values compatible with
    ordinary strings.

    Attributes:
        GLOBAL: Fold the whole gate body uniformly.
        RANDOM: Fold a random subset of the gates; the selection is driven by
            ``seed``.
        LOCAL: Split the gate body into contiguous chunks and fold each one;
            requires ``num_chunks``.

    """

    GLOBAL = "global"
    RANDOM = "random"
    LOCAL = "local"


def construct_circuits(
    circuit: Circuit,
    /,
    *,
    scale_factors: Sequence[float],
    scaling: ScalingMethod = ScalingMethod.GLOBAL,
    seed: int | None = None,
    num_chunks: int | None = None,
) -> list[Circuit]:
    r"""Fold ``circuit`` once per scale factor.

    The returned family preserves the logical action of ``circuit`` and
    carries its global phase. Measurements are removed before folding and
    re-appended afterwards.

    Args:
        circuit: Circuit to fold.
        scale_factors: Noise scale factors :math:`\lambda \ge 1`, in the order the results
            will later be reduced.
        scaling: Scaling strategy selecting the folding primitive.
        seed: Seed for :attr:`ScalingMethod.RANDOM`. When ``None`` a fixed default
            seed is used so the family stays reproducible.
        num_chunks: Number of contiguous chunks for :attr:`ScalingMethod.LOCAL`.

    Returns:
        One folded circuit per scale factor, in input order.

    Raises:
        ValueError: If ``scaling`` is not a recognized strategy, if
            ``scaling=ScalingMethod.LOCAL`` and ``num_chunks`` is ``None``, or if
            a scale factor is below one.

    """
    method = ScalingMethod(scaling)
    if method is ScalingMethod.GLOBAL:
        return [fold_global(circuit, factor) for factor in scale_factors]
    if method is ScalingMethod.RANDOM:
        resolved_seed = _DEFAULT_FOLD_SEED if seed is None else seed
        return [
            fold_gates_at_random(circuit, factor, seed=resolved_seed)
            for factor in scale_factors
        ]

    # Only LOCAL remains, and chunk-wise folding is undefined without a chunk
    # count, so the configuration is rejected before any circuit is built.
    if num_chunks is None:
        raise ValueError(
            "scaling=ScalingMethod.LOCAL requires num_chunks, but num_chunks "
            "was None."
        )
    return [
        fold_local(circuit, factor, num_chunks=num_chunks)
        for factor in scale_factors
    ]


def combine_results(
    results: Sequence[EstimatorResult],
    /,
    *,
    factory: Factory,
) -> float:
    r"""Reduce per-scale expectation values to the zero-noise limit.

    Args:
        results: Expectation values measured on the circuits returned by
            :func:`construct_circuits`, in the same order.
        factory: Extrapolation factory. It is mutated: every result is pushed onto it
            before the limit is fitted.

    Returns:
        The extrapolated zero-noise expectation value.

    Raises:
        ValueError: If more results are supplied than the factory expects.
        ExtrapolationError: If the factory has not collected enough points, or if the fit is
            numerically impossible.
    """
    for result in results:
        # The factory owns the noise schedule, so the scale factor paired with
        # each result comes from the factory itself rather than from an
        # independently supplied grid.
        scale_factor = factory.next_scale_factor()
        if scale_factor is None:
            raise ValueError(
                "The factory is already converged and cannot accept the "
                f"{len(results)} supplied result(s)."
            )
        factory.push(scale_factor, result.expectation)
    return factory.reduce()


def execute_with_zne(
    circuit: Circuit,
    executor: QemExecutor,
    observable: Hamiltonian | PauliString,
    /,
    *,
    scale_factors: Sequence[float] = (1.0, 2.0, 3.0),
    factory: Factory | None = None,
    scaling: ScalingMethod = ScalingMethod.GLOBAL,
    gate_set: Sequence[Instruction] | None = None,
    shots: int | None = None,
    seed: int | None = None,
    num_chunks: int | None = None,
) -> float:
    r"""Estimate one observable on ``circuit`` with zero-noise extrapolation.

    This is the immediate form of :func:`zne_executor`::

        execute_with_zne(c, ex, obs, **params)
        == zne_executor(ex, **params).estimate(
               [EstimatorRequest(c, obs)]
           )[0].expectation

    Args:
        circuit: Circuit whose expectation value is to be mitigated.
        executor: Execution backend supplying the expectation surface.
        observable: Observable to estimate. A bare Pauli string is interpreted as a
            Hamiltonian with unit coefficient.
        scale_factors: Noise scale factors :math:`\lambda \ge 1`.
        factory: Extrapolation factory. When ``None`` and the local route is taken,
            :class:`~cqlib_algorithms.error_mitigation.inference.RichardsonFactory`
            is used on ``scale_factors``.
        scaling: Scaling strategy used to build the folded circuits.
        gate_set: Gate subset folded by cqlib's selective folding. A non-empty set
            forces the cqlib route and therefore also requires an odd-integer
            scale grid and a factory with a cqlib equivalent.
        shots: Shot count for every folded circuit, or ``None`` for exact
            evaluation.
        seed: Seed of the folding draw and the shot sampling, or ``None`` for fresh
            randomness. Pinning it reproduces the whole estimate.
        num_chunks: Number of contiguous chunks for :attr:`ScalingMethod.LOCAL`.

    Returns:
        Mitigated expectation value.

    Raises:
        TypeError: If ``seed`` is neither an integer nor ``None``.
        ValueError: If the parameters describe an unusable configuration, such as a
            non-empty ``gate_set`` combined with a scale grid or factory that
            cqlib folding cannot reproduce.
        ExtrapolationError: If the extrapolation fit cannot be performed.
        cqlib.error_mitigation.ErrorMitigationError: Propagated unchanged from the cqlib route.
    """
    wrapped = zne_executor(
        executor,
        scale_factors=scale_factors,
        factory=factory,
        scaling=scaling,
        gate_set=gate_set,
        shots=shots,
        seed=seed,
        num_chunks=num_chunks,
    )
    request = EstimatorRequest(circuit=circuit, observable=observable, shots=shots)
    return wrapped.estimate([request])[0].expectation


def zne_executor(
    executor: QemExecutor,
    /,
    *,
    scale_factors: Sequence[float] = (1.0, 2.0, 3.0),
    factory: Factory | None = None,
    scaling: ScalingMethod = ScalingMethod.GLOBAL,
    gate_set: Sequence[Instruction] | None = None,
    shots: int | None = None,
    seed: int | None = None,
    num_chunks: int | None = None,
) -> QemExecutor:
    r"""Return an executor with zero-noise extrapolation applied.

    The returned executor keeps the wrapped executor's sampling surface and
    caching flag, so ZNE can be chained on top of another technique without
    changing either semantics.

    Args:
        executor: Execution backend to wrap.
        scale_factors: Noise scale factors :math:`\lambda \ge 1`.
        factory: Extrapolation factory, or ``None`` to select the default.
        scaling: Scaling strategy used to build the folded circuits.
        gate_set: Gate subset folded by cqlib's selective folding.
        shots: Shot count bound to every request, or ``None`` to defer to the
            request's own shot count.
        seed: Seed of the folding draw and the shot sampling, or ``None`` to defer
            to the request's own seed.
        num_chunks: Number of contiguous chunks for :attr:`ScalingMethod.LOCAL`.

    Returns:
        Executor whose expectation surface applies zero-noise extrapolation.

    Raises:
        TypeError: If ``seed`` is neither an integer nor ``None``.
        ValueError: If ``scaling`` is not a recognized strategy.
    """
    _validate_seed(seed)
    estimator = _ZneEstimator(
        executor,
        scale_factors=tuple(float(factor) for factor in scale_factors),
        factory=factory,
        scaling=ScalingMethod(scaling),
        gate_set=None if gate_set is None else tuple(gate_set),
        shots=shots,
        seed=seed,
        num_chunks=num_chunks,
    )
    return QemExecutor(
        estimator=estimator,
        sampler=executor.sampler,
        use_cache=executor.use_cache,
    )


class _ZneEstimator(Estimator):
    r"""Expectation surface that folds, measures, and extrapolates.

    Args:
        executor: Execution backend supplying the expectation surface for each folded
            circuit.
        scale_factors: Normalized noise scale factors.
        factory: Extrapolation factory, or ``None`` for the default.
        scaling: Normalized scaling strategy.
        gate_set: Normalized folded-gate subset, or ``None``.
        shots: Shot count bound to every request, or ``None`` to defer to the
            request.
        seed: Seed of the folding draw and the shot sampling, or ``None`` to defer
            to the request's own seed.
        num_chunks: Chunk count for local folding.

    """

    __slots__ = (
        "_executor",
        "_factory",
        "_gate_set",
        "_num_chunks",
        "_scale_factors",
        "_scaling",
        "_seed",
        "_shots",
    )

    _executor: QemExecutor
    _factory: Factory | None
    _gate_set: tuple[Instruction, ...] | None
    _num_chunks: int | None
    _scale_factors: tuple[float, ...]
    _scaling: ScalingMethod
    _seed: int | None
    _shots: int | None

    def __init__(
        self,
        executor: QemExecutor,
        /,
        *,
        scale_factors: tuple[float, ...],
        factory: Factory | None,
        scaling: ScalingMethod,
        gate_set: tuple[Instruction, ...] | None,
        shots: int | None,
        seed: int | None,
        num_chunks: int | None,
    ) -> None:
        self._executor = executor
        self._scale_factors = scale_factors
        self._factory = factory
        self._scaling = scaling
        self._gate_set = gate_set
        self._shots = shots
        self._seed = seed
        self._num_chunks = num_chunks

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        r"""Mitigate every request and return results in input order."""
        return [self._mitigate(request) for request in requests]

    def _mitigate(self, request: EstimatorRequest) -> EstimatorResult:
        r"""Estimate one request through the route its parameters select."""
        _warn_if_ill_conditioned(self._scale_factors)
        shots = self._shots if self._shots is not None else request.shots
        seed = self._seed if self._seed is not None else request.seed
        plan = _cqlib_plan(
            scaling=self._scaling,
            scale_factors=self._scale_factors,
            factory=self._factory,
            gate_set=self._gate_set,
        )
        if plan is not None:
            method, degree, fold_levels = plan
            expectation = _run_cqlib_zne(
                request.circuit,
                self._executor,
                request.observable,
                fold_levels=fold_levels,
                gate_set=self._gate_set,
                shots=shots,
                method=method,
                degree=degree,
                seed=seed,
            )
            return EstimatorResult(expectation=expectation)

        circuits = construct_circuits(
            request.circuit,
            scale_factors=self._scale_factors,
            scaling=self._scaling,
            seed=seed,
            num_chunks=self._num_chunks,
        )
        seeds = spawn_seeds(seed, len(circuits))
        requests = [
            EstimatorRequest(
                circuit=circuit,
                observable=request.observable,
                shots=shots,
                seed=seeds[index],
            )
            for index, circuit in enumerate(circuits)
        ]
        results = self._executor.estimate(requests)
        return EstimatorResult(
            expectation=combine_results(results, factory=self._make_factory())
        )

    def _make_factory(self) -> Factory:
        r"""Return an extrapolation factory private to one request.

        A factory accumulates the values pushed onto it, so a copy is handed
        out per request; sharing one instance would let a batch mix the
        measurements of different requests.
        """
        if self._factory is None:
            return RichardsonFactory(self._scale_factors)
        return copy.deepcopy(self._factory)


def _cqlib_plan(
    *,
    scaling: ScalingMethod,
    scale_factors: Sequence[float],
    factory: Factory | None,
    gate_set: Sequence[Instruction] | None,
) -> tuple[ExtrapolateMethod, int, list[int]] | None:
    r"""Return the cqlib extrapolation plan, or ``None`` for the local route.

    The cqlib route needs an odd-integer scale grid folded globally *and* a
    factory cqlib can reproduce. A non-empty ``gate_set`` can only be honoured
    by cqlib folding, so it turns a rejected configuration into an error
    instead of a silent fallback.

    Returns:
        Fitted model, polynomial degree, and cqlib fold levels, or ``None``
            when the local route must be used.

    Raises:
        ValueError: If a non-empty ``gate_set`` was requested but the remaining
            parameters cannot be served by cqlib folding.

    """
    grid_is_foldable = scaling is ScalingMethod.GLOBAL and all(
        _is_odd_fold_level(factor) for factor in scale_factors
    )
    extrapolation = _cqlib_extrapolation(factory, scale_factors)
    gate_set_forces_cqlib = gate_set is not None and len(gate_set) > 0

    if not grid_is_foldable or extrapolation is None:
        if gate_set_forces_cqlib:
            raise ValueError(
                "A non-empty gate_set can only be folded by cqlib, which "
                "requires scaling=ScalingMethod.GLOBAL with odd-integer scale "
                "factors and a factory cqlib can reproduce: a polynomial "
                "factory, or an ExpFactory whose asymptote is exactly zero."
            )
        return None

    # cqlib derives its noise factors as 2 * level + 1, so an odd scale factor
    # lambda maps to the level (lambda - 1) / 2.
    method, degree = extrapolation
    fold_levels = [int((float(factor) - 1.0) / 2.0) for factor in scale_factors]
    return method, degree, fold_levels


def _cqlib_extrapolation(
    factory: Factory | None,
    scale_factors: Sequence[float],
    /,
) -> tuple[ExtrapolateMethod, int] | None:
    r"""Map a factory onto the cqlib model that reproduces it.

    Returns:
        The cqlib extrapolation method and polynomial degree, or ``None``
            when no cqlib model matches the factory.

    """
    if factory is None:
        return ExtrapolateMethod.polynomial(), len(scale_factors) - 1
    if isinstance(factory, PolyFactory):
        return ExtrapolateMethod.polynomial(), factory.degree
    # cqlib's exponential fit forces a zero asymptote, so an ExpFactory that
    # estimates its asymptote or fixes a non-zero one has no cqlib image.
    if isinstance(factory, ExpFactory) and factory.asymptote == 0.0:
        return ExtrapolateMethod.exponential(), len(scale_factors) - 1
    return None


def _run_cqlib_zne(
    circuit: Circuit,
    executor: QemExecutor,
    observable: Hamiltonian | PauliString,
    /,
    *,
    fold_levels: Sequence[int],
    gate_set: Sequence[Instruction] | None,
    shots: int | None,
    method: ExtrapolateMethod,
    degree: int,
    seed: int | None,
) -> float:
    r"""Fold, measure, and extrapolate through cqlib's ZNE helper."""
    hamiltonian = _as_hamiltonian(observable)
    # Folding inverts the whole gate body, and a measurement has no unitary
    # inverse, so the measurement tail is removed before folding.
    body, _measured = strip_measurements(circuit)
    mitigation = ZNEMitigation(body, list(fold_levels))
    callback = make_cqlib_estimator(executor, hamiltonian, shots=shots, seed=seed)
    values = mitigation.run_em_sequence_with_shots(
        gate_set,
        hamiltonian,
        shots,
        callback,
    )
    return float(mitigation.extrapolate(list(values), method, degree))


def _as_hamiltonian(observable: Hamiltonian | PauliString, /) -> Hamiltonian:
    r"""Wrap a bare Pauli string so the extrapolation can consume it."""
    if isinstance(observable, PauliString):
        return Hamiltonian.from_pauli(observable)
    return observable


def _is_odd_fold_level(scale_factor: float, /) -> bool:
    r"""Return whether ``scale_factor`` equals ``2n + 1`` for integer ``n >= 0``."""
    value = float(scale_factor)
    if value < 1.0 or not value.is_integer():
        return False
    return (int(value) - 1) % 2 == 0


def _validate_seed(seed: int | None, /) -> None:
    r"""Validate an optional random seed."""
    if seed is None:
        return
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer or None")
    if seed < 0:
        raise ValueError("seed must be non-negative")


def _warn_if_ill_conditioned(scale_factors: Sequence[float], /) -> None:
    r"""Warn when the extrapolation grid cannot be inverted stably.

    The extrapolation coefficients solve a Vandermonde system in the scale
    factors. When the points are too close together that system is
    ill-conditioned and the fit amplifies the measurement error, which is
    reported as a :class:`ConvergenceWarning` rather than an error because the
    result stays finite and usable.
    """
    factors = np.asarray(scale_factors, dtype=float)
    if factors.size < 2:
        return
    vandermonde = np.vander(factors, factors.size, increasing=True)
    condition = float(np.linalg.cond(vandermonde))
    if condition > _ILL_CONDITIONED_LIMIT:
        warnings.warn(
            "The scale factors are too close together for a stable "
            f"extrapolation: the fit matrix has condition number "
            f"{condition:.3e}, above the {_ILL_CONDITIONED_LIMIT:.0e} "
            "threshold. Re-sample the scale factors on a wider, evenly "
            "spaced grid.",
            ConvergenceWarning,
            stacklevel=2,
        )
