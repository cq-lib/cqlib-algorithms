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

r"""Layerwise Richardson extrapolation.

Layerwise Richardson extrapolation (LRE) lowers the granularity of zero-noise
extrapolation from the whole circuit to its individual chunks. The gate body
is split into contiguous chunks and every chunk is folded by its own scale
factor, so one circuit yields a family of measurements indexed by a scale
*vector* instead of a single number.

The noise response is modelled as a multivariate polynomial in those scale
factors,

.. math::

    \langle O(\lambda) \rangle = \sum_{|\alpha| \le d} c_\alpha
    \lambda^\alpha,

and the mitigated value is the linear combination that cancels every
non-constant monomial while keeping the constant term,

.. math::

    \langle O(0) \rangle = \sum_i \gamma_i \langle O(\lambda_i) \rangle,
    \qquad
    \sum_i \gamma_i = 1, \qquad
    \sum_i \gamma_i \lambda_i^\alpha = 0 \quad (1 \le |\alpha| \le d).

The scale vectors are the graded monomial patterns of degree :math:`d` over
the :math:`k` chunks, so one call executes :math:`\binom{k + d}{d}` circuits.

The two-piece API is :func:`construct_circuits` (build the scaled family and
its scale vectors) plus :func:`combine_results` (reduce the measured values to
the extrapolated limit), orchestrated by :func:`execute_with_lre`;
:func:`lre_executor` returns the same computation as a reusable executor.

References
----------
[1] A. Russo and A. Mari, "Quantum error mitigation by layerwise Richardson
    extrapolation," Physical Review A 110, 062420, 2024. arXiv:2402.04000.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Mapping, Sequence

import numpy as np
from cqlib import Circuit
from cqlib.qis import Hamiltonian, PauliString

from ...protocols import Estimator, EstimatorRequest, EstimatorResult, spawn_seeds
from ...protocols.state_preparation import StatePreparation
from .._circuits import (
    bind_request_circuit,
    build_estimator_request,
    strip_measurements,
)
from ..core import QemExecutor
from ..inference import (
    ConvergenceWarning,
    multivariate_richardson_coefficients,
)
from ..inference.multivariate import (
    _infer_degree,
    _monomial_exponents,
    _sample_matrix,
)
from ..scaling import fold_layers
from ..scaling.folding import _restore_measurements, _split_into_chunks

# Condition number of the multivariate extrapolation system above which the
# fitted weights amplify their inputs by more than an order of magnitude per
# decade of scale spacing. Reaching it usually means the chunks are scaled too
# close together for the chosen degree.
_ILL_CONDITIONED_LIMIT = 1.0e8

# The weights are the exact solution of a linear system whose constant term is
# one, so a sum far from one means the system was solved to unexpected
# precision; the tolerance only guards against a degraded fit.
_WEIGHT_SUM_TOLERANCE = 1.0e-8


def construct_circuits(
    circuit: Circuit,
    /,
    *,
    degree: int,
    fold_multiplier: int,
    num_chunks: int,
) -> tuple[list[Circuit], list[list[float]]]:
    r"""Build the layerwise scaled family and its scale vectors.

    The gate body of ``circuit`` is split into ``num_chunks`` contiguous
    chunks and each chunk is folded by its own scale factor. Chunk :math:`i`
    of a pattern :math:`k` is folded
    :math:`\text{fold\_multiplier} \cdot k_i` times, i.e. scaled by

    .. math::

        \lambda_i = 1 + \text{fold\_multiplier} \cdot k_i .

    The patterns :math:`k` range over the graded monomial basis of total
    degree at most ``degree``, which is the smallest set of scale vectors that
    makes the multivariate extrapolation well posed.

    Args:
        circuit: Circuit to scale. Measurements are stripped before folding and
            re-appended afterwards.
        degree: Multivariate polynomial degree :math:`d \ge 1`. A higher degree
            cancels more noise orders at the cost of more executions.
        fold_multiplier: Positive integer step between adjacent graded levels.
        num_chunks: Number of contiguous chunks the gate body is split into, ``>= 1``.
            ``1`` puts every gate in one chunk, which reduces LRE to standard
            zero-noise extrapolation.

    Returns:
        The scaled circuits and, in the same order, their scale vectors. Each
            scale vector holds one factor per chunk.

    Raises:
        TypeError: If ``degree``, ``fold_multiplier``, or ``num_chunks`` is not an
            integer.
        ValueError: If ``degree < 1``, ``fold_multiplier < 1``, ``num_chunks < 1``, or
            ``num_chunks`` exceeds the number of gate operations of ``circuit``.

    Note:
        The family holds :math:`\binom{\text{num\_chunks} + \text{degree}}
        {\text{degree}}` circuits, the graded-monomial subset of the
        :math:`(\text{degree} + 1)^{\text{num\_chunks}}` full grid of per-chunk
        scale choices. That count is exponential in ``num_chunks`` for a fixed
        ``degree``, so the number of circuit executions grows quickly with the
        chunk count; keep ``num_chunks`` small.

    """
    _validate_degree(degree)
    _validate_fold_multiplier(fold_multiplier)
    _validate_num_chunks(num_chunks)

    body, measured = strip_measurements(circuit)
    num_operations = len(body.operations)
    if num_chunks > num_operations:
        raise ValueError(
            f"num_chunks={num_chunks} exceeds the {num_operations} gate "
            "operation(s) of the circuit, which would leave a chunk empty."
        )

    chunks = _split_into_chunks(list(body.operations), num_chunks)
    circuits: list[Circuit] = []
    scale_points: list[list[float]] = []
    for pattern in _graded_patterns(num_chunks, degree):
        # Every operation inherits the scale factor of the chunk it belongs
        # to, so ``fold_layers`` folds the body one chunk at a time.
        scales: dict[int, float] = {}
        offset = 0
        for chunk, level in zip(chunks, pattern, strict=True):
            factor = 1.0 + fold_multiplier * level
            for step in range(len(chunk)):
                scales[offset + step] = factor
            offset += len(chunk)
        folded = fold_layers(body, scales=scales)
        circuits.append(_restore_measurements(folded, measured))
        scale_points.append([1.0 + fold_multiplier * level for level in pattern])
    return circuits, scale_points


def combine_results(
    results: Sequence[EstimatorResult],
    scale_points: Sequence[Sequence[float]],
    /,
) -> float:
    r"""Reduce measured expectation values to the extrapolated limit.

    Args:
        results: Expectation values measured on the circuits returned by
            :func:`construct_circuits`, in the same order.
        scale_points: The scale vectors paired with ``results``, one vector per measurement.

    Returns:
        The extrapolated zero-noise expectation value.

    Raises:
        ValueError: If the number of results does not match the number of scale points, if
            the scale points do not span a graded monomial basis, if their
            multivariate sample matrix is singular, or if the resulting weights do
            not sum to one.

    Warns:
        ConvergenceWarning: If the multivariate sample matrix is ill-conditioned, in which case
            the extrapolation amplifies measurement error.

    """
    if len(results) != len(scale_points):
        raise ValueError(
            f"{len(results)} result(s) do not match {len(scale_points)} scale "
            "point(s); one result is needed per scale vector."
        )

    weights = multivariate_richardson_coefficients(scale_points)
    # The constant monomial must survive with unit weight so that the
    # extrapolation reproduces the noise-free value.
    total = float(np.sum(weights))
    if not np.isclose(total, 1.0, rtol=0.0, atol=_WEIGHT_SUM_TOLERANCE):
        raise ValueError(
            "The multivariate extrapolation weights do not sum to one "
            f"(sum={total!r}), so the noise-free term is not preserved; the "
            "scale points are too ill-conditioned for a stable fit."
        )

    _warn_if_ill_conditioned(scale_points)
    values = [result.expectation for result in results]
    return float(np.dot(values, weights))


def execute_with_lre(
    circuit: StatePreparation | Circuit,
    executor: QemExecutor,
    observable: Hamiltonian | PauliString,
    /,
    *,
    degree: int,
    fold_multiplier: int,
    num_chunks: int,
    shots: int | None = None,
    seed: int | None = None,
    parameter_bindings: Mapping[str, float] | None = None,
    prefix: str = "qem",
) -> float:
    r"""Estimate one observable on ``circuit`` with layerwise extrapolation.

    This is the immediate form of :func:`lre_executor`. The same task
    can be evaluated by calling ``run()`` on that executor.

    Args:
        circuit: State-preparation component or circuit to evaluate.
        executor: Execution backend supplying the expectation surface.
        observable: Observable to estimate.
        degree: Multivariate polynomial degree :math:`d \ge 1`.
        fold_multiplier: Positive integer step between adjacent graded levels.
        num_chunks: Number of contiguous chunks the gate body is split into.
        shots: Shot count for every scaled circuit, or ``None`` for exact evaluation.
        seed: Seed of the sampled scaled circuits, or ``None`` for fresh randomness.
            Every scaled circuit draws from its own child seed, so one value
            reproduces the whole family.
        parameter_bindings: Finite values covering exactly all active symbols.
            Values are bound before circuit transformations; None requires no symbols.
        prefix: Build prefix for a preparation component, defaulting to ``qem``;
            existing Circuit symbols keep their names.

    Returns:
        Mitigated expectation value.

    Raises:
        TypeError: If an input or configuration argument has the wrong type.
        ValueError: If bindings are non-finite or do not cover active symbols exactly,
            or if ``degree < 1``, ``fold_multiplier < 1``, ``num_chunks < 1``, or
            ``num_chunks`` exceeds the number of gate operations of ``circuit``.

    Note:
        One call executes :math:`\binom{\text{num\_chunks} + \text{degree}}
        {\text{degree}}` circuits, the graded-monomial subset of the
        :math:`(\text{degree} + 1)^{\text{num\_chunks}}` full grid of per-chunk
        scale choices. The count is exponential in ``num_chunks`` for a fixed
        ``degree``, so the number of executions grows quickly with the chunk
        count; keep ``num_chunks`` small.
    """
    wrapped = lre_executor(
        executor,
        degree=degree,
        fold_multiplier=fold_multiplier,
        num_chunks=num_chunks,
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


def lre_executor(
    executor: QemExecutor,
    /,
    *,
    degree: int,
    fold_multiplier: int,
    num_chunks: int,
    shots: int | None = None,
    seed: int | None = None,
) -> QemExecutor:
    r"""Return an executor with layerwise extrapolation applied.

    The returned executor keeps the wrapped executor's sampling surface and
    caching flag, so LRE can be chained on top of another technique without
    changing either semantics.

    Args:
        executor: Execution backend to wrap.
        degree: Multivariate polynomial degree :math:`d \ge 1`.
        fold_multiplier: Positive integer step between adjacent graded levels.
        num_chunks: Number of contiguous chunks the gate body is split into.
        shots: Shot count bound to every request, or ``None`` to defer to the
            request's own shot count.
        seed: Seed of the sampled scaled circuits, or ``None`` to defer to the
            request's own seed.

    Returns:
        Executor whose expectation surface applies layerwise extrapolation.

    Raises:
        TypeError: If ``degree``, ``fold_multiplier``, or ``num_chunks`` is not an
            integer, or if ``seed`` is neither an integer nor ``None``.
        ValueError: If ``degree < 1``, ``fold_multiplier < 1``, or ``num_chunks < 1``.

    """
    _validate_degree(degree)
    _validate_fold_multiplier(fold_multiplier)
    _validate_num_chunks(num_chunks)
    _validate_seed(seed)

    estimator = _LreEstimator(
        executor,
        degree=degree,
        fold_multiplier=fold_multiplier,
        num_chunks=num_chunks,
        shots=shots,
        seed=seed,
    )
    return QemExecutor(
        estimator=estimator,
        sampler=executor.sampler,
        use_cache=executor.use_cache,
    )


class _LreEstimator(Estimator):
    r"""Expectation surface that folds chunks, measures, and extrapolates.

    Args:
        executor: Execution backend supplying the expectation surface for each scaled
            circuit.
        degree: Multivariate polynomial degree.
        fold_multiplier: Step between adjacent graded levels.
        num_chunks: Number of chunks the gate body is split into.
        shots: Shot count bound to every request, or ``None`` to defer to the
            request.
        seed: Seed of the sampled scaled circuits, or ``None`` to defer to the
            request's own seed.

    """

    __slots__ = (
        "_degree",
        "_executor",
        "_fold_multiplier",
        "_num_chunks",
        "_seed",
        "_shots",
    )

    _degree: int
    _executor: QemExecutor
    _fold_multiplier: int
    _num_chunks: int
    _seed: int | None
    _shots: int | None

    def __init__(
        self,
        executor: QemExecutor,
        /,
        *,
        degree: int,
        fold_multiplier: int,
        num_chunks: int,
        shots: int | None,
        seed: int | None,
    ) -> None:
        self._executor = executor
        self._degree = degree
        self._fold_multiplier = fold_multiplier
        self._num_chunks = num_chunks
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
        r"""Fold one request's circuit family, measure it, and combine it."""
        bound_circuit = bind_request_circuit(request)
        shots = self._shots if self._shots is not None else request.shots
        seed = self._seed if self._seed is not None else request.seed
        circuits, scale_points = construct_circuits(
            bound_circuit,
            degree=self._degree,
            fold_multiplier=self._fold_multiplier,
            num_chunks=self._num_chunks,
        )
        # Each scaled circuit is measured independently, so it takes its own
        # child seed while one run seed still reproduces the whole family.
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
        # TODO: the extrapolation weights depend only on the scale points,
        # which are the same for every request of a batch, so this re-solves
        # the identical linear system once per request. Cache the weights on
        # the estimator if the per-request cost ever becomes significant.
        return EstimatorResult(expectation=combine_results(results, scale_points))


def _graded_patterns(
    num_chunks: int,
    degree: int,
    /,
) -> list[tuple[int, ...]]:
    r"""Return the graded monomial patterns of the layerwise scale grid.

    A pattern is a length-``num_chunks`` vector of non-negative integer levels
    whose total does not exceed ``degree``; there are
    ``comb(num_chunks + degree, degree)`` of them, one per multivariate
    Richardson weight. The all-zero pattern scales nothing and the uniform
    pattern ``(k, ..., k)`` folds every chunk identically.
    """
    return [
        pattern
        for pattern in itertools.product(range(degree + 1), repeat=num_chunks)
        if sum(pattern) <= degree
    ]


def _warn_if_ill_conditioned(
    scale_points: Sequence[Sequence[float]],
    /,
) -> None:
    r"""Warn when the multivariate extrapolation system is ill-conditioned.

    The weights solve a multivariate sample matrix built from the graded
    monomial basis evaluated at the scale points. When the points are too
    close together that system is ill-conditioned and the fit amplifies the
    measurement error, which is reported as a :class:`ConvergenceWarning`
    rather than an error because the result stays finite and usable.
    """
    points = np.asarray(scale_points, dtype=float)
    num_points, num_layers = points.shape
    degree = _infer_degree(num_layers, num_points)
    sample = _sample_matrix(points, _monomial_exponents(num_layers, degree))
    condition = float(np.linalg.cond(sample))
    if condition > _ILL_CONDITIONED_LIMIT:
        warnings.warn(
            "The scale points are too close together for a stable "
            f"extrapolation: the fit matrix has condition number "
            f"{condition:.3e}, above the {_ILL_CONDITIONED_LIMIT:.0e} "
            "threshold. Re-sample the chunks on a wider, evenly spaced grid.",
            ConvergenceWarning,
            stacklevel=2,
        )


def _validate_degree(degree: int, /) -> None:
    r"""Validate a positive polynomial degree."""
    if isinstance(degree, bool) or not isinstance(degree, int):
        raise TypeError("degree must be an integer")
    if degree < 1:
        raise ValueError("degree must be greater than or equal to 1")


def _validate_fold_multiplier(fold_multiplier: int, /) -> None:
    r"""Validate a positive fold multiplier."""
    if isinstance(fold_multiplier, bool) or not isinstance(fold_multiplier, int):
        raise TypeError("fold_multiplier must be an integer")
    if fold_multiplier < 1:
        raise ValueError("fold_multiplier must be greater than or equal to 1")


def _validate_num_chunks(num_chunks: int, /) -> None:
    r"""Validate a positive chunk count."""
    if isinstance(num_chunks, bool) or not isinstance(num_chunks, int):
        raise TypeError("num_chunks must be an integer")
    if num_chunks < 1:
        raise ValueError("num_chunks must be greater than or equal to 1")


def _validate_seed(seed: int | None, /) -> None:
    r"""Validate an optional random seed."""
    if seed is None:
        return
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer or None")
    if seed < 0:
        raise ValueError("seed must be non-negative")
