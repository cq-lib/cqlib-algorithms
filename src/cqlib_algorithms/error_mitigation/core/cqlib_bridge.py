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

r"""Adapters bridging this library's executor with cqlib mitigation entry points.

The functions here let cqlib mitigation routines (for example
``ZNEMitigation`` and ``VirtualDistillation``) drive a
:class:`~cqlib_algorithms.error_mitigation.core.executor.QemExecutor` through
cqlib's callback contract, and convert the results back into this library's
:class:`~cqlib_algorithms.protocols.EstimatorResult` so that chaining,
caching, and variance propagation stay inside one result type.
"""

from __future__ import annotations

from cqlib import Circuit

# cqlib's estimator contract and this library's Estimator protocol share a
# short name but differ, so the cqlib alias is imported under an explicit name.
from cqlib.error_mitigation import Estimator as CqlibEstimator
from cqlib.qis import Hamiltonian

from ...protocols import EstimatorRequest, EstimatorResult
from .executor import QemExecutor


def make_cqlib_estimator(
    executor: QemExecutor,
    observable: Hamiltonian,
    *,
    shots: int | None = None,
    seed: int | None = None,
) -> CqlibEstimator:
    r"""Bind a library executor and one observable into a cqlib callback.

    The returned callback satisfies cqlib's mitigation estimator contract.
    The observable and shot count cqlib passes in are ignored in favour of
    the ones bound here, so every call inherits the executor's caching and
    request validation behaviour.

    Parameters
    ----------
    executor
        Executor supplying the expectation-value backend.
    observable
        Observable fixed for the lifetime of the returned callback.
    shots
        Shot count attached to every request, or ``None`` for exact
        evaluation.
    seed
        Seed attached to every request, or ``None`` for fresh randomness.

    Returns
    -------
    CqlibEstimator
        Callback ``(circuit, hamiltonian, shots) -> (mean, variance)``.
    """

    def callback(
        circuit: Circuit,
        _hamiltonian: Hamiltonian | None,
        _shots: int | None,
    ) -> tuple[float, float]:
        r"""Return the bound observable's ``(mean, variance)`` for ``circuit``."""
        request = EstimatorRequest(
            circuit=circuit, observable=observable, shots=shots, seed=seed
        )
        result = executor.estimate([request])[0]
        return (result.expectation, result.variance or 0.0)

    return callback


def pair_to_result(pair: tuple[float, float]) -> EstimatorResult:
    r"""Convert a cqlib ``(mean, variance)`` pair into an ``EstimatorResult``.

    cqlib mitigation callbacks return the bare pair rather than a value
    object. Converting it at the boundary keeps one result type across the
    branch, which is what makes technique chaining and variance propagation
    well defined.

    Parameters
    ----------
    pair
        cqlib callback output: mean followed by variance.

    Returns
    -------
    EstimatorResult
        The same information as a frozen library result value.

    Raises
    ------
    TypeError
        If a member of ``pair`` is not a real number.
    ValueError
        If a member of ``pair`` is non-finite or the variance is negative.
    """
    mean, variance = pair
    return EstimatorResult(expectation=mean, variance=variance)
