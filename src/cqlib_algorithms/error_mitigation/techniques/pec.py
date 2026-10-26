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

r"""Probabilistic error cancellation.

Probabilistic error cancellation (PEC) recovers the ideal expectation value
of a circuit from a signed Monte-Carlo average over implementable circuits.
Each ideal operation is expanded against the device noise channel
:math:`\mathcal{D}` into a quasi-probability distribution

.. math::

    \mathcal{U} = \sum_\alpha \eta_\alpha \,
        \mathcal{D} \circ \mathcal{P}_\alpha \circ \mathcal{U},

where the corrections :math:`\mathcal{P}_\alpha` are Paulis. Because the noise
channel is a premise and only the Paulis are gates, every term is directly
executable: the device applies :math:`\mathcal{D}` to the corrected operation
:math:`\mathcal{P}_\alpha \mathcal{U}` on its own. Sampling :math:`\alpha`
with probability :math:`|\eta_\alpha| / \Gamma`, weighting each measured
expectation value by :math:`\mathrm{sign}(\eta_\alpha)`, and multiplying the
average by the one-norm :math:`\Gamma = \sum_\alpha |\eta_\alpha|` reproduces
the ideal value in expectation.

The price is variance inflated by :math:`\Gamma^2`: :math:`\Gamma` grows
exponentially with circuit depth, so it is accumulated in the log domain and
refused past a ceiling, and the sample count derived from it is reported when
it becomes large.

The two-piece API is :func:`construct_circuits` (sample the variant circuits
and return their signs together with :math:`\Gamma`) plus
:func:`combine_results` (form the :math:`\Gamma`-compensated signed average),
orchestrated by :func:`execute_with_pec`; :func:`pec_executor` returns the
same computation as a reusable executor for chaining.

References
----------
.. [1] K. Temme, S. Bravyi, and J. M. Gambetta, "Error mitigation for
   short-depth quantum circuits," Physical Review Letters 119, 180509, 2017.
   arXiv:1612.02058.
.. [2] S. Endo, S. C. Benjamin, and Y. Li, "Practical quantum error mitigation
   for near-future applications," Physical Review X 8, 031027, 2018.
   arXiv:1712.09271.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from numbers import Real

import numpy as np

from cqlib import Circuit
from cqlib.circuit import ValueOperation
from cqlib.qis import Hamiltonian, PauliString

from ...protocols import Estimator, EstimatorRequest, EstimatorResult, spawn_seeds
from .._circuits import expand, is_measurement
from ..core import QemExecutor
from ..quasiprobability import QuasiprobabilityRepresentation

# The one-norm grows exponentially with circuit depth while the sample count
# grows with its square, so past this one-norm the required number of circuit
# executions is unattainable and the request is refused instead of started.
_MAX_NORM = 1.0e12

# Above this sample count a run takes long enough that the caller is warned
# before it begins.
_MAX_SAMPLES = 100_000


def construct_circuits(
    circuit: Circuit,
    /,
    *,
    quasiprobabilities: Sequence[QuasiprobabilityRepresentation],
    precision: float = 0.03,
    seed: int | None = None,
) -> tuple[list[Circuit], list[int], float]:
    r"""Sample sign-carrying variant circuits from a quasi-probability basis.

    Each of the ``num_samples`` variants is built independently: every
    non-measurement operation of ``circuit`` draws one correction
    :math:`\alpha` with probability :math:`|\eta_\alpha| / \Gamma_k` and
    appends it, so the variant realises :math:`\mathcal{P}_\alpha
    \mathcal{U}` for that operation on the device. The identity correction is
    the operation itself and appends nothing, matching the representation's
    contract that ``basis[0]`` is the identity. The sign of a variant is the
    product of the signs of the corrections it drew, and ``norm`` is the
    one-norm :math:`\Gamma` of the whole circuit.

    Parameters
    ----------
    circuit
        Circuit to expand. Measurements are passed through unchanged and need
        no representation.
    quasiprobabilities
        One representation per non-measurement operation, in circuit order.
        Each must act on at most two qubits.
    precision
        Target precision :math:`\epsilon` of the estimate. The sample count is
        :math:`\lceil (\Gamma / \epsilon)^2 \rceil`.
    seed
        Seed of the sampling stream, or ``None`` for fresh randomness.

    Returns
    -------
    tuple[list[Circuit], list[int], float]
        The sampled variant circuits, their signs in :math:`\{-1, +1\}`, and
        the one-norm :math:`\Gamma` of the circuit representation.

    Raises
    ------
    TypeError
        If ``precision`` is not a real number or ``seed`` is neither an
        integer nor ``None``.
    ValueError
        If ``precision`` is not in :math:`(0, 1]`, if a representation acts on
        more than two qubits, if the representation count does not match the
        circuit's operation count, or if :math:`\Gamma` exceeds
        :data:`_MAX_NORM`.

    Warns
    -----
    UserWarning
        If the deduced sample count exceeds :data:`_MAX_SAMPLES`.
    """
    _validate_precision(precision)
    _validate_seed(seed)
    _validate_two_qubit_operations(circuit)
    representations = tuple(quasiprobabilities)
    _validate_representations(circuit, representations)

    # Gamma is a product of per-operation one-norms, so its logarithm is a sum;
    # accumulating in the log domain keeps the exponential growth from
    # overflowing before the ceiling below can reject it.
    log_norm = _log_total_norm(representations)
    if log_norm > math.log(_MAX_NORM):
        raise ValueError(
            "input circuit violates the condition: the quasi-probability "
            f"one-norm Gamma = exp({log_norm:.3f}) exceeds the "
            f"{_MAX_NORM:.0e} ceiling above which the sample count is "
            "unattainable, see Temme 2017, arXiv:1612.02058, in detail; the "
            "noise model is too strong for PEC at this circuit depth."
        )
    norm = float(math.exp(log_norm))

    num_samples = math.ceil((norm / precision) ** 2)
    if num_samples > _MAX_SAMPLES:
        warnings.warn(
            "PEC needs "
            f"{num_samples} circuit executions, above the {_MAX_SAMPLES} "
            "threshold; this may take a long time. Increase precision or "
            "reduce the circuit depth.",
            UserWarning,
            stacklevel=2,
        )

    rng = np.random.default_rng(seed)
    sampled, signs = _sample_circuits(
        circuit,
        representations,
        num_samples=num_samples,
        rng=rng,
    )
    return sampled, signs, norm


def combine_results(
    results: Sequence[EstimatorResult],
    norm: float,
    signs: Sequence[int],
    /,
) -> float:
    r"""Form the :math:`\Gamma`-compensated signed average of the results.

    The compensation is performed here, where the one-norm is a required
    argument, so a caller cannot drop it by accident: a single sample's
    sign-weighted mean is :math:`1 / \Gamma` of the ideal value and only the
    multiplication by ``norm`` restores it.

    Parameters
    ----------
    results
        Expectation values measured on the circuits returned by
        :func:`construct_circuits`, in the same order.
    norm
        One-norm :math:`\Gamma` returned by :func:`construct_circuits`.
    signs
        Per-circuit signs returned by :func:`construct_circuits`, in the same
        order.

    Returns
    -------
    float
        The PEC estimate of the ideal expectation value.

    Raises
    ------
    ValueError
        If ``results`` and ``signs`` have different lengths, or if no result
        was supplied.
    """
    if len(results) != len(signs):
        raise ValueError(
            f"got {len(results)} result(s) but {len(signs)} sign(s); they "
            "must be paired one to one."
        )
    if not results:
        raise ValueError("PEC needs at least one sign-weighted result.")
    unbiased = [
        norm * sign * result.expectation
        for result, sign in zip(results, signs, strict=True)
    ]
    return float(np.average(unbiased))


def execute_with_pec(
    circuit: Circuit,
    executor: QemExecutor,
    observable: Hamiltonian | PauliString,
    /,
    *,
    quasiprobabilities: Sequence[QuasiprobabilityRepresentation],
    precision: float = 0.03,
    shots: int | None = None,
    seed: int | None = None,
) -> float:
    r"""Estimate one observable on ``circuit`` with probabilistic error cancellation.

    This is the immediate form of :func:`pec_executor`::

        execute_with_pec(c, ex, obs, **params)
        == pec_executor(ex, **params).estimate(
               [EstimatorRequest(c, obs)]
           )[0].expectation

    Parameters
    ----------
    circuit
        Circuit whose expectation value is to be mitigated.
    executor
        Execution backend supplying the expectation surface.
    observable
        Observable to estimate. A bare Pauli string is interpreted as a
        Hamiltonian with unit coefficient.
    quasiprobabilities
        One representation per non-measurement operation of ``circuit``, in
        circuit order.
    precision
        Target precision :math:`\epsilon` of the estimate.
    shots
        Shot count for every variant circuit, or ``None`` for exact
        evaluation.
    seed
        Seed of the sampling stream, or ``None`` for fresh randomness.

    Returns
    -------
    float
        Mitigated expectation value.

    Raises
    ------
    TypeError
        If ``precision`` is not a real number or ``seed`` is neither an
        integer nor ``None``.
    ValueError
        If a parameter describes an unusable configuration, such as a
        representation for a gate acting on more than two qubits or a
        one-norm above the ceiling.
    """
    wrapped = pec_executor(
        executor,
        quasiprobabilities=quasiprobabilities,
        precision=precision,
        shots=shots,
        seed=seed,
    )
    request = EstimatorRequest(circuit=circuit, observable=observable, shots=shots)
    return wrapped.estimate([request])[0].expectation


def pec_executor(
    executor: QemExecutor,
    /,
    *,
    quasiprobabilities: Sequence[QuasiprobabilityRepresentation],
    precision: float = 0.03,
    shots: int | None = None,
    seed: int | None = None,
) -> QemExecutor:
    r"""Return an executor with probabilistic error cancellation applied.

    The returned executor keeps the wrapped executor's sampling surface and
    caching flag, so PEC can be chained on top of another technique without
    changing either semantics.

    Parameters
    ----------
    executor
        Execution backend to wrap.
    quasiprobabilities
        One representation per non-measurement operation of every circuit
        later submitted, in circuit order.
    precision
        Target precision :math:`\epsilon` of the estimate.
    shots
        Shot count bound to every request, or ``None`` to defer to the
        request's own shot count.
    seed
        Seed of the sampling stream, or ``None`` for fresh randomness.

    Returns
    -------
    QemExecutor
        Executor whose expectation surface applies probabilistic error
        cancellation.

    Raises
    ------
    TypeError
        If ``precision`` is not a real number or ``seed`` is neither an
        integer nor ``None``.
    ValueError
        If ``precision`` is not in :math:`(0, 1]`.
    """
    estimator = _PecEstimator(
        executor,
        quasiprobabilities=tuple(quasiprobabilities),
        precision=precision,
        shots=shots,
        seed=seed,
    )
    return QemExecutor(
        estimator=estimator,
        sampler=executor.sampler,
        use_cache=executor.use_cache,
    )


class _PecEstimator(Estimator):
    r"""Expectation surface that samples, measures, and recombines variants.

    Parameters
    ----------
    executor
        Execution backend supplying the expectation surface for each variant
        circuit.
    quasiprobabilities
        One representation per non-measurement operation, in circuit order.
    precision
        Target precision of the estimate.
    shots
        Shot count bound to every request, or ``None`` to defer to the
        request.
    seed
        Seed of the sampling stream, or ``None`` for fresh randomness.
    """

    __slots__ = (
        "_executor",
        "_precision",
        "_quasiprobabilities",
        "_seed",
        "_shots",
    )

    _executor: QemExecutor
    _precision: float
    _quasiprobabilities: tuple[QuasiprobabilityRepresentation, ...]
    _seed: int | None
    _shots: int | None

    def __init__(
        self,
        executor: QemExecutor,
        /,
        *,
        quasiprobabilities: tuple[QuasiprobabilityRepresentation, ...],
        precision: float,
        shots: int | None,
        seed: int | None,
    ) -> None:
        _validate_precision(precision)
        _validate_seed(seed)
        self._executor = executor
        self._quasiprobabilities = quasiprobabilities
        self._precision = float(precision)
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
        r"""Sample one request, measure every variant, and recombine."""
        shots = self._shots if self._shots is not None else request.shots
        seed = self._seed if self._seed is not None else request.seed
        circuits, signs, norm = construct_circuits(
            request.circuit,
            quasiprobabilities=self._quasiprobabilities,
            precision=self._precision,
            seed=seed,
        )
        # Each sampled variant is measured independently, so it takes its own
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
        return EstimatorResult(expectation=combine_results(results, norm, signs))


def _log_total_norm(
    representations: Sequence[QuasiprobabilityRepresentation],
    /,
) -> float:
    r"""Return :math:`\log \Gamma` of a whole-circuit representation.

    The circuit one-norm is the product of the per-operation one-norms, so its
    logarithm is the sum of their logarithms and stays finite however many
    operations the circuit has.
    """
    total = 0.0
    for representation in representations:
        total += _log_local_norm(representation.coefficients)
    return total


def _log_local_norm(coefficients: Sequence[float], /) -> float:
    r"""Return :math:`\log \sum_\alpha |\eta_\alpha|` in the log domain."""
    magnitudes = np.abs(np.asarray(coefficients, dtype=float))
    # Zero weights contribute nothing to the sum; dropping them avoids the
    # ``log(0)`` that would otherwise have to be reduced away.
    nonzero = magnitudes[magnitudes > 0.0]
    if nonzero.size == 0:
        return -math.inf
    return float(np.logaddexp.reduce(np.log(nonzero)))


def _sample_circuits(
    circuit: Circuit,
    representations: Sequence[QuasiprobabilityRepresentation],
    /,
    *,
    num_samples: int,
    rng: np.random.Generator,
) -> tuple[list[Circuit], list[int]]:
    r"""Sample ``num_samples`` variant circuits and their signs.

    Parameters
    ----------
    circuit
        Circuit to expand.
    representations
        One representation per non-measurement operation, in circuit order.
    num_samples
        Number of variants to draw.
    rng
        Random stream the corrections are drawn from.

    Returns
    -------
    tuple[list[Circuit], list[int]]
        The variant circuits, then their signs, both in draw order.
    """
    sampled: list[Circuit] = []
    signs: list[int] = []
    for _ in range(num_samples):
        variant, sign = _sample_variant(circuit, representations, rng)
        sampled.append(variant)
        signs.append(sign)
    return sampled, signs


def _sample_variant(
    circuit: Circuit,
    representations: Sequence[QuasiprobabilityRepresentation],
    rng: np.random.Generator,
    /,
) -> tuple[Circuit, int]:
    r"""Draw one variant circuit and its sign.

    Returns
    -------
    tuple[Circuit, int]
        The variant circuit and the product of the signs of the corrections
        it drew.
    """
    position = 0
    sign = 1

    def expander(operation: ValueOperation) -> tuple[ValueOperation, ...]:
        nonlocal position, sign
        if is_measurement(operation):
            return (operation,)
        representation = representations[position]
        position += 1
        choice = _draw_correction(representation.coefficients, rng)
        if representation.coefficients[choice] < 0.0:
            sign = -sign
        # ``basis[0]`` is the identity by the representation's contract, so the
        # identity term needs no correction appended and the variant is the
        # operation itself.
        if choice == 0:
            return (operation,)
        correction = ValueOperation.from_instruction(
            representation.basis[choice],
            list(operation.qubits),
        )
        return (operation, correction)

    return expand(circuit, expander), sign


def _draw_correction(coefficients: Sequence[float], rng: np.random.Generator, /) -> int:
    r"""Draw a basis index with probability proportional to its weight.

    Raises
    ------
    ValueError
        If the weights do not form a finite, non-zero distribution.
    """
    magnitudes = np.abs(np.asarray(coefficients, dtype=float))
    total = float(magnitudes.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(
            "a quasi-probability representation must have a finite, non-zero "
            "one-norm before it can be sampled."
        )
    return int(rng.choice(magnitudes.size, p=magnitudes / total))


def _validate_precision(precision: float, /) -> None:
    r"""Reject a precision outside the half-open interval :math:`(0, 1]`."""
    if isinstance(precision, bool) or not isinstance(precision, Real):
        raise TypeError("precision must be a real number")
    if not 0.0 < float(precision) <= 1.0:
        raise ValueError(
            f"precision must be within (0, 1] but was {precision!r}."
        )


def _validate_seed(seed: int | None, /) -> None:
    r"""Reject a seed that is neither an integer nor ``None``."""
    if seed is None:
        return
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer or None")


def _validate_two_qubit_operations(circuit: Circuit, /) -> None:
    r"""Reject any gate acting on more than two qubits (trap T10)."""
    for operation in circuit.operations:
        if is_measurement(operation):
            continue
        if len(operation.qubits) > 2:
            raise ValueError(
                "input circuit violates the condition: a quasi-probability "
                "decomposition exists only for one- and two-qubit channels, "
                "see Temme 2017, arXiv:1612.02058, in detail; operation "
                f"{operation.name!r} acts on {len(operation.qubits)} qubits, "
                "so pre-compile it into one- and two-qubit gates first."
            )


def _validate_representations(
    circuit: Circuit,
    representations: Sequence[QuasiprobabilityRepresentation],
    /,
) -> None:
    r"""Reject a representation list that does not match the circuit."""
    if not representations:
        raise ValueError(
            "quasiprobabilities must contain at least one representation."
        )
    expected = sum(
        1 for operation in circuit.operations if not is_measurement(operation)
    )
    if len(representations) != expected:
        raise ValueError(
            f"quasiprobabilities has {len(representations)} entries but the "
            f"circuit has {expected} non-measurement operation(s); supply one "
            "representation per operation in circuit order."
        )
    for representation in representations:
        coefficients = np.asarray(representation.coefficients, dtype=float)
        if not bool(np.all(np.isfinite(coefficients))):
            raise ValueError(
                "a quasi-probability representation has a non-finite "
                "coefficient."
            )
        if float(np.sum(np.abs(coefficients))) <= 0.0:
            raise ValueError(
                "a quasi-probability representation has a zero one-norm and "
                "cannot be sampled."
            )
