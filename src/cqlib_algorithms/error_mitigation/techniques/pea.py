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

r"""Probabilistic error amplification.

Probabilistic error amplification (PEA) trades the sampling overhead of PEC
for a bias-free extrapolation. Instead of cancelling the device noise, it
strengthens it by a controlled factor and measures the expectation value at
several amplified strengths; the values are then extrapolated back to zero
amplification.

At the amplified strength :math:`\epsilon' = s\,\epsilon` the ideal operation
:math:`\mathcal{U}` is expanded as a genuine probability distribution over the
ideal operation and every Pauli-corrected copy,

.. math::

    \mathcal{U} = (1 - \epsilon') \mathcal{U}
        + \frac{\epsilon'}{4^k - 1}
          \sum_{\alpha \neq 0} \mathcal{D} \circ \mathcal{P}_\alpha
          \circ \mathcal{U},

whose weights are non-negative and sum to one, so :math:`\Gamma = 1` and
sampling the expansion costs no variance amplification. It only injects the
extra noise the extrapolation is built on, in contrast to the signed weights
of probabilistic error cancellation.

The two-piece API is :func:`construct_circuits` (sample the variant circuits
at one scale factor) plus :func:`combine_results` (average them into one
estimate), orchestrated by :func:`execute_with_pea`, which repeats the pair at
every scale factor and extrapolates the resulting values to zero
amplification; :func:`pea_executor` returns the same computation as a reusable
executor for chaining.

References
----------
[1] Y. Kim, A. Eddins, S. Anand, K. X. Wei, E. van den Berg, S. Rosenblatt, H. Nayfeh,
    Y. Wu, M. Zaletel, K. Temme, and A. Kandala, "Evidence for the utility of quantum
    computing before fault tolerance," Nature 618, 500-505, 2023.
    doi:10.1038/s41586-023-06096-3.
[2] A. Mari, N. Shammah, and W. J. Zeng, "Extending quantum probabilistic error
    cancellation by noise scaling," Physical Review A 104, 052607, 2021.
    arXiv:2108.02237.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real

import numpy as np
from cqlib import Circuit
from cqlib.qis import Hamiltonian, PauliString

from ...protocols import Estimator, EstimatorRequest, EstimatorResult, spawn_seeds
from ...protocols.state_preparation import StatePreparation
from .._circuits import bind_request_circuit, build_estimator_request, is_measurement
from ..core import QemExecutor
from ..inference import (
    ExpFactory,
    Factory,
    LinearFactory,
    PolyFactory,
    RichardsonFactory,
)
from ..quasiprobability import QuasiprobabilityRepresentation, amplify_depolarizing
from ..quasiprobability.depolarizing import _storage_instruction
from .pec import (
    _sample_variant,
    _validate_precision,
    _validate_seed,
    _validate_two_qubit_operations,
)

# Names of the extrapolation models that can be built from the scale grid
# alone. The remaining factories in ``inference`` need extra parameters (a
# polynomial degree or a fixed asymptote) that a bare string cannot carry, so
# they are reachable only by constructing the factory directly.
_SUPPORTED_EXTRAPOLATION_METHODS = frozenset(
    {"exponential", "linear", "polynomial", "richardson"}
)


def construct_circuits(
    circuit: Circuit,
    /,
    *,
    noise_level: float,
    scale_factor: float,
    precision: float = 0.1,
    seed: int | None = None,
) -> list[Circuit]:
    r"""Sample variant circuits from one amplified quasi-probability expansion.

    Every non-measurement operation is expanded with
    :func:`~cqlib_algorithms.error_mitigation.quasiprobability.amplify_depolarizing`
    at the amplified level :math:`\epsilon' = \text{scale\_factor} \times
    \text{noise\_level}`, and ``num_samples`` variant circuits are drawn from
    the resulting non-negative weights. The weights sum to one, so no sign is
    carried and the family is recombined by a plain average.

    Args:
        circuit: Circuit to expand. Measurements are passed through unchanged and need
            no representation.
        noise_level: Baseline depolarizing strength of the device, in :math:`[0, 1)`.
        scale_factor: Amplification factor :math:`s > 0` applied to ``noise_level``.
        precision: Target precision :math:`\epsilon` of the estimate. The sample count is
            :math:`\lceil 1 / \epsilon^2 \rceil`.
        seed: Seed of the sampling stream, or ``None`` for fresh randomness.

    Returns:
        The sampled variant circuits, in draw order.

    Raises:
        TypeError: If ``precision`` is not a real number or ``seed`` is neither an
            integer nor ``None``.
        ValueError: If ``precision`` is not in :math:`(0, 1]`, if a gate acts on more than
            two qubits, or if the amplified level reaches one so that the
            expansion stops being a probability distribution.

    """
    _validate_precision(precision)
    _validate_seed(seed)
    _validate_two_qubit_operations(circuit)
    representations = _amplified_representations(
        circuit,
        noise_level,
        scale_factor,
    )

    # The amplified weights sum to one, hence Gamma = 1 and the sample count
    # does not carry the squared one-norm that PEC has to pay.
    num_samples = math.ceil(1.0 / float(precision) ** 2)
    rng = np.random.default_rng(seed)
    return [
        _sample_variant(circuit, representations, rng)[0] for _ in range(num_samples)
    ]


def combine_results(results: Sequence[EstimatorResult], /) -> float:
    r"""Average the expectation values of the variant circuits.

    The amplified weights are non-negative and sum to one, so the estimate is
    an ordinary mean with no :math:`\Gamma` compensation.

    Args:
        results: Expectation values measured on the circuits returned by
            :func:`construct_circuits`, in the same order.

    Returns:
        The amplified expectation value at one scale factor.

    Raises:
        ValueError: If no result was supplied.

    """
    if not results:
        raise ValueError(
            "probabilistic error amplification needs at least one result to average."
        )
    return float(np.mean([result.expectation for result in results]))


def execute_with_pea(
    circuit: StatePreparation | Circuit,
    executor: QemExecutor,
    observable: Hamiltonian | PauliString,
    /,
    *,
    noise_level: float,
    scale_factors: Sequence[float],
    extrapolation_method: str,
    precision: float = 0.1,
    shots: int | None = None,
    seed: int | None = None,
    parameter_bindings: Mapping[str, float] | None = None,
    prefix: str = "qem",
) -> float:
    r"""Estimate one observable on ``circuit`` with probabilistic error amplification.

    This is the immediate form of :func:`pea_executor`. The same task
    can be evaluated by calling ``run()`` on that executor.

    Args:
        circuit: State-preparation component or circuit to evaluate.
        executor: Execution backend supplying the expectation surface.
        observable: Observable to estimate. A bare Pauli string is interpreted as a
            Hamiltonian with unit coefficient.
        noise_level: Baseline depolarizing strength of the device, in :math:`[0, 1)`.
        scale_factors: Amplification factors :math:`s > 0` at which the circuit is measured.
        extrapolation_method: Name of the extrapolation model that reduces the amplified values to
            the zero-amplification limit. Supported names are ``"richardson"``,
            ``"linear"``, ``"polynomial"``, and ``"exponential"``.
        precision: Target precision :math:`\epsilon` of every amplified estimate.
        shots: Shot count for every variant circuit, or ``None`` for exact
            evaluation.
        seed: Seed of the sampling stream, or ``None`` for fresh randomness.
        parameter_bindings: Finite values covering exactly all active symbols.
            Values are bound before circuit transformations; None requires no symbols.
        prefix: Build prefix for a preparation component, defaulting to ``qem``;
            existing Circuit symbols keep their names.

    Returns:
        Mitigated expectation value.

    Raises:
        TypeError: If an input or configuration argument has the wrong type.
        ValueError: If bindings are non-finite or do not cover active symbols exactly,
            or if ``extrapolation_method`` is not a supported name, if ``precision``
            is not in :math:`(0, 1]`, if a parameter describes an unusable
            configuration, or if ``noise_level * max(scale_factors)`` reaches one.
    """
    wrapped = pea_executor(
        executor,
        noise_level=noise_level,
        scale_factors=scale_factors,
        extrapolation_method=extrapolation_method,
        precision=precision,
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


def pea_executor(
    executor: QemExecutor,
    /,
    *,
    noise_level: float,
    scale_factors: Sequence[float],
    extrapolation_method: str,
    precision: float = 0.1,
    shots: int | None = None,
    seed: int | None = None,
) -> QemExecutor:
    r"""Return an executor with probabilistic error amplification applied.

    The returned executor keeps the wrapped executor's sampling surface and
    caching flag, so PEA can be chained on top of another technique without
    changing either semantics.

    Args:
        executor: Execution backend to wrap.
        noise_level: Baseline depolarizing strength of the device, in :math:`[0, 1)`.
        scale_factors: Amplification factors :math:`s > 0` at which every later circuit is
            measured.
        extrapolation_method: Name of the extrapolation model that reduces the amplified values to
            the zero-amplification limit. Supported names are ``"richardson"``,
            ``"linear"``, ``"polynomial"``, and ``"exponential"``.
        precision: Target precision :math:`\epsilon` of every amplified estimate.
        shots: Shot count bound to every request, or ``None`` to defer to the
            request's own shot count.
        seed: Seed of the sampling stream, or ``None`` for fresh randomness.

    Returns:
        Executor whose expectation surface applies probabilistic error
            amplification.

    Raises:
        TypeError: If ``extrapolation_method`` is not a string, if ``precision`` is not
            a real number, or if ``seed`` is neither an integer nor ``None``.
        ValueError: If ``extrapolation_method`` is not a supported name, if ``precision``
            is not in :math:`(0, 1]`, or if ``noise_level * max(scale_factors)``
            reaches one.

    """
    estimator = _PeaEstimator(
        executor,
        noise_level=noise_level,
        scale_factors=tuple(float(factor) for factor in scale_factors),
        extrapolation_method=extrapolation_method,
        precision=precision,
        shots=shots,
        seed=seed,
    )
    return QemExecutor(
        estimator=estimator,
        sampler=executor.sampler,
        use_cache=executor.use_cache,
    )


class _PeaEstimator(Estimator):
    r"""Expectation surface that amplifies, measures, and extrapolates.

    Args:
        executor: Execution backend supplying the expectation surface for each variant
            circuit.
        noise_level: Baseline depolarizing strength of the device.
        scale_factors: Amplification factors at which every request is measured.
        extrapolation_method: Normalized name of the extrapolation model.
        precision: Target precision of every amplified estimate.
        shots: Shot count bound to every request, or ``None`` to defer to the
            request.
        seed: Seed of the sampling stream, or ``None`` for fresh randomness.

    """

    __slots__ = (
        "_executor",
        "_extrapolation_method",
        "_noise_level",
        "_precision",
        "_scale_factors",
        "_seed",
        "_shots",
    )

    _executor: QemExecutor
    _extrapolation_method: str
    _noise_level: float
    _precision: float
    _scale_factors: tuple[float, ...]
    _seed: int | None
    _shots: int | None

    def __init__(
        self,
        executor: QemExecutor,
        /,
        *,
        noise_level: float,
        scale_factors: tuple[float, ...],
        extrapolation_method: str,
        precision: float,
        shots: int | None,
        seed: int | None,
    ) -> None:
        _validate_precision(precision)
        _validate_seed(seed)
        _validate_extrapolation_method(extrapolation_method)
        _validate_amplification(noise_level, scale_factors)
        self._executor = executor
        self._noise_level = float(noise_level)
        self._scale_factors = scale_factors
        self._extrapolation_method = extrapolation_method.strip().lower()
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
        r"""Amplify, measure, and extrapolate one request."""
        bound_circuit = bind_request_circuit(request)
        shots = self._shots if self._shots is not None else request.shots
        seed = self._seed if self._seed is not None else request.seed
        # Each scale factor draws its own amplified variants and measures them
        # independently, so one run seed still reproduces the whole family.
        scale_seeds = spawn_seeds(seed, len(self._scale_factors))
        amplified: list[float] = []
        for scale_factor, scale_seed in zip(self._scale_factors, scale_seeds):
            variants = construct_circuits(
                bound_circuit,
                noise_level=self._noise_level,
                scale_factor=scale_factor,
                precision=self._precision,
                seed=scale_seed,
            )
            variant_seeds = spawn_seeds(scale_seed, len(variants))
            amplified.append(
                combine_results(
                    self._executor.estimate(
                        [
                            EstimatorRequest(
                                circuit=variant,
                                observable=request.observable,
                                shots=shots,
                                seed=variant_seeds[index],
                            )
                            for index, variant in enumerate(variants)
                        ]
                    )
                )
            )
        # A factory accumulates the points pushed onto it, so a private one is
        # built per request; sharing an instance would mix requests together.
        factory = _make_factory(
            self._extrapolation_method,
            self._scale_factors,
        )
        for scale_factor, value in zip(
            self._scale_factors,
            amplified,
            strict=True,
        ):
            factory.push(scale_factor, value)
        return EstimatorResult(expectation=factory.reduce())


def _amplified_representations(
    circuit: Circuit,
    noise_level: float,
    scale_factor: float,
    /,
) -> list[QuasiprobabilityRepresentation]:
    r"""Return one amplified representation per non-measurement operation.

    Args:
        circuit: Circuit whose operations are expanded.
        noise_level: Baseline depolarizing strength of the device.
        scale_factor: Amplification factor applied to ``noise_level``.

    Returns:
        One amplified expansion per non-measurement operation, in circuit
            order.

    """
    representations = []
    for operation in circuit.operations:
        if is_measurement(operation):
            continue
        representations.append(
            amplify_depolarizing(
                _storage_instruction(operation),
                noise_level,
                scale_factor,
            )
        )
    return representations


def _make_factory(
    extrapolation_method: str,
    scale_factors: Sequence[float],
    /,
) -> Factory:
    r"""Build the extrapolation factory named by ``extrapolation_method``.

    The polynomial model is fitted at the highest degree the grid supports,
    which makes it an exact interpolant of the measured points.
    """
    name = extrapolation_method.strip().lower()
    if name == "richardson":
        return RichardsonFactory(scale_factors)
    if name == "linear":
        return LinearFactory(scale_factors)
    if name == "polynomial":
        return PolyFactory(scale_factors, degree=len(scale_factors) - 1)
    return ExpFactory(scale_factors)


def _validate_extrapolation_method(extrapolation_method: str, /) -> None:
    r"""Validate an extrapolation-method name."""
    if not isinstance(extrapolation_method, str):
        raise TypeError("extrapolation_method must be a string")
    if extrapolation_method.strip().lower() not in _SUPPORTED_EXTRAPOLATION_METHODS:
        supported = ", ".join(sorted(_SUPPORTED_EXTRAPOLATION_METHODS))
        raise ValueError(
            f"extrapolation_method {extrapolation_method!r} is unknown; "
            f"supported names are {supported}."
        )


def _validate_amplification(
    noise_level: float,
    scale_factors: Sequence[float],
    /,
) -> None:
    r"""Validate a noise level and amplification grid.

    The amplified level :math:`s\,\epsilon` must stay below one for the
    expansion to remain a probability distribution, so the product of the
    baseline level and the largest scale factor is checked once here.
    """
    if isinstance(noise_level, bool) or not isinstance(noise_level, Real):
        raise TypeError("noise_level must be a real number")
    if not np.isfinite(noise_level) or noise_level < 0.0:
        raise ValueError(
            f"noise_level must be finite and non-negative but was {noise_level!r}."
        )
    if not scale_factors:
        raise ValueError("scale_factors must contain at least one factor.")
    for factor in scale_factors:
        if isinstance(factor, bool) or not isinstance(factor, Real):
            raise TypeError("every scale factor must be a real number")
        if not np.isfinite(factor) or factor <= 0.0:
            raise ValueError(
                f"every scale factor must be finite and positive but one was "
                f"{factor!r}."
            )
    if float(noise_level) * max(float(factor) for factor in scale_factors) >= 1.0:
        raise ValueError(
            "input noise violates the condition: noise_level * "
            "max(scale_factors) must be < 1 for the amplification weights to "
            "stay non-negative, see Kim et al. 2023, "
            "doi:10.1038/s41586-023-06096-3, in detail; lower the baseline "
            "noise level or the largest scale factor."
        )
