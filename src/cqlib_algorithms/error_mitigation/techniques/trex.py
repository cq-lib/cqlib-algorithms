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

r"""Twirled readout error extinction.

TREX estimates the ideal expectation value of an observable without ever
building a readout confusion matrix. Immediately before the measurement a
random Pauli is applied on every qubit, and each recorded bit is flipped back
classically, so that averaged over the twirl an asymmetric readout error
becomes a symmetric single-parameter bit-flip. Calibration circuits prepare
:math:`|0\cdots0\rangle`, receive the same twirl, and expose that parameter;
dividing the twirled expectation value by the calibration eigenvalue removes
the readout bias.

Because the correction is a per-qubit sign factor applied to sampled counts,
the method never materialises a :math:`2^n \times 2^n` matrix, and its cost
grows linearly in the qubit count and the shot count.

:func:`construct_circuits` builds the twirled and calibration circuits and
:func:`combine_results` turns their counts into the mitigated value; the two
are independent. :func:`execute_with_trex` and :func:`trex_executor`
orchestrate them on an executor.

References
----------
.. [1] E. van den Berg, Z. K. Minev, and K. Temme, "Model-free readout-error
   mitigation for quantum expectation values," Physical Review A 105,
   032620, 2022. arXiv:2012.09738.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from cqlib.circuit import StandardGate, ValueOperation
from cqlib.device import ExecutionResult
from cqlib.qis import Hamiltonian, PauliString, Phase

from cqlib import Circuit

from ...protocols import (
    Estimator,
    EstimatorRequest,
    EstimatorResult,
    SamplerRequest,
    spawn_seeds,
)
from .._circuits import rebuild
from ..core import QemExecutor
from .rem import _validate_z_diagonal

# A bit of one in the twirl table puts this Pauli on the qubit before the
# measurement. Both X and Y flip the recorded Z-basis bit, so either axis
# serves as the twirl; X is used by default and the choice is statistically
# invisible from the Z-basis readout.
_TWIRL_PAULI = StandardGate.X

# Below this magnitude the calibration eigenvalue is dominated by shot noise,
# so dividing by it would amplify noise instead of removing readout bias.
_CALIBRATION_FLOOR = 1.0e-10


def construct_circuits(
    circuit: Circuit,
    /,
    *,
    num_randomizations: int = 32,
    seed: int | None = None,
) -> list[Circuit]:
    r"""Build the twirled and calibration circuits for one measurement.

    The returned list holds ``num_randomizations`` twirled copies of
    ``circuit`` followed by ``num_randomizations`` calibration circuits, so it
    is twice as long as the number of randomizations. The twirled copy and the
    calibration circuit at offsets ``i`` and ``i + num_randomizations`` share
    the Pauli draw of randomization ``i``: the first is ``circuit`` with a
    Pauli on every qubit whose randomization bit is one, and the second is the
    same draw applied to the all-zero state.

    The Pauli bits are drawn afresh from ``seed`` on every call. A caller that
    wants to recombine through :func:`combine_results` must therefore reproduce
    the same draw by passing the same ``seed``; the orchestrators in this
    module draw the table once and thread it through both stages instead, so
    they never depend on reproduction.

    Parameters
    ----------
    circuit
        Circuit whose expectation value is to be mitigated.
    num_randomizations
        Number of independent twirls to build, at least one.
    seed
        Seed of the twirl draw, or ``None`` for a fresh draw.

    Returns
    -------
    list[Circuit]
        ``num_randomizations`` twirled circuits followed by
        ``num_randomizations`` calibration circuits.

    Raises
    ------
    TypeError
        If ``num_randomizations`` is not an integer, or if ``seed`` is neither
        an integer nor ``None``.
    ValueError
        If ``num_randomizations`` is below one.
    """
    _validate_num_randomizations(num_randomizations)
    _validate_seed(seed)
    twirls = _draw_twirls(num_randomizations, circuit.num_qubits, seed)
    return _build_circuits(circuit, twirls, _TWIRL_PAULI)


def combine_results(
    results: Sequence[ExecutionResult],
    twirls: Sequence[Sequence[int]],
    /,
    *,
    observable: Hamiltonian | PauliString,
) -> float:
    r"""Mitigated expectation value from the twirled and calibration counts.

    The estimate recombines the two stages of the method. Writing
    :math:`f_k(D)` for the expectation of the :math:`k`-th Pauli term on the
    recorded distribution of a circuit after its bits are flipped back by the
    twirl, the mitigated value is

    .. math::

        \langle O \rangle = \sum_k c_k\, \frac{1}{R} \sum_{i=1}^{R}
            \frac{f_k(D_1, s_i)}{f_k(D_0, s_i)},

    where :math:`R` is the number of randomizations, :math:`c_k` the
    coefficient of term :math:`k`, :math:`D_1` the twirled data circuit,
    :math:`D_0` the calibration circuit, and :math:`s_i` the Pauli draw of
    randomization :math:`i`. The ratio is the readout eigenvalue the twirl
    symmetrizes the error down to. The observable must be diagonal in the
    :math:`Z` basis, the restriction every sampling surface imposes.

    Parameters
    ----------
    results
        Measured counts of the circuits returned by :func:`construct_circuits`,
        in that order: the ``R`` twirled results first, then the ``R``
        calibration results.
    twirls
        The ``R`` per-randomization bit tables, each holding one bit per
        qubit, in the same order used to build the circuits.
    observable
        Observable to estimate. It must be diagonal in the :math:`Z` basis,
        that is, a sum of :math:`Z` and identity products.

    Returns
    -------
    float
        Mitigated expectation value.

    Raises
    ------
    TypeError
        If an entry of ``results`` is not a :class:`cqlib.device.ExecutionResult`.
    ValueError
        If ``observable`` is not diagonal in the :math:`Z` basis, if ``twirls``
        is empty or a row has the wrong width, or if ``results`` does not hold
        exactly two entries per twirl.
    """
    _validate_z_diagonal(observable)
    num_randomizations = len(twirls)
    if num_randomizations < 1:
        raise ValueError(
            "trex needs at least one randomization to recombine."
        )
    if len(results) != 2 * num_randomizations:
        raise ValueError(
            "trex needs one twirled and one calibration result per "
            f"randomization, so {2 * num_randomizations} results in total, "
            f"but got {len(results)}."
        )
    for result in results:
        if not isinstance(result, ExecutionResult):
            raise TypeError(
                "results must contain cqlib.device.ExecutionResult values"
            )

    hamiltonian = _as_hamiltonian(observable)
    num_qubits = hamiltonian.num_qubits
    _validate_twirls(twirls, num_qubits)

    # The twirl only relabels computational-basis outcomes, so a single all-Z
    # measurement basis serves every diagonal term of the observable.
    measurement = PauliString.from_str("Z" * num_qubits)
    twirled_distributions = [
        _xor_distribution(results[i], twirls[i], num_qubits)
        for i in range(num_randomizations)
    ]
    calibration_distributions = [
        _xor_distribution(
            results[num_randomizations + i], twirls[i], num_qubits
        )
        for i in range(num_randomizations)
    ]

    total = 0.0j
    for pauli, coefficient in hamiltonian.terms:
        # The measurement carries the bare Pauli, so its expectation is the
        # readout parity alone; the term's overall sign is applied once below,
        # folded into the scalar weight, so no sign is counted twice.
        bare = pauli.copy()
        bare.phase = Phase.plus()
        single = Hamiltonian.from_pauli(bare)
        scalar = complex(coefficient) * complex(pauli.phase.to_complex())
        corrected = 0.0
        for i in range(num_randomizations):
            # The correction is a per-qubit sign ratio over sampled counts. No
            # 2**n readout matrix is ever built or inverted here, so the cost of
            # the recombination stays linear in the qubit count.
            twirled = single.expectation_probs(
                [(measurement, twirled_distributions[i])]
            )
            calibration = single.expectation_probs(
                [(measurement, calibration_distributions[i])]
            )
            if abs(calibration) > _CALIBRATION_FLOOR:
                corrected += twirled / calibration
            else:
                # A vanishing eigenvalue means the readout is nearly fully
                # depolarized, where division amplifies shot noise; the raw
                # twirled value is used as the more stable fallback.
                corrected += twirled
        total += scalar * corrected / num_randomizations
    return float(total.real)


def execute_with_trex(
    circuit: Circuit,
    executor: QemExecutor,
    observable: Hamiltonian | PauliString,
    /,
    *,
    shots: int,
    num_randomizations: int = 32,
    seed: int | None = None,
) -> float:
    r"""Estimate one observable on ``circuit`` with readout twirling.

    This is the immediate form of :func:`trex_executor`::

        execute_with_trex(c, ex, obs, shots=s)
        == trex_executor(ex, shots=s).estimate(
               [EstimatorRequest(c, obs, shots=s)]
           )[0].expectation

    Parameters
    ----------
    circuit
        Circuit whose expectation value is to be mitigated.
    executor
        Execution backend supplying the sampling surface.
    observable
        Observable to estimate. It must be diagonal in the :math:`Z` basis.
    shots
        Shot count for every twirled and calibration circuit.
    num_randomizations
        Number of independent twirls to average over.
    seed
        Seed of the twirl draw and the measurements, or ``None`` for fresh
        randomness.

    Returns
    -------
    float
        Mitigated expectation value.

    Raises
    ------
    TypeError
        If ``shots`` or ``num_randomizations`` is not an integer, or if
        ``seed`` is neither an integer nor ``None``.
    ValueError
        If ``shots`` or ``num_randomizations`` is below one, or if
        ``observable`` is not diagonal in the :math:`Z` basis.
    """
    wrapped = trex_executor(
        executor,
        shots=shots,
        num_randomizations=num_randomizations,
        seed=seed,
    )
    request = EstimatorRequest(circuit=circuit, observable=observable, shots=shots)
    return wrapped.estimate([request])[0].expectation


def trex_executor(
    executor: QemExecutor,
    /,
    *,
    shots: int,
    num_randomizations: int = 32,
    seed: int | None = None,
) -> QemExecutor:
    r"""Return an executor with readout twirling applied.

    The returned executor keeps the wrapped executor's sampling surface and
    caching flag, so readout twirling can be chained on top of another
    technique. Readout twirling consumes counts and produces an expectation
    value, so it replaces the expectation surface while leaving sampling
    untouched.

    Parameters
    ----------
    executor
        Execution backend to wrap.
    shots
        Shot count bound to every twirled and calibration circuit.
    num_randomizations
        Number of independent twirls averaged for every request.
    seed
        Seed of the twirl draw and the measurements, or ``None`` to defer to
        the request's own seed.

    Returns
    -------
    QemExecutor
        Executor whose expectation surface applies readout twirling.

    Raises
    ------
    TypeError
        If ``shots`` or ``num_randomizations`` is not an integer, or if
        ``seed`` is neither an integer nor ``None``.
    ValueError
        If ``shots`` or ``num_randomizations`` is below one.
    """
    estimator = _TrexEstimator(
        executor,
        shots=shots,
        num_randomizations=num_randomizations,
        seed=seed,
    )
    return QemExecutor(
        estimator=estimator,
        sampler=executor.sampler,
        use_cache=executor.use_cache,
    )


class _TrexEstimator(Estimator):
    r"""Expectation surface that twirls the readout and divides by calibration.

    Parameters
    ----------
    executor
        Execution backend supplying the sampling surface.
    shots
        Shot count bound to every twirled and calibration circuit.
    num_randomizations
        Number of independent twirls averaged for every request.
    seed
        Seed of the twirl draw and the measurements, or ``None`` to defer to
        the request's own seed.
    """

    __slots__ = ("_executor", "_num_randomizations", "_seed", "_shots")

    _executor: QemExecutor
    _num_randomizations: int
    _seed: int | None
    _shots: int

    def __init__(
        self,
        executor: QemExecutor,
        /,
        *,
        shots: int,
        num_randomizations: int,
        seed: int | None,
    ) -> None:
        _validate_shots(shots)
        _validate_num_randomizations(num_randomizations)
        _validate_seed(seed)
        self._executor = executor
        self._shots = shots
        self._num_randomizations = num_randomizations
        self._seed = seed

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        r"""Mitigate every request and return results in input order."""
        return [self._mitigate(request) for request in requests]

    def _mitigate(self, request: EstimatorRequest) -> EstimatorResult:
        r"""Sample the twirled and calibration circuits and recombine."""
        # Reject a non-diagonal observable before spending any shots on it.
        _validate_z_diagonal(request.observable)
        num_qubits = request.circuit.num_qubits
        seed = self._seed if self._seed is not None else request.seed
        # The table is drawn once and threaded through both stages, so the
        # circuits and the recombination always see the same twirl even when
        # the seed is None.
        twirls = _draw_twirls(self._num_randomizations, num_qubits, seed)
        circuits = _build_circuits(request.circuit, twirls, _TWIRL_PAULI)
        # Each twirled and calibration circuit is sampled independently, so it
        # takes its own child seed while one run seed still reproduces the
        # whole randomized estimate.
        seeds = spawn_seeds(seed, len(circuits))
        results = self._executor.sample(
            [
                SamplerRequest(
                    circuit=circuit,
                    qubits=tuple(circuit.qubits),
                    shots=self._shots,
                    seed=seeds[index],
                )
                for index, circuit in enumerate(circuits)
            ]
        )
        value = combine_results(results, twirls.tolist(), observable=request.observable)
        return EstimatorResult(expectation=value)


def _build_circuits(
    circuit: Circuit,
    twirls: Sequence[Sequence[int]] | np.ndarray,
    twirl_pauli: StandardGate,
) -> list[Circuit]:
    r"""Build the twirled circuits followed by the calibration circuits.

    Parameters
    ----------
    circuit
        Circuit whose expectation value is to be mitigated.
    twirls
        Per-randomization bit tables, each holding one bit per qubit.
    twirl_pauli
        Pauli placed on every qubit whose twirl bit is one.

    Returns
    -------
    list[Circuit]
        One twirled copy and one calibration circuit per twirl, twirled
        circuits first.
    """
    num_qubits = circuit.num_qubits
    twirled = [
        _apply_twirl(circuit, row, num_qubits, twirl_pauli) for row in twirls
    ]
    # A calibration circuit is the same twirl applied to the all-zero state, so
    # the readout error can be read off without any state preparation.
    calibration = [
        _apply_twirl(Circuit(num_qubits), row, num_qubits, twirl_pauli)
        for row in twirls
    ]
    return [*twirled, *calibration]


def _apply_twirl(
    circuit: Circuit,
    row: Sequence[int],
    num_qubits: int,
    twirl_pauli: StandardGate,
) -> Circuit:
    r"""Append one Pauli per set twirl bit, keeping the circuit's global phase.

    Parameters
    ----------
    circuit
        Circuit receiving the Paulis.
    row
        One bit per qubit; a set bit puts ``twirl_pauli`` on that qubit.
    num_qubits
        Number of qubits in ``row`` and in ``circuit``.
    twirl_pauli
        Pauli placed on every qubit whose twirl bit is one.

    Returns
    -------
    Circuit
        The rebuilt circuit with the twirl Paulis appended.
    """
    paulis: list[ValueOperation] = []
    for qubit in range(num_qubits):
        if int(row[qubit]) == 1:
            paulis.append(
                ValueOperation.from_standard_gate(
                    twirl_pauli, [circuit.qubits[qubit]]
                )
            )
    return rebuild(circuit, [*circuit.operations, *paulis])


def _draw_twirls(
    num_randomizations: int,
    num_qubits: int,
    seed: int | None,
) -> np.ndarray:
    r"""Draw one independent twirl bit per qubit per randomization.

    Parameters
    ----------
    num_randomizations
        Number of randomizations, one row per twirl.
    num_qubits
        Number of qubits, one column per qubit.
    seed
        Seed of the draw, or ``None`` for entropy.

    Returns
    -------
    numpy.ndarray
        An integer array of shape ``(num_randomizations, num_qubits)`` holding
        zeros and ones.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(num_randomizations, num_qubits))


def _xor_distribution(
    result: ExecutionResult,
    row: Sequence[int],
    num_qubits: int,
) -> dict[str, float]:
    r"""Flip the recorded bitstring of every set twirl bit and normalise.

    Parameters
    ----------
    result
        Sampled counts whose bitstrings are as wide as the circuit.
    row
        One bit per qubit; a set bit flips that qubit's recorded bit.
    num_qubits
        Width of the recorded bitstrings.

    Returns
    -------
    dict[str, float]
        The measured distribution after the classical un-twirl, keyed by
        little-endian bitstring.
    """
    # cqlib bitstrings are little-endian, so qubit q is bit q of the integer
    # value of the bitstring and the whole flip is one XOR with a bit mask.
    mask = 0
    for qubit in range(num_qubits):
        if int(row[qubit]) == 1:
            mask |= 1 << qubit

    shots = result.shots
    distribution: dict[str, float] = {}
    for bitstring, count in result.counts.items():
        index = int(bitstring, 2) ^ mask
        key = format(index, f"0{num_qubits}b")
        distribution[key] = distribution.get(key, 0.0) + count / shots
    return distribution


def _as_hamiltonian(observable: Hamiltonian | PauliString, /) -> Hamiltonian:
    r"""Wrap a bare Pauli string so every observable shares one code path."""
    if isinstance(observable, PauliString):
        return Hamiltonian.from_pauli(observable)
    return observable


def _validate_twirls(twirls: Sequence[Sequence[int]], num_qubits: int) -> None:
    r"""Reject a twirl table that does not cover every qubit per row."""
    for row in twirls:
        if len(row) != num_qubits:
            raise ValueError(
                "every twirl row must hold one bit per qubit, so "
                f"{num_qubits} bits, but got {len(row)}."
            )


def _validate_num_randomizations(num_randomizations: int, /) -> None:
    r"""Reject a variant count that is not a positive integer."""
    if isinstance(num_randomizations, bool) or not isinstance(
        num_randomizations, int
    ):
        raise TypeError("num_randomizations must be an integer")
    if num_randomizations < 1:
        raise ValueError(
            "num_randomizations must be at least one, but was "
            f"{num_randomizations!r}."
        )


def _validate_shots(shots: int, /) -> None:
    r"""Reject a shot count that is not a positive integer."""
    if isinstance(shots, bool) or not isinstance(shots, int):
        raise TypeError("shots must be an integer")
    if shots < 1:
        raise ValueError(f"shots must be at least one, but was {shots!r}.")


def _validate_seed(seed: int | None, /) -> None:
    r"""Reject a seed that is neither an integer nor ``None``."""
    if seed is None:
        return
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer or None")
