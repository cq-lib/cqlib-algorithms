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

r"""Readout-error mitigation.

Measurement noise acts on the readout record rather than on the circuit: a
prepared computational basis state :math:`|x\rangle` is reported as
:math:`|y\rangle` with probability :math:`A_{yx}`, so the measured
distribution relates to the ideal one by :math:`p_\text{obs} = A
p_\text{true}` and the correction is :math:`p_\text{true} = A^{-1}
p_\text{obs}`. When the readout errors are uncorrelated the stochastic matrix
factorises into per-qubit blocks, :math:`A = A_{n-1} \otimes \cdots \otimes
A_0`, which makes calibration linear in the qubit count.

Because the circuit is left untouched and only the counts are post-processed,
the technique consumes the sampling surface and can only recombine
observables diagonal in the :math:`Z` basis (:math:`Z` products and the
identity).

Three components make up the module: :func:`calibrate_readout` estimates the
per-qubit flip rates, :func:`mitigate_counts` inverts the confusion matrix and
returns a valid probability vector, and :func:`execute_with_rem` measures the
counts and recombines the expectation value. The first two are independent of
any executor and can be exercised on their own.

References
----------
[1] F. B. Maciejewski, Z. Zimboras, and M. Oszmaniec, "Mitigation of readout noise in
    near-term quantum devices by classical post-processing based on detector
    tomography," Quantum 4, 257, 2020. arXiv:1907.08518.
[2] S. Bravyi, S. Sheldon, A. Kandala, D. C. McKay, and J. M. Gambetta, "Mitigating
    measurement errors in multiqubit experiments," Physical Review A 103, 042605, 2021.
    arXiv:2006.14044.
[3] J. Duchi, S. Shalev-Shwartz, Y. Singer, and T. Chandra, "Efficient projections onto
    the l1-ball for learning in high dimensions," Proceedings of the 25th International
    Conference on Machine Learning (ICML), pp. 272-279, 2008.
    doi:10.1145/1390156.1390191.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import reduce

import numpy as np

from cqlib import Circuit
from cqlib.device import ExecutionResult, ReadoutError
from cqlib.qis import Hamiltonian, PauliString

from ...protocols import (
    Estimator,
    EstimatorRequest,
    EstimatorResult,
    SamplerRequest,
    spawn_seeds,
)
from ..core import QemExecutor
from ..inference import ConvergenceWarning

# Assembling the full 2**n inverse costs O(4**n) memory, which stops being
# practical past this qubit count; larger systems take the tensored route that
# contracts one 2x2 factor per qubit instead.
_FULL_MATRIX_MAX_QUBITS = 10

# Condition number of the confusion matrix above which inverting it amplifies
# the measurement noise more than an order of magnitude per decade of readout
# error. Reaching it usually means at least one qubit is close to completely
# depolarized readout.
_ILL_CONDITIONED_LIMIT = 1.0e4


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    r"""Per-qubit readout confusion data.

    Args:
        per_qubit: One :class:`cqlib.device.ReadoutError` per qubit, ordered by qubit
            index, each holding the two single-qubit flip rates of that qubit.

    Attributes:
        p0: Tuple of :math:`P(\text{read } 1 \mid \text{prepared } 0)` per qubit.
        p1: Tuple of :math:`P(\text{read } 0 \mid \text{prepared } 1)` per qubit.

    """

    per_qubit: tuple[ReadoutError, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.per_qubit, tuple):
            raise TypeError("per_qubit must be a tuple of ReadoutError values")
        if not self.per_qubit:
            raise ValueError("per_qubit must contain at least one ReadoutError")
        if any(not isinstance(item, ReadoutError) for item in self.per_qubit):
            raise TypeError("per_qubit must contain only ReadoutError values")

    @property
    def p0(self) -> tuple[float, ...]:
        r"""Return :math:`P(\text{read } 1 \mid \text{prepared } 0)` per qubit."""
        return tuple(error.p_1_given_0 for error in self.per_qubit)

    @property
    def p1(self) -> tuple[float, ...]:
        r"""Return :math:`P(\text{read } 0 \mid \text{prepared } 1)` per qubit."""
        return tuple(error.p_0_given_1 for error in self.per_qubit)


def calibrate_readout(
    executor: QemExecutor,
    num_qubits: int,
    /,
    *,
    shots: int,
    seed: int | None = None,
) -> ConfusionMatrix:
    r"""Estimate per-qubit readout flip rates from two calibration circuits.

    The all-zero and the all-one basis state are prepared and measured, and
    each qubit's marginal flip rate is read off its own bit. Both extremes are
    needed because they expose opposite flips: an all-zero preparation only
    shows :math:`0 \to 1` transitions and an all-one preparation only shows
    :math:`1 \to 0` transitions. Only the marginals are taken, since the
    uncorrelated model needs the single-qubit columns alone.

    Args:
        executor: Execution backend supplying the sampling surface.
        num_qubits: Number of qubits to calibrate.
        shots: Shot count for each of the two calibration circuits.
        seed: Seed of the two calibration measurements, or ``None`` for fresh
            randomness. Each calibration circuit draws from its own child seed, so
            one value reproduces the whole calibration.

    Returns:
        Estimated per-qubit readout errors.

    Raises:
        TypeError: If ``num_qubits`` or ``shots`` is not an integer, or if ``seed`` is
            neither an integer nor ``None``.
        ValueError: If ``num_qubits`` or ``shots`` is not positive, or if the executor
            holds no sampling backend.

    """
    if isinstance(num_qubits, bool) or not isinstance(num_qubits, int):
        raise TypeError("num_qubits must be an integer")
    if num_qubits <= 0:
        raise ValueError("num_qubits must be greater than zero")
    if isinstance(shots, bool) or not isinstance(shots, int):
        raise TypeError("shots must be an integer")
    if shots <= 0:
        raise ValueError("shots must be greater than zero")
    _validate_seed(seed)

    zero_circuit = Circuit(num_qubits)
    one_circuit = Circuit(num_qubits)
    for qubit in range(num_qubits):
        one_circuit.x(qubit)

    # The two calibration circuits are measured independently, so each takes its
    # own child seed while the calibration as a whole stays reproducible.
    zero_seed, one_seed = spawn_seeds(seed, 2)
    prepared = executor.sample(
        [
            SamplerRequest(
                circuit=zero_circuit,
                qubits=tuple(zero_circuit.qubits),
                shots=shots,
                seed=zero_seed,
            ),
            SamplerRequest(
                circuit=one_circuit,
                qubits=tuple(one_circuit.qubits),
                shots=shots,
                seed=one_seed,
            ),
        ]
    )

    # ReadoutError takes the false-negative rate first, so the mixed-up
    # preparation supplies p_0_given_1 and the all-zero one supplies
    # p_1_given_0; passing them by keyword keeps the directions explicit.
    per_qubit = tuple(
        ReadoutError(
            p_0_given_1=1.0 - _marginal_ones(prepared[1], index, num_qubits),
            p_1_given_0=_marginal_ones(prepared[0], index, num_qubits),
        )
        for index in range(num_qubits)
    )
    return ConfusionMatrix(per_qubit=per_qubit)


def mitigate_counts(
    counts: Mapping[str, int],
    confusion: ConfusionMatrix,
    /,
    *,
    post_select: bool = False,
) -> np.ndarray:
    r"""Correct a measured count distribution for readout error.

    Args:
        counts: Measured counts keyed by little-endian bitstring, the rightmost
            character being qubit 0. Every key must be exactly
            ``len(confusion.per_qubit)`` characters long.
        confusion: Per-qubit readout errors describing the noise to invert.
        post_select: Repair strategy for a corrected vector that leaves the probability
            simplex. ``True`` screens out the negative entries and renormalises
            the surviving mass; ``False`` projects the vector onto the simplex in
            the Euclidean norm.

    Returns:
        Corrected probability vector of length :math:`2^n`, indexed by the
            integer value of the bitstring, so bit ``i`` of an index belongs to
            qubit ``i``. Entries are non-negative and sum to one.

    Raises:
        TypeError: If ``counts`` is not a mapping or ``confusion`` is not a
            :class:`ConfusionMatrix`.
        ValueError: If a count key has the wrong width or is not binary, if a count is
            negative, if no shot was recorded, or if ``post_select`` screening
            leaves no mass to normalise.

    Warns:
        ConvergenceWarning: If the confusion matrix condition number exceeds
            :data:`_ILL_CONDITIONED_LIMIT`, in which case the correction amplifies
            measurement noise.
    """
    if not isinstance(confusion, ConfusionMatrix):
        raise TypeError("confusion must be a ConfusionMatrix")
    if not isinstance(counts, Mapping):
        raise TypeError("counts must be a mapping from bitstring to count")

    num_qubits = len(confusion.per_qubit)
    observed = np.zeros(1 << num_qubits, dtype=float)
    for bitstring, count in counts.items():
        if not isinstance(bitstring, str) or len(bitstring) != num_qubits:
            raise ValueError(
                "count keys must be little-endian bitstrings of length "
                f"{num_qubits}"
            )
        if any(bit not in "01" for bit in bitstring):
            raise ValueError("count keys must be binary strings")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("counts must be non-negative integers")
        observed[int(bitstring, 2)] += count

    recorded = float(observed.sum())
    if recorded <= 0.0:
        raise ValueError("counts must contain at least one recorded shot")
    observed /= recorded

    _warn_if_ill_conditioned(confusion)
    corrected = _apply_inverse_confusion(observed, confusion)
    if post_select:
        return _post_select(corrected)
    return _project_to_simplex(corrected)


def execute_with_rem(
    circuit: Circuit,
    executor: QemExecutor,
    observable: Hamiltonian | PauliString,
    /,
    *,
    shots: int,
    confusion: ConfusionMatrix | None = None,
    p0: float = 0.01,
    p1: float = 0.01,
    post_select: bool = False,
    seed: int | None = None,
) -> float:
    r"""Estimate one observable on ``circuit`` with readout-error mitigation.

    This is the immediate form of :func:`rem_executor`::

        execute_with_rem(c, ex, obs, shots=s)
        == rem_executor(ex, shots=s).estimate(
               [EstimatorRequest(c, obs, shots=s)]
           )[0].expectation

    Args:
        circuit: Circuit whose expectation value is to be mitigated.
        executor: Execution backend supplying the sampling surface.
        observable: Observable to estimate. It must be diagonal in the :math:`Z` basis,
            that is, a sum of :math:`Z` and identity products.
        shots: Shot count for the sampled circuit.
        confusion: Measured per-qubit readout errors, or ``None`` to synthesize the same
            single-qubit model from ``p0`` and ``p1`` on every qubit.
        p0: :math:`P(\text{read } 1 \mid \text{prepared } 0)`, used only when
            ``confusion`` is ``None``.
        p1: :math:`P(\text{read } 0 \mid \text{prepared } 1)`, used only when
            ``confusion`` is ``None``.
        post_select: Repair strategy for a corrected vector that leaves the simplex.
        seed: Seed of the sampled circuit, or ``None`` for fresh randomness.

    Returns:
        Mitigated expectation value.

    Raises:
        TypeError: If ``seed`` is neither an integer nor ``None``.
        ValueError: If ``observable`` is not diagonal in the :math:`Z` basis, if
            ``confusion`` does not cover the observable's qubits, or if the
            executor holds no sampling backend.

    """
    wrapped = rem_executor(
        executor,
        shots=shots,
        confusion=confusion,
        p0=p0,
        p1=p1,
        post_select=post_select,
        seed=seed,
    )
    request = EstimatorRequest(circuit=circuit, observable=observable, shots=shots)
    return wrapped.estimate([request])[0].expectation


def rem_executor(
    executor: QemExecutor,
    /,
    *,
    shots: int,
    confusion: ConfusionMatrix | None = None,
    p0: float = 0.01,
    p1: float = 0.01,
    post_select: bool = False,
    seed: int | None = None,
) -> QemExecutor:
    r"""Return an executor with readout-error mitigation applied.

    The returned executor keeps the wrapped executor's sampling surface and
    caching flag, so REM can be chained on top of another technique. Readout
    correction consumes counts and produces an expectation value, so it
    replaces the expectation surface while leaving sampling untouched.

    Args:
        executor: Execution backend to wrap.
        shots: Shot count bound to every request.
        confusion: Measured per-qubit readout errors, or ``None`` to synthesize the same
            single-qubit model from ``p0`` and ``p1`` on every qubit.
        p0: :math:`P(\text{read } 1 \mid \text{prepared } 0)`, used only when
            ``confusion`` is ``None``.
        p1: :math:`P(\text{read } 0 \mid \text{prepared } 1)`, used only when
            ``confusion`` is ``None``.
        post_select: Repair strategy for a corrected vector that leaves the simplex.
        seed: Seed of the sampled circuit, or ``None`` to defer to the request's own
            seed.

    Returns:
        Executor whose expectation surface applies readout-error mitigation.

    Raises:
        TypeError: If ``seed`` is neither an integer nor ``None``.

    """
    _validate_seed(seed)
    estimator = _RemEstimator(
        executor,
        shots=shots,
        confusion=confusion,
        p0=p0,
        p1=p1,
        post_select=post_select,
        seed=seed,
    )
    return QemExecutor(
        estimator=estimator,
        sampler=executor.sampler,
        use_cache=executor.use_cache,
    )


class _RemEstimator(Estimator):
    r"""Expectation surface that samples counts and corrects readout error.

    Args:
        executor: Execution backend supplying the sampling surface.
        shots: Shot count bound to every request.
        confusion: Measured per-qubit readout errors, or ``None`` to synthesize the same
            single-qubit model from ``p0`` and ``p1`` on every qubit.
        p0: :math:`P(\text{read } 1 \mid \text{prepared } 0)` of the synthesized
            model.
        p1: :math:`P(\text{read } 0 \mid \text{prepared } 1)` of the synthesized
            model.
        post_select: Repair strategy for a corrected vector that leaves the simplex.
        seed: Seed of the sampled circuit, or ``None`` to defer to the request's own
            seed.

    """

    __slots__ = (
        "_confusion",
        "_executor",
        "_p0",
        "_p1",
        "_post_select",
        "_seed",
        "_shots",
    )

    _confusion: ConfusionMatrix | None
    _executor: QemExecutor
    _p0: float
    _p1: float
    _post_select: bool
    _seed: int | None
    _shots: int

    def __init__(
        self,
        executor: QemExecutor,
        /,
        *,
        shots: int,
        confusion: ConfusionMatrix | None,
        p0: float,
        p1: float,
        post_select: bool,
        seed: int | None,
    ) -> None:
        self._executor = executor
        self._shots = shots
        self._confusion = confusion
        self._p0 = p0
        self._p1 = p1
        self._post_select = post_select
        self._seed = seed

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        r"""Mitigate every request and return results in input order."""
        return [self._mitigate(request) for request in requests]

    def _mitigate(self, request: EstimatorRequest) -> EstimatorResult:
        r"""Measure one request's counts, correct them, and recombine."""
        observable = request.observable
        _validate_z_diagonal(observable)

        num_qubits = observable.num_qubits
        confusion = self._confusion
        if confusion is None:
            confusion = _uniform_confusion(self._p0, self._p1, num_qubits)
        elif len(confusion.per_qubit) != num_qubits:
            raise ValueError(
                "the confusion matrix covers "
                f"{len(confusion.per_qubit)} qubit(s) but the observable acts "
                f"on {num_qubits}"
            )

        sampled = self._executor.sample(
            [
                SamplerRequest(
                    circuit=request.circuit,
                    qubits=tuple(request.circuit.qubits),
                    shots=self._shots,
                    seed=self._seed if self._seed is not None else request.seed,
                )
            ]
        )[0]
        corrected = mitigate_counts(
            sampled.counts,
            confusion,
            post_select=self._post_select,
        )

        # Corrected entry i is the probability of the basis state whose
        # bitstring reads as i, so padding the index to num_qubits characters
        # produces the little-endian key the expectation expects.
        distribution = {
            format(index, f"0{num_qubits}b"): float(corrected[index])
            for index in range(1 << num_qubits)
        }
        # A single computational-basis record serves every diagonal term, so
        # one all-Z measurement basis is enough for the whole observable.
        measurement = PauliString.from_str("Z" * num_qubits)
        expectation = observable.expectation_probs([(measurement, distribution)])
        return EstimatorResult(expectation=float(expectation))


def _marginal_ones(
    result: ExecutionResult,
    index: int,
    num_qubits: int,
) -> float:
    r"""Return the measured-one frequency of qubit ``index``.

    Args:
        result: Sampled counts record whose bitstrings are as wide as the circuit.
        index: Qubit whose marginal is extracted.
        num_qubits: Width of the recorded bitstrings.

    Returns:
        Fraction of shots in which qubit ``index`` was read as one.

    """
    ones = 0
    for bitstring, count in result.counts.items():
        # cqlib count keys are little-endian, so qubit q sits at position
        # num_qubits - 1 - q.
        if bitstring[num_qubits - 1 - index] == "1":
            ones += count
    return ones / result.shots


def _local_matrix(error: ReadoutError) -> np.ndarray:
    r"""Return one qubit's :math:`2 \times 2` confusion matrix.

    The matrix is indexed row-major by the observed bit and column-major by
    the prepared bit, matching :math:`A_{yx} = P(y \mid x)`.
    """
    p0 = error.p_1_given_0
    p1 = error.p_0_given_1
    return np.array([[1.0 - p0, p1], [p0, 1.0 - p1]], dtype=float)


def _local_inverse(error: ReadoutError) -> np.ndarray:
    r"""Return the pseudo-inverse of one qubit's confusion matrix.

    The pseudo-inverse is used rather than the inverse so that a qubit whose
    readout is fully depolarized degrades the correction instead of
    aborting it.
    """
    return np.linalg.pinv(_local_matrix(error))


def _apply_inverse_confusion(
    observed: np.ndarray,
    confusion: ConfusionMatrix,
) -> np.ndarray:
    r"""Return :math:`A^{-1} p` for the confusion data of ``confusion``."""
    inverses = [_local_inverse(error) for error in confusion.per_qubit]
    if len(inverses) > _FULL_MATRIX_MAX_QUBITS:
        return _apply_tensored_inverse(observed, inverses)

    # The factors are listed highest qubit first so that the assembled row
    # index has bit i belonging to qubit i, which is the convention shared by
    # the little-endian count keys and the returned vector.
    full = reduce(np.kron, inverses[::-1])
    return np.asarray(full @ observed)


def _apply_tensored_inverse(
    observed: np.ndarray,
    inverses: list[np.ndarray],
) -> np.ndarray:
    r"""Apply the per-qubit inverses axis by axis without forming :math:`2^n`.

    Args:
        observed: Probability vector indexed by bitstring value.
        inverses: Pseudo-inverse of each qubit's confusion matrix, in qubit order.

    Returns:
        Corrected probability vector in the same index convention.

    """
    num_qubits = len(inverses)
    tensored = observed.reshape((2,) * num_qubits)
    for qubit, inverse in enumerate(inverses):
        # Reshaping in C order makes the last axis qubit 0, so qubit q lives
        # on axis num_qubits - 1 - q.
        axis = num_qubits - 1 - qubit
        contracted = np.tensordot(inverse, tensored, axes=([1], [axis]))
        tensored = np.moveaxis(contracted, 0, axis)
    return tensored.reshape(-1)


def _post_select(corrected: np.ndarray) -> np.ndarray:
    r"""Screen out negative entries and renormalise the remaining mass.

    Raises:
        ValueError: If every entry is non-positive, leaving nothing to normalise.

    """
    screened = np.clip(corrected, 0.0, None)
    mass = float(screened.sum())
    if mass <= 0.0:
        raise ValueError(
            "post_select removed every entry of the corrected distribution, "
            "so it cannot be renormalised. This happens when A^-1 p_obs has "
            "no positive mass, which means the confusion matrix is not a "
            "faithful model of the readout noise; see Maciejewski 2020, "
            "arXiv:1907.08518, Eq. 25."
        )
    return np.asarray(screened / mass)


def _project_to_simplex(vector: np.ndarray) -> np.ndarray:
    r"""Project ``vector`` onto the probability simplex in the Euclidean norm.

    The projection is the sorted cumulative-sum threshold of Duchi et al.
    (2008), which runs in :math:`O(d \log d)` because only a sort dominates.
    It is the minimum Euclidean distance repair of a non-physical readout
    distribution described by Maciejewski 2020, arXiv:1907.08518, Eq. 25.
    """
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered)
    indices = np.arange(1, vector.size + 1)
    # The active set is the prefix of the sorted vector whose shifted values
    # stay positive; its size is the threshold denominator.
    positive = ordered - (cumulative - 1.0) / indices > 0.0
    rho = indices[positive][-1]
    threshold = (cumulative[rho - 1] - 1.0) / rho
    return np.asarray(np.clip(vector - threshold, 0.0, None))


def _warn_if_ill_conditioned(confusion: ConfusionMatrix) -> None:
    r"""Warn when inverting the confusion matrix amplifies readout noise.

    For a Kronecker product of square factors the two-norm condition number
    multiplies, so the assembled condition number is the product of the
    per-qubit ones and is available without forming the full matrix. A large
    value means at least one qubit is close to fully depolarized readout, and
    the pseudo-inverse will then amplify the shot noise along the weak
    direction.
    """
    condition = 1.0
    for error in confusion.per_qubit:
        condition *= float(np.linalg.cond(_local_matrix(error)))
    if condition > _ILL_CONDITIONED_LIMIT:
        warnings.warn(
            "The readout confusion matrix is ill-conditioned: its condition "
            f"number is {condition:.3e}, above the "
            f"{_ILL_CONDITIONED_LIMIT:.0e} threshold, so the correction "
            "amplifies measurement noise. Re-calibrate with more shots or on "
            "a less noisy device.",
            ConvergenceWarning,
            stacklevel=2,
        )


def _validate_z_diagonal(observable: Hamiltonian | PauliString) -> None:
    r"""Validate a Z-diagonal observable.
    """
    if isinstance(observable, PauliString):
        diagonal = observable.x_mask == 0
    else:
        diagonal = all(pauli.x_mask == 0 for pauli, _ in observable.terms)
    if not diagonal:
        raise ValueError(
            "The execution surface of this method only supports observables "
            "that are diagonal in the Z basis: every Pauli term must be a "
            "product of Z and identity, but an X or Y factor was found. "
            "Non-diagonal terms have no computational-basis expectation; see "
            "Bravyi 2021, arXiv:2006.14044, Sec. II."
        )


def _validate_seed(seed: int | None) -> None:
    r"""Validate an optional random seed."""
    if seed is None:
        return
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer or None")
    if seed < 0:
        raise ValueError("seed must be non-negative")


def _uniform_confusion(p0: float, p1: float, num_qubits: int) -> ConfusionMatrix:
    r"""Build the same single-qubit readout error for every qubit.

    Args:
        p0: :math:`P(\text{read } 1 \mid \text{prepared } 0)`.
        p1: :math:`P(\text{read } 0 \mid \text{prepared } 1)`.
        num_qubits: Number of qubits to build.

    Returns:
        Uniform per-qubit readout errors.

    Raises:
        ValueError: If either rate lies outside ``[0, 1]``.

    """
    error = ReadoutError(p_0_given_1=p1, p_1_given_0=p0)
    return ConfusionMatrix(per_qubit=tuple(error for _ in range(num_qubits)))
