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

r"""Debiasing by qubit-label symmetrization.

DEB reduces a coherent, direction-dependent error by randomising the assignment
of logical qubits to physical qubits. A device error attached to a physical
qubit shows up, from the point of view of a fixed logical observable, as a bias
pointing along one direction. Drawing several random permutations of the qubit
labels and applying each permutation to both the circuit and the observable
leaves the estimated physical quantity unchanged while rotating the mismatch
between the fixed error and the observable. Aggregating the per-permutation
estimates then cancels the part of the bias that the average no longer shares.

Two aggregations are available. ``averaging`` returns the mean of the
per-permutation expectation values and is the lower-variance choice.
``sharpening`` recombines the recorded bitstrings shot by shot, taking a
majority vote across permutations at each shot index before evaluating the
expectation value, and trades variance for a smaller residual bias.

The method consumes the sampling surface: it reads ``counts`` and can only
recombine observables that are diagonal in the :math:`Z` basis.

:func:`construct_circuits` builds the permuted circuits and observables and
:func:`combine_results` turns their counts into the mitigated value; the two
are independent. :func:`execute_with_deb` and :func:`deb_executor` orchestrate
them on an executor.

References
----------
.. [1] A. Maksymov, J. Nguyen, Y. Nam, and I. Markov, "Enhancing quantum
   computer performance via symmetrization," arXiv:2301.07233, 2023.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from cqlib.circuit import Qubit, ValueOperation
from cqlib.device import ExecutionResult
from cqlib.qis import Hamiltonian, PauliString

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

# The supported aggregations. Averaging and sharpening differ only in how the
# per-permutation counts are reduced, so the mode selects one reduction path.
_MODES = frozenset({"averaging", "sharpening"})


def construct_circuits(
    circuit: Circuit,
    observable: Hamiltonian | PauliString,
    /,
    *,
    num_variants: int,
    seed: int | None = None,
) -> tuple[list[Circuit], list[Hamiltonian | PauliString]]:
    r"""Build one permuted circuit and observable per random relabelling.

    Each variant is a full permutation :math:`\pi` of the qubit labels applied
    to the circuit and to the observable simultaneously. Writing :math:`q` for
    a logical qubit, the permutation sends it to the physical qubit
    :math:`\pi(q)`: the circuit's operation on :math:`q` is rebuilt on
    :math:`\pi(q)`, and the observable's Pauli on :math:`q` is rebuilt on
    :math:`\pi(q)`. The pair therefore estimates the same physical quantity as
    the unpermuted pair, which is the symmetry the aggregation exploits.

    Every permutation is drawn afresh from ``seed`` on each call. A caller
    that wants to recombine through :func:`combine_results` must reproduce the
    same draw by passing the same ``seed``; the orchestrators in this module
    draw the permutations once and thread them through both stages instead, so
    they never depend on reproduction.

    Parameters
    ----------
    circuit
        Circuit whose expectation value is to be mitigated.
    observable
        Observable to estimate, permuted in lockstep with the circuit. It must
        be diagonal in the :math:`Z` basis, that is, a sum of :math:`Z` and
        identity products.
    num_variants
        Number of independent permutations to build, at least one.
    seed
        Seed of the permutation draw, or ``None`` for a fresh draw.

    Returns
    -------
    tuple[list[Circuit], list[Hamiltonian | PauliString]]
        The ``num_variants`` permuted circuits and the matching permuted
        observables, aligned by index. A bare
        :class:`~cqlib.qis.PauliString` input yields permuted
        :class:`~cqlib.qis.PauliString` outputs and a
        :class:`~cqlib.qis.Hamiltonian` input yields permuted Hamiltonians.

    Raises
    ------
    TypeError
        If ``num_variants`` is not an integer, or if ``seed`` is neither an
        integer nor ``None``.
    ValueError
        If ``num_variants`` is below one, if the observable is not diagonal in
        the :math:`Z` basis, or if the observable and the circuit do not act
        on the same number of qubits.
    """
    _validate_num_variants(num_variants)
    _validate_seed(seed)
    _validate_z_diagonal(observable)
    if observable.num_qubits != circuit.num_qubits:
        raise ValueError(
            "the observable and the circuit must act on the same number of "
            f"qubits, but got {observable.num_qubits} and "
            f"{circuit.num_qubits}."
        )
    permutations = _draw_permutations(num_variants, circuit.num_qubits, seed)
    circuits = [_permute_circuit(circuit, row) for row in permutations]
    observables = [
        _permute_observable(observable, row) for row in permutations
    ]
    return circuits, observables


def combine_results(
    results: Sequence[ExecutionResult],
    /,
    *,
    mode: str,
    observable: Hamiltonian | PauliString,
    shots: int,
) -> float:
    r"""Mitigated expectation value from the per-permutation counts.

    With :math:`R` permutations and per-permutation estimates
    :math:`\hat{o}_i`, ``averaging`` returns

    .. math::

        \hat{o} = \frac{1}{R} \sum_{i=1}^{R} \hat{o}_i .

    ``sharpening`` replaces the :math:`R` shot sequences :math:`(s_{i,1},
    \dots, s_{i,S})` by the per-shot majority :math:`\tilde{s}_j =
    \mathrm{majority}_i\, s_{i,j}`, drops a shot whose vote between the two
    most frequent records ties, and evaluates the expectation value on the
    surviving sharpened record. Because the relabelling of
    :func:`construct_circuits` keeps recorded bit :math:`q` attached to logical
    qubit :math:`q`, the caller's unpermuted observable is the right one to
    evaluate here.

    The observable must be diagonal in the :math:`Z` basis, and every variant
    must have been sampled with the common shot count ``shots``; sharpening in
    particular has no well-defined per-shot vote otherwise.

    Parameters
    ----------
    results
        Measured counts of the circuits returned by :func:`construct_circuits`,
        one :class:`cqlib.device.ExecutionResult` per variant, in that order.
    mode
        Aggregation to apply: ``"averaging"`` or ``"sharpening"``.
    observable
        Observable to estimate. It must be diagonal in the :math:`Z` basis,
        that is, a sum of :math:`Z` and identity products.
    shots
        Shot count sampled for every variant. Each result must carry this many
        shots.

    Returns
    -------
    float
        Mitigated expectation value.

    Raises
    ------
    TypeError
        If ``mode`` is not a string, if ``shots`` is not an integer, or if an
        entry of ``results`` is not a
        :class:`cqlib.device.ExecutionResult`.
    ValueError
        If ``mode`` is not a supported aggregation, if ``shots`` is below one,
        if ``observable`` is not diagonal in the :math:`Z` basis, if
        ``results`` is empty, or if a variant was not sampled with ``shots``
        shots. The last case is the guard that sharpening needs one shared
        shot count across variants.
    """
    _validate_mode(mode)
    _validate_shots(shots)
    _validate_z_diagonal(observable)
    _validate_variants(results, shots)
    hamiltonian = _as_hamiltonian(observable)
    if mode == "averaging":
        return _average(results, hamiltonian)
    return _sharpen(results, hamiltonian, shots)


def execute_with_deb(
    circuit: Circuit,
    executor: QemExecutor,
    observable: Hamiltonian | PauliString,
    /,
    *,
    mode: str,
    shots: int,
    num_variants: int,
    seed: int | None = None,
) -> float:
    r"""Estimate one observable on ``circuit`` with qubit-label symmetrization.

    This is the immediate form of :func:`deb_executor`::

        execute_with_deb(c, ex, obs, mode=m, shots=s, num_variants=n)
        == deb_executor(ex, mode=m, shots=s, num_variants=n).estimate(
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
    mode
        Aggregation to apply: ``"averaging"`` or ``"sharpening"``.
    shots
        Shot count for every permuted circuit.
    num_variants
        Number of independent permutations to aggregate over.
    seed
        Seed of the permutation draw, or ``None`` for a fresh draw.

    Returns
    -------
    float
        Mitigated expectation value.

    Raises
    ------
    TypeError
        If ``mode`` is not a string, if ``shots`` or ``num_variants`` is not an
        integer, or if ``seed`` is neither an integer nor ``None``.
    ValueError
        If ``mode`` is not a supported aggregation, if ``shots`` or
        ``num_variants`` is below one, or if ``observable`` is not diagonal in
        the :math:`Z` basis.
    """
    wrapped = deb_executor(
        executor,
        mode=mode,
        shots=shots,
        num_variants=num_variants,
        seed=seed,
    )
    request = EstimatorRequest(circuit=circuit, observable=observable, shots=shots)
    return wrapped.estimate([request])[0].expectation


def deb_executor(
    executor: QemExecutor,
    /,
    *,
    mode: str,
    shots: int,
    num_variants: int,
    seed: int | None = None,
) -> QemExecutor:
    r"""Return an executor with qubit-label symmetrization applied.

    The returned executor keeps the wrapped executor's sampling surface and
    caching flag, so symmetrization can be chained on top of another technique.
    Symmetrization consumes counts and produces an expectation value, so it
    replaces the expectation surface while leaving sampling untouched.

    Parameters
    ----------
    executor
        Execution backend to wrap.
    mode
        Aggregation to apply: ``"averaging"`` or ``"sharpening"``.
    shots
        Shot count bound to every permuted circuit.
    num_variants
        Number of independent permutations aggregated for every request.
    seed
        Seed of the permutation draw, or ``None`` to defer to the request's own
        seed.

    Returns
    -------
    QemExecutor
        Executor whose expectation surface applies qubit-label symmetrization.

    Raises
    ------
    TypeError
        If ``mode`` is not a string, if ``shots`` or ``num_variants`` is not an
        integer, or if ``seed`` is neither an integer nor ``None``.
    ValueError
        If ``mode`` is not a supported aggregation, or if ``shots`` or
        ``num_variants`` is below one.
    """
    estimator = _DebEstimator(
        executor,
        mode=mode,
        shots=shots,
        num_variants=num_variants,
        seed=seed,
    )
    return QemExecutor(
        estimator=estimator,
        sampler=executor.sampler,
        use_cache=executor.use_cache,
    )


class _DebEstimator(Estimator):
    r"""Expectation surface that permutes qubit labels and aggregates variants.

    Parameters
    ----------
    executor
        Execution backend supplying the sampling surface.
    mode
        Aggregation to apply: ``"averaging"`` or ``"sharpening"``.
    shots
        Shot count bound to every permuted circuit.
    num_variants
        Number of independent permutations aggregated for every request.
    seed
        Seed of the permutation draw, or ``None`` to defer to the request's own
        seed.
    """

    __slots__ = ("_executor", "_mode", "_num_variants", "_seed", "_shots")

    _executor: QemExecutor
    _mode: str
    _num_variants: int
    _seed: int | None
    _shots: int

    def __init__(
        self,
        executor: QemExecutor,
        /,
        *,
        mode: str,
        shots: int,
        num_variants: int,
        seed: int | None,
    ) -> None:
        _validate_mode(mode)
        _validate_shots(shots)
        _validate_num_variants(num_variants)
        _validate_seed(seed)
        self._executor = executor
        self._mode = mode
        self._shots = shots
        self._num_variants = num_variants
        self._seed = seed

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        r"""Mitigate every request and return results in input order."""
        return [self._mitigate(request) for request in requests]

    def _mitigate(self, request: EstimatorRequest) -> EstimatorResult:
        r"""Sample the permuted circuits and aggregate their counts."""
        # Reject a non-diagonal observable before spending any shots on it.
        _validate_z_diagonal(request.observable)
        seed = self._seed if self._seed is not None else request.seed
        # The permutations are drawn once here and only the permuted circuits
        # are needed downstream; the matching permuted observables are left to
        # the direct two-stage callers of construct_circuits.
        circuits, _ = construct_circuits(
            request.circuit,
            request.observable,
            num_variants=self._num_variants,
            seed=seed,
        )
        # Each permuted circuit is sampled independently, so it takes its own
        # child seed while one run seed still reproduces the whole aggregate.
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
        value = combine_results(
            results,
            mode=self._mode,
            observable=request.observable,
            shots=self._shots,
        )
        return EstimatorResult(expectation=value)


def _draw_permutations(
    num_variants: int,
    num_qubits: int,
    seed: int | None,
) -> list[np.ndarray]:
    r"""Draw one independent qubit-label permutation per variant.

    Parameters
    ----------
    num_variants
        Number of permutations to draw.
    num_qubits
        Number of qubit labels permuted, ``0`` to ``num_qubits - 1``.
    seed
        Seed of the draw, or ``None`` for entropy.

    Returns
    -------
    list[numpy.ndarray]
        ``num_variants`` permutations, each an integer array of length
        ``num_qubits``; entry ``q`` is the physical label of logical qubit
        ``q``.
    """
    rng = np.random.default_rng(seed)
    return [rng.permutation(num_qubits) for _ in range(num_variants)]


def _permute_circuit(circuit: Circuit, permutation: np.ndarray, /) -> Circuit:
    r"""Rebuild ``circuit`` with qubit ``q`` moved to ``permutation[q]``.

    The rebuilt circuit lists its qubit register in permuted physical order and
    remaps every operation's qubits through the permutation, so the recorded
    bitstring bit ``q`` stays attached to logical qubit ``q``. The global phase
    is carried over by going through :func:`_circuits.rebuild`.

    Parameters
    ----------
    circuit
        Circuit to relabel.
    permutation
        Physical label of every logical qubit.

    Returns
    -------
    Circuit
        The relabelled circuit.
    """
    register = [int(permutation[index]) for index in range(circuit.num_qubits)]
    shell = Circuit(register)
    shell.set_global_phase(circuit.global_phase)
    operations = [
        ValueOperation(
            operation.instruction,
            [Qubit(int(permutation[qubit.index])) for qubit in operation.qubits],
            operation.params,
            operation.label,
        )
        for operation in circuit.operations
    ]
    return rebuild(shell, operations)


def _permute_observable(
    observable: Hamiltonian | PauliString,
    permutation: np.ndarray,
    /,
) -> Hamiltonian | PauliString:
    r"""Relabel every Pauli of ``observable`` by the qubit permutation.

    The term that acted on logical qubit ``q`` is rebuilt on physical qubit
    ``permutation[q]``, matching the relabelling of :func:`_permute_circuit`, so
    the permuted pair still estimates the same physical quantity.

    Parameters
    ----------
    observable
        Observable to relabel.
    permutation
        Physical label of every logical qubit.

    Returns
    -------
    Hamiltonian | PauliString
        The relabelled observable, with the same container type as the input.
    """
    if isinstance(observable, PauliString):
        return _permute_pauli(observable, permutation)
    terms = [
        (_permute_pauli(pauli, permutation), coefficient)
        for pauli, coefficient in observable.terms
    ]
    return Hamiltonian.from_list(terms)


def _permute_pauli(pauli: PauliString, permutation: np.ndarray, /) -> PauliString:
    r"""Relabel one Pauli string, keeping its global phase.

    Parameters
    ----------
    pauli
        Pauli string to relabel.
    permutation
        Physical label of every logical qubit.

    Returns
    -------
    PauliString
        The relabelled Pauli string.
    """
    permuted = PauliString(pauli.num_qubits)
    for qubit in range(pauli.num_qubits):
        permuted.set_pauli(int(permutation[qubit]), pauli.get_pauli(qubit))
    permuted.phase = pauli.phase
    return permuted


def _average(
    results: Sequence[ExecutionResult],
    hamiltonian: Hamiltonian,
    /,
) -> float:
    r"""Mean of the per-variant expectation values.

    The relabelling keeps recorded bit ``q`` attached to logical qubit ``q``,
    so one all-:math:`Z` measurement basis serves every variant of a diagonal
    observable.

    Parameters
    ----------
    results
        One sampled result per variant.
    hamiltonian
        Observable to estimate, already wrapped as a Hamiltonian.

    Returns
    -------
    float
        Mean of the per-variant expectation values.
    """
    num_qubits = hamiltonian.num_qubits
    measurement = PauliString.from_str("Z" * num_qubits)
    total = 0.0
    for result in results:
        distribution = _normalised_distribution(result, num_qubits)
        total += float(
            hamiltonian.expectation_probs([(measurement, distribution)])
        )
    return total / len(results)


def _sharpen(
    results: Sequence[ExecutionResult],
    hamiltonian: Hamiltonian,
    shots: int,
    /,
) -> float:
    r"""Expectation value on the shot-by-shot majority record.

    Parameters
    ----------
    results
        One sampled result per variant, all with ``shots`` shots.
    hamiltonian
        Observable to estimate, already wrapped as a Hamiltonian.
    shots
        Shot count shared by every variant.

    Returns
    -------
    float
        Expectation value on the sharpened record.

    Raises
    ------
    ValueError
        If every shot ties and the sharpened record is empty.
    """
    num_qubits = hamiltonian.num_qubits
    sequences = [_expand_shots(result, shots, num_qubits) for result in results]
    sharpened: dict[str, int] = {}
    kept = 0
    for index in range(shots):
        winner = _majority([sequence[index] for sequence in sequences])
        if winner is None:
            # A tie leaves no majority record for this shot, so the shot
            # carries no information that survives the vote and is dropped.
            continue
        sharpened[winner] = sharpened.get(winner, 0) + 1
        kept += 1
    if kept == 0:
        raise ValueError(
            "sharpening dropped every shot: no shot index had a majority "
            "vote across the variants."
        )
    measurement = PauliString.from_str("Z" * num_qubits)
    distribution = {key: count / kept for key, count in sharpened.items()}
    return float(hamiltonian.expectation_probs([(measurement, distribution)]))


def _normalised_distribution(
    result: ExecutionResult,
    num_qubits: int,
    /,
) -> dict[str, float]:
    r"""Return a variant's recorded counts as a normalised distribution.

    Parameters
    ----------
    result
        Sampled counts whose bitstrings are as wide as the circuit.
    num_qubits
        Width of the recorded bitstrings.

    Returns
    -------
    dict[str, float]
        Probability of each little-endian bitstring, keyed by a bitstring of
        exactly ``num_qubits`` characters.
    """
    shots = result.shots
    distribution: dict[str, float] = {}
    for bitstring, count in result.counts.items():
        key = format(int(bitstring, 2), f"0{num_qubits}b")
        distribution[key] = distribution.get(key, 0.0) + count / shots
    return distribution


def _expand_shots(
    result: ExecutionResult,
    shots: int,
    num_qubits: int,
    /,
) -> list[str]:
    r"""Expand a variant's aggregated counts into one bitstring per shot.

    The counts hold no per-shot sequence, so the shot order is a deterministic
    reconstruction: each recorded bitstring is repeated as many times as it was
    counted, in sorted bitstring order. The reconstruction is identical for
    every variant, so a shot index is a consistent key across variants.

    Parameters
    ----------
    result
        Sampled counts for one variant.
    shots
        Shot count the variant declares.
    num_qubits
        Width of the recorded bitstrings.

    Returns
    -------
    list[str]
        ``shots`` little-endian bitstrings of width ``num_qubits``.

    Raises
    ------
    ValueError
        If the recorded counts do not sum to ``shots``.
    """
    sequence: list[str] = []
    # Sort the keys so the reconstruction never depends on the order the
    # recorded counts happen to be iterated in, which cqlib leaves unspecified.
    for bitstring, count in sorted(result.counts.items()):
        key = format(int(bitstring, 2), f"0{num_qubits}b")
        sequence.extend([key] * count)
    if len(sequence) != shots:
        raise ValueError(
            "the recorded counts of a variant sum to "
            f"{len(sequence)} shots, but the variant declares {shots}."
        )
    return sequence


def _majority(votes: Sequence[str], /) -> str | None:
    r"""Return the strict majority of ``votes``, or ``None`` on a tie.

    Parameters
    ----------
    votes
        One recorded bitstring per variant for a single shot index.

    Returns
    -------
    str | None
        The most frequent bitstring, or ``None`` when the two most frequent
        bitstrings share the same count.
    """
    ranking = Counter(votes).most_common(2)
    if len(ranking) > 1 and ranking[0][1] == ranking[1][1]:
        return None
    return ranking[0][0]


def _as_hamiltonian(observable: Hamiltonian | PauliString, /) -> Hamiltonian:
    r"""Wrap a bare Pauli string so every observable shares one code path."""
    if isinstance(observable, PauliString):
        return Hamiltonian.from_pauli(observable)
    return observable


def _validate_mode(mode: str, /) -> None:
    r"""Reject an aggregation mode outside the supported set."""
    if not isinstance(mode, str):
        raise TypeError("mode must be a string")
    if mode not in _MODES:
        raise ValueError(
            f"mode must be one of {sorted(_MODES)}, but was {mode!r}."
        )


def _validate_variants(
    results: Sequence[ExecutionResult],
    shots: int,
    /,
) -> None:
    r"""Reject an empty variant set or one with mismatched shot counts."""
    if len(results) < 1:
        raise ValueError("deb needs at least one variant to combine.")
    for result in results:
        if not isinstance(result, ExecutionResult):
            raise TypeError(
                "results must contain cqlib.device.ExecutionResult values"
            )
        if result.shots != shots:
            raise ValueError(
                "every variant must be sampled with the same shot count, "
                f"{shots}, but a variant carries {result.shots} shots."
            )


def _validate_num_variants(num_variants: int, /) -> None:
    r"""Reject a variant count that is not a positive integer."""
    if isinstance(num_variants, bool) or not isinstance(num_variants, int):
        raise TypeError("num_variants must be an integer")
    if num_variants < 1:
        raise ValueError(
            "num_variants must be at least one, but was "
            f"{num_variants!r}."
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
