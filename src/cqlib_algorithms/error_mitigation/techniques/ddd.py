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

r"""Dynamical decoupling.

An idling qubit still collects the phase its environment imposes on it. A
dynamical-decoupling rule answers that by spending an idle stretch on a short
pulse sequence whose pulses cancel each other, so the qubit is refocused while
the logical action of the circuit stays untouched.

:class:`DynamicalDecoupling` places a rule into a circuit. The circuit is first
re-ordered into its earliest-possible layers, which turns every idle stretch of
a qubit into a contiguous run of layers, and the rule is then laid into those
runs. The rule itself is built by :func:`~._ddd_rules.general_rule` or taken
from the :data:`~._ddd_rules.XX`, :data:`~._ddd_rules.XYXY` and
:data:`~._ddd_rules.YY` sequences.

The two-piece API is :func:`construct_circuits` plus :func:`combine_results`,
orchestrated by :func:`execute_with_ddd`; :func:`ddd_executor` returns the same
computation as a reusable executor for chaining.

References
----------
.. [1] L. Viola and S. Lloyd, "Dynamical suppression of decoherence in
   two-state quantum systems," Physical Review A 58, 2733, 1998.
   arXiv:quant-ph/9803057.
.. [2] L. Viola, E. Knill, and S. Lloyd, "Dynamical decoupling of open quantum
   systems," Physical Review Letters 82, 2417, 1999.
   doi:10.1103/PhysRevLett.82.2417.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from cqlib import Circuit
from cqlib.circuit import ValueOperation
from cqlib.qis import Hamiltonian, PauliString

from ...protocols import Estimator, EstimatorRequest, EstimatorResult, spawn_seeds
from .._circuits import rebuild, splice
from ..core import QemExecutor
from ._ddd_rules import DDDRule


@dataclass(frozen=True, slots=True)
class DynamicalDecoupling:
    r"""Line transform inserting decoupling pulses into idle windows.

    Every maximal idle window of a qubit is filled with the pulse layout that
    ``rule`` produces for a window of that many layers. A window too short for
    the rule, or one that the rule's spacing cannot fill, is left untouched.

    Parameters
    ----------
    rule
        Pulse sequence laid into every idle window.

    Raises
    ------
    TypeError
        If ``rule`` is not a :class:`DDDRule`.
    """

    rule: DDDRule

    def __post_init__(self) -> None:
        _validate_rule(self.rule)

    def apply(self, circuit: Circuit, /) -> Circuit:
        r"""Return ``circuit`` with the rule laid into its idle windows.

        The pulses are inserted on the wires that are idle at that point, so
        the circuit acts on the same qubits and keeps its global phase. The
        pulses of the rule multiply to the identity, so the transformed
        circuit realises the same unitary as the input.

        Parameters
        ----------
        circuit
            Circuit whose idle windows receive the pulses.

        Returns
        -------
        Circuit
            The decoupled circuit, or ``circuit`` itself when no window can
            hold the rule.

        Raises
        ------
        ValueError
            If the rule's pulses do not multiply to the identity up to a
            global phase.

        References
        ----------
        .. [1] B. Pokharel, N. Anand, B. Fortman, and D. A. Lidar,
           "Demonstration of fidelity improvement using dynamical decoupling
           with superconducting qubits," Physical Review Letters 121, 220502,
           2018. arXiv:1807.08768.
        """
        layers = _operation_layers(circuit)
        insertions = _idle_window_insertions(circuit, layers, self.rule)
        if not insertions:
            return circuit
        # The insertion addresses the rebuilt operation list, so the circuit is
        # relayered first, which makes every idle window a contiguous run of
        # indices that splice can address.
        relayered = rebuild(
            circuit,
            [operation for layer in layers for operation in layer],
        )
        return splice(relayered, insertions)


def construct_circuits(
    circuit: Circuit,
    /,
    *,
    rule: DDDRule,
) -> list[Circuit]:
    r"""Decouple ``circuit`` with ``rule``.

    Parameters
    ----------
    circuit
        Circuit whose idle windows receive the pulses.
    rule
        Pulse sequence laid into every idle window.

    Returns
    -------
    list[Circuit]
        The single decoupled circuit, in the one-element family that
        :func:`combine_results` reduces.

    Raises
    ------
    TypeError
        If ``rule`` is not a :class:`DDDRule`.
    """
    return [DynamicalDecoupling(rule=rule).apply(circuit)]


def combine_results(results: Sequence[EstimatorResult], /) -> float:
    r"""Return the expectation value measured on the decoupled circuit.

    Inserting a decoupling sequence is deterministic, so the family built by
    :func:`construct_circuits` holds one circuit and the reduction is the
    identity.

    Parameters
    ----------
    results
        Expectation values measured on the circuits returned by
        :func:`construct_circuits`.

    Returns
    -------
    float
        The mitigated expectation value.

    Raises
    ------
    ValueError
        If ``results`` does not hold exactly one result.
    """
    if len(results) != 1:
        raise ValueError(
            "dynamical decoupling combines exactly one result, but "
            f"{len(results)} were supplied."
        )
    return results[0].expectation


def execute_with_ddd(
    circuit: Circuit,
    executor: QemExecutor,
    observable: Hamiltonian | PauliString,
    /,
    *,
    rule: DDDRule,
    shots: int | None = None,
    seed: int | None = None,
) -> float:
    r"""Estimate one observable on ``circuit`` with dynamical decoupling.

    This is the immediate form of :func:`ddd_executor`::

        execute_with_ddd(c, ex, obs, **params)
        == ddd_executor(ex, **params).estimate(
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
    rule
        Pulse sequence laid into every idle window.
    shots
        Shot count for the decoupled circuit, or ``None`` for exact
        evaluation.
    seed
        Seed of the shot-based estimation, or ``None`` for fresh randomness.
        The pulse layout is deterministic, so the seed only reaches the
        estimator backend.

    Returns
    -------
    float
        Mitigated expectation value.

    Raises
    ------
    TypeError
        If ``rule`` is not a :class:`DDDRule`.
    ValueError
        If the rule's pulses do not multiply to the identity up to a global
        phase.

    References
    ----------
    .. [1] B. Pokharel, N. Anand, B. Fortman, and D. A. Lidar, "Demonstration
       of fidelity improvement using dynamical decoupling with superconducting
       qubits," Physical Review Letters 121, 220502, 2018. arXiv:1807.08768.
    """
    wrapped = ddd_executor(executor, rule=rule, shots=shots, seed=seed)
    request = EstimatorRequest(circuit=circuit, observable=observable, shots=shots)
    return wrapped.estimate([request])[0].expectation


def ddd_executor(
    executor: QemExecutor,
    /,
    *,
    rule: DDDRule,
    shots: int | None = None,
    seed: int | None = None,
) -> QemExecutor:
    r"""Return an executor with dynamical decoupling applied.

    The returned executor keeps the wrapped executor's sampling surface and
    caching flag, so dynamical decoupling can be chained on top of another
    technique without changing either semantics.

    Parameters
    ----------
    executor
        Execution backend to wrap.
    rule
        Pulse sequence laid into every idle window.
    shots
        Shot count bound to every request, or ``None`` to defer to the
        request's own shot count.
    seed
        Seed of the shot-based estimation, or ``None`` to defer to the
        request's own seed.

    Returns
    -------
    QemExecutor
        Executor whose expectation surface applies dynamical decoupling.

    Raises
    ------
    TypeError
        If ``rule`` is not a :class:`DDDRule`, or if ``seed`` is neither an
        integer nor ``None``.
    """
    estimator = _DddEstimator(executor, rule=rule, shots=shots, seed=seed)
    return QemExecutor(
        estimator=estimator,
        sampler=executor.sampler,
        use_cache=executor.use_cache,
    )


class _DddEstimator(Estimator):
    r"""Expectation surface that decouples each circuit before measuring it.

    Parameters
    ----------
    executor
        Execution backend supplying the expectation surface for the decoupled
        circuit.
    rule
        Pulse sequence laid into every idle window.
    shots
        Shot count bound to every request, or ``None`` to defer to the
        request.
    seed
        Seed of the shot-based estimation, or ``None`` to defer to the
        request.
    """

    __slots__ = ("_executor", "_rule", "_seed", "_shots")

    _executor: QemExecutor
    _rule: DDDRule
    _seed: int | None
    _shots: int | None

    def __init__(
        self,
        executor: QemExecutor,
        /,
        *,
        rule: DDDRule,
        shots: int | None,
        seed: int | None,
    ) -> None:
        _validate_rule(rule)
        _validate_seed(seed)
        self._executor = executor
        self._rule = rule
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
        r"""Estimate one request on its decoupled circuit."""
        shots = self._shots if self._shots is not None else request.shots
        seed = self._seed if self._seed is not None else request.seed
        circuits = construct_circuits(request.circuit, rule=self._rule)
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
        return EstimatorResult(expectation=combine_results(results))


def _operation_layers(circuit: Circuit, /) -> list[list[ValueOperation]]:
    r"""Group the operations of ``circuit`` into earliest-possible layers.

    Parameters
    ----------
    circuit
        Circuit to layer.

    Returns
    -------
    list[list[ValueOperation]]
        One entry per layer, each holding the operations that can start as
        early as that layer. Empty layers are dropped, so the returned list
        indexes the time slots a decoupling rule is laid into.
    """
    dag = circuit.dag()
    layers: list[list[ValueOperation]] = []
    for node_layer in dag.layers():
        operations: list[ValueOperation] = []
        for node in node_layer:
            operation = dag.operation(node)
            if operation is not None:
                operations.append(operation)
        if operations:
            layers.append(operations)
    return layers


def _idle_window_insertions(
    circuit: Circuit,
    layers: Sequence[Sequence[ValueOperation]],
    rule: DDDRule,
    /,
) -> dict[int, list[ValueOperation]]:
    r"""Map operation indices to the pulses a rule places after them.

    Parameters
    ----------
    circuit
        Circuit whose wires are scanned for idle windows.
    layers
        Layered operations of ``circuit``, in execution order.
    rule
        Pulse sequence laid into every idle window.

    Returns
    -------
    dict[int, list[ValueOperation]]
        Insertions keyed by the index, in the relayered circuit, of the last
        operation of the layer that a pulse follows.
    """
    qubits = list(circuit.qubits)
    occupied = {qubit: [False] * len(layers) for qubit in qubits}
    for layer_index, layer in enumerate(layers):
        for operation in layer:
            for qubit in operation.qubits:
                occupied[qubit][layer_index] = True

    # A pulse laid into layer t has to sit among the operations of layer t,
    # because every qubit it touches is idle in that whole layer and therefore
    # commutes with them.
    last_index: list[int] = []
    running = -1
    for layer in layers:
        running += len(layer)
        last_index.append(running)

    insertions: dict[int, list[ValueOperation]] = {}
    for qubit in qubits:
        for start, length in _idle_windows(occupied[qubit]):
            for offset, gate in enumerate(rule.sequence(length)):
                if gate is None:
                    continue
                insertions.setdefault(last_index[start + offset], []).append(
                    ValueOperation.from_standard_gate(gate, [qubit])
                )
    return insertions


def _idle_windows(occupied: Sequence[bool], /) -> Iterator[tuple[int, int]]:
    r"""Yield the start and length of every maximal idle stretch.

    Parameters
    ----------
    occupied
        One flag per layer, ``True`` where the wire carries an operation.

    Yields
    ------
    tuple[int, int]
        Start layer and length, in layers, of one maximal stretch of idle
        layers.
    """
    start: int | None = None
    for index, is_occupied in enumerate(occupied):
        if is_occupied:
            if start is not None:
                yield start, index - start
                start = None
        elif start is None:
            start = index
    if start is not None:
        yield start, len(occupied) - start


def _validate_rule(rule: DDDRule, /) -> None:
    r"""Reject a rule that is not a decoupling rule."""
    if not isinstance(rule, DDDRule):
        raise TypeError(
            f"rule must be a DDDRule instance, but received {type(rule).__name__}"
        )


def _validate_seed(seed: int | None, /) -> None:
    r"""Reject a seed that is neither an integer nor ``None``."""
    if seed is None:
        return
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer or None")
