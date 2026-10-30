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

r"""Virtual distillation.

Virtual distillation estimates an observable from the moment ratio

.. math::

    \langle O \rangle_\text{VD}
        = \frac{\operatorname{Tr}(O \rho^M)}{\operatorname{Tr}(\rho^M)},

where :math:`\rho` is the noisy state the circuit prepares and :math:`M` is
the number of copies. Raising :math:`\rho` to the :math:`M`-th power suppresses
the weight of its low-fidelity components, so the ratio moves towards the ideal
value as soon as the fidelity of :math:`\rho` with the ideal state exceeds one
half. Below one half the ordering of the components reverses and the estimate
is driven away from the ideal value.

Both moments are read from one copy-swap circuit that holds :math:`M`
independent copies of the circuit, a Hadamard gate on an ancillary qubit, and
a ladder of ancillary-controlled SWAP gates between adjacent copies. The
ancilla measured in the :math:`X` basis gives the denominator moment; the
ancilla measured together with the observable placed on the first copy gives
the numerator moment. Both the circuit and the two observables come from
:class:`cqlib.error_mitigation.VirtualDistillation`.

:func:`construct_circuits` builds the copy-swap circuit and
:func:`combine_results` turns a numerator/denominator pair into the distilled
value; :func:`execute_with_vd` and :func:`vd_executor` orchestrate the two on
an executor.

References
----------
[1] W. J. Huggins, B. A. O'Gorman, N. C. Rubin, D. R. Hamel, H. Fu, Y. Qiu, and K. B.
    Whaley, "Virtual distillation for quantum error mitigation," Physical Review X 11,
    041036, 2021. arXiv:2011.05165.
"""

from __future__ import annotations

from collections.abc import Sequence

from cqlib.error_mitigation import VirtualDistillation
from cqlib.qis import Hamiltonian, PauliString

from cqlib import Circuit

from ...protocols import Estimator, EstimatorRequest, EstimatorResult, spawn_seeds
from ..core import QemExecutor, make_cqlib_estimator


def _as_hamiltonian(observable: Hamiltonian | PauliString, /) -> Hamiltonian:
    r"""Wrap a bare Pauli string so every observable shares one code path."""
    if isinstance(observable, PauliString):
        return Hamiltonian.from_pauli(observable)
    return observable


def construct_circuits(
    circuit: Circuit,
    /,
    *,
    copies: int = 2,
) -> list[Circuit]:
    r"""Build the copy-swap circuit for one virtual-distillation measurement.

    The circuit holds ``circuit`` laid out on :math:`M` copies over a register
    of :math:`Mn + 1` qubits, a Hadamard gate on the last (ancillary) qubit,
    and a ladder of :math:`(M - 1) n` ancillary-controlled SWAP gates that
    exchanges adjacent copies. Measuring the ancillary qubit yields
    :math:`\operatorname{Tr}(\rho^M)`, and measuring it together with the
    observable placed on the first copy yields
    :math:`\operatorname{Tr}(O\rho^M)`.

    Args:
        circuit: Circuit whose expectation value is to be mitigated.
        copies: Number of density-matrix copies :math:`M`, at least two.

    Returns:
        A single-element list holding the copy-swap circuit.

    Raises:
        TypeError: If ``copies`` is not an integer.
        ValueError: If ``copies`` is below two.

    """
    _validate_copies(copies)
    return [VirtualDistillation(circuit, copies).build_copy_swap_circuit()]


def combine_results(results: Sequence[EstimatorResult], /) -> float:
    r"""Return the distilled moment ratio from a numerator/denominator pair.

    The first result is the numerator estimate and the second the denominator
    estimate, both measured on the copy-swap circuit. The return value is

    .. math::

        \langle O \rangle_\text{VD} = \frac{\mu_\text{num}}{\mu_\text{den}},

    where :math:`\mu_\text{num}` and :math:`\mu_\text{den}` are the two
    expectation values.

    Args:
        results: Exactly two results, ordered numerator first and denominator second.

    Returns:
        The distilled expectation value.

    Raises:
        TypeError: If an entry is not an :class:`EstimatorResult`.
        ValueError: If ``results`` does not hold exactly two entries, or if the
            denominator expectation is zero so the ratio is undefined.

    """
    if len(results) != 2:
        raise ValueError(
            "virtual distillation needs the numerator result and the "
            f"denominator result, so exactly two, but got {len(results)}."
        )
    for result in results:
        if not isinstance(result, EstimatorResult):
            raise TypeError("results must contain EstimatorResult values")
    numerator, denominator = results
    if denominator.expectation == 0.0:
        raise ValueError(
            "virtual distillation denominator expectation is zero, so the "
            "moment ratio is undefined."
        )
    return float(numerator.expectation / denominator.expectation)


def execute_with_vd(
    circuit: Circuit,
    executor: QemExecutor,
    observable: Hamiltonian | PauliString,
    /,
    *,
    copies: int = 2,
    shots_numerator: int,
    shots_denominator: int,
    seed: int | None = None,
) -> float:
    r"""Estimate one observable on ``circuit`` with virtual distillation.

    This is the immediate form of :func:`vd_executor`::

        execute_with_vd(c, ex, obs, **params)
        == vd_executor(ex, **params).estimate(
               [EstimatorRequest(c, obs)]
           )[0].expectation

    The estimate is the moment ratio :math:`\operatorname{Tr}(O\rho^M) /
    \operatorname{Tr}(\rho^M)` read from the copy-swap circuit, where
    :math:`\rho` is the state ``circuit`` prepares. Distillation suppresses the
    error only while the fidelity of :math:`\rho` with the ideal state exceeds
    one half; above that error rate the ordering of the components of
    :math:`\rho` reverses and the estimate is driven away from the ideal value.
    The method also assumes independent noise across copies; correlated noise
    shared by the copies invalidates the correction model.

    Args:
        circuit: Circuit whose expectation value is to be mitigated.
        executor: Execution backend supplying the expectation surface.
        observable: Observable to estimate, acting on exactly the circuit's qubits.
        copies: Number of density-matrix copies :math:`M`, at least two. Larger values
            suppress the error more strongly but widen the circuit to
            :math:`Mn + 1` qubits.
        shots_numerator: Shot count for the numerator observable, at least one.
        shots_denominator: Shot count for the denominator observable, at least one. An odd count
            keeps the estimated moment from vanishing identically.
        seed: Non-negative seed of the sampled numerator and denominator, or
            ``None`` for fresh randomness. The two measurements draw from separate
            child seeds, so one value reproduces the whole estimate.

    Returns:
        The distilled expectation value.

    Raises:
        TypeError: If ``copies``, ``shots_numerator``, or ``shots_denominator`` is not an
            integer, or if ``seed`` is neither an integer nor ``None``.
        ValueError: If ``copies`` is below two, either shot count is below one, or ``seed``
            is negative.

    """
    wrapped = vd_executor(
        executor,
        copies=copies,
        shots_numerator=shots_numerator,
        shots_denominator=shots_denominator,
        seed=seed,
    )
    request = EstimatorRequest(circuit=circuit, observable=observable)
    return wrapped.estimate([request])[0].expectation


def vd_executor(
    executor: QemExecutor,
    /,
    *,
    copies: int = 2,
    shots_numerator: int,
    shots_denominator: int,
    seed: int | None = None,
) -> QemExecutor:
    r"""Return an executor with virtual distillation applied.

    The returned executor keeps the wrapped executor's sampling surface and
    caching flag, so virtual distillation can be chained on top of another
    technique. Virtual distillation consumes expectation values and produces an
    expectation value, so it replaces the expectation surface while leaving
    sampling untouched.

    Args:
        executor: Execution backend to wrap.
        copies: Number of density-matrix copies :math:`M`, at least two.
        shots_numerator: Shot count bound to the numerator observable, at least one.
        shots_denominator: Shot count bound to the denominator observable, at least one.
        seed: Non-negative seed of the sampled numerator and denominator, or
            ``None`` to defer to the request's own seed.

    Returns:
        Executor whose expectation surface applies virtual distillation.

    Raises:
        TypeError: If ``copies``, ``shots_numerator``, or ``shots_denominator`` is not an
            integer, or if ``seed`` is neither an integer nor ``None``.
        ValueError: If ``copies`` is below two, either shot count is below one, or ``seed``
            is negative.

    """
    _validate_copies(copies)
    _validate_shots(shots_numerator, "shots_numerator")
    _validate_shots(shots_denominator, "shots_denominator")
    _validate_seed(seed)

    estimator = _VdEstimator(
        executor,
        copies=copies,
        shots_numerator=shots_numerator,
        shots_denominator=shots_denominator,
        seed=seed,
    )
    return QemExecutor(
        estimator=estimator,
        sampler=executor.sampler,
        use_cache=executor.use_cache,
    )


class _VdEstimator(Estimator):
    r"""Expectation surface that distils the moment ratio of a request.

    Args:
        executor: Execution backend supplying the expectation surface of the copy-swap
            circuit.
        copies: Number of density-matrix copies :math:`M`, at least two.
        shots_numerator: Shot count bound to the numerator observable, at least one.
        shots_denominator: Shot count bound to the denominator observable, at least one.
        seed: Non-negative seed of the sampled numerator and denominator, or
            ``None`` to defer to the request's own seed.

    """

    __slots__ = ("_copies", "_executor", "_seed", "_shots_denominator", "_shots_numerator")

    _copies: int
    _executor: QemExecutor
    _seed: int | None
    _shots_denominator: int
    _shots_numerator: int

    def __init__(
        self,
        executor: QemExecutor,
        /,
        *,
        copies: int,
        shots_numerator: int,
        shots_denominator: int,
        seed: int | None,
    ) -> None:
        _validate_copies(copies)
        _validate_shots(shots_numerator, "shots_numerator")
        _validate_shots(shots_denominator, "shots_denominator")
        self._executor = executor
        self._copies = copies
        self._shots_numerator = shots_numerator
        self._shots_denominator = shots_denominator
        self._seed = seed

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        r"""Mitigate every request and return results in input order."""
        return [self._mitigate(request) for request in requests]

    def _mitigate(self, request: EstimatorRequest) -> EstimatorResult:
        r"""Evaluate the moment ratio of one request on its copy-swap circuit."""
        observable = _as_hamiltonian(request.observable)
        runner = VirtualDistillation(request.circuit, self._copies)
        seed = self._seed if self._seed is not None else request.seed
        # The numerator and denominator are independent measurements, so each
        # takes its own child seed while one run seed still reproduces the
        # moment ratio.
        numerator_seed, denominator_seed = spawn_seeds(seed, 2)
        # cqlib reports both moments on the ancillary qubit: the denominator is
        # a bare X on the ancilla and the numerator is the observable on the
        # first copy together with that X. The callbacks forward whichever one
        # cqlib supplies, so only the circuit, the shot plumbing, and the
        # result contract of the runner are reused here.
        numerator_callback = make_cqlib_estimator(
            self._executor,
            shots=self._shots_numerator,
            seed=numerator_seed,
        )
        denominator_callback = make_cqlib_estimator(
            self._executor,
            shots=self._shots_denominator,
            seed=denominator_seed,
        )
        numerator_mean, numerator_variance = runner.run_numerator_circuit(
            observable,
            self._shots_numerator,
            numerator_callback,
        )
        denominator_mean, denominator_variance = runner.run_denominator_circuit(
            self._shots_denominator,
            denominator_callback,
        )
        value = combine_results(
            [
                EstimatorResult(numerator_mean, numerator_variance),
                EstimatorResult(denominator_mean, denominator_variance),
            ]
        )
        return EstimatorResult(expectation=value)


def _validate_copies(copies: int, /) -> None:
    r"""Validate a virtual-distillation copy count."""
    if isinstance(copies, bool) or not isinstance(copies, int):
        raise TypeError("copies must be an integer")
    if copies < 2:
        raise ValueError(f"copies must be at least two, but was {copies!r}.")


def _validate_shots(shots: int, name: str, /) -> None:
    r"""Validate a positive shot count."""
    if isinstance(shots, bool) or not isinstance(shots, int):
        raise TypeError(f"{name} must be an integer")
    if shots < 1:
        raise ValueError(f"{name} must be at least one, but was {shots!r}.")


def _validate_seed(seed: int | None, /) -> None:
    r"""Validate an optional random seed."""
    if seed is None:
        return
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer or None")
    if seed < 0:
        raise ValueError("seed must be non-negative")
