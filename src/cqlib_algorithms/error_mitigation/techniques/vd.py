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
independent copies of the circuit and applies a SWAP network that exchanges
the first copy with every other copy. The observable whose expectation is
:math:`\operatorname{Tr}(\rho^M)` is the Hermitian part of that network; the
observable whose expectation is :math:`\operatorname{Tr}(O \rho^M)` is the
Hermitian part of the network composed with the copy-symmetrised observable
:math:`O_\text{sym} = \frac{1}{M}\sum_c O^{(c)}`. Both are assembled here as
Pauli strings over the widened register of :math:`Mn` qubits and evaluated on
the copy-swap circuit.

:func:`construct_circuits` builds the copy-swap circuit and
:func:`combine_results` turns a numerator/denominator pair into the distilled
value; :func:`execute_with_vd` and :func:`vd_executor` orchestrate the two on
an executor.

References
----------
.. [1] W. J. Huggins, B. A. O'Gorman, N. C. Rubin, D. R. Hamel, H. Fu,
   Y. Qiu, and K. B. Whaley, "Virtual distillation for quantum error
   mitigation," Physical Review X 11, 041036, 2021. arXiv:2011.05165.
"""

from __future__ import annotations

from collections.abc import Sequence

from cqlib.error_mitigation import VirtualDistillation
from cqlib.qis import Hamiltonian, Pauli, PauliString

from cqlib import Circuit

from ...protocols import Estimator, EstimatorRequest, EstimatorResult, spawn_seeds
from ..core import QemExecutor, make_cqlib_estimator
    
_PAULI_BY_BITS: dict[tuple[int, int], Pauli] = {
    (0, 0): Pauli.i(),
    (1, 0): Pauli.x(),
    (1, 1): Pauli.y(),
    (0, 1): Pauli.z(),
}

# The complex value of each phase exponent of Z4.
_PHASE_OF_EXPONENT: dict[int, complex] = {
    0: 1.0 + 0.0j,
    1: 1.0j,
    2: -1.0 + 0.0j,
    3: -1.0j,
}


def _build_product_table() -> dict[tuple[int, int, int, int], tuple[int, int, int]]:
    r"""Return the product table of the four single-qubit Paulis.

    Each entry is keyed by the symplectic codes of the two factors and holds
    the symplectic code of their product together with its phase exponent. The
    table is built from cqlib's own multiplication, so the expansions below can
    never drift from cqlib's phase convention.
    """
    table: dict[tuple[int, int, int, int], tuple[int, int, int]] = {}
    for left in _PAULI_BY_BITS.values():
        for right in _PAULI_BY_BITS.values():
            product, phase = left.mul_with_phase(right)
            table[(*left.to_symplectic(), *right.to_symplectic())] = (
                *product.to_symplectic(),
                phase.exponent,
            )
    return table


_PRODUCT = _build_product_table()

_Symplectic = tuple[int, int]
_Expansion = dict[_Symplectic, complex]


def _multiply(
    left: _Symplectic, right: _Symplectic, num_qubits: int
) -> tuple[_Symplectic, complex]:
    r"""Return the symplectic code and phase of a product of two Pauli strings.

    Parameters
    ----------
    left, right
        Symplectic codes ``(x_mask, z_mask)`` of the two factors, each acting
        on ``num_qubits`` qubits.
    num_qubits
        Width of both factors.

    Returns
    -------
    tuple[tuple[int, int], complex]
        The symplectic code of the product and its complex phase.
    """
    left_x, left_z = left
    right_x, right_z = right
    # The qubits of a Pauli string commute, so the product phase is the sum of
    # the per-qubit phases of the four Pauli products.
    exponent = 0
    for qubit in range(num_qubits):
        key = (
            (left_x >> qubit) & 1,
            (left_z >> qubit) & 1,
            (right_x >> qubit) & 1,
            (right_z >> qubit) & 1,
        )
        exponent += _PRODUCT[key][2]
    return (left_x ^ right_x, left_z ^ right_z), _PHASE_OF_EXPONENT[exponent & 3]


def _swap_expansion(num_qubits: int, first: int, second: int) -> _Expansion:
    r"""Return the Pauli expansion of the SWAP between two register copies.

    Parameters
    ----------
    num_qubits
        Width :math:`n` of one copy.
    first, second
        Copy indices the SWAP exchanges.

    Returns
    -------
    dict[tuple[int, int], complex]
        The expansion of
        :math:`\mathrm{SWAP} = \frac{1}{d}\sum_P P \otimes P` transcribed to
        the two copies, one :math:`n`-qubit Pauli per term with weight
        :math:`1/d`, :math:`d = 2^n`.
    """
    dimension = 1 << num_qubits
    expansion: _Expansion = {}
    for x in range(dimension):
        for z in range(dimension):
            key = (
                (x << (first * num_qubits)) | (x << (second * num_qubits)),
                (z << (first * num_qubits)) | (z << (second * num_qubits)),
            )
            expansion[key] = expansion.get(key, 0.0j) + 1.0 / dimension
    return expansion


def _network_expansion(num_qubits: int, copies: int, *, adjoint: bool) -> _Expansion:
    r"""Return the Pauli expansion of the copy-swap network or of its adjoint.

    Parameters
    ----------
    num_qubits
        Width :math:`n` of one copy.
    copies
        Number of copies :math:`M`.
    adjoint
        ``False`` for the network and ``True`` for its adjoint, which is the
        same SWAP sequence multiplied in the opposite order.

    Returns
    -------
    dict[tuple[int, int], complex]
        The expansion of the star SWAP network
        :math:`S = \mathrm{SWAP}_{0,M-1}\cdots \mathrm{SWAP}_{0,1}` over the
        widened register of :math:`Mn` qubits.
    """
    total = num_qubits * copies
    network: _Expansion = {(0, 0): 1.0 + 0.0j}
    for other in range(1, copies):
        gate = _swap_expansion(num_qubits, 0, other)
        combined: _Expansion = {}
        for gate_key, gate_coefficient in gate.items():
            for net_key, net_coefficient in network.items():
                if adjoint:
                    key, phase = _multiply(net_key, gate_key, total)
                else:
                    key, phase = _multiply(gate_key, net_key, total)
                combined[key] = combined.get(key, 0.0j) + (
                    gate_coefficient * net_coefficient * phase
                )
        network = combined
    return network


def _accumulate(destination: _Expansion, source: _Expansion, scale: float) -> None:
    r"""Add ``scale`` times one expansion into another, in place."""
    for key, coefficient in source.items():
        destination[key] = destination.get(key, 0.0j) + scale * coefficient


def _pauli_expansion(observable: Hamiltonian) -> _Expansion:
    r"""Return the symplectic-keyed Pauli expansion of an observable.

    Each term's Pauli phase is folded into its coefficient, so terms that
    differ only by a phase share one key and are combined.
    """
    expansion: _Expansion = {}
    for pauli, coefficient in observable.terms:
        key = (pauli.x_mask, pauli.z_mask)
        effective = complex(coefficient) * pauli.phase.to_complex()
        expansion[key] = expansion.get(key, 0.0j) + effective
    return expansion


def _hamiltonian_from_expansion(expansion: _Expansion, num_qubits: int) -> Hamiltonian:
    r"""Rebuild a Hamiltonian from a real symplectic-keyed Pauli expansion.

    Parameters
    ----------
    expansion
        Pauli expansion over ``num_qubits`` qubits.
    num_qubits
        Width of the widened register.

    Returns
    -------
    Hamiltonian
        A Hermitian Hamiltonian holding one term per non-zero expansion entry.
    """
    entries: list[tuple[PauliString, float]] = []
    for (x, z), coefficient in expansion.items():
        if coefficient == 0.0j:
            continue
        pauli = PauliString(num_qubits)
        for qubit in range(num_qubits):
            axis = ((x >> qubit) & 1, (z >> qubit) & 1)
            pauli.set_pauli(qubit, _PAULI_BY_BITS[axis])
        # Every expansion below is Hermitian by construction, so a coefficient
        # has no imaginary part beyond floating-point residue.
        entries.append((pauli, float(coefficient.real)))
    return Hamiltonian.from_list(entries)


def _denominator_observable(num_qubits: int, copies: int) -> Hamiltonian:
    r"""Return the observable whose expectation is :math:`\operatorname{Tr}(\rho^M)`.

    Parameters
    ----------
    num_qubits
        Width :math:`n` of one copy.
    copies
        Number of copies :math:`M`.

    Returns
    -------
    Hamiltonian
        The Hermitian part :math:`(S + S^\dagger)/2` of the copy-swap network
        :math:`S`, acting on the widened register of :math:`Mn` qubits.
    """
    total = num_qubits * copies
    expansion: _Expansion = {}
    _accumulate(expansion, _network_expansion(num_qubits, copies, adjoint=False), 0.5)
    _accumulate(expansion, _network_expansion(num_qubits, copies, adjoint=True), 0.5)
    return _hamiltonian_from_expansion(expansion, total)


def _numerator_observable(observable: Hamiltonian, copies: int) -> Hamiltonian:
    r"""Return the observable whose expectation is :math:`\operatorname{Tr}(O\rho^M)`.

    The returned operator is the Hermitian part of the copy-swap network
    composed with the copy-symmetrised observable
    :math:`O_\text{sym} = \frac{1}{M}\sum_c O^{(c)}`, where :math:`O^{(c)}`
    is ``observable`` placed on copy :math:`c`.

    Parameters
    ----------
    observable
        Observable :math:`O` acting on one copy, of width :math:`n`.
    copies
        Number of copies :math:`M`.

    Returns
    -------
    Hamiltonian
        The numerator observable acting on the widened register of
        :math:`Mn` qubits.
    """
    num_qubits = observable.num_qubits
    total = num_qubits * copies
    network = _network_expansion(num_qubits, copies, adjoint=False)
    adjoint_network = _network_expansion(num_qubits, copies, adjoint=True)
    observable_expansion = _pauli_expansion(observable)
    # The factor 1/2 hermitises S O_sym + O_sym S^dagger, and 1/copies
    # symmetrises the observable over the copies.
    weight = 0.5 / copies
    expansion: _Expansion = {}
    for copy in range(copies):
        placed = {
            (x << copy * num_qubits, z << copy * num_qubits): coefficient
            for (x, z), coefficient in observable_expansion.items()
        }
        for net_key, net_coefficient in network.items():
            for placed_key, placed_coefficient in placed.items():
                key, phase = _multiply(net_key, placed_key, total)
                expansion[key] = expansion.get(key, 0.0j) + (
                    weight * net_coefficient * placed_coefficient * phase
                )
        for placed_key, placed_coefficient in placed.items():
            for adjoint_key, adjoint_coefficient in adjoint_network.items():
                key, phase = _multiply(placed_key, adjoint_key, total)
                expansion[key] = expansion.get(key, 0.0j) + (
                    weight * placed_coefficient * adjoint_coefficient * phase
                )
    return _hamiltonian_from_expansion(expansion, total)


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

    The circuit decomposes ``circuit``, appends it once per copy on a register
    of :math:`Mn` qubits, and then exchanges the first copy with every other
    copy. Measuring the numerator and the denominator observables built by this
    module on that state yields :math:`\operatorname{Tr}(O\rho^M)` and
    :math:`\operatorname{Tr}(\rho^M)` respectively.

    Parameters
    ----------
    circuit
        Circuit whose expectation value is to be mitigated.
    copies
        Number of density-matrix copies :math:`M`, at least two.

    Returns
    -------
    list[Circuit]
        A single-element list holding the copy-swap circuit.

    Raises
    ------
    TypeError
        If ``copies`` is not an integer.
    ValueError
        If ``copies`` is below two.
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

    Parameters
    ----------
    results
        Exactly two results, ordered numerator first and denominator second.

    Returns
    -------
    float
        The distilled expectation value.

    Raises
    ------
    TypeError
        If an entry is not an :class:`EstimatorResult`.
    ValueError
        If ``results`` does not hold exactly two entries, or if the
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

    Parameters
    ----------
    circuit
        Circuit whose expectation value is to be mitigated.
    executor
        Execution backend supplying the expectation surface.
    observable
        Observable to estimate, acting on exactly the circuit's qubits.
    copies
        Number of density-matrix copies :math:`M`, at least two. Larger values
        suppress the error more strongly but widen the circuit to :math:`Mn`
        qubits.
    shots_numerator
        Shot count for the numerator observable, at least one.
    shots_denominator
        Shot count for the denominator observable, at least one. An odd count
        keeps the estimated moment from vanishing identically.
    seed
        Seed of the sampled numerator and denominator, or ``None`` for fresh
        randomness. The two measurements draw from separate child seeds, so one
        value reproduces the whole estimate.

    Returns
    -------
    float
        The distilled expectation value.

    Raises
    ------
    TypeError
        If ``copies``, ``shots_numerator``, or ``shots_denominator`` is not an
        integer, or if ``seed`` is neither an integer nor ``None``.
    ValueError
        If ``copies`` is below two, or if either shot count is below one.
    Warning
        The method assumes the two copies are subject to independent noise; a
        correlated noise channel shared by both copies voids the correction.
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

    Parameters
    ----------
    executor
        Execution backend to wrap.
    copies
        Number of density-matrix copies :math:`M`, at least two.
    shots_numerator
        Shot count bound to the numerator observable, at least one.
    shots_denominator
        Shot count bound to the denominator observable, at least one.
    seed
        Seed of the sampled numerator and denominator, or ``None`` to defer to
        the request's own seed.

    Returns
    -------
    QemExecutor
        Executor whose expectation surface applies virtual distillation.

    Raises
    ------
    TypeError
        If ``copies``, ``shots_numerator``, or ``shots_denominator`` is not an
        integer, or if ``seed`` is neither an integer nor ``None``.
    ValueError
        If ``copies`` is below two, or if either shot count is below one.
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

    Parameters
    ----------
    executor
        Execution backend supplying the expectation surface of the copy-swap
        circuit.
    copies
        Number of density-matrix copies :math:`M`, at least two.
    shots_numerator
        Shot count bound to the numerator observable, at least one.
    shots_denominator
        Shot count bound to the denominator observable, at least one.
    seed
        Seed of the sampled numerator and denominator, or ``None`` to defer to
        the request's own seed.
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
        copies = self._copies
        observable = _as_hamiltonian(request.observable)
        runner = VirtualDistillation(request.circuit, copies)
        seed = self._seed if self._seed is not None else request.seed
        # The numerator and denominator are independent measurements, so each
        # takes its own child seed while one run seed still reproduces the
        # moment ratio.
        numerator_seed, denominator_seed = spawn_seeds(seed, 2)
        # cqlib sizes its own numerator observable from the Hamiltonian passed
        # below, but expands it under an observation convention this library
        # replaces. Each callback therefore binds the copy-swap observable
        # built here; only the circuit, the shot plumbing, and the result
        # contract of the runner are reused. See the virtual-distillation entry
        # of the self-built-component table in the interface specification.
        numerator_callback = make_cqlib_estimator(
            self._executor,
            _numerator_observable(observable, copies),
            shots=self._shots_numerator,
            seed=numerator_seed,
        )
        denominator_callback = make_cqlib_estimator(
            self._executor,
            _denominator_observable(request.circuit.num_qubits, copies),
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
    r"""Reject a copy count that is not an integer of at least two."""
    if isinstance(copies, bool) or not isinstance(copies, int):
        raise TypeError("copies must be an integer")
    if copies < 2:
        raise ValueError(f"copies must be at least two, but was {copies!r}.")


def _validate_shots(shots: int, name: str, /) -> None:
    r"""Reject a shot count that is not a positive integer."""
    if isinstance(shots, bool) or not isinstance(shots, int):
        raise TypeError(f"{name} must be an integer")
    if shots < 1:
        raise ValueError(f"{name} must be at least one, but was {shots!r}.")


def _validate_seed(seed: int | None, /) -> None:
    r"""Reject a seed that is neither an integer nor ``None``."""
    if seed is None:
        return
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer or None")
