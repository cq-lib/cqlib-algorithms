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

"""Tests for virtual distillation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from cqlib import Circuit
from cqlib.qis import Hamiltonian, PauliString

from cqlib_algorithms.error_mitigation.core import QemExecutor
from cqlib_algorithms.error_mitigation.techniques.vd import (
    _denominator_observable,
    _numerator_observable,
    combine_results,
    construct_circuits,
    vd_executor,
)
from cqlib_algorithms.protocols import (
    Estimator,
    EstimatorRequest,
    EstimatorResult,
    spawn_seeds,
)

_ATOL = 1e-9


class _RecordingEstimator(Estimator):
    """Record the seed of every request and return a fixed expectation."""

    __slots__ = ("seeds",)

    def __init__(self) -> None:
        self.seeds: list[int | None] = []

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        """Return one fixed result per request, recording its seed."""
        results: list[EstimatorResult] = []
        for request in requests:
            self.seeds.append(request.seed)
            results.append(EstimatorResult(expectation=0.5, variance=0.0))
        return results


def _bit_flip_state(marginal: float, copies: int, /) -> np.ndarray:
    """Return the ``copies``-fold tensor power of a bit-flip density matrix.

    The single-copy state is :math:`\\rho = (1 - p)|0\\rangle\\langle 0| +
    p|1\\rangle\\langle 1|` with ``marginal`` the excitation probability.
    """
    density = np.array(
        [[1.0 - marginal, 0.0], [0.0, marginal]], dtype=complex
    )
    state = np.array([[1.0 + 0.0j]])
    for _ in range(copies):
        state = np.kron(state, density)
    return state


def _module_moments(
    observable: Hamiltonian,
    state: np.ndarray,
    copies: int,
) -> tuple[float, float]:
    """Return the numerator and denominator moments of the copy-swap layer."""
    numerator = float(
        np.trace(
            _numerator_observable(observable, copies).to_matrix() @ state
        ).real
    )
    denominator = float(
        np.trace(
            _denominator_observable(observable.num_qubits, copies).to_matrix()
            @ state
        ).real
    )
    return numerator, denominator


def _swap_operator(num_qubits: int, /) -> np.ndarray:
    """Return the SWAP of two ``num_qubits``-wide copies as a permutation."""
    block = 1 << num_qubits
    dimension = block * block
    swap = np.zeros((dimension, dimension), dtype=complex)
    for high in range(block):
        for low in range(block):
            swap[low * block + high, high * block + low] = 1.0
    return swap


def _symmetrised_observable(
    operator: np.ndarray,
    num_qubits: int,
    copies: int,
) -> np.ndarray:
    """Return the copy symmetrisation :math:`\\frac{1}{M}\\sum_c O^{(c)}`."""
    block = 1 << num_qubits
    identity = np.eye(block, dtype=complex)
    total = np.zeros((block**copies, block**copies), dtype=complex)
    for copy in range(copies):
        factors = [operator if index == copy else identity for index in range(copies)]
        placed = factors[0]
        for factor in factors[1:]:
            placed = np.kron(placed, factor)
        total += placed
    return total / copies


def _symmetrised_moments(
    observable: Hamiltonian,
    state: np.ndarray,
    copies: int,
) -> tuple[float, float]:
    """Return the moments from the closed-form symplectic definitions.

    This is an independent reference for the two-copy case: the network
    reduces to a single SWAP and the numerator is
    :math:`\\operatorname{Tr}(S O_\\text{sym} \\rho^{\\otimes M})`, both of
    which are convention independent once the observable is symmetrised.
    """
    assert copies == 2
    swap = _swap_operator(observable.num_qubits)
    o_sym = _symmetrised_observable(observable.to_matrix(), observable.num_qubits, copies)
    numerator = float(np.trace(swap @ o_sym @ state).real)
    denominator = float(np.trace(swap @ state).real)
    return numerator, denominator


def test_two_copy_expectation_matches_the_analytic_bit_flip_state() -> None:
    marginal, copies = 0.1, 2
    observable = Hamiltonian.from_pauli(PauliString.from_str("Z"))
    state = _bit_flip_state(marginal, copies)

    numerator, denominator = _module_moments(observable, state, copies)
    distilled = combine_results(
        [EstimatorResult(numerator), EstimatorResult(denominator)]
    )

    # On rho = (1 - eps)|0><0| + eps|1><1| the two-copy ratio is
    # (1 - 2 eps) / (1 - 2 eps + 2 eps^2).
    expected = (1.0 - 2.0 * marginal) / (
        1.0 - 2.0 * marginal + 2.0 * marginal**2
    )
    assert distilled == pytest.approx(expected, abs=_ATOL)
    # Distillation moves the estimate closer to the ideal value of one.
    assert abs(distilled - 1.0) < abs((1.0 - 2.0 * marginal) - 1.0)


def test_the_output_is_unchanged_without_noise() -> None:
    copies = 2
    observable = Hamiltonian.from_pauli(PauliString.from_str("Z"))
    state = _bit_flip_state(0.0, copies)

    numerator, denominator = _module_moments(observable, state, copies)
    distilled = combine_results(
        [EstimatorResult(numerator), EstimatorResult(denominator)]
    )

    # A pure |0...0> leaves both moments at the ideal value, so the ratio is
    # the ordinary expectation value.
    assert numerator == pytest.approx(1.0, abs=_ATOL)
    assert denominator == pytest.approx(1.0, abs=_ATOL)
    assert distilled == pytest.approx(1.0, abs=_ATOL)


def test_beyond_fifty_percent_error_reverses_the_direction() -> None:
    marginal, copies = 0.75, 2
    observable = Hamiltonian.from_pauli(PauliString.from_str("Z"))
    state = _bit_flip_state(marginal, copies)

    numerator, denominator = _module_moments(observable, state, copies)
    distilled = combine_results(
        [EstimatorResult(numerator), EstimatorResult(denominator)]
    )

    # Past a fidelity of one half the ordering of the components reverses, so
    # the ratio lands further from the ideal value instead of closer to it.
    raw = 1.0 - 2.0 * marginal
    assert abs(distilled - 1.0) > abs(raw - 1.0)
    assert abs(distilled) > abs(raw)


def test_combine_results_is_the_ratio_of_the_symmetrised_moments() -> None:
    marginal, copies = 0.1, 2
    observable = Hamiltonian.from_pauli(PauliString.from_str("Z"))
    state = _bit_flip_state(marginal, copies)

    module_numerator, module_denominator = _module_moments(
        observable, state, copies
    )
    reference_numerator, reference_denominator = _symmetrised_moments(
        observable, state, copies
    )

    # The self-built observation layer must reproduce the closed-form
    # Hermitian parts (S + S^\dagger)/2 and (S O_sym + O_sym S^\dagger)/2.
    assert module_numerator == pytest.approx(reference_numerator, abs=_ATOL)
    assert module_denominator == pytest.approx(
        reference_denominator, abs=_ATOL
    )
    assert combine_results(
        [
            EstimatorResult(module_numerator),
            EstimatorResult(module_denominator),
        ]
    ) == pytest.approx(
        reference_numerator / reference_denominator, abs=_ATOL
    )

    with pytest.raises(ValueError, match="exactly two"):
        combine_results([EstimatorResult(1.0)])
    with pytest.raises(ValueError, match="undefined"):
        combine_results([EstimatorResult(1.0), EstimatorResult(0.0)])


def test_the_two_copies_use_independent_draws() -> None:
    recorder = _RecordingEstimator()
    executor = QemExecutor(estimator=recorder)
    circuit = Circuit(1)
    circuit.h(0)
    observable = PauliString.from_str("Z")
    request = EstimatorRequest(circuit=circuit, observable=observable)
    settings = {"copies": 2, "shots_numerator": 4096, "shots_denominator": 4097}

    def run_seeds(seed: int) -> list[int | None]:
        recorder.seeds.clear()
        vd_executor(executor, seed=seed, **settings).estimate([request])
        return list(recorder.seeds)

    first = run_seeds(11)
    repeat = run_seeds(11)
    other = run_seeds(12)

    # The numerator and denominator each take their own child seed, and one run
    # seed reproduces that pair while a different seed resamples it. Asserting
    # on the seed set keeps the check independent of how many times cqlib
    # replays each callback.
    expected = set(spawn_seeds(11, 2))
    assert set(first) == expected
    assert len(expected) == 2
    assert set(repeat) == expected
    assert set(other) == set(spawn_seeds(12, 2))


def test_copy_swap_circuit_is_built_by_cqlib(rotation_circuit) -> None:
    circuit = rotation_circuit(0.9)

    circuits = construct_circuits(circuit, copies=2)

    assert len(circuits) == 1
    assert circuits[0].width == 2 * circuit.num_qubits
