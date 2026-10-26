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

"""Tests for zero-noise extrapolation."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pytest
from cqlib import Circuit
from cqlib.circuit import Instruction, StandardGate

from cqlib_algorithms.error_mitigation._circuits import strip_measurements
from cqlib_algorithms.error_mitigation.core import QemExecutor
from cqlib_algorithms.error_mitigation.inference import (
    ConvergenceWarning,
    RichardsonFactory,
)
from cqlib_algorithms.error_mitigation.scaling import fold_global
from cqlib_algorithms.error_mitigation.techniques.zne import (
    ScalingMethod,
    combine_results,
    construct_circuits,
    execute_with_zne,
    zne_executor,
)
from cqlib_algorithms.protocols import (
    Estimator,
    EstimatorRequest,
    EstimatorResult,
)

_ATOL = 1e-9


def _linear_response_executor(
    circuit: Circuit,
    slope: float,
    intercept: float,
    /,
) -> QemExecutor:
    """Return a fake executor reporting ``slope * lambda + intercept``.

    The scale factor of a globally folded circuit is recovered from its gate
    count relative to the unfolded ``circuit``, so the value is linear in the
    noise scale rather than in the raw gate count.
    """
    base = len(strip_measurements(circuit)[0].operations)

    class _LinearResponse(Estimator):
        """Fake backend whose value depends only on the fold count."""

        __slots__ = ()

        def run(
            self,
            requests: Sequence[EstimatorRequest],
            /,
        ) -> list[EstimatorResult]:
            """Return the linear response of each request's fold count."""
            results: list[EstimatorResult] = []
            for request in requests:
                folded = len(strip_measurements(request.circuit)[0].operations)
                scale = folded / base
                results.append(EstimatorResult(slope * scale + intercept))
            return results

    return QemExecutor(estimator=_LinearResponse())


def test_construct_circuits_returns_one_circuit_per_scale_factor(
    bell_circuit: Circuit,
) -> None:
    scale_factors = (1.0, 2.0, 3.0)

    circuits = construct_circuits(bell_circuit, scale_factors=scale_factors)

    assert len(circuits) == len(scale_factors)
    assert all(isinstance(circuit, Circuit) for circuit in circuits)


def test_folded_expectation_is_invariant_without_noise(
    exact_executor, bell_circuit: Circuit, zz_observable
) -> None:
    base = exact_executor.estimate(
        [EstimatorRequest(circuit=bell_circuit, observable=zz_observable)]
    )[0].expectation

    circuits = construct_circuits(
        bell_circuit,
        scale_factors=(1.0, 2.0, 3.0, 4.0),
    )
    folded = exact_executor.estimate(
        [
            EstimatorRequest(circuit=circuit, observable=zz_observable)
            for circuit in circuits
        ]
    )

    # Folding multiplies the error of a block without changing its logical
    # action, so in the noise-free limit every fold keeps the ideal value.
    assert all(
        result.expectation == pytest.approx(base, abs=_ATOL) for result in folded
    )


def test_global_folding_multiplies_the_gate_count_by_two_n_plus_one(
    bell_circuit: Circuit, gate_count
) -> None:
    base = gate_count(bell_circuit)

    # A level n global fold scales the gate count by 2n + 1, i.e. lambda = 3
    # triples the body and lambda = 5 quintuples it.
    assert gate_count(fold_global(bell_circuit, 3.0)) == 3 * base
    assert gate_count(fold_global(bell_circuit, 5.0)) == 5 * base


def test_richardson_reproduces_a_linear_noise_response(
    bell_circuit: Circuit, zz_observable
) -> None:
    slope, intercept = 0.5, -0.3
    executor = _linear_response_executor(bell_circuit, slope, intercept)

    mitigated = execute_with_zne(
        bell_circuit,
        executor,
        zz_observable,
        scale_factors=(1.0, 2.0, 3.0),
    )

    # A noise response linear in the scale factor is reproduced exactly by any
    # Richardson order, so the extrapolation returns the constant term.
    assert mitigated == pytest.approx(intercept, abs=_ATOL)


def test_combine_results_matches_the_factory_reduce() -> None:
    scale_factors = (1.0, 2.0, 3.0)
    results = [
        EstimatorResult(expectation=value) for value in (0.5, 0.4, 0.3)
    ]

    combined = combine_results(results, factory=RichardsonFactory(scale_factors))

    manual = RichardsonFactory(scale_factors)
    for factor, result in zip(scale_factors, results, strict=True):
        manual.push(factor, result.expectation)
    assert combined == pytest.approx(manual.reduce(), abs=_ATOL)


def test_local_scaling_requires_num_chunks(bell_circuit: Circuit) -> None:
    with pytest.raises(ValueError, match="num_chunks"):
        construct_circuits(
            bell_circuit,
            scale_factors=(1.0, 2.0),
            scaling=ScalingMethod.LOCAL,
        )


def test_random_scaling_is_reproducible_for_a_seed(
    wide_circuit: Circuit, names
) -> None:
    def build() -> Circuit:
        return construct_circuits(
            wide_circuit,
            scale_factors=(3.0, 5.0),
            scaling=ScalingMethod.RANDOM,
            seed=7,
        )[0]

    # A pinned seed reproduces the random gate selection, so both draws share
    # the same folded operation sequence.
    assert names(build()) == names(build())


def test_a_gate_set_forces_the_cqlib_route(
    exact_executor, rotation_circuit, z_observable
) -> None:
    circuit = rotation_circuit(0.9)
    gate_set = [Instruction.from_standard_gate(StandardGate.RY)]

    mitigated = execute_with_zne(
        circuit,
        exact_executor,
        z_observable,
        scale_factors=(1.0, 3.0, 5.0),
        gate_set=gate_set,
    )

    # The cqlib route still returns the noise-free value, and a grid cqlib
    # folding cannot reproduce is rejected instead of silently falling back.
    assert mitigated == pytest.approx(math.cos(0.9), abs=1e-6)
    with pytest.raises(ValueError, match="gate_set"):
        execute_with_zne(
            circuit,
            exact_executor,
            z_observable,
            scale_factors=(1.0, 2.0, 3.0),
            gate_set=gate_set,
        )


def test_an_ill_conditioned_grid_emits_a_convergence_warning(
    exact_executor, bell_circuit: Circuit, zz_observable
) -> None:
    with pytest.warns(ConvergenceWarning):
        execute_with_zne(
            bell_circuit,
            exact_executor,
            zz_observable,
            scale_factors=(1.0, 1.0 + 1e-7, 1.0 + 2e-7),
        )


def test_execute_with_zne_equals_the_orchestrated_two_piece_result(
    exact_executor, rotation_circuit, z_observable
) -> None:
    circuit = rotation_circuit(0.9)
    scale_factors = (1.0, 2.0, 3.0)

    direct = execute_with_zne(
        circuit,
        exact_executor,
        z_observable,
        scale_factors=scale_factors,
    )

    circuits = construct_circuits(circuit, scale_factors=scale_factors)
    results = exact_executor.estimate(
        [
            EstimatorRequest(circuit=folded, observable=z_observable)
            for folded in circuits
        ]
    )
    assembled = combine_results(
        results,
        factory=RichardsonFactory(scale_factors),
    )

    assert direct == pytest.approx(assembled, abs=_ATOL)


def test_zne_executor_preserves_the_cache_flag(exact_estimator) -> None:
    source = QemExecutor(estimator=exact_estimator, use_cache=False)

    wrapped = zne_executor(source, scale_factors=(1.0, 2.0, 3.0))

    assert wrapped.use_cache is source.use_cache
