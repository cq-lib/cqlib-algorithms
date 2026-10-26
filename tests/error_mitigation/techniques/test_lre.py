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

"""Tests for layerwise Richardson extrapolation."""

from __future__ import annotations

import math

import numpy as np
import pytest
from cqlib import Circuit

from cqlib_algorithms.error_mitigation._circuits import strip_measurements
from cqlib_algorithms.error_mitigation.scaling import fold_global, fold_layers
from cqlib_algorithms.error_mitigation.techniques.lre import (
    construct_circuits,
    execute_with_lre,
)
from cqlib_algorithms.protocols import EstimatorRequest

_ATOL = 1e-9


def _layer_scales(circuit: Circuit, factor: float, /) -> dict[int, float]:
    """Return a per-operation scale map scaling the whole body uniformly."""
    body, _ = strip_measurements(circuit)
    return {index: factor for index in range(len(body.operations))}


def test_the_grid_size_is_the_graded_monomial_basis_size(
    wide_circuit: Circuit,
) -> None:
    circuits, scale_points = construct_circuits(
        wide_circuit,
        degree=2,
        fold_multiplier=2,
        num_chunks=3,
    )

    # The graded monomial basis over three chunks up to total degree two has
    # comb(3 + 2, 2) terms, one scale vector and one circuit each.
    assert len(circuits) == len(scale_points) == math.comb(3 + 2, 2)
    assert all(len(vector) == 3 for vector in scale_points)


def test_all_layers_at_one_scale_reduce_to_standard_zne(
    wide_circuit: Circuit,
) -> None:
    factor = 3.0

    uniform = fold_layers(wide_circuit, scales=_layer_scales(wide_circuit, factor))
    global_fold = fold_global(wide_circuit, factor)

    # Scaling every chunk by the same factor is exactly global folding, so a
    # single chunk reproduces the standard ZNE odd grid {1, 3, 5}.
    assert np.allclose(uniform.to_matrix(), global_fold.to_matrix(), atol=_ATOL)
    assert len(uniform.operations) == len(global_fold.operations)

    circuits, scale_points = construct_circuits(
        wide_circuit,
        degree=2,
        fold_multiplier=2,
        num_chunks=1,
    )
    assert [vector[0] for vector in scale_points] == [1.0, 3.0, 5.0]
    for circuit, scale in zip(circuits, (1.0, 3.0, 5.0), strict=True):
        reference = fold_global(wide_circuit, scale)
        assert np.allclose(
            circuit.to_matrix(), reference.to_matrix(), atol=_ATOL
        )


def test_layerwise_scaling_is_invariant_without_noise(
    exact_executor, bell_circuit: Circuit, zz_observable
) -> None:
    base = exact_executor.estimate(
        [EstimatorRequest(circuit=bell_circuit, observable=zz_observable)]
    )[0].expectation

    circuits, _ = construct_circuits(
        bell_circuit,
        degree=2,
        fold_multiplier=2,
        num_chunks=2,
    )
    scaled = exact_executor.estimate(
        [
            EstimatorRequest(circuit=circuit, observable=zz_observable)
            for circuit in circuits
        ]
    )

    # Folding is unitary, so every scaled circuit keeps the ideal expectation
    # and the extrapolated limit coincides with the mitigation-free value.
    assert all(
        result.expectation == pytest.approx(base, abs=_ATOL) for result in scaled
    )
    mitigated = execute_with_lre(
        bell_circuit,
        exact_executor,
        zz_observable,
        degree=2,
        fold_multiplier=2,
        num_chunks=2,
    )
    assert mitigated == pytest.approx(base, abs=_ATOL)


def test_layer_multipliers_stay_on_the_odd_grid(wide_circuit: Circuit) -> None:
    _, scale_points = construct_circuits(
        wide_circuit,
        degree=2,
        fold_multiplier=2,
        num_chunks=2,
    )
    values = [value for vector in scale_points for value in vector]

    # An even fold multiplier makes 1 + fold_multiplier * level always odd, so
    # every chunk folds an integer number of times and no remainder remains.
    assert all(value == int(value) for value in values)
    assert all(int(value) % 2 == 1 for value in values)


def test_degree_fold_multiplier_and_num_chunks_are_required_keywords(
    bell_circuit: Circuit,
) -> None:
    # The three grid parameters are keyword-only by contract; a positional call
    # must be rejected rather than silently reinterpreted.
    with pytest.raises(TypeError):
        construct_circuits(bell_circuit, 2, 2, 2)
