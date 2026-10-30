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

"""Tests for probabilistic error amplification."""

from __future__ import annotations

import math
from itertools import pairwise

import pytest
from cqlib import Circuit
from cqlib.circuit import Instruction, StandardGate

from cqlib_algorithms.error_mitigation.quasiprobability import (
    amplify_depolarizing,
    depolarizing_quasiprobability_of_operation,
)
from cqlib_algorithms.error_mitigation.techniques.pea import (
    combine_results,
    construct_circuits,
    execute_with_pea,
)
from cqlib_algorithms.protocols import EstimatorRequest, EstimatorResult

_ATOL = 1e-9


def _amplified_weights(
    noise_level: float,
    scale_factor: float,
    /,
    *,
    num_qubits: int = 1,
) -> tuple[float, float]:
    """Return the identity weight and the shared Pauli-correction weight.

    The amplified expansion keeps the untouched gate at
    :math:`1 - s\\epsilon` and spreads :math:`s\\epsilon` evenly over the
    :math:`4^k - 1` non-identity Pauli corrections.
    """
    level = noise_level * scale_factor
    corrections = 4**num_qubits - 1
    return 1.0 - level, level / corrections


def test_the_amplified_representation_at_scale_one_is_the_original() -> None:
    circuit = Circuit(1)
    circuit.h(0)
    body_names = [operation.name for operation in circuit.operations]

    # Zero baseline noise keeps the amplified level at zero, so the identity
    # correction carries the whole weight and every sampled variant is the
    # untouched circuit.
    family = construct_circuits(
        circuit,
        noise_level=0.0,
        scale_factor=1.0,
        precision=0.01,
        seed=0,
    )

    assert family
    assert all(
        [operation.name for operation in variant.operations] == body_names
        for variant in family
    )


def test_the_amplified_coefficients_match_the_closed_form() -> None:
    noise_level, scale_factor = 0.05, 3.0
    circuit = Circuit(1)
    circuit.h(0)
    representation = amplify_depolarizing(
        Instruction.from_standard_gate(StandardGate.H),
        noise_level,
        scale_factor,
    )
    identity_weight, correction_weight = _amplified_weights(
        noise_level, scale_factor
    )

    # The amplified weights follow the Mari closed form directly: the identity
    # keeps 1 - s * eps and each of the three Pauli corrections takes
    # s * eps / (4 - 1).
    coefficients = representation.coefficients
    assert coefficients[0] == pytest.approx(identity_weight, abs=_ATOL)
    assert all(
        value == pytest.approx(correction_weight, abs=_ATOL)
        for value in coefficients[1:]
    )
    assert all(value >= 0.0 for value in coefficients)
    assert sum(coefficients) == pytest.approx(1.0, abs=_ATOL)
    # Non-negative weights that sum to one put the one-norm at exactly one, so
    # the family needs no sign and no Gamma compensation.
    assert representation.norm == pytest.approx(1.0, abs=_ATOL)

    pair = amplify_depolarizing(
        Instruction.from_standard_gate(StandardGate.CX),
        noise_level,
        scale_factor,
    )
    pair_identity, pair_correction = _amplified_weights(
        noise_level, scale_factor, num_qubits=2
    )
    # A two-qubit gate spreads the same level over fifteen corrections.
    assert pair.coefficients[0] == pytest.approx(pair_identity, abs=_ATOL)
    assert pair.coefficients[1] == pytest.approx(pair_correction, abs=_ATOL)
    assert pair.norm == pytest.approx(1.0, abs=_ATOL)


def test_the_amplified_overhead_is_monotone_in_the_scale_factor() -> None:
    noise_level = 0.02
    scale_factors = (1.0, 2.0, 3.0, 4.0)
    instruction = Instruction.from_standard_gate(StandardGate.H)

    # Amplifying the channel makes the PEC overhead needed to invert it grow
    # monotonically in the scale factor, which is the variance the amplified
    # families buy back at the extrapolation step.
    overheads = [
        depolarizing_quasiprobability_of_operation(
            instruction, factor * noise_level
        ).norm
        for factor in scale_factors
    ]

    assert all(a < b for a, b in pairwise(overheads))
    assert all(
        amplify_depolarizing(instruction, noise_level, factor).norm
        == pytest.approx(1.0, abs=_ATOL)
        for factor in scale_factors
    )


def test_combine_results_averages_without_a_gamma_compensation() -> None:
    values = (0.2, 0.6, 1.0)

    # The amplified weights are a genuine probability distribution, so the
    # estimate is a plain mean with no multiplication by a one-norm.
    assert combine_results(
        [EstimatorResult(expectation=value) for value in values]
    ) == pytest.approx(sum(values) / len(values), abs=_ATOL)

    with pytest.raises(ValueError, match="at least one result"):
        combine_results([])


def test_the_equivalent_noise_level_stays_below_one(
    exact_executor, rotation_circuit, z_observable
) -> None:
    circuit = rotation_circuit(0.9)

    # noise_level * max(scale_factors) = 1.0 reaches the boundary where the
    # identity weight would flip sign, so the configuration is rejected before
    # any circuit is built.
    with pytest.raises(ValueError, match="must be < 1"):
        execute_with_pea(
            circuit,
            exact_executor,
            z_observable,
            noise_level=0.5,
            scale_factors=(1.0, 2.0),
            extrapolation_method="linear",
        )


def test_rejects_an_unsupported_extrapolation_method(
    exact_executor, rotation_circuit, z_observable
) -> None:
    circuit = rotation_circuit(0.9)

    with pytest.raises(ValueError, match="extrapolation_method"):
        execute_with_pea(
            circuit,
            exact_executor,
            z_observable,
            noise_level=0.02,
            scale_factors=(1.0, 2.0, 3.0),
            extrapolation_method="cubic",
        )


def test_extrapolation_recovers_the_zero_amplification_limit(
    exact_executor, rotation_circuit, z_observable
) -> None:
    circuit = rotation_circuit(0.9)
    noise_level = 0.02
    scale_factors = (1.0, 2.0, 3.0)
    precision = 0.02

    bare = exact_executor.estimate(
        [EstimatorRequest(circuit=circuit, observable=z_observable)]
    )[0].expectation

    # Sampling the amplified family on a noiseless device realises a
    # depolarizing channel whose strength grows linearly with the scale
    # factor, so extrapolating to zero amplification restores the bare value
    # up to the sampling error of the family.
    mitigated = execute_with_pea(
        circuit,
        exact_executor,
        z_observable,
        noise_level=noise_level,
        scale_factors=scale_factors,
        extrapolation_method="richardson",
        precision=precision,
        seed=0,
    )

    num_samples = math.ceil(1.0 / precision**2)
    assert abs(mitigated - bare) <= 5.0 / math.sqrt(num_samples)
