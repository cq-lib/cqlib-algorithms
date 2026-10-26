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

"""Tests for the amplified quasi-probability expansions used by PEA."""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise

import pytest
from cqlib.circuit import Instruction, StandardGate

from cqlib_algorithms.error_mitigation.quasiprobability import (
    amplify_depolarizing,
    depolarizing_quasiprobability_of_operation,
)

_NOISE_LEVEL = 0.02


def _hadamard() -> Instruction:
    """Return the one-qubit instruction used across the tests."""
    return Instruction.from_standard_gate(StandardGate.H)


def _cx() -> Instruction:
    """Return the two-qubit instruction used across the tests."""
    return Instruction.from_standard_gate(StandardGate.CX)


def test_identity_weight_is_one_minus_the_amplified_level() -> None:
    expansion = amplify_depolarizing(_hadamard(), _NOISE_LEVEL, 1.0)
    assert expansion.coefficients[0] == pytest.approx(1.0 - _NOISE_LEVEL)


def test_weights_are_non_negative_and_sum_to_one() -> None:
    expansion = amplify_depolarizing(_hadamard(), _NOISE_LEVEL, 3.0)
    assert all(weight >= 0.0 for weight in expansion.coefficients)
    assert sum(expansion.coefficients) == pytest.approx(1.0)
    assert expansion.norm == pytest.approx(1.0)


@pytest.mark.parametrize("scale_factor", [1.0, 2.0, 3.0, 4.0])
def test_one_qubit_correction_weight_scales_with_the_scale_factor(
    scale_factor: float,
) -> None:
    expansion = amplify_depolarizing(_hadamard(), _NOISE_LEVEL, scale_factor)
    assert expansion.coefficients[1] == pytest.approx(scale_factor * _NOISE_LEVEL / 3.0)


def test_two_qubit_expansion_spreads_over_fifteen_corrections() -> None:
    expansion = amplify_depolarizing(_cx(), _NOISE_LEVEL, 2.0)
    assert len(expansion.basis) == 16
    assert expansion.coefficients[0] == pytest.approx(1.0 - 2.0 * _NOISE_LEVEL)
    assert expansion.coefficients[1] == pytest.approx(2.0 * _NOISE_LEVEL / 15.0)
    assert expansion.norm == pytest.approx(1.0)


def test_scale_one_is_the_baseline_amplification() -> None:
    baseline = amplify_depolarizing(_hadamard(), _NOISE_LEVEL, 1.0)
    assert baseline.coefficients == (1.0 - _NOISE_LEVEL, *3 * [_NOISE_LEVEL / 3.0])


def test_amplify_rejects_a_three_qubit_gate() -> None:
    gate = Instruction.from_standard_gate(StandardGate.CCX)
    with pytest.raises(ValueError):
        amplify_depolarizing(gate, _NOISE_LEVEL, 1.0)


def test_amplify_rejects_an_over_amplified_level() -> None:
    with pytest.raises(ValueError):
        amplify_depolarizing(_hadamard(), _NOISE_LEVEL, 60.0)


def test_amplify_rejects_a_non_positive_scale_factor() -> None:
    with pytest.raises(ValueError):
        amplify_depolarizing(_hadamard(), _NOISE_LEVEL, 0.0)


def test_amplify_rejects_a_negative_noise_level() -> None:
    with pytest.raises(ValueError):
        amplify_depolarizing(_hadamard(), -0.1, 1.0)


@pytest.mark.parametrize("noise_level", [float("inf"), float("nan")])
def test_amplify_rejects_a_non_finite_noise_level(noise_level: float) -> None:
    with pytest.raises(ValueError):
        amplify_depolarizing(_hadamard(), noise_level, 1.0)


def test_the_amplified_representation_at_scale_one_is_the_original() -> None:
    expansion = amplify_depolarizing(_hadamard(), _NOISE_LEVEL, 1.0)

    # At scale one the amplified level is the bare noise level, so the first
    # term is still the untouched gate: the identity correction carries weight
    # 1 - noise_level and leaves the remaining weight for the Pauli
    # corrections. This is the quasiprobability-layer reading of checkpoint
    # PEA-1, where the originally sampled variant is the unmodified circuit.
    assert expansion.ideal.standard_gate == StandardGate.H
    assert expansion.basis[0].standard_gate == StandardGate.I
    assert expansion.coefficients[0] == pytest.approx(1.0 - _NOISE_LEVEL)
    assert expansion.coefficients[0] > max(expansion.coefficients[1:])


@pytest.mark.parametrize(
    ("gate_factory", "corrections"),
    [(_hadamard, 3), (_cx, 15)],
)
def test_the_amplified_coefficients_match_the_closed_form(
    gate_factory: Callable[[], Instruction],
    corrections: int,
) -> None:
    scale_factor = 2.5
    amplified_level = scale_factor * _NOISE_LEVEL
    expansion = amplify_depolarizing(gate_factory(), _NOISE_LEVEL, scale_factor)

    # Mari 2021 closed form: the identity keeps 1 - s * noise_level and each of
    # the 4**k - 1 Pauli corrections takes s * noise_level / (4**k - 1), so the
    # weights stay non-negative and sum to one.
    assert len(expansion.coefficients) == corrections + 1
    assert expansion.coefficients[0] == pytest.approx(1.0 - amplified_level)
    assert all(
        weight == pytest.approx(amplified_level / corrections)
        for weight in expansion.coefficients[1:]
    )


def test_the_amplified_overhead_is_monotone_in_the_scale_factor() -> None:
    scale_factors = (1.0, 2.0, 3.0, 4.0)
    gate = _hadamard()

    # Checkpoint PEA-2: the amplified family itself carries no overhead
    # (Gamma = 1), but the PEC overhead needed to invert the amplified channel
    # -- the variance the extrapolation buys back -- grows with the factor.
    overheads = [
        depolarizing_quasiprobability_of_operation(
            gate,
            factor * _NOISE_LEVEL,
        ).norm
        for factor in scale_factors
    ]
    assert all(a < b for a, b in pairwise(overheads))


def test_the_scale_factor_times_the_noise_level_must_stay_below_one() -> None:
    # The identity weight 1 - s * noise_level flips sign once s * noise_level
    # reaches one, so the expansion stops being a probability distribution.
    with pytest.raises(ValueError, match="reaches one"):
        amplify_depolarizing(_hadamard(), 0.25, 4.0)

    # Just below the boundary the weights are still non-negative and sum to
    # one, so the configuration is accepted.
    accepted = amplify_depolarizing(_hadamard(), 0.25, 3.9)
    assert min(accepted.coefficients) > 0.0
    assert sum(accepted.coefficients) == pytest.approx(1.0)
