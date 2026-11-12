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

"""Tests for the zero-noise extrapolation factories."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import exp

import pytest

from cqlib_algorithms.error_mitigation.inference.factories import (
    AdaExpFactory,
    ConvergenceWarning,
    ExpFactory,
    ExtrapolationError,
    Factory,
    LinearFactory,
    PolyFactory,
    RichardsonFactory,
    _richardson_coefficients,
)

_ATOL = 1e-9
_SEED_GRID = (1.0, 2.0, 3.0)


def _push(factory: Factory, values: Sequence[float], /) -> None:
    """Push one value per scale factor of the fixed seed grid."""
    for factor, value in zip(_SEED_GRID, values, strict=True):
        factory.push(factor, value)


def _collect(factory: Factory, sampler: Callable[[float], float]) -> list[float]:
    """Run a factory to convergence, returning the scale factors it visited."""
    plan: list[float] = []
    while (factor := factory.next_scale_factor()) is not None:
        plan.append(factor)
        factory.push(factor, sampler(factor))
    return plan


def test_richardson_coefficients_at_one_two_three_are_exactly_three_minus_three_one() -> (
    None
):
    weights = _richardson_coefficients(_SEED_GRID)

    assert weights.tolist() == pytest.approx([3.0, -3.0, 1.0], abs=_ATOL)
    # The weights annihilate every power of the scale factor below the number
    # of points and sum to one, which is what makes the fit exact.
    assert float(weights.sum()) == pytest.approx(1.0, abs=_ATOL)
    assert float(weights @ [1.0, 2.0, 3.0]) == pytest.approx(0.0, abs=_ATOL)
    assert float(weights @ [1.0, 4.0, 9.0]) == pytest.approx(0.0, abs=_ATOL)


def test_a_linear_noise_response_is_recovered_exactly_by_richardson() -> None:
    factory = RichardsonFactory(_SEED_GRID)
    _push(factory, [0.3 + 0.7 * factor for factor in _SEED_GRID])

    assert factory.reduce() == pytest.approx(0.3, abs=_ATOL)


def test_poly_factory_recovers_a_polynomial_of_its_degree() -> None:
    factory = PolyFactory(_SEED_GRID, degree=2)
    _push(
        factory,
        [0.25 + 0.5 * factor + 0.125 * factor**2 for factor in _SEED_GRID],
    )

    assert factory.degree == 2
    assert factory.reduce() == pytest.approx(0.25, abs=_ATOL)


def test_linear_factory_is_a_poly_factory_of_degree_one() -> None:
    factory = LinearFactory(_SEED_GRID)

    assert isinstance(factory, PolyFactory)
    assert factory.degree == 1


def test_exp_factory_recovers_a_decaying_exponential() -> None:
    factory = ExpFactory(_SEED_GRID)
    _push(factory, [0.6 + 1.8 * exp(-0.9 * factor) for factor in _SEED_GRID])

    assert factory.asymptote is None
    assert factory.reduce() == pytest.approx(2.4, abs=_ATOL)


def test_exp_factory_with_a_zero_asymptote_matches_the_closed_form() -> None:
    factory = ExpFactory(_SEED_GRID, asymptote=0.0)
    _push(factory, [1.8 * exp(-0.9 * factor) for factor in _SEED_GRID])

    assert factory.asymptote == 0.0
    assert factory.reduce() == pytest.approx(1.8, abs=_ATOL)


def test_exp_factory_rejects_a_sign_changing_residual() -> None:
    factory = ExpFactory(_SEED_GRID, asymptote=0.0)
    _push(factory, [0.4, -0.3, 0.1])

    with pytest.raises(ExtrapolationError, match="cross the assumed asymptote"):
        factory.reduce()


def test_ada_exp_factory_estimates_the_asymptote_from_equally_spaced_seeds() -> None:
    factory = AdaExpFactory(steps=4, scale_factor=2.0)
    plan = _collect(factory, lambda factor: 0.6 + 1.8 * exp(-0.9 * factor))

    # The first three points form the equally spaced seed triplet that the
    # asymptote estimate relies on; the fourth is adaptively placed.
    assert plan[:3] == pytest.approx([1.0, 2.0, 3.0], abs=_ATOL)
    assert factory.is_converged()
    assert factory.next_scale_factor() is None
    assert factory.reduce() == pytest.approx(2.4, abs=_ATOL)


def test_ada_exp_factory_raises_when_the_asymptote_cannot_be_estimated() -> None:
    factory = AdaExpFactory(steps=4, scale_factor=2.0)
    factory.push(1.0, 2.0)
    factory.push(2.0, 1.0)

    # Fewer than three seed points leaves the closed-form asymptote estimate
    # without enough data.
    with pytest.raises(ExtrapolationError, match="at least 3"):
        factory.reduce()


def test_ada_exp_factory_caches_the_resolved_asymptote() -> None:
    factory = AdaExpFactory(steps=4, scale_factor=2.0)
    plan = _collect(factory, lambda factor: 0.6 + 1.8 * exp(-0.9 * factor))

    first = factory.reduce()
    cached = factory._estimated_asymptote
    assert cached is not None
    assert cached == pytest.approx(0.6, abs=_ATOL)
    # The estimate is taken once and reused, because the adaptive points are
    # deliberately not equally spaced.
    assert factory._resolved_asymptote(plan) is cached
    assert factory.reduce() == first


def test_ill_conditioned_scale_factors_are_rejected_by_the_fit() -> None:
    spacing = 1.0e-9
    factors = (1.0, 1.0 + spacing, 1.0 + 2.0 * spacing)
    factory = PolyFactory(factors, degree=2)
    for factor in factors:
        factory.push(factor, 0.25 + 0.5 * factor + 0.125 * factor**2)

    # The normal equations square the condition number, so points this close
    # together make the Gram matrix singular instead of yielding a limit.
    with pytest.raises(ExtrapolationError, match="singular"):
        factory.reduce()


def test_convergence_warning_subclasses_user_warning() -> None:
    assert issubclass(ConvergenceWarning, UserWarning)


def test_the_base_factory_raises_not_implemented_error() -> None:
    # Factory is the shared protocol; its methods document the contract and
    # raise when they are called without a concrete model behind them.
    with pytest.raises(NotImplementedError):
        Factory.push(object(), 1.0, 1.0)
    with pytest.raises(NotImplementedError):
        Factory.reduce(object())
    with pytest.raises(TypeError):
        Factory()  # type: ignore[misc]


def test_a_batched_factory_needs_at_least_two_scale_factors() -> None:
    with pytest.raises(ValueError, match="At least 2 scale factors"):
        RichardsonFactory((1.0,))


def test_reduce_before_convergence_raises_extrapolation_error() -> None:
    factory = PolyFactory(_SEED_GRID, degree=2)
    factory.push(1.0, 1.0)

    assert not factory.is_converged()
    assert factory.next_scale_factor() == 2.0
    with pytest.raises(ExtrapolationError, match="2 more expectation value"):
        factory.reduce()
