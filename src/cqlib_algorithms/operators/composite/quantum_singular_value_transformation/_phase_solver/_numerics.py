r"""Shared numerical primitives for QSP phase synthesis.

This module contains the building blocks common to both the real and complex
polynomial solvers:

* the QSP signal / phase 2x2 blocks and the full response evaluation,
* the BFGS optimisation loop (loss, gradient, line search, update, driver),
* phase-symmetry expansion and wraparound,
* small scalar / integer validators.

The optimisation primitives (``_loss``, ``_finite_difference_gradient``,
``_line_search``) and the driver ``_run_bfgs`` take a callable ``loss_fn`` so
that each solver can inject its own phase parameterisation (symmetric phases
for real polynomials, the full phase sequence for complex polynomials) without
duplicating the loop body.

All routines here are pure ``numpy`` functions with no dependency on the
circuit layer.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

_COEFFICIENT_TOLERANCE = 1e-14
_BOUND_TOLERANCE = 1e-9
_ARMIJO = 1e-4
_MIN_STEP = 1e-8
_GRADIENT_STEP = 1e-7
_MAX_RESTARTS = 8


def _validate_positive_scalar(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, float | int):
        raise TypeError(f"{name} must be a real finite scalar")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise TypeError(f"{name} must be a real finite scalar")
    if scalar <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return scalar


def _validate_positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _chebyshev_nodes(count: int) -> np.ndarray:
    r"""Return the first-kind Chebyshev nodes on ``(0, 1]`` for training."""
    indices = np.arange(1, count + 1, dtype=np.float64)
    return np.cos((2.0 * indices - 1.0) * math.pi / (4.0 * count))


def _initial_inverse_hessian(num_free_phases: int, degree: int) -> np.ndarray:
    r"""Return the initial BFGS inverse-Hessian approximation."""
    diagonal = np.full(num_free_phases, 0.5, dtype=np.float64)
    if degree % 2 == 0:
        diagonal[-1] = 1.0
    return np.diag(diagonal)


def _symmetric_phases(hat_phases: np.ndarray, degree: int) -> np.ndarray:
    r"""Expand the free-phase vector to the full symmetric QSP phase sequence."""
    if degree % 2 == 1:
        return np.concatenate((hat_phases, hat_phases[::-1]))
    return np.concatenate((hat_phases, hat_phases[::-1][1:]))


def _wrap_phases(phases: np.ndarray) -> np.ndarray:
    r"""Wrap phase angles into the half-open interval ``(-pi, pi]``."""
    wrapped = (phases + math.pi) % (2.0 * math.pi) - math.pi
    wrapped[np.isclose(wrapped, -math.pi, atol=1e-15)] = math.pi
    return wrapped.astype(np.float64, copy=False)


def _phase_matrix(phase: float) -> np.ndarray:
    r"""Return the phase-rotation block ``diag(e^{i phi}, e^{-i phi})``."""
    return np.asarray(
        [
            [np.exp(1j * phase), 0.0],
            [0.0, np.exp(-1j * phase)],
        ],
        dtype=np.complex128,
    )


def _signal_matrix(signal_value: float) -> np.ndarray:
    r"""Return the QSP signal block ``W(x) = e^{i theta X}``."""
    amplitude = math.sqrt(max(0.0, 1.0 - signal_value * signal_value))
    return np.asarray(
        [
            [signal_value, 1j * amplitude],
            [1j * amplitude, signal_value],
        ],
        dtype=np.complex128,
    )


def _qsp_response(signal_value: float, phases: np.ndarray) -> complex:
    r"""Return the ``(0, 0)`` entry of the full QSP response matrix.

    Unlike the legacy ``_qsp_real_response`` helper, this returns the full
    complex value so the same routine can drive both real and complex
    polynomial fitting.
    """
    matrix = _phase_matrix(float(phases[0]))
    signal = _signal_matrix(signal_value)
    for phase in phases[1:]:
        matrix = matrix @ signal @ _phase_matrix(float(phase))
    return complex(matrix[0, 0])


def _loss(
    full_phases: np.ndarray,
    nodes: np.ndarray,
    targets: np.ndarray,
    *,
    real_only: bool = False,
) -> float:
    r"""Return the mean-squared fitting error on the Chebyshev-node grid.

    Parameters
    ----------
    full_phases
        The *expanded* QSP phase sequence (length ``degree + 1``).  Each
        solver is responsible for expanding its free parameter vector into
        this full sequence before calling ``_loss``.
    real_only
        When ``True`` the fitted quantity is ``Re(P(x))`` only (Corollary 10
        real-polynomial mode); the complex residual collapses to its real part.
        When ``False`` the full complex ``(0, 0)`` entry is fitted
        (Corollary 8 complex-polynomial mode).
    """
    predictions = np.asarray(
        [_qsp_response(float(node), full_phases) for node in nodes],
        dtype=np.complex128,
    )
    if real_only:
        predictions = predictions.real.astype(np.float64)
    residual = predictions - targets
    return float(np.mean(np.abs(residual) ** 2))


def _finite_difference_gradient(
    params: np.ndarray,
    loss_fn: Callable[[np.ndarray], float],
) -> np.ndarray:
    r"""Return the central-finite-difference gradient of ``loss_fn``."""
    gradient = np.empty_like(params)
    for index in range(params.size):
        step = _GRADIENT_STEP * max(1.0, abs(float(params[index])))
        forward = params.copy()
        backward = params.copy()
        forward[index] += step
        backward[index] -= step
        gradient[index] = (loss_fn(forward) - loss_fn(backward)) / (2.0 * step)
    return gradient


def _line_search(
    params: np.ndarray,
    direction: np.ndarray,
    current_loss: float,
    directional_derivative: float,
    loss_fn: Callable[[np.ndarray], float],
) -> float:
    r"""Return an Armijo backtracking step length."""
    step = 1.0
    while step >= _MIN_STEP:
        trial_loss = loss_fn(params + step * direction)
        if trial_loss <= current_loss + _ARMIJO * step * directional_derivative:
            return step
        step *= 0.5
    return 0.0


def _bfgs_update(
    inverse_hessian: np.ndarray,
    displacement: np.ndarray,
    curvature: np.ndarray,
    curvature_dot: float,
) -> np.ndarray:
    r"""Return the BFGS-updated inverse-Hessian approximation."""
    rho = 1.0 / curvature_dot
    identity = np.eye(inverse_hessian.shape[0], dtype=np.float64)
    outer_sy = np.outer(displacement, curvature)
    outer_ys = np.outer(curvature, displacement)
    outer_ss = np.outer(displacement, displacement)
    updated: np.ndarray = np.asarray(
        (identity - rho * outer_sy)
        @ inverse_hessian
        @ (identity - rho * outer_ys)
        + rho * outer_ss,
        dtype=np.float64,
    )
    return updated


def _run_bfgs(
    *,
    num_free_phases: int,
    degree: int,
    loss_fn: Callable[[np.ndarray], float],
    tolerance: float,
    max_iterations: int,
    gradient_tolerance: float,
    wrap: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    r"""Run the shared BFGS loop driven by an arbitrary ``loss_fn``.

    The first attempt starts from the canonical point ``(pi/4, 0, ..., 0)``.
    Convergence means ``loss <= tolerance``; a vanishing gradient is *not*
    treated as success, because the canonical start can sit on a saddle point
    (zero gradient, large loss).  Whenever an attempt ends without reaching the
    tolerance -- a stationary but non-converged point, a stalled line search, or
    an exhausted iteration budget -- the solve is restarted from a uniformly
    random point.  After ``_MAX_RESTARTS`` extra attempts a ``RuntimeError`` is
    raised.

    Parameters
    ----------
    loss_fn
        Callable mapping the free parameter vector to the scalar loss.
    wrap
        Callable mapping the converged free parameter vector to the final
        (full, wrapped) phase sequence.
    """
    generator = np.random.default_rng()
    for attempt in range(_MAX_RESTARTS + 1):
        if attempt == 0:
            current = np.zeros(num_free_phases, dtype=np.float64)
            current[0] = math.pi / 4.0
        else:
            current = generator.uniform(
                -math.pi,
                math.pi,
                size=num_free_phases,
            )

        inverse_hessian = _initial_inverse_hessian(num_free_phases, degree)
        loss = loss_fn(current)
        gradient = _finite_difference_gradient(current, loss_fn)

        for _ in range(max_iterations):
            if loss <= tolerance:
                return wrap(current)
            if np.linalg.norm(gradient, ord=np.inf) <= gradient_tolerance:
                break

            direction = -inverse_hessian @ gradient
            directional_derivative = float(np.dot(gradient, direction))
            if directional_derivative >= 0.0:
                direction = -gradient
                directional_derivative = -float(np.dot(gradient, gradient))

            step = _line_search(
                current,
                direction,
                loss,
                directional_derivative,
                loss_fn,
            )
            if step < _MIN_STEP:
                break

            next_current = current + step * direction
            next_loss = loss_fn(next_current)
            next_gradient = _finite_difference_gradient(next_current, loss_fn)

            displacement = next_current - current
            curvature = next_gradient - gradient
            curvature_dot = float(np.dot(displacement, curvature))
            if curvature_dot > 1e-12:
                inverse_hessian = _bfgs_update(
                    inverse_hessian,
                    displacement,
                    curvature,
                    curvature_dot,
                )

            current = next_current
            loss = next_loss
            gradient = next_gradient

    raise RuntimeError(
        "QSP phase solver did not converge within max_iterations",
    )
