"""Internal numerical helpers for quantum signal processing phase synthesis."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

import numpy as np
from numpy.polynomial.chebyshev import chebval

_COEFFICIENT_TOLERANCE = 1e-14
_BOUND_TOLERANCE = 1e-9
_ARMIJO = 1e-4
_MIN_STEP = 1e-8
_GRADIENT_STEP = 1e-7


def solve_qsp_phases(
    coefficients: Sequence[float] | np.ndarray,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 300,
    gradient_tolerance: float = 1e-10,
) -> np.ndarray:
    """Return a symmetric QSP phase sequence for a real Chebyshev polynomial.

    Parameters
    ----------
    coefficients
        Chebyshev-basis coefficients in ascending order. The implied degree is
        the highest non-zero coefficient index after trimming trailing entries
        smaller than ``1e-14`` in magnitude.
    tolerance
        Target mean-squared error on the Chebyshev-node training grid.
    max_iterations
        Maximum number of BFGS iterations.
    gradient_tolerance
        Infinity-norm tolerance used to stop when the loss surface is locally
        flat enough.

    Returns
    -------
    numpy.ndarray
        Full symmetric phase sequence of length ``degree + 1``.

    Raises
    ------
    TypeError
        If coefficients or solver controls have invalid types.
    ValueError
        If the polynomial violates the QSP parity or boundedness constraints,
        or if the solver controls are outside their valid ranges.
    RuntimeError
        If the internal quasi-Newton solve does not converge.
    """
    trimmed = _normalize_coefficients(coefficients)
    degree = len(trimmed) - 1

    tolerance_float = _validate_positive_scalar(tolerance, name="tolerance")
    gradient_tolerance_float = _validate_positive_scalar(
        gradient_tolerance,
        name="gradient_tolerance",
    )
    max_iterations_int = _validate_positive_integer(
        max_iterations,
        name="max_iterations",
    )

    _validate_qsp_constraints(trimmed)

    if degree == 0:
        return np.asarray([math.acos(float(trimmed[0]))], dtype=np.float64)
    if degree == 1:
        phase = 0.5 * math.acos(float(trimmed[1]))
        return np.asarray([phase, phase], dtype=np.float64)

    num_free_phases = (degree + 2) // 2
    nodes = _chebyshev_nodes(num_free_phases)
    targets = chebval(nodes, trimmed)
    current = np.zeros(num_free_phases, dtype=np.float64)
    current[0] = math.pi / 4.0

    inverse_hessian = _initial_inverse_hessian(num_free_phases, degree)
    loss = _loss(current, degree, nodes, targets)
    gradient = _finite_difference_gradient(current, degree, nodes, targets)

    for _ in range(max_iterations_int):
        if loss <= tolerance_float or np.linalg.norm(gradient, ord=np.inf) <= (
            gradient_tolerance_float
        ):
            return _wrap_phases(_symmetric_phases(current, degree))

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
            degree,
            nodes,
            targets,
        )
        if step < _MIN_STEP:
            break

        next_current = current + step * direction
        next_loss = _loss(next_current, degree, nodes, targets)
        next_gradient = _finite_difference_gradient(
            next_current,
            degree,
            nodes,
            targets,
        )

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

    raise RuntimeError("QSP phase solver did not converge within max_iterations")


def _normalize_coefficients(
    coefficients: Sequence[float] | np.ndarray,
) -> np.ndarray:
    if isinstance(coefficients, np.ndarray):
        flattened = coefficients.reshape(-1).tolist()
    elif isinstance(coefficients, Sequence) and not isinstance(
        coefficients,
        (str, bytes),
    ):
        flattened = list(coefficients)
    else:
        raise TypeError("coefficients must be a one-dimensional real sequence")

    if not flattened:
        raise ValueError("coefficients must not be empty")

    values = np.empty(len(flattened), dtype=np.float64)
    for index, value in enumerate(flattened):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("coefficients must contain real finite scalars")
        scalar = float(value)
        if not math.isfinite(scalar):
            raise TypeError("coefficients must contain real finite scalars")
        values[index] = scalar

    nonzero = np.flatnonzero(np.abs(values) > _COEFFICIENT_TOLERANCE)
    if nonzero.size == 0:
        return np.asarray([0.0], dtype=np.float64)
    return values[: int(nonzero[-1]) + 1]


def _validate_qsp_constraints(coefficients: np.ndarray) -> None:
    degree = len(coefficients) - 1
    parity = degree % 2
    for index, value in enumerate(coefficients):
        if index % 2 != parity and abs(float(value)) > _COEFFICIENT_TOLERANCE:
            raise ValueError(
                "coefficients must have Chebyshev parity matching the degree"
            )

    grid_size = max(129, 16 * degree + 1)
    grid = np.linspace(-1.0, 1.0, grid_size, dtype=np.float64)
    max_magnitude = float(np.max(np.abs(chebval(grid, coefficients))))
    if max_magnitude > 1.0 + _BOUND_TOLERANCE:
        raise ValueError("target polynomial must satisfy |f(x)| <= 1 on [-1, 1]")


def _validate_positive_scalar(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
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
    indices = np.arange(1, count + 1, dtype=np.float64)
    return np.cos((2.0 * indices - 1.0) * math.pi / (4.0 * count))


def _initial_inverse_hessian(num_free_phases: int, degree: int) -> np.ndarray:
    diagonal = np.full(num_free_phases, 0.5, dtype=np.float64)
    if degree % 2 == 0:
        diagonal[-1] = 1.0
    return np.diag(diagonal)


def _loss(
    hat_phases: np.ndarray,
    degree: int,
    nodes: np.ndarray,
    targets: np.ndarray,
) -> float:
    phases = _symmetric_phases(hat_phases, degree)
    predictions = np.asarray(
        [_qsp_real_response(float(node), phases) for node in nodes],
        dtype=np.float64,
    )
    residual = predictions - targets
    return float(np.mean(residual * residual))


def _finite_difference_gradient(
    hat_phases: np.ndarray,
    degree: int,
    nodes: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    gradient = np.empty_like(hat_phases)
    for index in range(hat_phases.size):
        step = _GRADIENT_STEP * max(1.0, abs(float(hat_phases[index])))
        forward = hat_phases.copy()
        backward = hat_phases.copy()
        forward[index] += step
        backward[index] -= step
        gradient[index] = (
            _loss(forward, degree, nodes, targets)
            - _loss(backward, degree, nodes, targets)
        ) / (2.0 * step)
    return gradient


def _line_search(
    hat_phases: np.ndarray,
    direction: np.ndarray,
    current_loss: float,
    directional_derivative: float,
    degree: int,
    nodes: np.ndarray,
    targets: np.ndarray,
) -> float:
    step = 1.0
    while step >= _MIN_STEP:
        trial_loss = _loss(hat_phases + step * direction, degree, nodes, targets)
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


def _symmetric_phases(hat_phases: np.ndarray, degree: int) -> np.ndarray:
    if degree % 2 == 1:
        return np.concatenate((hat_phases, hat_phases[::-1]))
    return np.concatenate((hat_phases, hat_phases[::-1][1:]))


def _wrap_phases(phases: np.ndarray) -> np.ndarray:
    wrapped = (phases + math.pi) % (2.0 * math.pi) - math.pi
    wrapped[np.isclose(wrapped, -math.pi, atol=1e-15)] = math.pi
    return wrapped.astype(np.float64, copy=False)


def _qsp_real_response(signal_value: float, phases: np.ndarray) -> float:
    matrix = _phase_matrix(float(phases[0]))
    signal = _signal_matrix(signal_value)
    for phase in phases[1:]:
        matrix = matrix @ signal @ _phase_matrix(float(phase))
    return float(np.real(matrix[0, 0]))


def _signal_matrix(signal_value: float) -> np.ndarray:
    amplitude = math.sqrt(max(0.0, 1.0 - signal_value * signal_value))
    return np.asarray(
        [
            [signal_value, 1j * amplitude],
            [1j * amplitude, signal_value],
        ],
        dtype=np.complex128,
    )


def _phase_matrix(phase: float) -> np.ndarray:
    return np.asarray(
        [
            [np.exp(1j * phase), 0.0],
            [0.0, np.exp(-1j * phase)],
        ],
        dtype=np.complex128,
    )
