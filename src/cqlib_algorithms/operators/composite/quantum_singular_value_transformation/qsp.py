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

r"""Single-qubit quantum signal processing circuit construction."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

import numpy as np
from cqlib import Circuit
from cqlib.circuit.gates import StandardGate

from ._phase_solver import (
    polynomial_to_chebyshev,
    solve_complex_qsp_phases,
    solve_qsp_phases,
)


class QuantumSignalProcessing:
    r"""Construct the single-qubit SU(2) core of a QSP sequence.

    This component builds the one-qubit circuit for

    .. math::

        U(x, \Phi) = e^{i\phi_0 Z} W(x) e^{i\phi_1 Z} \cdots W(x) e^{i\phi_d Z},

    where

    .. math::

        W(x) =
        \begin{pmatrix}
            x & i\sqrt{1 - x^2} \\
            i\sqrt{1 - x^2} & x
        \end{pmatrix}
        = e^{i\theta X}, \quad x = \cos\theta.

    The implementation realizes the exact same SU(2) unitary as an alternating
    sequence of ``RZ`` and ``RX`` gates on one qubit:

    .. math::

        e^{i\phi Z} = R_Z(-2\phi), \quad W(x) = R_X(-2\theta).

    This class intentionally covers the single-qubit signal-processing core
    only. It does not build a block-encoding or qubitization iterate.

    Args:
        phases: Full QSP phase sequence ``(phi_0, ..., phi_d)``.
        signal_value: Real signal value ``x`` in ``[-1, 1]``. Exactly one of
            ``signal_value`` and ``signal_angle`` must be provided.
        signal_angle: Real angle ``theta`` satisfying ``x = cos(theta)``. Exactly one of
            ``signal_value`` and ``signal_angle`` must be provided.

    Raises:
        TypeError: If the phase sequence or signal arguments have invalid types.
        ValueError: If the phase sequence is empty, if the signal specification is
            ambiguous, or if ``signal_value`` lies outside ``[-1, 1]``.

    """

    __slots__ = ("_phases", "_signal_angle", "_signal_value")

    _phases: tuple[float, ...]
    _signal_angle: float
    _signal_value: float

    def __init__(
        self,
        phases: Sequence[float] | np.ndarray,
        *,
        signal_value: float | None = None,
        signal_angle: float | None = None,
    ) -> None:
        self._phases = _normalize_phases(phases)
        self._signal_value, self._signal_angle = _resolve_signal(
            signal_value=signal_value,
            signal_angle=signal_angle,
        )

    @property
    def phases(self) -> tuple[float, ...]:
        r"""Return the immutable full QSP phase sequence."""
        return self._phases

    @property
    def signal_value(self) -> float:
        r"""Return the configured signal value ``x = cos(theta)``."""
        return self._signal_value

    @property
    def signal_angle(self) -> float:
        r"""Return the configured signal angle ``theta`` in radians."""
        return self._signal_angle

    @property
    def num_qubits(self) -> int:
        r"""Return the width of the local QSP core circuit."""
        return 1

    def build_circuit(self) -> Circuit:
        r"""Build and return a fresh one-qubit QSP circuit."""
        circuit = Circuit(1)
        circuit.append_gate(StandardGate.RZ(-2.0 * self.phases[0]), [0])
        for phase in self.phases[1:]:
            circuit.append_gate(StandardGate.RX(-2.0 * self.signal_angle), [0])
            circuit.append_gate(StandardGate.RZ(-2.0 * phase), [0])
        return circuit

    @staticmethod
    def solve_phases(
        coefficients: Sequence[float] | np.ndarray,
        *,
        tolerance: float = 1e-12,
        max_iterations: int = 300,
        gradient_tolerance: float = 1e-10,
    ) -> np.ndarray:
        r"""Return QSP phases for a real Chebyshev polynomial.

        Args:
            coefficients: Chebyshev-basis coefficients in ascending order.
            tolerance: Target mean-squared fitting loss on the Chebyshev-node grid.
            max_iterations: Maximum number of internal BFGS iterations.
            gradient_tolerance: Infinity-norm gradient stopping tolerance.

        Returns:
            Full symmetric phase sequence.

        Raises:
            TypeError: If the coefficient sequence or solver controls have invalid types.
            ValueError: If the polynomial violates the standard real-QSP constraints.
            RuntimeError: If the internal solver does not converge.

        """
        return solve_qsp_phases(
            coefficients,
            tolerance=tolerance,
            max_iterations=max_iterations,
            gradient_tolerance=gradient_tolerance,
        )

    @classmethod
    def from_chebyshev_coefficients(
        cls,
        coefficients: Sequence[float] | np.ndarray,
        *,
        signal_value: float | None = None,
        signal_angle: float | None = None,
        tolerance: float = 1e-12,
        max_iterations: int = 300,
        gradient_tolerance: float = 1e-10,
    ) -> QuantumSignalProcessing:
        r"""Solve the real-QSP phase problem for a Chebyshev-basis polynomial.

        ``P(x)`` is specified by its Chebyshev-basis coefficients
        ``P(x) = \sum_k c_k T_k(x)`` and the returned operator realises the
        real polynomial ``P`` on the single-qubit signal register.

        Args:
            coefficients: Chebyshev-basis coefficients ``(c_0, c_1, ..., c_d)`` in ascending
                order.
            signal_value: Real signal value ``x`` in ``[-1, 1]``.
            signal_angle: Real signal angle ``theta`` satisfying ``x = cos(theta)``.
            tolerance: Target mean-squared fitting loss on the Chebyshev-node grid.
            max_iterations: Maximum number of internal BFGS iterations.
            gradient_tolerance: Infinity-norm gradient stopping tolerance.

        Returns:
            Configured QSP operator with solved phases and bound signal.

        """
        return cls(
            cls.solve_phases(
                coefficients,
                tolerance=tolerance,
                max_iterations=max_iterations,
                gradient_tolerance=gradient_tolerance,
            ),
            signal_value=signal_value,
            signal_angle=signal_angle,
        )

    @classmethod
    def from_polynomial_coefficients(
        cls,
        coefficients: Sequence[float] | np.ndarray,
        *,
        signal_value: float | None = None,
        signal_angle: float | None = None,
        tolerance: float = 1e-12,
        max_iterations: int = 300,
        gradient_tolerance: float = 1e-10,
    ) -> QuantumSignalProcessing:
        r"""Solve the real-QSP phase problem for a power-basis polynomial.

        ``P(x)`` is specified by its standard power (monomial) coefficients
        ``P(x) = a_0 + a_1 x + ... + a_d x^d``.  The coefficients are first
        converted to the Chebyshev basis before solving.

        Args:
            coefficients: Power-basis coefficients ``(a_0, a_1, ..., a_d)`` in ascending
                order.
            signal_value: Real signal value ``x`` in ``[-1, 1]``.
            signal_angle: Real signal angle ``theta`` satisfying ``x = cos(theta)``.
            tolerance: Target mean-squared fitting loss on the Chebyshev-node grid.
            max_iterations: Maximum number of internal BFGS iterations.
            gradient_tolerance: Infinity-norm gradient stopping tolerance.

        Returns:
            Configured QSP operator with solved phases and bound signal.

        """
        return cls.from_chebyshev_coefficients(
            polynomial_to_chebyshev(coefficients),
            signal_value=signal_value,
            signal_angle=signal_angle,
            tolerance=tolerance,
            max_iterations=max_iterations,
            gradient_tolerance=gradient_tolerance,
        )

    @staticmethod
    def solve_complex_phases(
        coefficients: Sequence[complex] | np.ndarray,
        *,
        tolerance: float = 1e-12,
        max_iterations: int = 300,
        gradient_tolerance: float = 1e-10,
    ) -> np.ndarray:
        r"""Return QSP phases for a complex Chebyshev polynomial.

        Unlike :meth:`solve_phases`, this fits the full complex ``(0, 0)``
        entry of the QSP response matrix to the target polynomial
        ``P in C[x]`` (Corollary 8).  All ``degree + 1`` phases are free
        real parameters; no symmetry constraint is imposed.

        Args:
            coefficients: Chebyshev-basis coefficients ``(c_0, c_1, ..., c_d)`` in ascending
                order.  Coefficients may be complex.
            tolerance: Target mean-squared fitting loss on the Chebyshev-node grid.
            max_iterations: Maximum number of internal BFGS iterations.
            gradient_tolerance: Infinity-norm gradient stopping tolerance.

        Returns:
            Full (non-symmetric) phase sequence of length ``degree + 1``.

        Raises:
            TypeError: If the coefficient sequence or solver controls have invalid types.
            ValueError: If the polynomial violates the QSP parity or boundedness
                constraints.
            RuntimeError: If the internal solver does not converge.

        """
        return solve_complex_qsp_phases(
            coefficients,
            tolerance=tolerance,
            max_iterations=max_iterations,
            gradient_tolerance=gradient_tolerance,
        )

    @classmethod
    def from_complex_chebyshev_coefficients(
        cls,
        coefficients: Sequence[complex] | np.ndarray,
        *,
        signal_value: float | None = None,
        signal_angle: float | None = None,
        tolerance: float = 1e-12,
        max_iterations: int = 300,
        gradient_tolerance: float = 1e-10,
    ) -> QuantumSignalProcessing:
        r"""Solve the complex-QSP phase problem for a Chebyshev-basis polynomial.

        ``P(x)`` is specified by its complex Chebyshev-basis coefficients
        ``P(x) = \sum_k c_k T_k(x)`` and the returned operator realises the
        full complex polynomial ``P`` on the single-qubit signal register
        (Corollary 8).

        Args:
            coefficients: Chebyshev-basis coefficients ``(c_0, c_1, ..., c_d)`` in ascending
                order.  May be complex.
            signal_value: Real signal value ``x`` in ``[-1, 1]``.
            signal_angle: Real angle ``theta`` satisfying ``x = cos(theta)``.
            tolerance: Target mean-squared fitting loss on the Chebyshev-node grid.
            max_iterations: Maximum number of internal BFGS iterations.
            gradient_tolerance: Infinity-norm gradient stopping tolerance.

        Returns:
            Configured QSP operator with solved phases and bound signal.

        """
        return cls(
            cls.solve_complex_phases(
                coefficients,
                tolerance=tolerance,
                max_iterations=max_iterations,
                gradient_tolerance=gradient_tolerance,
            ),
            signal_value=signal_value,
            signal_angle=signal_angle,
        )

    @classmethod
    def from_complex_polynomial_coefficients(
        cls,
        coefficients: Sequence[complex] | np.ndarray,
        *,
        signal_value: float | None = None,
        signal_angle: float | None = None,
        tolerance: float = 1e-12,
        max_iterations: int = 300,
        gradient_tolerance: float = 1e-10,
    ) -> QuantumSignalProcessing:
        r"""Solve the complex-QSP phase problem for a power-basis polynomial.

        ``P(x)`` is specified by its standard power (monomial) coefficients
        ``P(x) = a_0 + a_1 x + ... + a_d x^d``.  The coefficients are first
        converted to the Chebyshev basis before solving.

        Args:
            coefficients: Power-basis coefficients ``(a_0, a_1, ..., a_d)`` in ascending
                order.  May be complex.
            signal_value: Real signal value ``x`` in ``[-1, 1]``.
            signal_angle: Real angle ``theta`` satisfying ``x = cos(theta)``.
            tolerance: Target mean-squared fitting loss on the Chebyshev-node grid.
            max_iterations: Maximum number of internal BFGS iterations.
            gradient_tolerance: Infinity-norm gradient stopping tolerance.

        Returns:
            Configured QSP operator with solved phases and bound signal.

        """
        return cls.from_complex_chebyshev_coefficients(
            polynomial_to_chebyshev(coefficients),
            signal_value=signal_value,
            signal_angle=signal_angle,
            tolerance=tolerance,
            max_iterations=max_iterations,
            gradient_tolerance=gradient_tolerance,
        )


def _normalize_phases(phases: Sequence[float] | np.ndarray) -> tuple[float, ...]:
    r"""Validate and return a QSP phase sequence in normalized form."""
    if isinstance(phases, np.ndarray):
        flattened = phases.reshape(-1).tolist()
    elif isinstance(phases, Sequence) and not isinstance(phases, (str, bytes)):
        flattened = list(phases)
    else:
        raise TypeError("phases must be a one-dimensional real sequence")

    if not flattened:
        raise ValueError("phases must not be empty")

    normalized: list[float] = []
    for value in flattened:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("phases must contain real finite scalars")
        scalar = float(value)
        if not math.isfinite(scalar):
            raise TypeError("phases must contain real finite scalars")
        normalized.append(scalar)
    return tuple(normalized)


def _resolve_signal(
    *,
    signal_value: float | None,
    signal_angle: float | None,
) -> tuple[float, float]:
    r"""Resolve ``signal_value`` / ``signal_angle`` into ``(x, theta)``."""
    if (signal_value is None) == (signal_angle is None):
        raise ValueError(
            "exactly one of signal_value and signal_angle must be provided"
        )

    if signal_value is not None:
        if isinstance(signal_value, bool) or not isinstance(signal_value, Real):
            raise TypeError("signal_value must be a real finite scalar")
        x_value = float(signal_value)
        if not math.isfinite(x_value):
            raise TypeError("signal_value must be a real finite scalar")
        if not -1.0 <= x_value <= 1.0:
            raise ValueError("signal_value must satisfy -1 <= signal_value <= 1")
        return x_value, float(math.acos(x_value))

    assert signal_angle is not None
    if isinstance(signal_angle, bool) or not isinstance(signal_angle, Real):
        raise TypeError("signal_angle must be a real finite scalar")
    theta = float(signal_angle)
    if not math.isfinite(theta):
        raise TypeError("signal_angle must be a real finite scalar")
    return float(math.cos(theta)), theta
