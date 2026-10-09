"""Single-qubit quantum signal processing circuit construction."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

import numpy as np
from cqlib.circuit.gates import StandardGate

from cqlib import Circuit

from ._phase_solver import solve_qsp_phases


class QuantumSignalProcessing:
    """Construct the single-qubit SU(2) core of a QSP sequence.

    This component builds the one-qubit circuit for

    .. math::

        U(x, \\Phi) = e^{i\\phi_0 Z} W(x) e^{i\\phi_1 Z} \\cdots W(x) e^{i\\phi_d Z},

    where

    .. math::

        W(x) =
        \\begin{pmatrix}
            x & i\\sqrt{1 - x^2} \\\\
            i\\sqrt{1 - x^2} & x
        \\end{pmatrix}
        = e^{i\\theta X}, \\quad x = \\cos\\theta.

    The implementation realizes the exact same SU(2) unitary as an alternating
    sequence of ``RZ`` and ``RX`` gates on one qubit:

    .. math::

        e^{i\\phi Z} = R_Z(-2\\phi), \\quad W(x) = R_X(-2\\theta).

    This class intentionally covers the single-qubit signal-processing core
    only. It does not build a block-encoding or qubitization iterate.

    Parameters
    ----------
    phases
        Full QSP phase sequence ``(phi_0, ..., phi_d)``.
    signal_value
        Real signal value ``x`` in ``[-1, 1]``. Exactly one of
        ``signal_value`` and ``signal_angle`` must be provided.
    signal_angle
        Real angle ``theta`` satisfying ``x = cos(theta)``. Exactly one of
        ``signal_value`` and ``signal_angle`` must be provided.

    Raises
    ------
    TypeError
        If the phase sequence or signal arguments have invalid types.
    ValueError
        If the phase sequence is empty, if the signal specification is
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
        """Return the immutable full QSP phase sequence."""
        return self._phases

    @property
    def signal_value(self) -> float:
        """Return the configured signal value ``x = cos(theta)``."""
        return self._signal_value

    @property
    def signal_angle(self) -> float:
        """Return the configured signal angle ``theta`` in radians."""
        return self._signal_angle

    @property
    def num_qubits(self) -> int:
        """Return the width of the local QSP core circuit."""
        return 1

    def build_circuit(self) -> Circuit:
        """Build and return a fresh one-qubit QSP circuit."""
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
        """Return QSP phases for a real Chebyshev polynomial.

        Parameters
        ----------
        coefficients
            Chebyshev-basis coefficients in ascending order.
        tolerance
            Target mean-squared fitting loss on the Chebyshev-node grid.
        max_iterations
            Maximum number of internal BFGS iterations.
        gradient_tolerance
            Infinity-norm gradient stopping tolerance.

        Returns
        -------
        numpy.ndarray
            Full symmetric phase sequence.

        Raises
        ------
        TypeError
            If the coefficient sequence or solver controls have invalid types.
        ValueError
            If the polynomial violates the standard real-QSP constraints.
        RuntimeError
            If the internal solver does not converge.
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
        """Solve a real-QSP instance and bind it to one signal value.

        Parameters
        ----------
        coefficients
            Chebyshev-basis coefficients in ascending order.
        signal_value
            Real signal value ``x`` in ``[-1, 1]``.
        signal_angle
            Real signal angle ``theta`` satisfying ``x = cos(theta)``.
        tolerance
            Target mean-squared fitting loss on the Chebyshev-node grid.
        max_iterations
            Maximum number of internal BFGS iterations.
        gradient_tolerance
            Infinity-norm gradient stopping tolerance.

        Returns
        -------
        QuantumSignalProcessing
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


def _normalize_phases(phases: Sequence[float] | np.ndarray) -> tuple[float, ...]:
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
